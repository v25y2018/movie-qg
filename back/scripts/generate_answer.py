import os
import sqlite3
import requests
import json
from datetime import datetime
from chromadb import PersistentClient

# 環境設定
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2"  # 必要に応じて変更可

# Chromaからvoice取得
def get_voice_from_chroma(video_id):
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")
    results = collection.get(where={"video_id": video_id})
    return " ".join(results["documents"])

# voiceを指定文字数で分割
def split_voice(voice, chunk_size=2000):
    return [voice[i:i + chunk_size] for i in range(0, len(voice), chunk_size)]

# プロンプト作成
def make_prompt(course, voice_chunk):
    return f"""
あなたは{course}の先生です。以下の講義文字起こしであるvoice_chunkを考慮してテキスト内の表現をそのまま使って、説明文の形式に整えてください。
説明文は複数になっても問題ありません。ただし、出力形式(JSON)を守ってください。

出力形式:
[
  {{"answer": "説明文1"}},
  {{"answer": "説明文2"}},
  ...
]

※注意:
- JSON形式の配列だけを出力してください
- 説明文以外の解説や補足は不要です
--- 講義文字起こし ---
{voice_chunk}
"""

# Ollama呼び出し
def call_ollama(prompt):
    res = requests.post(OLLAMA_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    res.raise_for_status()
    return res.json()["response"]

# SQLiteに保存
def save_explains(video_id, course, section, voice_chunk, explains, chunk_index):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # テーブルがなければ作成
    cur.execute('''
        CREATE TABLE IF NOT EXISTS qg (
            qgid TEXT PRIMARY KEY,
            videoId TEXT,
            voice_chunk TEXT,
            explain TEXT,
            question TEXT,
            course TEXT,
            section TEXT,
            chunk_index INTEGER,
            createdat TEXT
        )
    ''')

    now = datetime.utcnow()
    createdat = now.isoformat()

    for i, explain in enumerate(explains):
        qgid = now.strftime("%Y%m%d%H%M%S") + f"{chunk_index:02d}{i:02d}"
        cur.execute("""
            INSERT INTO qg (qgid, videoId, voice_chunk, explain, question, course, section, chunk_index, createdat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (qgid, video_id, voice_chunk, explain, "", course, section, chunk_index, createdat))

    conn.commit()
    conn.close()
    print(f"{len(explains)}件の説明文を保存しました。")

# メイン処理
def main():
    video_id = "202509999888"  # ← 必要に応じて変更
    course = "大分大学入門"
    section = "セキュリティ"

    voice = get_voice_from_chroma(video_id)
    if not voice:
        print("voiceが見つかりません")
        return

    chunks = split_voice(voice, chunk_size=2000)
    print(f"チャンク数: {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        print(f"[{idx+1}/{len(chunks)}] 説明文生成中...")
        prompt = make_prompt(course, chunk)

        try:
            response = call_ollama(prompt)
            explains_json = json.loads(response)
            explains = [item["answer"].strip() for item in explains_json if item.get("answer")]
            save_explains(video_id, course, section, chunk, explains, idx)
        except Exception as e:
            print(f"[{idx+1}] 生成エラー:", e)
            print("Ollama応答:", response[:500] if 'response' in locals() else "(取得不可)")

if __name__ == "__main__":
    main()
