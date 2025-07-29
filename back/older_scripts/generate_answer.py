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
MODEL_NAME = "qwen3:32b"  # 必要に応じて変更可

# Chromaからvoice取得
def get_voice_from_chroma(video_id):
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")
    results = collection.get(where={"video_id": video_id})
    return " ".join(results["documents"])

# voiceを指定文字数で分割
def split_voice(voice, chunk_size=22000):
    return [voice[i:i + chunk_size] for i in range(0, len(voice), chunk_size)]

# プロンプト作成
def make_prompt(course, voice_chunk):
    return f"""
あなたは{course}の先生です。以下の講義文字起こしである『voice_chunk』を考慮してテキスト内の表現をそのまま使って、振り返りテスト用の質問と回答を複数生成してください。
質問は、質問単体で見ても学生がわかるように明瞭に、かつ簡潔に書いてください。

--- 講義文字起こし『voice_chunk』 ---
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

# SQLiteに保存（question + answer + priority 形式）
def save_qna(video_id, course, section, voice_chunk, qna_list, chunk_index):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute('''
        CREATE TABLE IF NOT EXISTS qg (
            qgid TEXT PRIMARY KEY,
            videoId TEXT,
            voice_chunk TEXT,
            explain TEXT,
            question TEXT,
            priority REAL,
            course TEXT,
            section TEXT,
            chunk_index INTEGER,
            createdat TEXT
        )
    ''')

    now = datetime.utcnow()
    createdat = now.isoformat()

    for i, item in enumerate(qna_list):
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        priority = float(item.get("priority", 0.0))
        if not question or not answer:
            continue
        qgid = now.strftime("%Y%m%d%H%M%S") + f"{chunk_index:02d}{i:02d}"
        cur.execute("""
            INSERT INTO qg (qgid, videoId, voice_chunk, explain, question, priority, course, section, chunk_index, createdat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (qgid, video_id, voice_chunk, answer, question, priority, course, section, chunk_index, createdat))

    conn.commit()
    conn.close()
    print(f"{len(qna_list)}件のQ&Aを保存しました。")

# メイン処理
def main():
    video_id = "Oita-01"  # ← 必要に応じて変更
    course = "大分大学入門"
    section = "第一回"

    voice = get_voice_from_chroma(video_id)
    if not voice:
        print("voiceが見つかりません")
        return

    chunks = split_voice(voice, chunk_size=22000)
    print(f"チャンク数: {len(chunks)}")

    for idx, chunk in enumerate(chunks):
        print(f"[{idx+1}/{len(chunks)}] Q&A生成中...")
        prompt = make_prompt(course, chunk)

        try:
            response = call_ollama(prompt)
            qna_list = json.loads(response)
            save_qna(video_id, course, section, chunk, qna_list, idx)
        except Exception as e:
            print(f"[{idx+1}] 生成エラー: {e}")
            print("Ollama応答:", response[:5000] if 'response' in locals() else "(取得不可)")

if __name__ == "__main__":
    main()
