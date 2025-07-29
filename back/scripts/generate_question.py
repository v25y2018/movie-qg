import os
import re
import sqlite3
import requests
import json
from datetime import datetime
from chromadb import PersistentClient

# 環境設定
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3:32b"  # 使用するモデル名

# Chromaからセグメント単位で取得（スライド単位）
def get_slide_segments(video_id):
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")
    results = collection.get(where={"video_id": video_id})
    return zip(results["documents"], results["metadatas"])

# プロンプト作成（OCR結果も含める）
def make_prompt(course, ocr, voice):
    return f"""
あなたは{course}の講師です。以下のスライドの文字起こし{voice}と、スライド上のキーワード{ocr}に基づき、
学生の理解度を確認する振り返りテストの質問と解答を複数生成してください。なお、講義において重要でないと判断した場合は質問を生成しなくても構わない。

質問は、学生が単独で見ても意味が通るように明確で簡潔にしてください。
重要度（priority）は0.0〜10.0の範囲で適切に設定してください。

絶対にJSON形式のみで出力してください：

すべての項目（question, answer, priority）は必ず "（ダブルクォーテーション）で囲ってください。
JSONパーサーでエラーが出ないよう、文字列内の改行や記号にも注意してください。

出力形式---
[
  {{"question": "質問文1", "answer": "解答1", "priority": "重要度"}},
  {{"question": "質問文2", "answer": "解答2", "priority": "重要度"}}
]

"""

#繰り返しパースに失敗するとき
def save_failed_output(video_id, chunk_index, content):
    fail_dir = "failures"
    os.makedirs(fail_dir, exist_ok=True)
    filename = os.path.join(fail_dir, f"{video_id}_{chunk_index:02d}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"⚠️ LLM出力を {filename} に保存しました")

#JSONパース失敗時のみJSON取り出し
def extract_json_array(text):
    try:
        match = re.search(r'\[\s*{.*?}\s*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception as e:
        print(f"JSON抽出・整形に失敗: {e}")
    return None

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
def save_qna(video_id, course, section, voice, qna_list, slide_index):
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
        try:
            priority = float(item.get("priority", 0.0))
        except:
            priority = 0.0

        if not question or not answer:
            continue

        qgid = now.strftime("%Y%m%d%H%M%S") + f"{slide_index:02d}{i:02d}"
        cur.execute("""
            INSERT INTO qg (qgid, videoId, voice_chunk, explain, question, priority, course, section, chunk_index, createdat)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (qgid, video_id, voice, answer, question, priority, course, section, slide_index, createdat))

    conn.commit()
    conn.close()
    print(f"{len(qna_list)}件のQ&Aを保存しました。")

# メイン処理
def main():
    video_id = "Oita-01"  # 必要に応じて変更
    course = "大分大学入門"
    section = "1"

    slides = list(get_slide_segments(video_id))
    if not slides:
        print("Chromaに該当するセグメントが見つかりません")
        return

    for idx, (voice, meta) in enumerate(slides):
        ocr = meta.get("ocr", "")
        prompt = make_prompt(course, ocr, voice)
        print(f"[{idx+1}/{len(slides)}] Q&A生成中...")

        try:
            response = call_ollama(prompt)
            qna_list = json.loads(response)
            save_qna(video_id, course, section, voice, qna_list, idx)
        except Exception as e:
            print(f"[{idx+1}] 生成エラー: {e}")
            print("Ollama応答:", response[:5000] if 'response' in locals() else "(取得不可)")

if __name__ == "__main__":
    main()
