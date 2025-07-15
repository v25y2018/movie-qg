import os
import sqlite3
import requests
import json

# 設定
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2"

# 質問生成用プロンプト
def make_question_prompt(course, section, voice_chunk, explain):
    return f"""
あなたは{course}の「{section}」を担当する先生です。
講義の文字起こし結果である(voice_chunk)をもとに、以下の説明文(explain)に対応する学生の理解を確認するための質問を1つ作ってください。
質問は説明文に対応した内容で、答えをから見つけられるようにしてください。

出力形式（この形だけを出力）:
{{"question": "質問文"}}

--- voice（講義内容の文字起こし） ---
{voice_chunk}

--- 説明文（explain） ---
{explain}
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

# メイン処理
def generate_questions():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # questionが未設定のレコードを取得
    cur.execute("""
        SELECT qgid, course, section, voice_chunk, explain
        FROM qg
        WHERE question IS NULL OR question = ''
    """)
    rows = cur.fetchall()
    print(f"未処理レコード数: {len(rows)}")

    for row in rows:
        qgid, course, section, voice_chunk, explain = row
        print(f"[{qgid}] 質問生成中...")

        try:
            prompt = make_question_prompt(course, section, voice_chunk, explain)
            response = call_ollama(prompt)
            data = json.loads(response)
            question = data.get("question", "").strip()

            if question:
                cur.execute("UPDATE qg SET question = ? WHERE qgid = ?", (question, qgid))
                conn.commit()
                print(f"[{qgid}] 質問保存完了")
            else:
                print(f"[{qgid}] 質問が生成されませんでした")

        except Exception as e:
            print(f"[{qgid}] エラー: {e}")

    conn.close()

if __name__ == "__main__":
    generate_questions()
