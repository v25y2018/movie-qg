import sqlite3
import os

# DBパス
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))

def list_qg(video_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 対象videoIdで取得（explainとquestionとpriority）
    cur.execute("""
        SELECT qgid, explain, question, priority FROM qg
        WHERE videoId = ?
        ORDER BY createdat ASC
    """, (video_id,))
    rows = cur.fetchall()

    if not rows:
        print(f"videoId = {video_id} に該当するデータはありません。")
        return

    print(f"videoId = {video_id} の説明文と質問一覧（重要度付き）\n")
    for i, (qgid, explain, question, priority) in enumerate(rows, 1):
        print(f"【{i:02d}】QGID: {qgid}")
        print(f"説明文 : {explain}")
        print(f"質問文 : {question if question else '（未生成）'}")
        print(f"重要度 : {priority if priority is not None else '（未設定）'}")
        print("-" * 40)

    conn.close()

if __name__ == "__main__":
    # 任意のvideoIdに変更してください
    video_id = "2"
    list_qg(video_id)
