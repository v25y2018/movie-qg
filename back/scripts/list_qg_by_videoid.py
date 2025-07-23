import sqlite3
import os

# DBパス
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))

def list_qg(video_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # videoIdに一致するレコードを取得
    cur.execute("""
        SELECT qgid, videoId, question, explain, model, createdat
        FROM qg
        WHERE videoId = ?
        ORDER BY createdat ASC
    """, (video_id,))
    rows = cur.fetchall()

    if not rows:
        print(f"videoId = {video_id} に該当するデータはありません。")
        return

    print(f"videoId = {video_id} のQ&A一覧\n")
    for i, (qgid, vid, question, explain, model, createdat) in enumerate(rows, 1):
        print(f"【{i:02d}】QGID: {qgid}")
        print(f"movieId : {vid}")
        print(f"質問文   : {question}")
        print(f"解答文   : {explain}")
        print(f"生成モデル: {model}")
        print(f"生成日時 : {createdat}")
        print("-" * 40)

    conn.close()

if __name__ == "__main__":
    # 任意のvideoIdを指定
    video_id = "2"
    list_qg(video_id)
