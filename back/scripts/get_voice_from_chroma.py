from chromadb import PersistentClient
import os
import sqlite3

# Chromaのパスを指定
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("video-transcripts")

# 全文取得
docs = collection.get(include=["documents"])
all_voice = "\n".join(docs["documents"])

# SQLite DBを準備
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/qg.sqlite3"))
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS qg (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT,
    voice TEXT,
    answer TEXT,
    question TEXT
)
''')

# 既に登録済みか確認して、なければ追加
course = "情報基盤入門"
c.execute("SELECT id FROM qg WHERE course = ?", (course,))
if not c.fetchone():
    c.execute("INSERT INTO qg (course, voice) VALUES (?, ?)", (course, all_voice))
    print("Chromaから取得したvoiceを保存しました。")
else:
    print("すでにvoiceは保存済みです。")

conn.commit()
conn.close()

