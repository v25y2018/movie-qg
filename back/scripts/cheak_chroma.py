import os
from chromadb import PersistentClient

# ChromaDBの保存パス（あなたの環境に合わせて修正）
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))

# 検索対象の video_id（必要に応じて変更）
TARGET_VIDEO_ID = "oita-ocr3"

def show_chroma_documents(video_id):
    # Chromaクライアントの初期化
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")

    # クエリ実行
    results = collection.get(where={"video_id": video_id})

    # 出力
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    for idx, (doc, meta, id_) in enumerate(zip(documents, metadatas, ids)):
        print(f"--- Document {idx + 1} ---")
        print(f"ID: {id_}")
        print(f"Metadata: {meta}")
        print(f"Document: {doc[:3000]}")  # 長すぎる場合は冒頭3000文字だけ
        print()

if __name__ == "__main__":
    show_chroma_documents(TARGET_VIDEO_ID)

