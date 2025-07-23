from chromadb import PersistentClient

CHROMA_PATH = "../src/chroma_db"  # 環境に応じて修正

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("video-transcripts")
results = collection.get(include=["metadatas"])

# video_id を一覧で表示
ids = set(m["video_id"] for m in results["metadatas"] if "video_id" in m)
print("登録されている video_id:", ids)

