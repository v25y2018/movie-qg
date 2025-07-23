from chromadb import PersistentClient
import sys
import os
from sentence_transformers import SentenceTransformer

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# クエリ文を引数から取得
query = sys.argv[1]

# ChromaのDBパス
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("video-transcripts")

print("ドキュメント数:", len(collection.get()["ids"]))

# 🔽 ここで ruri-small を使ってクエリをベクトル化
embedding_model = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
query_embedding = embedding_model.encode([query]).tolist()

# 🔽 query_embeddings を使う
result = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

# 🔽 結果の表示
print("検索結果:")
for i, doc in enumerate(result["documents"][0]):
    print(f"#{i+1}: {doc}")
    print(f"  OCR: {result['metadatas'][0][i].get('ocr', '')}")
    print(f"  start-end: {result['metadatas'][0][i]['start']} ～ {result['metadatas'][0][i]['end']}")

