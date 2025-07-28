import os
os.environ["TESSDATA_PREFIX"] = "/opt/homebrew/share/tessdata/"

import sys
import io
import subprocess
import MeCab
from datetime import datetime
import whisper
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import chromadb

# 引数
video_path = sys.argv[1]
video_name = sys.argv[2]
course = sys.argv[3]
section = sys.argv[4]
video_id = sys.argv[5]

# 各種設定
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
INTERVAL = 5  # 秒
SIM_THRESHOLD = 0.85
createdat = datetime.utcnow().isoformat()

# モデル
embedder = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
whisper_model = whisper.load_model("small")

# MeCab初期化（名詞抽出用）
tagger = MeCab.Tagger()

# 動画長取得
def get_duration(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(out.stdout.decode().strip())

# OCR＋画像前処理＋キーワード抽出
def extract_keywords_from_frame(video_path, time_sec):
    result = subprocess.run(
        ["ffmpeg", "-ss", str(time_sec), "-i", video_path, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    try:
        image = Image.open(io.BytesIO(result.stdout)).convert("L")  # グレースケール変換
        image = image.point(lambda x: 0 if x < 128 else 255)  # 二値化
        text = pytesseract.image_to_string(image, lang="jpn").strip()
        if not text:
            return ""
        # 名詞抽出
        keywords = []
        # 名詞抽出（tagger.parseToNodeは使用可能）
        node = tagger.parseToNode(text)
        while node:
            if "名詞" in node.feature:
                keyword = node.surface.strip()
                if keyword:
                    keywords.append(keyword)
            node = node.next

        return " ".join(keywords)
    except Exception as e:
        print(f"[ERROR] OCR/処理失敗: {e}")
        return ""

# スライド境界の検出（キーワードベース）
def detect_slide_boundaries(duration):
    ocr_texts = []
    embeddings = []
    boundaries = [0.0]

    for t in range(0, int(duration), INTERVAL):
        keywords = extract_keywords_from_frame(video_path, t)
        if not keywords:
            continue
        ocr_texts.append((t, keywords))
        emb = embedder.encode(keywords, convert_to_tensor=True)
        embeddings.append(emb)

        if len(embeddings) >= 2:
            sim = util.cos_sim(embeddings[-1], embeddings[-2]).item()
            if sim < SIM_THRESHOLD:
                boundaries.append(float(t))

    boundaries.append(duration)
    return ocr_texts, boundaries

def process_and_store(ocr_texts, boundaries):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")
    documents = []
    metadatas = []
    ids = []

    # 音声認識は最初に一度だけ実行
    result = whisper_model.transcribe(video_path, verbose=False, fp16=False, language="ja", initial_prompt="")
    all_segments = result["segments"]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]

        # セグメントをスライド範囲で抽出
        segments = [s for s in all_segments if start <= s["start"] < end]
        if not segments:
            continue

        voice = " ".join(s["text"].strip() for s in segments if s.get("text"))
        if not voice.strip():
            continue

        # OCR統合（中心時刻に最も近いキーワード）
        ocr = ""
        for t, txt in reversed(ocr_texts):
            if t <= (start + end) / 2:
                ocr = txt
                break

        documents.append(voice)
        metadatas.append({
            "video": video_name,
            "video_id": video_id,
            "course": course,
            "section": section,
            "start": start,
            "end": end,
            "createdat": createdat,
            "ocr": ocr
        })
        ids.append(f"{video_id}-slide{i}")

    if documents:
        embeddings = embedder.encode(documents)
        collection.add(documents=documents, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

    print("スライド分割・音声認識・Chroma保存が完了しました")


if __name__ == "__main__":
    duration = get_duration(video_path)
    ocr_texts, boundaries = detect_slide_boundaries(duration)
    process_and_store(ocr_texts, boundaries)