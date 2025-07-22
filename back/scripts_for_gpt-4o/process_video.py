#!/usr/bin/env python3

import sys
import os
import io
import subprocess
import whisper
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
import chromadb
from datetime import datetime

# 引数の取得
video_path = sys.argv[1]    # 例: uploads/movie.mp4
video_name = sys.argv[2]    # 例: movie
course = sys.argv[3]        # コース名
section = sys.argv[4]       # セクション名
video_id = sys.argv[5]      # 動画ID（将来自動生成予定）

# 現在のUTC時刻（ISO8601形式）を記録
createdat = datetime.utcnow().isoformat()

# Chromaの保存先パス
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
CHUNK_SIZE = 200

# Whisperモデルをロード
whisper_model = whisper.load_model("small")

# 初回の仮チャンク（OCR用プロンプト作成）
result = whisper_model.transcribe(video_path, verbose=False)
segments = result.get("segments", [])
if not segments:
    print("セグメントが取得できませんでした")
    sys.exit(1)

# 仮OCRプロンプト作成のために使用
ocr_list_for_prompt = []

# チャンク処理1回目（OCRプロンプト作成用）
texts = []
metadatas = []
chunk_text = ""
chunk_start = None

for seg in segments:
    text = seg.get("text", "").strip()
    if not text:
        continue
    if chunk_text == "":
        chunk_start = seg["start"]
    chunk_text += text + " "
    chunk_end = seg["end"]

    if len(chunk_text) >= CHUNK_SIZE:
        texts.append(chunk_text.strip())
        metadatas.append({
            "video": video_name,
            "video_id": video_id,
            "course": course,
            "section": section,
            "start": float(chunk_start),
            "end": float(chunk_end),
            "createdat": createdat
        })
        chunk_text = ""
        chunk_start = None

# 最後のチャンクが残っていれば追加
if chunk_text.strip():
    texts.append(chunk_text.strip())
    metadatas.append({
        "video": video_name,
        "video_id": video_id,
        "course": course,
        "section": section,
        "start": float(chunk_start),
        "end": float(chunk_end),
        "createdat": createdat
    })

# OCR処理（プロンプト用）
for i, meta in enumerate(metadatas):
    mid_time = (meta["start"] + meta["end"]) / 2
    try:
        result = subprocess.run([
            "ffmpeg", "-ss", str(mid_time), "-i", video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        image = Image.open(io.BytesIO(result.stdout))
        ocr_text = pytesseract.image_to_string(image, lang="jpn").strip()
        metadatas[i]["ocr"] = ocr_text
        ocr_list_for_prompt.append(ocr_text)
    except Exception:
        metadatas[i]["ocr"] = ""

# OCRを initial_prompt として再度音声認識
ocr_prompt = " ".join(set(ocr_list_for_prompt)).strip()[:200]
result = whisper_model.transcribe(video_path, initial_prompt=ocr_prompt, verbose=False)
segments = result.get("segments", [])

# チャンク処理2回目（最終データ用）
texts = []
metadatas = []
chunk_text = ""
chunk_start = None

for seg in segments:
    text = seg.get("text", "").strip()
    if not text:
        continue
    if chunk_text == "":
        chunk_start = seg["start"]
    chunk_text += text + " "
    chunk_end = seg["end"]

    if len(chunk_text) >= CHUNK_SIZE:
        texts.append(chunk_text.strip())
        metadatas.append({
            "video": video_name,
            "video_id": video_id,
            "course": course,
            "section": section,
            "start": float(chunk_start),
            "end": float(chunk_end),
            "createdat": createdat
        })
        chunk_text = ""
        chunk_start = None

# 最後のチャンク
if chunk_text.strip():
    texts.append(chunk_text.strip())
    metadatas.append({
        "video": video_name,
        "video_id": video_id,
        "course": course,
        "section": section,
        "start": float(chunk_start),
        "end": float(chunk_end),
        "createdat": createdat
    })

# OCR再処理（再チャンクに対して）
for i, meta in enumerate(metadatas):
    mid_time = (meta["start"] + meta["end"]) / 2
    try:
        result = subprocess.run([
            "ffmpeg", "-ss", str(mid_time), "-i", video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        image = Image.open(io.BytesIO(result.stdout))
        ocr_text = pytesseract.image_to_string(image, lang="jpn").strip()
        metadatas[i]["ocr"] = ocr_text
    except Exception:
        metadatas[i]["ocr"] = ""

# ベクトル化（日本語対応モデル）
embedding_model = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
embeddings = embedding_model.encode(texts)

# ChromaDB に保存
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("video-transcripts")
ids = [f"{video_id}-chunk{i}" for i in range(len(texts))]
collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

print("動画処理とChroma保存完了")

