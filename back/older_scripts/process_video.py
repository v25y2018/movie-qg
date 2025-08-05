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

# ファイル存在確認と補完
if not os.path.exists(video_path):
    maybe_path = os.path.join(os.path.dirname(__file__), "../src/uploads", os.path.basename(video_path))
    if os.path.exists(maybe_path):
        video_path = maybe_path
    else:
        print(f"[ERROR] 動画ファイルが見つかりません: {video_path}")
        sys.exit(1)
else:
    print(f"[DEBUG] 読み込みファイルパス: {video_path}")

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

ocr_list_for_prompt = []
texts = []
metadatas = []
chunk_text = ""
chunk_start = None

# チャンク処理1回目
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

# OCR処理（プロンプト作成用）
for i, meta in enumerate(metadatas):
    mid_time = (meta["start"] + meta["end"]) / 2
    try:
        result = subprocess.run([
            "ffmpeg", "-ss", str(mid_time), "-i", video_path,
            "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        image = Image.open(io.BytesIO(result.stdout))
        ocr_text = pytesseract.image_to_string(image, lang="jpn").strip()
        metadatas[i]["ocr"] = ocr_text
        ocr_list_for_prompt.append(ocr_text)
    except subprocess.TimeoutExpired:
        print(f"[WARN] OCR用ffmpegタイムアウト: {mid_time}秒地点")
        metadatas[i]["ocr"] = ""
    except Exception:
        metadatas[i]["ocr"] = ""

# OCRを initial_prompt にして再度文字起こし
ocr_prompt = " ".join(set(ocr_list_for_prompt)).strip()[:200]
result = whisper_model.transcribe(video_path, initial_prompt=ocr_prompt, verbose=False)
segments = result.get("segments", [])

texts = []
metadatas = []
chunk_text = ""
chunk_start = None

# チャンク処理2回目
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
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=5)
        image = Image.open(io.BytesIO(result.stdout))
        ocr_text = pytesseract.image_to_string(image, lang="jpn").strip()
        metadatas[i]["ocr"] = ocr_text
    except subprocess.TimeoutExpired:
        print(f"[WARN] OCR用ffmpegタイムアウト: {mid_time}秒地点（再処理）")
        metadatas[i]["ocr"] = ""
    except Exception:
        metadatas[i]["ocr"] = ""

# ベクトル化と保存
embedding_model = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
embeddings = embedding_model.encode(texts)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("video-transcripts")
ids = [f"{video_id}-chunk{i}" for i in range(len(texts))]
collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

print("動画処理とChroma保存完了")