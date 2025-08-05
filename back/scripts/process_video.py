import os
import sys
import io
import uuid
import shutil
import subprocess
import MeCab
from datetime import datetime
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import chromadb
from mlx_whisper import transcribe

# ====== デバッグ用フラグ ======
DEBUG = False
SKIP_OCR = False
SKIP_ASR = False
SKIP_SAVE = False
VERBOSE_TRANSCRIBE = False


# 引数
video_path = sys.argv[1]
video_name = sys.argv[2]
course = sys.argv[3]
section = sys.argv[4]
video_id = sys.argv[5]

# 各種設定
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/chroma_db"))
TMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/tmp_movies"))
TMP_DIR = os.path.join(TMP_ROOT, str(uuid.uuid4()))
INTERVAL = 5
SIM_THRESHOLD = 0.85
createdat = datetime.utcnow().isoformat()

# モデルとツール
embedder = SentenceTransformer("cl-nagoya/ruri-small", trust_remote_code=True)
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
        image = Image.open(io.BytesIO(result.stdout)).convert("L")
        image = image.point(lambda x: 0 if x < 128 else 255)  # 二値化
        text = pytesseract.image_to_string(image, lang="jpn").strip()
        if not text:
            return ""

        # 除外対象の表層形（助詞や不要語）
        stop_words = {"に", "は", "を", "で", "の", "と", "が", "や", "など", "そして", "です", "ます", "から", "より", "まで", "へ", "ね", "よ"}

        keywords = []
        node = tagger.parseToNode(text)
        while node:
            features = node.feature.split(",")
            surface = node.surface.strip()
            # 名詞かつ除外語でないものを追加
            if "名詞" in features[0] and surface and surface not in stop_words:
                keywords.append(surface)
            node = node.next

        return " ".join(keywords)

    except Exception as e:
        print(f"[ERROR] OCR/処理失敗: {e}")
        return ""


# スライド境界の検出
def detect_slide_boundaries(duration):
    ocr_texts = []
    embeddings = []
    boundaries = [0.0]

    for t in range(0, int(duration), INTERVAL):
        if SKIP_OCR:
            continue
        keywords = extract_keywords_from_frame(video_path, t)
        if DEBUG:
            print(f"[DEBUG] OCR抽出中: {t}s -> {keywords}")
        if not keywords:
            continue
        ocr_texts.append((t, keywords))
        emb = embedder.encode(keywords, convert_to_tensor=True)
        embeddings.append(emb)

        if len(embeddings) >= 2:
            sim = util.cos_sim(embeddings[-1], embeddings[-2]).item()
            if DEBUG:
                print(f"[DEBUG] 類似度: {sim:.3f}")
            if sim < SIM_THRESHOLD:
                boundaries.append(float(t))

    boundaries.append(duration)
    return ocr_texts, boundaries

# 動画を一時ファイルとして保存
def extract_segment(video_path, start, end, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
        "-i", video_path, "-c:v", "copy", "-c:a", "copy", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Whisper処理
def transcribe_with_mlx_whisper(audio_path, ocr_text=""):
    return transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        language="ja",
        task="transcribe",
        initial_prompt=ocr_text,
        verbose=VERBOSE_TRANSCRIBE,
        condition_on_previous_text=False,
        carry_initial_prompt=True
    )

# メイン処理
def process_and_store(ocr_texts, boundaries):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("video-transcripts")
    documents, metadatas, ids = [], [], []

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        segment_path = os.path.join(TMP_DIR, f"slide-{i}.mp4")

        if DEBUG:
            print(f"[DEBUG] スライド{i}: {start}〜{end} 秒 切り出し中...")
        extract_segment(video_path, start, end, segment_path)

        ocr = ""
        for t, txt in reversed(ocr_texts):
            if t <= (start + end) / 2:
                ocr = txt
                break

        try:
            if SKIP_ASR:
                voice = "[SKIPPED ASR]"
            else:
                if DEBUG:
                    print(f"[DEBUG] 音声認識: slide-{i}.mp4 + prompt='{ocr[:30]}...'")
                result = transcribe_with_mlx_whisper(segment_path, ocr)
                voice = result.get("text", "").strip()
            if not voice:
                continue
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
        except Exception as e:
            print(f"[ERROR] スライド{i}の音声認識に失敗: {e}")

    if not SKIP_SAVE and documents:
        if DEBUG:
            print(f"[DEBUG] {len(documents)}件のドキュメントをChromaに保存中...")
        embeddings = embedder.encode(documents)
        collection.add(documents=documents, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)

    print("スライド分割・音声認識・Chroma保存が完了しました")

# 実行部分
if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    try:
        duration = get_duration(video_path)
        if DEBUG:
            print(f"[DEBUG] 動画長：{duration:.2f} 秒")
        ocr_texts, boundaries = detect_slide_boundaries(duration)
        process_and_store(ocr_texts, boundaries)
    finally:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)
