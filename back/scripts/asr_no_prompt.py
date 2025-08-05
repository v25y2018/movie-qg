import os
import sys
import subprocess
import tempfile
from mlx_whisper import transcribe

# 引数：動画ファイルパス
video_path = sys.argv[1]  # 例: uploads/lecture.mp4

# 動画から音声を一時ファイルに抽出
def extract_audio(video_path, output_wav_path):
    command = [
        "ffmpeg",
        "-y",  # 上書き許可
        "-i", video_path,
        "-vn",  # 映像無視
        "-acodec", "pcm_s16le",  # リニアPCM
        "-ar", "16000",  # サンプリングレート16kHz（Whisperの推奨）
        "-ac", "1",  # モノラル
        output_wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    # 一時ファイルとして.wav作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        print("[INFO] 音声抽出中...")
        extract_audio(video_path, tmp_audio_path)

        print("[INFO] 音声認識中（イニシャルプロンプトなし）...")
        result = transcribe(
            tmp_audio_path,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="ja",
            task="transcribe",
            verbose=True,
            condition_on_previous_text=False,
            carry_initial_prompt=False
        )

        print("\n--- 認識結果 ---\n")
        print(result.get("text", "").strip())

    except Exception as e:
        print(f"[ERROR] 処理中に例外が発生: {e}")

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

if __name__ == "__main__":
    main()

