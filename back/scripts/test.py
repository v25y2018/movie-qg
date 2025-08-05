from mlx_whisper import transcribe

result = transcribe(
    "example.wav",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
)

