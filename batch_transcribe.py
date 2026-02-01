# batch_transcribe.py - 完整版 (Step 1~3)

import os
from pathlib import Path
from faster_whisper import WhisperModel
import torch

# =======================
# Step 1: 基本設定
# =======================

# 輸入輸出資料夾
VIDEOS_DIR = Path("/home/jay/whisper/videos")          # 放影片的資料夾
TRANSCRIPTS_DIR = Path("/home/jay/whisper/transcripts")  # 生成 TXT/SRT
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# 模型路徑 (CTranslate2 格式)
MODEL_PATH = Path("/home/jay/whisper/models/medium")
DEVICE = "cuda"     # GPU，可改 "cpu"

# SRT 開關
GENERATE_SRT = False  # True -> 生成 SRT，False -> 不生成

print("Step 1 完成：導入庫，設置路徑和模型參數")
print(f"VIDEOS_DIR: {VIDEOS_DIR}")
print(f"TRANSCRIPTS_DIR: {TRANSCRIPTS_DIR}")
print(f"DEVICE: {DEVICE}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"GENERATE_SRT: {GENERATE_SRT}")

# =======================
# Step 2: 加載模型並檢查 GPU
# =======================

print("\nStep 2: 開始加載模型...")
model = WhisperModel(
    str(MODEL_PATH),
    device=DEVICE,
    compute_type="float16"  # GPU 用 float16
)

# 檢查 GPU 可用性
if torch.cuda.is_available():
    print(f"GPU 可用，設備數量: {torch.cuda.device_count()}")
    print(f"當前設備: {torch.cuda.get_device_name(0)}")
else:
    print("GPU 不可用，將使用 CPU")

print("Step 2 完成：模型加載成功")

# =======================
# Step 3: 批量轉寫
# =======================

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi"]

def format_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

video_files = [f for f in VIDEOS_DIR.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

if not video_files:
    print("\nvideos/ 資料夾沒有可轉寫的視頻。")
else:
    print(f"\n找到 {len(video_files)} 個視頻，開始轉寫...")

for video_path in video_files:
    print(f"\n正在轉寫: {video_path.name}")

    # 轉寫
    segments, info = model.transcribe(str(video_path))

    # 生成 TXT
    txt_file = TRANSCRIPTS_DIR / f"{video_path.stem}.txt"
    with open(txt_file, "w", encoding="utf-8") as f_txt:
        for segment in segments:
            f_txt.write(segment.text + "\n")

    # 生成 SRT（如果需要）
    if GENERATE_SRT:
        srt_file = TRANSCRIPTS_DIR / f"{video_path.stem}.srt"
        with open(srt_file, "w", encoding="utf-8") as f_srt:
            for i, segment in enumerate(segments, start=1):
                start = segment.start
                end = segment.end
                f_srt.write(f"{i}\n")
                f_srt.write(f"{format_time(start)} --> {format_time(end)}\n")
                f_srt.write(f"{segment.text}\n\n")

    # 打印完成訊息
    if GENERATE_SRT:
        print(f"完成: TXT -> {txt_file.name}, SRT -> {srt_file.name}")
    else:
        print(f"完成: TXT -> {txt_file.name}")

print("\nStep 3 完成：所有視頻轉寫結束")
