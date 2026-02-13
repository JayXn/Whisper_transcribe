import os
import gc
import argparse
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
import torch
import shutil
import sys
import traceback
import subprocess
import json

# =======================
# 參數設定
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="暫存音檔目錄 (temp_audio)")
parser.add_argument("--base_name", required=True, help="原影片檔名（程式會取 stem）")
parser.add_argument("--generate_txt", type=str, default="true", help="是否輸出 TXT")
parser.add_argument("--generate_srt", type=str, default="true", help="是否輸出 SRT")
parser.add_argument("--cleanup_temp", type=str, default="true", help="是否刪除暫存音檔")
parser.add_argument("--language", type=str, default="zh", help="語言代碼，例如 zh、en")
args = parser.parse_args()

TEMP_DIR = Path(args.input_dir)
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_NAME = Path(args.base_name).stem

MODEL_PATH = Path("/home/jay/whisper/models/medium")  # 模型路徑
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def str2bool(s):
    return str(s).lower() in ("1", "true", "yes", "y")

generate_txt = str2bool(args.generate_txt)
generate_srt = str2bool(args.generate_srt)
cleanup_temp = str2bool(args.cleanup_temp)
language = args.language

# VAD 設定
USE_VAD = True
VAD_PARAMS = dict(min_silence_duration_ms=100)

# =======================
# 工具函數
# =======================
def format_time(t: float) -> str:
    """將秒數轉為 SRT 時間格式 HH:MM:SS,mmm"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms >= 1000:
        s += 1
        ms -= 1000
    if s >= 60:
        m += 1
        s -= 60
    if m >= 60:
        h += 1
        m -= 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def transcribe_one(model, audio_path):
    """使用 Whisper 模型轉寫單個音檔，回傳 segments 與 info"""
    if USE_VAD:
        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language=language,
                vad_filter=True,
                vad_parameters=VAD_PARAMS
            )
        except TypeError:
            # 某些版本不支援 vad_parameters
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language=language,
                vad_filter=True
            )
    else:
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            language=language,
            vad_filter=False
        )
    return list(segments), info

def calculate_duration(segments, info):
    """計算音檔總時長，優先 info.duration"""
    total_secs = None
    if getattr(info, "duration", None):
        try:
            total_secs = float(info.duration)
        except:
            total_secs = None
    if total_secs is None:
        total_secs = sum(max(0.0, s.end - s.start) for s in segments)
    return total_secs or 0.0

def write_segments(segments, cumulative_time, final_txt_f, final_srt_f, segment_counter):
    """將 segments 寫入 TXT 與 SRT，返回更新後的 segment_counter"""
    for s in segments:
        start_adj = s.start + cumulative_time
        end_adj = s.end + cumulative_time

        if final_txt_f:
            final_txt_f.write(s.text.strip() + "\n")

        if final_srt_f:
            final_srt_f.write(f"{segment_counter}\n")
            final_srt_f.write(f"{format_time(start_adj)} --> {format_time(end_adj)}\n")
            final_srt_f.write(s.text.strip() + "\n\n")
            segment_counter += 1
    return segment_counter

def get_ffprobe_duration(audio_path):
    """使用 ffprobe 快速獲取音檔時長，不跑模型"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration", "-of", "json",
             str(audio_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        data = json.loads(result.stdout)
        duration = float(data["streams"][0]["duration"])
        return duration
    except Exception:
        return 0.0

# =======================
# 檢查資料
# =======================
if not TEMP_DIR.exists():
    print(f"找不到暫存目錄 {TEMP_DIR}", file=sys.stderr)
    sys.exit(1)

audio_files = sorted(TEMP_DIR.glob("part*.mp3"))
if not audio_files:
    print("沒有找到 part*.mp3 檔案", file=sys.stderr)
    sys.exit(1)

# =======================
# 載入模型
# =======================
print(f"開始載入 Whisper 模型: {MODEL_PATH} on {DEVICE} ...")
model = WhisperModel(str(MODEL_PATH), device=DEVICE, compute_type="float16")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# =======================
# 預掃描總長度（ffprobe）
# =======================
print("掃描所有分段長度（使用 ffprobe，不會跑模型）...")
durations = [get_ffprobe_duration(p) for p in audio_files]
total_video_secs = sum(durations)
print(f"發現 {len(audio_files)} 個分段，總長度 {total_video_secs:.1f} 秒")

# 使用簡單總進度條（每個音檔完成就更新）
global_pbar = tqdm(total=len(audio_files), unit="file", desc="影片總進度", ncols=100)

# =======================
# 準備輸出
# =======================
final_txt_f = open(TRANSCRIPTS_DIR / f"{BASE_NAME}.txt", "w", encoding="utf-8") if generate_txt else None
final_srt_f = open(TRANSCRIPTS_DIR / f"{BASE_NAME}.srt", "w", encoding="utf-8") if generate_srt else None

segment_counter = 1
cumulative_time = 0.0  # 用於 SRT 累積時間

# =======================
# 正式轉寫
# =======================
try:
    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"\n🎬 處理段落 ({idx}/{len(audio_files)}): {audio_path.name}")
        try:
            segments, info = transcribe_one(model, audio_path)
        except Exception as e:
            print(f"❌ 轉寫失敗: {audio_path.name}, {e}", file=sys.stderr)
            traceback.print_exc()
            continue

        total_secs = calculate_duration(segments, info)

        # 寫入 TXT / SRT
        segment_counter = write_segments(
            segments,
            cumulative_time,
            final_txt_f,
            final_srt_f,
            segment_counter
        )

        # 更新累積時間
        cumulative_time += total_secs

        # 更新總進度條：每完成一個 mp3 就更新
        global_pbar.update(1)

        # VRAM 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

finally:
    if final_txt_f:
        final_txt_f.close()
    if final_srt_f:
        final_srt_f.close()
    global_pbar.close()

# =======================
# 刪除暫存音檔（可選）
# =======================
if cleanup_temp:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

print("\n✅ 轉寫完成")
