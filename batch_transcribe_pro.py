import os
import gc
import argparse
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
import torch
import re
import shutil
import sys
import traceback

# =======================
# 參數設定
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="暫存音檔目錄 (temp_audio)")
parser.add_argument("--base_name", required=True, help="原影片檔名（含副檔名均可，程式會取 stem）")
parser.add_argument("--generate_txt", type=str, default="true", help="是否輸出 TXT，True/False")
parser.add_argument("--generate_srt", type=str, default="true", help="是否輸出 SRT，True/False")
parser.add_argument("--cleanup_temp", type=str, default="true", help="是否刪除暫存音檔，True/False")
parser.add_argument("--language", type=str, default="zh", help="指定語言，預設 zh")
args = parser.parse_args()

TEMP_DIR = Path(args.input_dir)
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_NAME = Path(args.base_name).stem

MODEL_PATH = Path("/home/jay/whisper/models/medium")  # 請確認模型路徑正確
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

generate_txt = args.generate_txt.lower() == "true"
generate_srt = args.generate_srt.lower() == "true"
cleanup_temp = args.cleanup_temp.lower() == "true"

USE_VAD = False  # 先不啟用 VAD 參數以避免版本相容問題（你可以改回 True，如果確定 faster-whisper 支援）

# =======================
# 工具函數
# =======================
def format_time(t):
    """把秒數（float）格式化為 SRT 時間格式 HH:MM:SS,mmm"""
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    # 保險處理 ms = 1000 的情況
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

# =======================
# 檢查資料夾
# =======================
if not TEMP_DIR.exists() or not TEMP_DIR.is_dir():
    print(f"⚠️ 找不到暫存音訊目錄: {TEMP_DIR}", file=sys.stderr)
    sys.exit(1)

audio_files = sorted(TEMP_DIR.glob("part*.mp3"))
if not audio_files:
    print("⚠️ 沒有找到任何 part*.mp3 檔案，請先用 ffmpeg 切片到 temp_audio/。", file=sys.stderr)
    sys.exit(1)

# =======================
# 載入模型
# =======================
print(f"Loading Whisper model on {DEVICE}...")
try:
    model = WhisperModel(str(MODEL_PATH), device=DEVICE, compute_type="float16")
except Exception as e:
    print("❌ 模型載入失敗:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# =======================
# 準備最終輸出（stream writing）
# =======================
final_txt_path = TRANSCRIPTS_DIR / f"{BASE_NAME}.txt"
final_srt_path = TRANSCRIPTS_DIR / f"{BASE_NAME}.srt"

# 若已存在，覆蓋（或你可改為先備份）
if generate_txt and final_txt_path.exists():
    final_txt_path.unlink()
if generate_srt and final_srt_path.exists():
    final_srt_path.unlink()

final_txt_f = open(final_txt_path, "w", encoding="utf-8") if generate_txt else None
final_srt_f = open(final_srt_path, "w", encoding="utf-8") if generate_srt else None

segment_counter = 1
cumulative_time = 0.0  # 每個片段的時間偏移（秒）

# =======================
# 逐段轉寫並直接寫入最終檔（不建立 per-part 暫存檔）
# =======================
try:
    for idx, audio_path in enumerate(audio_files):
        print(f"\n🎬 處理段落 ({idx}): {audio_path.name}")

        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
                language=args.language,
                vad_filter=USE_VAD
            )
        except Exception as e:
            # 單段失敗 -> 記錄並繼續下一段（避免整個流程中斷）
            print(f"❌ 轉寫段落失敗 {audio_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            # 嘗試釋放資源再繼續
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            # 把 cumulative_time 加上該檔實際長度（若 info 未回傳，使用 0）
            try:
                cumulative_time += info.duration
            except:
                pass
            continue

        # 段內進度條（以該 audio 的本地 time 更新）
        local_last = 0.0
        pbar = tqdm(total=info.duration, unit="sec", desc=f"段落進度 {audio_path.name}", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

        # segments 是 generator 或 list，逐個寫入最終檔
        for s in segments:
            # s.start / s.end 是相對於該片段的時間，需加上 cumulative_time
            start_adj = s.start + cumulative_time
            end_adj = s.end + cumulative_time

            # TXT: 每個 segment 的文字直接追加（換行）
            if final_txt_f:
                final_txt_f.write(s.text.strip() + "\n")

            # SRT: 使用全局計數器與調整後時間寫入
            if final_srt_f:
                final_srt_f.write(f"{segment_counter}\n")
                final_srt_f.write(f"{format_time(start_adj)} --> {format_time(end_adj)}\n")
                final_srt_f.write(s.text.strip() + "\n\n")
                segment_counter += 1

            # 更新進度條（以該段的相對時間差計算）
            pbar.update(max(0.0, s.end - local_last))
            local_last = s.end

        pbar.close()

        # 處理完此片段後，把片段總長加到 cumulative_time
        try:
            cumulative_time += info.duration
        except:
            # 若 info.duration 無效，保守不累加
            pass

        # 釋放段落資源與 VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 確保資料 flush 到硬碟
    if final_txt_f:
        final_txt_f.flush()
    if final_srt_f:
        final_srt_f.flush()

    print(f"\n✅ 轉寫完成：輸出位置 -> {TRANSCRIPTS_DIR.resolve()}")

finally:
    # 關檔與清理（不會因未生成而錯誤）
    try:
        if final_txt_f:
            final_txt_f.close()
    except:
        pass
    try:
        if final_srt_f:
            final_srt_f.close()
    except:
        pass

# =======================
# 刪除暫存音檔（由 cleanup_temp 控制）
# =======================
if cleanup_temp:
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"🧹 已刪除暫存音檔 {TEMP_DIR}")
    except Exception as e:
        print(f"⚠️ 刪除暫存音檔失敗: {e}", file=sys.stderr)
