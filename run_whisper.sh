#!/bin/bash

# run_whisper.sh
# 一鍵完成 mp4 -> temp_audio -> batch_transcribe -> 最終 srt/txt

VIDEOS_DIR="videos"
SEGMENT_TIME=${1:-1200}      # 切片長度（秒）
GENERATE_TXT=${2:-false}     # 是否生成 TXT（傳 "true"/"false"）
GENERATE_SRT=${3:-true}      # 是否生成 SRT（傳 "true"/"false"）
CLEANUP_TEMP=${4:-true}      # 是否刪除暫存音檔（傳 "true"/"false"）
LANGUAGE=${5:-zh}          # 語言代碼（例如 zh、en 等，預設為 zh）  

# 容錯：忽略空值，確保變數為字串
SEGMENT_TIME="${SEGMENT_TIME}"
GENERATE_TXT="${GENERATE_TXT}"
GENERATE_SRT="${GENERATE_SRT}"
CLEANUP_TEMP="${CLEANUP_TEMP}"

# 啟用 nocasematch，讓副檔名比對不分大小寫
shopt -s nocasematch

# 檢查 videos 目錄
if [ ! -d "$VIDEOS_DIR" ]; then
    echo "❌ 沒有找到 videos/ 目錄: $VIDEOS_DIR"
    exit 1
fi

# 檢查 .venv 是否存在並啟動
if [ -f ".venv/bin/activate" ]; then
    echo "啟動虛擬環境 .venv..."
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "⚠ 找不到 .venv，請手動啟動環境或確認 .venv 路徑，會嘗試用當前 python 執行"
fi


# 遍歷影片（支援空格與特殊字元）
for VIDEO_FILE in "$VIDEOS_DIR"/*; do
    # 只處理常見影片副檔名（不區分大小寫）
    if [[ ! "$VIDEO_FILE" =~ \.(mp4|mkv|mov|avi)$ ]]; then
        continue
    fi

    echo
    echo "=============================="
    echo "開始處理影片: $VIDEO_FILE"
    echo "=============================="

    TMP_DIR="temp_audio"
    mkdir -p "$TMP_DIR"

    # 清除舊的 part 檔，以免與前一次混合（僅刪 part*.mp3）
    rm -f "$TMP_DIR"/part*.mp3 2>/dev/null || true

    # 切片
    echo "切片中，每段 ${SEGMENT_TIME} 秒..."
    ffmpeg -hide_banner -loglevel info -i "$VIDEO_FILE" -f segment -segment_time "$SEGMENT_TIME" -vn -acodec libmp3lame -q:a 4 "$TMP_DIR/part%03d.mp3"
    ff_exit=$?
    if [ $ff_exit -ne 0 ]; then
        echo "⚠ ffmpeg 切片失敗（exit $ff_exit），跳過此影片: $VIDEO_FILE"
        continue
    fi

    # 呼叫 Python 轉寫（把控制參數以字串形式傳入）
    echo "開始轉寫..."
    python batch_transcribe.py \
        --input_dir "$TMP_DIR" \
        --base_name "$VIDEO_FILE" \
        --generate_txt "$GENERATE_TXT" \
        --generate_srt "$GENERATE_SRT" \
        --cleanup_temp "$CLEANUP_TEMP" \
        --language "$LANGUAGE"
    py_exit=$?
    if [ $py_exit -ne 0 ]; then
        echo "⚠ batch_transcribe.py 執行異常（exit $py_exit），請檢查日誌。跳過此影片。"
        # 若 Python 已刪除 temp_audio（cleanup=true），就重建一個空目錄以免下一輪錯誤
        mkdir -p "$TMP_DIR"
        continue
    fi

    echo "影片處理完成: $VIDEO_FILE"
done

echo
echo "所有影片處理完成。"
if [ "$CLEANUP_TEMP" = "true" ]; then
    echo "暫存音檔由 Python 處理應已刪除（若 cleanup=true）。"
else
    echo "暫存音檔保留在 temp_audio/，如需刪除請手動執行 rm -rf temp_audio/ 。"
fi