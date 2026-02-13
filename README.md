# 🎙️ Whisper Batch Transcribe

這是一個基於 **Python** 與 **Bash** 的自動化工具，旨在將長影片自動切片、轉寫為文字 (**TXT**) 與字幕 (**SRT**)。支援 **faster-whisper** 核心，提供強大的 GPU 加速功能。

---

## ✨ 功能概覽

* **自動切片**：自動將影片音訊提取並切割成多段（預設 MP3 格式），提升處理穩定性。
* **高效轉寫**：採用 `faster-whisper`，支援 NVIDIA GPU (CUDA) 加速。
* **語音偵測 (VAD)**：內建 VAD 過濾靜音片段，產出的字幕更精確、不冗餘。
* **多格式輸出**：同步產生 `.txt` 全文與帶時間軸的 `.srt` 字幕檔。
* **進度追蹤**：視覺化總進度條，即時掌握處理狀態。
* **自動清理**：轉寫完成後自動移除暫存音檔，節省磁碟空間。

---

## 📂 目錄結構

```text
whisper/
├── videos/              # 放置來源影片 (.mp4, .mkv 等)
│   └── example.mp4
├── temp_audio/          # [自動建立] 存放切片後的音訊
├── transcripts/         # [自動建立] 最終輸出的 TXT/SRT 結果
├── batch_transcribe.py  # 主要 Python 轉寫邏輯
├── run_whisper.sh       # 一鍵執行 Bash 腳本
└── .venv/               # Python 虛擬環境

```

---

## 🚀 安裝與設定

### 1. 建立虛擬環境
建議使用虛擬環境以保持系統乾淨：
```bash
# 建立環境
python3 -m venv .venv

# 啟動環境 (Linux/WSL2)
source .venv/bin/activate
```

### 2. 安裝依賴套件
> **注意**：請根據你的 CUDA 版本選擇對應的 PyTorch 安裝指令。

```bash
# 安裝 PyTorch (以 CUDA 12.1 為例)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 安裝 Whisper 相關套件
pip install faster-whisper tqdm
```
---

## 🛠️ 使用方法

### 方案 A：使用一鍵 Bash 腳本 (推薦)
```bash
bash run_whisper.sh [SEGMENT_TIME] [GENERATE_TXT] [GENERATE_SRT] [CLEANUP_TEMP] [LANGUAGE]
```

參數名稱,預設值,說明
SEGMENT_TIME,1200,每段音訊長度 (秒)
GENERATE_TXT,false,是否輸出 TXT
GENERATE_SRT,true,是否輸出 SRT
CLEANUP_TEMP,true,是否刪除暫存音訊
LANGUAGE,zh,語言代碼

---

### 方案 B：單獨執行 Python

```text
bash
python batch_transcribe.py \
    --input_dir "temp_audio" \
    --base_name "example.mp4" \
    --generate_txt true \
    --generate_srt true \
    --cleanup_temp true \
    --language zh
```

---

## 🔄 運作流程

1. **影片切片**：利用 `ffmpeg` 將影片音訊提取並切成多個小段落（如 part000.mp3）。
2. **時長掃描**：使用 `ffprobe` 取得每個檔案的精確秒數，以此建立精準的進度條。
3. **模型轉寫**：逐段送入 `faster-whisper` 模型，並套用 VAD（語音活動偵測）過濾靜音。
4. **合併輸出**：自動累加各段的時間偏移量，產出最終的 `.srt` 或 `.txt` 檔案。
5. **清理暫存**：若設定開啟，轉寫結束後會自動清空 `temp_audio/` 資料夾。

---

## ⚠️ 注意事項

* **硬體加速**：程式會自動偵測 CUDA，若有 NVIDIA GPU 建議優先使用以大幅提升速度。
* **VRAM 管理**：若遇到顯存不足（Out of Memory），請嘗試將 `SEGMENT_TIME` 設定短一點（例如 600）。
* **檔名建議**：請避免影片檔名包含空格或特殊符號，以降低 Bash 處理路徑時出錯的機率。
* **語言設定**：預設為 `zh`（中文），如需轉寫英文影片請手動改為 `en`。

---
