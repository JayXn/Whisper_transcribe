from faster_whisper import download_model

# 原 Hugging Face 模型 snapshot 路徑
source_model = "/home/jay/whisper/models/medium"

# 輸出 CTranslate2 模型到同一資料夾
output_dir = "/home/jay/whisper/models/medium"

# medium 模型
download_model("medium", output_dir=output_dir)
print("模型已轉成 CTranslate2 格式，輸出到", output_dir)
