from pathlib import Path
from faster_whisper import WhisperModel
import torch

# 模型路徑（CTranslate2 格式）
MODEL_PATH = Path("/home/jay/whisper/models/medium")
DEVICE = "cuda"

print("Step 2: 開始加載模型...")

# 加載模型
model = WhisperModel(
    str(MODEL_PATH),
    device=DEVICE,
    compute_type="float16"  # 適合 GPU
)

# 檢查 GPU 可用性
if torch.cuda.is_available():
    print(f"GPU 可用，設備數量: {torch.cuda.device_count()}")
    print(f"當前設備: {torch.cuda.get_device_name(0)}")
else:
    print("GPU 不可用，將使用 CPU")

print("Step 2 完成：模型加載成功")
