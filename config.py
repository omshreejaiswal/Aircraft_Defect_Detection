import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

DATABASE_PATH = LOGS_DIR / "inspection_logs.db"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
UNET_MODEL_PATH = MODELS_DIR / "unet.pth"

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT", "aircraft-defect-detection")
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION", "1")
ROBOFLOW_OUTPUT_FORMAT = os.getenv("ROBOFLOW_OUTPUT_FORMAT", "yolov8")

DEFAULT_IMAGE_SIZE = (640, 640)
TARGET_CLASSES = ["defect"]
USER_AGENT = "AI-Driven-Aircraft-Inspection/1.0"

TRAINING_CONFIG = {
    "detection_epochs": int(os.getenv("DETECTION_EPOCHS", 30)),
    "segmentation_epochs": int(os.getenv("SEGMENTATION_EPOCHS", 20)),
    "batch_size": int(os.getenv("BATCH_SIZE", 8)),
}
