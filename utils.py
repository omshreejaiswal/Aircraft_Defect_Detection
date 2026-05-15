import io
import json
import logging
import random
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from config import (
    DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ROBOFLOW_API_KEY,
    ROBOFLOW_OUTPUT_FORMAT,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    ROBOFLOW_WORKSPACE,
    USER_AGENT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_from_bytes(image_bytes: bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image bytes.")
    return image


def validate_image_file(image_path: Path) -> bool:
    try:
        image = cv2.imread(str(image_path))
        return image is not None and image.size > 0
    except Exception:
        return False


def resize_and_normalize(image: np.ndarray, target_size=(640, 640)) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def augment_image(image: np.ndarray) -> np.ndarray:
    augmented = image.copy()
    if random.random() < 0.5:
        augmented = cv2.flip(augmented, 1)
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=0)
    if random.random() < 0.35:
        kernel_size = random.choice([3, 5])
        augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
    if random.random() < 0.2:
        angle = random.uniform(-15, 15)
        h, w = augmented.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        augmented = cv2.warpAffine(augmented, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return augmented


def fetch_roboflow_universe_dataset(project: str, version: str, api_key: str, output_dir: Path) -> Path:
    if not api_key:
        raise EnvironmentError("Roboflow API key is required. Set ROBOFLOW_API_KEY in your .env file.")

    safe_makedirs(output_dir)
    headers = {"User-Agent": USER_AGENT}
    workspace = ROBOFLOW_WORKSPACE
    try:
        if not workspace:
            logging.info("Resolving Roboflow workspace for API key")
            workspace_response = requests.get(
                "https://api.roboflow.com/",
                params={"api_key": api_key},
                headers=headers,
                timeout=30,
            )
            workspace_response.raise_for_status()
            workspace_payload = workspace_response.json()
            workspace = workspace_payload.get("workspace", "")

        if not workspace:
            raise RuntimeError("Unable to determine Roboflow workspace. Set ROBOFLOW_WORKSPACE in your .env file.")

        export_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{ROBOFLOW_OUTPUT_FORMAT}"

        logging.info("Requesting Roboflow export link for %s/%s/%s", workspace, project, version)
        export_response = requests.get(export_url, params={"api_key": api_key}, headers=headers, timeout=120)
        export_response.raise_for_status()
        export_payload = export_response.json()
        if "error" in export_payload:
            raise RuntimeError(f"Roboflow export failed: {export_payload['error']}")

        download_url = export_payload.get("export", {}).get("link")
        if not download_url:
            raise RuntimeError("Roboflow export response did not include a dataset download link.")

        logging.info("Downloading Roboflow dataset archive")
        response = requests.get(download_url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Roboflow request failed. Check your internet connection, API key, workspace, project, and version."
        ) from None

    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type.lower():
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text[:300]}
        raise RuntimeError(f"Roboflow download failed: {payload.get('error', payload)}")

    archive_path = output_dir / "roboflow_dataset.zip"
    with open(archive_path, "wb") as archive_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                archive_file.write(chunk)

    logging.info("Extracting Roboflow data to %s", output_dir)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(output_dir)
    except zipfile.BadZipFile as exc:
        preview = archive_path.read_text(errors="replace")[:300]
        raise RuntimeError(f"Roboflow did not return a zip archive. Response started with: {preview}") from exc

    archive_path.unlink(missing_ok=True)
    return output_dir


def clean_dataset_directory(dataset_dir: Path) -> tuple[Path, Path]:
    image_dir = None
    label_dir = None
    for candidate in dataset_dir.rglob("*"):
        if candidate.is_dir() and candidate.name in {"images", "train", "valid", "test"}:
            if image_dir is None and any(str(candidate).endswith(ext) for ext in ["images", "train", "valid", "test"]):
                pass
    for family in dataset_dir.glob("**/*"):
        if family.is_dir() and any(ext in family.name.lower() for ext in ["images", "labels"]):
            pass

    for image_path in dataset_dir.rglob("*.jpg"):
        if not validate_image_file(image_path):
            logging.warning("Removing corrupt image: %s", image_path)
            image_path.unlink(missing_ok=True)

    return dataset_dir, dataset_dir


def create_pseudo_segmentation_mask(annotation_path: Path, image_shape: tuple, scale_factor: int = 2) -> np.ndarray:
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    with open(annotation_path, "r") as annotation_file:
        for line in annotation_file:
            values = line.strip().split()
            if len(values) != 5:
                continue
            _, x_center, y_center, width, height = map(float, values)
            x_center *= image_shape[1]
            y_center *= image_shape[0]
            width *= image_shape[1] * scale_factor
            height *= image_shape[0] * scale_factor
            x1 = int(max(x_center - width / 2, 0))
            y1 = int(max(y_center - height / 2, 0))
            x2 = int(min(x_center + width / 2, image_shape[1] - 1))
            y2 = int(min(y_center + height / 2, image_shape[0] - 1))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def safe_load_annotations(annotation_path: Path) -> list[tuple[int, float, float, float, float]]:
    annotations = []
    try:
        with open(annotation_path, "r") as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 5:
                    continue
                annotations.append(tuple(map(float, values)))
    except Exception:
        pass
    return annotations


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    scaled = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return scaled


def create_report_path() -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"inspection_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"


def extract_first_video_frame(video_bytes: bytes, suffix: str = ".mp4") -> np.ndarray | None:
    temp_path = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_path = Path(temp_file.name)

        capture = cv2.VideoCapture(str(temp_path))
        success, frame = capture.read()
        if success:
            return frame
        return None
    finally:
        if capture is not None:
            capture.release()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def get_dataset_yaml(dataset_root: Path, class_names: list[str]) -> str:
    data_yaml = {
        "path": str(dataset_root),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": class_names,
    }
    file_path = dataset_root / "data.yaml"
    with open(file_path, "w") as f:
        json.dump(data_yaml, f, indent=2)
    return str(file_path)
