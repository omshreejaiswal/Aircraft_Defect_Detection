import logging
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MPLCONFIG_DIR = BASE_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

import config
from segmentation import UNet, build_unet_model, save_unet_weights
from utils import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION,
    create_pseudo_segmentation_mask,
    fetch_roboflow_universe_dataset,
    get_dataset_yaml,
    normalize_to_uint8,
    safe_makedirs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def locate_dataset_root(base_dir: Path) -> Path:
    if (base_dir / "train").exists():
        return base_dir
    for child in base_dir.iterdir():
        if child.is_dir() and (child / "train").exists():
            return child
    return base_dir


def has_yolo_dataset(dataset_root: Path) -> bool:
    required_paths = [
        dataset_root / "train" / "images",
        dataset_root / "train" / "labels",
        dataset_root / "valid" / "images",
        dataset_root / "valid" / "labels",
    ]
    return all(path.exists() for path in required_paths)


def create_segmentation_masks(dataset_root: Path, mask_root: Path) -> None:
    safe_makedirs(mask_root)
    for subset in ["train", "valid", "test"]:
        image_dir = dataset_root / subset / "images"
        label_dir = dataset_root / subset / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue

        subset_mask_dir = mask_root / subset
        subset_mask_dir.mkdir(parents=True, exist_ok=True)

        for annotation_path in label_dir.glob("*.txt"):
            image_path = image_dir / f"{annotation_path.stem}.jpg"
            if not image_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            mask = create_pseudo_segmentation_mask(annotation_path, image.shape)
            cv2.imwrite(str(subset_mask_dir / f"{annotation_path.stem}.png"), mask)


def sanitize_yolo_labels(dataset_root: Path, single_class: bool = True) -> None:
    """Normalize labels for YOLO detection training.

    Roboflow exports can contain mixed detect/segment annotations. This project
    trains a detector, so each row must be exactly: class x_center y_center width height.
    """
    for subset in ["train", "valid", "test"]:
        label_dir = dataset_root / subset / "labels"
        if not label_dir.exists():
            continue

        for annotation_path in label_dir.glob("*.txt"):
            remapped_lines = []
            seen = set()
            changed = False
            for line in annotation_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    changed = True
                    continue

                row = parts[:5]
                if len(parts) > 5:
                    changed = True
                    try:
                        polygon = [float(value) for value in parts[1:]]
                    except ValueError:
                        continue

                    if len(polygon) < 6 or len(polygon) % 2 != 0:
                        continue

                    xs = polygon[0::2]
                    ys = polygon[1::2]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    row = [
                        parts[0],
                        str((x_min + x_max) / 2),
                        str((y_min + y_max) / 2),
                        str(x_max - x_min),
                        str(y_max - y_min),
                    ]
                elif len(parts) != 5:
                    changed = True
                    continue

                if single_class and row[0] != "0":
                    changed = True
                    row[0] = "0"

                try:
                    class_id = int(float(row[0]))
                    coords = [float(value) for value in row[1:]]
                except ValueError:
                    changed = True
                    continue

                if any(value < 0.0 or value > 1.0 for value in coords):
                    changed = True
                    coords = [min(1.0, max(0.0, value)) for value in coords]

                if coords[2] <= 0.0 or coords[3] <= 0.0:
                    changed = True
                    continue

                normalized_row = (
                    str(class_id),
                    *(f"{value:.6f}".rstrip("0").rstrip(".") for value in coords),
                )
                normalized_line = " ".join(normalized_row)
                if normalized_line in seen:
                    changed = True
                    continue

                seen.add(normalized_line)
                remapped_lines.append(normalized_line)

            if changed:
                annotation_path.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""))

        cache_path = dataset_root / subset / "labels.cache"
        cache_path.unlink(missing_ok=True)


def prepare_dataset() -> Path:
    output_dir = config.DATA_DIR / "roboflow"
    safe_makedirs(config.DATA_DIR)
    dataset_root = locate_dataset_root(output_dir) if output_dir.exists() else output_dir

    if has_yolo_dataset(dataset_root):
        logging.info("Using existing dataset at %s", dataset_root)
    else:
        dataset_dir = fetch_roboflow_universe_dataset(
            ROBOFLOW_PROJECT,
            ROBOFLOW_VERSION,
            ROBOFLOW_API_KEY,
            output_dir,
        )
        dataset_root = locate_dataset_root(dataset_dir)

    if not has_yolo_dataset(dataset_root):
        raise FileNotFoundError(
            f"Dataset at {dataset_root} is missing train/valid image and label folders."
        )

    get_dataset_yaml(dataset_root, config.TARGET_CLASSES)
    sanitize_yolo_labels(dataset_root, single_class=len(config.TARGET_CLASSES) == 1)
    mask_root = config.DATA_DIR / "segmentation_masks"
    create_segmentation_masks(dataset_root, mask_root)
    return dataset_root


class DefectSegmentationDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size=(256, 256)) -> None:
        self.image_paths = list(image_dir.glob("*.jpg"))
        self.mask_dir = mask_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        image = normalize_to_uint8(image).astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)


def train_segmentation(dataset_root: Path) -> None:
    mask_root = config.DATA_DIR / "segmentation_masks"
    train_dataset = DefectSegmentationDataset(dataset_root / "train" / "images", mask_root / "train")
    val_dataset = DefectSegmentationDataset(dataset_root / "valid" / "images", mask_root / "valid")

    train_loader = DataLoader(train_dataset, batch_size=config.TRAINING_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.TRAINING_CONFIG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    history = {
        "epochs": [],
        "training_loss": [],
        "validation_loss": [],
        "accuracy": [],
        "source": "segmentation_training",
    }

    for epoch in range(config.TRAINING_CONFIG["segmentation_epochs"]):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        proxy_accuracy = max(0.0, min(1.0, 1.0 - avg_val_loss))
        history["epochs"].append(epoch + 1)
        history["training_loss"].append(round(avg_train_loss, 4))
        history["validation_loss"].append(round(avg_val_loss, 4))
        history["accuracy"].append(round(proxy_accuracy, 4))

        logging.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f",
            epoch + 1,
            config.TRAINING_CONFIG["segmentation_epochs"],
            avg_train_loss,
            avg_val_loss,
        )

    save_unet_weights(model, config.UNET_MODEL_PATH)
    history_path = config.MODELS_DIR / "training_history.json"
    logs_path = config.MODELS_DIR / "training_logs.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    logs_path.write_text(json.dumps(history, indent=2))
    logging.info("Saved segmentation weights to %s", config.UNET_MODEL_PATH)


def train_detection(dataset_root: Path) -> None:
    model = YOLO("yolov8n.pt")
    data_yaml = str(dataset_root / "data.yaml")

    model.train(
        data=data_yaml,
        epochs=config.TRAINING_CONFIG["detection_epochs"],
        imgsz=640,
        batch=config.TRAINING_CONFIG["batch_size"],
        project=str(config.MODELS_DIR),
        name="yolo_training",
        exist_ok=True,
    )

    if (config.MODELS_DIR / "yolo_training" / "weights" / "best.pt").exists():
        final_path = config.MODELS_DIR / "best.pt"
        (config.MODELS_DIR / "yolo_training" / "weights" / "best.pt").rename(final_path)
        logging.info("Saved trained YOLO model to %s", final_path)


def main() -> None:
    dataset_root = prepare_dataset()
    train_detection(dataset_root)
    train_segmentation(dataset_root)
    logging.info("Training complete: YOLO detection and U-Net segmentation finished.")


if __name__ == "__main__":
    main()
