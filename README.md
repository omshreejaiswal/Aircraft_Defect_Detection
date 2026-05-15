# Aircraft Defect Detection and Predictive Maintenance System

AI-Driven Aircraft Defect Detection and Predictive Maintenance System is a Streamlit-based computer vision application for inspecting aircraft component images, detecting visible defects, estimating severity, and generating maintenance recommendations.

The project combines YOLOv8 object detection, PyTorch-based segmentation, OpenCV image processing, SQLite inspection logging, dashboard visualizations, and PDF report generation into a single prototype workflow for aircraft inspection and maintenance decision support.

## Features

- Detects aircraft surface defects from uploaded inspection images.
- Uses YOLOv8 for defect localization and a U-Net-style PyTorch model for segmentation support.
- Quantifies defect area, spread, and surface occupancy.
- Performs severity and risk analysis using rule-based maintenance intelligence.
- Recommends actions such as monitoring, repair, or grounding based on inspection signals.
- Stores inspection history in SQLite for dashboard review.
- Generates PDF inspection reports with visual evidence and charts.
- Provides a Streamlit dashboard for image upload, analysis, visualization, and report download.

## Technologies Used

- Python
- Streamlit
- YOLOv8 / Ultralytics
- PyTorch and TorchVision
- OpenCV
- NumPy
- Plotly
- SQLite
- ReportLab
- Roboflow dataset integration

## Installation

1. Clone the repository:

```bash
git clone https://github.com/omshreejaiswal/Aircraft_Defect_Detection.git
cd Aircraft_Defect_Detection
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
```

Update `.env` with your Roboflow workspace, project, version, and API key if you plan to train or download datasets.

## Run Commands

Start the Streamlit application:

```bash
streamlit run app.py
```

Train or refresh the detection and segmentation models:

```bash
python train.py
```

Generate training charts, if needed:

```bash
python training_chart_generator.py
```

## Folder Structure

```text
Aircraft_Def_Det/
├── app.py                       # Streamlit application entry point
├── model.py                     # Detection, segmentation, and decision pipeline
├── segmentation.py              # U-Net model and segmentation utilities
├── maintenance.py               # Maintenance recommendation logic
├── visualization.py             # Image and dashboard visualization helpers
├── chart_generation.py          # Chart assets for dashboard and reports
├── report_generator.py          # PDF inspection report generation
├── report.py                    # Report-related helpers
├── database.py                  # SQLite inspection logging
├── train.py                     # Dataset download and model training workflow
├── config.py                    # Paths and environment-driven settings
├── utils.py                     # Shared utility functions
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment configuration
├── yolov8n.pt                   # Small YOLOv8 fallback model weight
└── README.md
```

Generated folders such as `data/`, `datasets/`, `models/`, `runs/`, `logs/`, and `reports/` are intentionally excluded from Git because they may contain large datasets, trained weights, local databases, or generated inspection files.

## Model and Data Notes

- The repository includes `yolov8n.pt` as a small fallback YOLOv8 model for local development.
- Custom trained weights such as `best.pt` and `unet.pth` should be stored outside Git or published through a release asset, cloud drive, or model registry.
- Datasets should be downloaded through the configured Roboflow workflow rather than committed directly.

## Future Enhancements

- Add sample inspection images and anonymized demo outputs.
- Publish trained model weights as GitHub Releases or external model artifacts.
- Add unit tests for severity scoring, recommendation logic, and report generation.
- Add screenshots or a short demo video to the README.
- Add CI checks for linting and dependency validation.
- Package the application with Docker for reproducible deployment.
- Extend the dashboard with inspection trend analytics and aircraft/component metadata.

## Project Status

This project is intended as an academic and prototype showcase for AI-assisted aircraft inspection. Maintenance decisions should be validated by certified aviation maintenance professionals before operational use.
