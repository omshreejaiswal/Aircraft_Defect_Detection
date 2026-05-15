from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import streamlit as st

from database import InspectionDatabase
from model import InspectionResult, analyze_image, load_models
from chart_generation import build_report_assets
from report_generator import generate_inspection_report
from utils import create_report_path, load_image_from_bytes


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(220, 38, 38, 0.08), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, #eef6f4 100%);
            color: #0f172a;
        }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp li, .stApp small {
            color: #0f172a;
        }
        .stMarkdown, .stCaption, .stText, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
            color: #0f172a;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stFileUploaderFileName"],
        [data-testid="stFileUploaderFileData"], [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
            background: #172b4d;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div, [data-testid="stSidebar"] li, [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] [data-baseweb="radio"] label,
        [data-testid="stSidebar"] [data-baseweb="radio"] div {
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(248, 250, 252, 0.25);
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #172b4d;
            border: 1px solid rgba(148, 163, 184, 0.35);
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #f8fafc !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzone"] section {
            color: #e2e8f0 !important;
        }
        .stButton button, .stDownloadButton button, [data-testid="stFileUploaderDropzone"] button {
            color: #ffffff !important;
        }
        .stAlert, [data-testid="stNotification"], [data-testid="stMarkdownContainer"] code {
            color: #0f172a !important;
        }
        .hero-card, .metric-card, .content-card {
            background: rgba(255, 255, 255, 0.94);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.3rem;
        }
        .hero-subtitle {
            color: #475569;
            font-size: 1rem;
            line-height: 1.6;
        }
        .status-good, .status-alert {
            display: inline-block;
            padding: 0.3rem 0.75rem;
            border-radius: 999px;
            font-size: 0.92rem;
            font-weight: 700;
        }
        .status-good {
            color: #166534;
            background: #dcfce7;
        }
        .status-alert {
            color: #991b1b;
            background: #fee2e2;
        }
        .section-label {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }
        .section-subtitle {
            font-size: 0.95rem;
            color: #475569;
            margin-bottom: 1rem;
        }
        .dashboard-card {
            position: relative;
            min-height: 250px;
            border-radius: 24px;
            padding: 1.2rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
            border: 1px solid rgba(148, 163, 184, 0.14);
            color: #0f172a;
            overflow: hidden;
            transition: transform 0.22s ease, box-shadow 0.22s ease;
        }
        .dashboard-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 24px 50px rgba(15, 23, 42, 0.14);
        }
        .dashboard-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: var(--accent-color);
        }
        .dashboard-card.red {
            --accent-color: #dc2626;
            background: linear-gradient(180deg, rgba(254, 242, 242, 0.98) 0%, rgba(255,255,255,0.98) 100%);
        }
        .dashboard-card.orange {
            --accent-color: #ea580c;
            background: linear-gradient(180deg, rgba(255, 247, 237, 0.98) 0%, rgba(255,255,255,0.98) 100%);
        }
        .dashboard-card.green {
            --accent-color: #16a34a;
            background: linear-gradient(180deg, rgba(240, 253, 244, 0.98) 0%, rgba(255,255,255,0.98) 100%);
        }
        .dashboard-card.blue {
            --accent-color: #2563eb;
            background: linear-gradient(180deg, rgba(239, 246, 255, 0.98) 0%, rgba(255,255,255,0.98) 100%);
        }
        .dashboard-label {
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #334155;
            margin-bottom: 0.9rem;
        }
        .dashboard-icon {
            font-size: 1.2rem;
            margin-right: 0.35rem;
        }
        .dashboard-value {
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.65rem;
        }
        .dashboard-subvalue {
            font-size: 0.92rem;
            line-height: 1.5;
            color: #475569;
            min-height: 44px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.95rem;
            font-weight: 800;
            margin-top: 0.2rem;
        }
        .status-badge.red {
            color: #991b1b;
            background: #fee2e2;
        }
        .status-badge.green {
            color: #166534;
            background: #dcfce7;
        }
        .trend-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.82rem;
            font-weight: 700;
            background: rgba(15, 23, 42, 0.06);
            color: #334155;
            margin-top: 0.55rem;
        }
        .confidence-track {
            width: 100%;
            height: 10px;
            background: rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 0.7rem;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #0f766e 0%, #2563eb 100%);
        }
        .tooltip-note {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Aircraft Defect Detection Intelligence Suite</div>
            <div class="hero-subtitle">
                Streamlined visual inspection with defect localization, segmentation overlays,
                maintenance risk scoring, analytics, and downloadable PDF reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


def load_input_frame() -> tuple[str | None, Any]:
    st.sidebar.subheader("Input Source")
    source = st.sidebar.radio("Select inspection input", ["Image Upload"])

    frame = None
    source_name = None

    if source == "Image Upload":
        uploaded_image = st.file_uploader("Upload aircraft component image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            source_name = uploaded_image.name
            frame = load_image_from_bytes(uploaded_image.read())

    return source_name, frame


def derive_prediction_dashboard(defect_detected: bool, probability: float, segmentation_score: float | None = None) -> dict[str, Any]:
    probability = max(0.0, min(1.0, probability))
    segmentation_score = segmentation_score if segmentation_score is not None else 0.0

    if not defect_detected:
        status = "No Defect"
        color = "blue"
        status_color = "green"
        icon = "✅"
    else:
        status = "Defect Detected"
        if probability >= 0.85:
            color = "red"
            status_color = "red"
            icon = "❌"
        elif probability >= 0.60:
            color = "red"
            status_color = "red"
            icon = "⚠️"
        elif probability >= 0.30:
            color = "orange"
            status_color = "red"
            icon = "⚠️"
        else:
            color = "green"
            status_color = "green"
            icon = "✅"

    return {
        "defect_detected": defect_detected,
        "probability": probability,
        "segmentation_score": segmentation_score,
        "status": status,
        "confidence_percent": probability * 100.0,
        "accent_color": color,
        "status_color": status_color,
        "icon": icon,
    }


def _trend_data(probability: float) -> dict[str, str]:
    previous_probability = st.session_state.get("previous_probability")
    if previous_probability is None:
        return {"symbol": "•", "label": "First prediction", "delta": "0.0"}

    delta = probability - previous_probability
    if abs(delta) < 0.001:
        return {"symbol": "→", "label": "No significant change", "delta": f"{delta * 100:.1f}%"}
    if delta > 0:
        return {"symbol": "↑", "label": "Confidence increased", "delta": f"+{delta * 100:.1f}%"}
    return {"symbol": "↓", "label": "Confidence decreased", "delta": f"{delta * 100:.1f}%"}


def render_prediction_dashboard(result: InspectionResult) -> dict[str, Any]:
    intelligence = result.intelligence
    dashboard = derive_prediction_dashboard(
        result.defect_present,
        result.confidence,
        result.quantification["spread"],
    )
    dashboard.update(
        {
            "severity": intelligence["severity"],
            "risk": intelligence["risk"],
            "recommendation": intelligence["recommendation"],
        }
    )
    dashboard["accent_color"] = {
        "Critical": "red",
        "High": "red",
        "Moderate": "orange",
        "Low": "green",
    }.get(intelligence["severity"], dashboard["accent_color"])
    trend = _trend_data(result.confidence)

    cols = st.columns(5)
    card_specs = [
        {
            "label": "Inspection Status",
            "value": (
                f"<div class='status-badge {dashboard['status_color']}' title='Live classification from the uploaded image'>"
                f"<span>{dashboard['icon']}</span><span>{dashboard['status']}</span></div>"
            ),
            "subvalue": "Immediate defect presence decision generated from the current model prediction.",
            "tooltip": "Status updates immediately after model inference.",
            "color": dashboard["accent_color"] if result.defect_present else "blue",
        },
        {
            "label": "Severity",
            "value": dashboard["severity"],
            "subvalue": f"Derived from occupancy, detection confidence, and segmentation support score {result.quantification['spread'] * 100:.1f}%.",
            "tooltip": "Severity combines rule-based maintenance thresholds with the current model output.",
            "color": dashboard["accent_color"],
        },
        {
            "label": "Risk",
            "value": dashboard["risk"],
            "subvalue": "Operational risk level mapped directly from the current severity assessment.",
            "tooltip": "Risk mapping reflects the requested aviation monitoring policy.",
            "color": dashboard["accent_color"],
        },
        {
            "label": "Recommendation",
            "value": dashboard["recommendation"],
            "subvalue": f"{intelligence['maintenance_type']} with target schedule {intelligence['schedule']}.",
            "tooltip": "Recommendation updates automatically whenever the inspection result changes.",
            "color": dashboard["accent_color"],
        },
        {
            "label": "Confidence",
            "value": f"{dashboard['confidence_percent']:.0f}%",
            "subvalue": "Probability score reported by the current model prediction.",
            "tooltip": "Confidence score displayed as a percentage and trend-aware indicator.",
            "color": dashboard["accent_color"],
            "is_confidence": True,
        },
    ]

    for col, spec in zip(cols, card_specs):
        with col:
            st.markdown(
                f"""
                <div class="dashboard-card {spec['color']}" title="{spec['tooltip']}">
                    <div class="dashboard-label"><span class="dashboard-icon">{dashboard['icon'] if spec['label'] == 'Inspection Status' else ''}</span>{spec['label']}</div>
                    <div class="dashboard-value">{spec['value']}</div>
                    <div class="dashboard-subvalue">{spec['subvalue']}</div>
                    {f"<div class='trend-chip'>{trend['symbol']} {trend['label']} ({trend['delta']})</div>" if spec.get('is_confidence') else ""}
                    {f"<div class='confidence-track'><div class='confidence-fill' style='width:{dashboard['confidence_percent']:.1f}%;'></div></div>" if spec.get('is_confidence') else ""}
                    {f"<div class='tooltip-note'>Live probability bar for the current prediction.</div>" if spec.get('is_confidence') else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.session_state["dashboard_state"] = dashboard
    st.session_state["previous_probability"] = result.confidence
    return dashboard


def render_image_section(result: InspectionResult) -> None:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-label">Processed Output</div>
            <div class="section-subtitle">Compare the source image with the annotated and segmented outputs from the model pipeline.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.image(cv2.cvtColor(result.original, cv2.COLOR_BGR2RGB), caption="Original Input", width="stretch")
    with right:
        st.image(cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB), caption="Processed Defect Overlay", width="stretch")

    st.markdown(
        """
        <div class="section-card">
            <div class="section-label">Segmentation Result</div>
            <div class="section-subtitle">Review the detection rendering and binary segmentation output used for quantification.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    seg_left, seg_right = st.columns(2)
    with seg_left:
        st.image(cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB), caption="Detection Output", width="stretch")
    with seg_right:
        st.image(result.mask, caption="Segmentation Mask", width="stretch", clamp=True)


def render_prediction_section(result: InspectionResult) -> None:
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("Prediction Output")
    metrics_left, metrics_right = st.columns(2)
    with metrics_left:
        st.metric("Severity", result.intelligence["severity"])
    with metrics_right:
        st.metric("Confidence", f"{result.confidence * 100.0:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)


def create_report_payload(result: InspectionResult) -> dict[str, Any]:
    dashboard = result.intelligence
    return {
        "project_name": "Aircraft Defect Detection",
        "author": "OpenAI Codex",
        "timestamp": datetime.utcnow().isoformat(),
        "severity": dashboard["severity"],
        "risk": dashboard["risk"],
        "recommendation": dashboard["recommendation"],
        "action": dashboard["action"],
        "schedule": dashboard["schedule"],
        "priority": dashboard["priority"],
        "maintenance_type": dashboard["maintenance_type"],
        "next_inspection_date": dashboard["next_inspection_date"],
        "notes": dashboard["notes"],
        "insights": dashboard["insights"],
        "component": dashboard["component"],
        "failure_type": dashboard["failure_type"],
        "severity_reasoning": dashboard["severity_reasoning"],
        "defects": result.detections,
        "area": result.quantification["area"],
        "spread": result.quantification["spread"],
        "occupancy": result.quantification["surface_occupancy"],
        "confidence": result.confidence,
        "summary": result.summary,
    }


def persist_inspection(db: InspectionDatabase, source_name: str | None, result: InspectionResult, report_path: Path) -> None:
    dashboard = result.intelligence
    db.log_inspection(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "image_path": source_name or "uploaded_input",
            "defects": result.detections,
            "severity": dashboard["severity"],
            "risk": dashboard["risk"],
            "recommendation": dashboard["recommendation"],
            "area": result.quantification["area"],
            "spread": result.quantification["spread"],
            "occupancy": result.quantification["surface_occupancy"],
            "confidence": result.confidence,
            "report_path": str(report_path),
        }
    )


def render_recent_history(db: InspectionDatabase) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Recent Inspections")
    rows = db.fetch_recent_inspections(limit=8)
    if not rows:
        st.sidebar.caption("No inspection history available yet.")
        return
    for row in rows:
        st.sidebar.markdown(
            f"""
            <div style="padding:0.7rem 0.8rem;margin-bottom:0.55rem;border-radius:14px;background:rgba(255,255,255,0.08);">
                <div style="font-weight:700;color:#f8fafc;">{row["severity"]} • {row["risk"]}</div>
                <div style="font-size:0.85rem;color:#dbeafe;">{row["timestamp"]}</div>
                <div style="font-size:0.85rem;color:#e2e8f0;">Detections: {row["defects"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def run_app() -> None:
    st.set_page_config(
        page_title="Aircraft Inspection AI",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    render_header()

    st.sidebar.title("Inspection Controls")

    db = InspectionDatabase()
    detection_model, segmentation_model = get_models()

    try:
        source_name, frame = load_input_frame()

        if frame is None:
            st.markdown(
                '<div class="content-card">Upload an aircraft component image to begin the inspection workflow.</div>',
                unsafe_allow_html=True,
            )
            return

        progress = st.progress(0, text="Preparing inspection pipeline...")
        with st.spinner("Running detection and segmentation models..."):
            progress.progress(20, text="Loading and validating input...")
            result = analyze_image(frame, detection_model, segmentation_model)
            progress.progress(65, text="Building analytics and report assets...")
            inspection_history = [dict(row) for row in db.fetch_all_inspections()]
            report_assets = build_report_assets(result, history_rows=inspection_history)
            report_path = create_report_path()
            report_payload = create_report_payload(result)
            generate_inspection_report(report_payload, report_path, visual_assets=report_assets["report_images"])
            progress.progress(90, text="Saving inspection log...")
            persist_inspection(db, source_name, result, report_path)
            progress.progress(100, text="Inspection complete.")

        render_image_section(result)
        render_prediction_section(result)

        st.download_button(
            "Download Report",
            data=report_path.read_bytes(),
            file_name=report_path.name,
            mime="application/pdf",
            use_container_width=False,
        )
    except Exception as exc:
        st.error(f"Inference failed. {exc}")
    finally:
        db.close()


if __name__ == "__main__":
    run_app()
