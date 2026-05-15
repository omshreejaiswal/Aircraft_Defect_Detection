from __future__ import annotations

from typing import Any

from chart_generation import build_report_assets


def build_visual_assets(result: Any, history_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return build_report_assets(result, history_rows=history_rows)
