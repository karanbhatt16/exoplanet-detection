from __future__ import annotations

import base64
import io
import tempfile
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from exoplanet_pipeline import TransitAssessment, TransitCandidate, TransitSearcher

app = Flask(__name__)

ALLOWED_SUFFIXES = {".csv", ".txt", ".parquet", ".fits", ".fit", ".lc"}


def _to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(item) for item in value]
    return value


def _figure_to_data_uri(fig) -> str:  # type: ignore[no-untyped-def]
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _candidate_payload(candidate: TransitCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "source": candidate.source,
        "period_days": round(candidate.period, 6),
        "duration_hours": round(candidate.duration * 24.0, 4),
        "transit_time_days": round(candidate.transit_time, 6),
        "depth": _to_python(candidate.depth),
        "power": round(candidate.power, 4),
        "snr": _to_python(candidate.snr),
        "stellar_radius_rsun": _to_python(candidate.stellar_radius_rsun),
        "stellar_radius_err_rsun": _to_python(candidate.stellar_radius_err_rsun),
        "planet_radius_rearth": _to_python(candidate.planet_radius_rearth),
        "planet_radius_err_rearth": _to_python(candidate.planet_radius_err_rearth),
        "radius_ratio": _to_python(candidate.radius_ratio),
        "radius_source": candidate.radius_source,
        "notes": candidate.notes,
        "stats": _to_python(candidate.stats),
    }


def _assessment_payload(
    assessment: TransitAssessment,
    figure_data_uri: str,
    display_source: str,
    summary_text: str,
) -> dict[str, Any]:
    summary_lines = summary_text.splitlines()
    if summary_lines:
        summary_lines[0] = f"Source: {display_source}"
    return {
        "success": True,
        "display_source": display_source,
        "status": assessment.status,
        "reason": assessment.reason,
        "confidence_percent": round(float(assessment.confidence_percent), 1),
        "is_candidate": bool(assessment.is_candidate),
        "summary_lines": summary_lines,
        "candidate": _candidate_payload(assessment.candidate),
        "notes": list(assessment.notes),
        "metrics": _to_python(assessment.metrics),
        "figure_data_uri": figure_data_uri,
    }


def _analysis_error(message: str, code: int = 400) -> tuple[dict[str, Any], int]:
    return {"success": False, "error": message}, code


def _parse_float(form_value: str | None, field_name: str, default: float | None = None) -> float | None:
    value = (form_value or "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_int(form_value: str | None, field_name: str, default: int | None = None) -> int | None:
    value = (form_value or "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer.") from exc


def _build_searcher(form: dict[str, str]) -> TransitSearcher:
    return TransitSearcher(
        target_id=(form.get("target_id") or "").strip() or None,
        sector=_parse_int(form.get("sector"), "Sector"),
        use_all_sectors=(form.get("use_all_sectors") or "").lower() == "on",
        min_period=_parse_float(form.get("min_period"), "Min period", 0.5) or 0.5,
        max_period=_parse_float(form.get("max_period"), "Max period", 20.0) or 20.0,
        min_duration=(_parse_float(form.get("min_duration"), "Min duration", 1.2) or 1.2) / 24.0,
        max_duration=(_parse_float(form.get("max_duration"), "Max duration", 7.2) or 7.2) / 24.0,
        duration_steps=_parse_int(form.get("duration_steps"), "Duration steps", 8) or 8,
        stellar_radius_rsun=_parse_float(form.get("stellar_radius"), "Stellar radius"),
        stellar_radius_err_rsun=_parse_float(form.get("stellar_radius_err"), "Radius error"),
    )


def _analyze(searcher: TransitSearcher, source_mode: str, upload_path: Path | None) -> tuple[TransitAssessment, Any, str]:
    if source_mode == "file":
        if upload_path is None:
            raise ValueError("Please upload a local light-curve file.")
        preview = searcher.inspector.load_local(upload_path)
        lightcurve = searcher._preview_to_lightcurve(preview)
        assessment = searcher.analyze_lightcurve(lightcurve, preview.source)
        return assessment, lightcurve, preview.source

    lightcurve, source = searcher.load_target()
    assessment = searcher.analyze_lightcurve(lightcurve, source)
    return assessment, lightcurve, source


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    form = request.form.to_dict(flat=True)
    source_mode = (form.get("source_mode") or "target").strip().lower()

    try:
        searcher = _build_searcher(form)
    except ValueError as exc:
        payload, code = _analysis_error(str(exc), 400)
        return jsonify(payload), code

    upload_path: Path | None = None
    display_source = (form.get("target_id") or "").strip()
    if source_mode == "file":
        upload = request.files.get("lightcurve_file")
        if upload is None or upload.filename.strip() == "":
            return jsonify(_analysis_error("Please choose a light-curve file to upload.", 400)[0]), 400
        original_name = secure_filename(upload.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            suffix = ".csv"
        upload_path = Path(tempfile.gettempdir()) / f"exoplanet_upload_{uuid.uuid4().hex}{suffix}"
        upload.save(upload_path)
        display_source = original_name or upload_path.name
    elif not display_source:
        payload, code = _analysis_error("Please enter a TIC target ID.", 400)
        return jsonify(payload), code

    try:
        assessment, lightcurve, source = _analyze(searcher, source_mode, upload_path)
        figure = searcher.build_result_figure(assessment, source_lightcurve=lightcurve)
        summary_text = searcher.summarize(assessment)
        payload = _assessment_payload(assessment, _figure_to_data_uri(figure), display_source or source, summary_text)
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        payload, code = _analysis_error(str(exc), 500)
        return jsonify(payload), code
    finally:
        if upload_path is not None and upload_path.exists():
            try:
                upload_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    app.run(debug=True)
