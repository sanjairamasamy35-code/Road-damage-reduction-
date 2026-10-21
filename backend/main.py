"""FastAPI adapter for the existing RoadGuard Streamlit domain logic."""

from __future__ import annotations

import io
import os
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

import app as roadguard
import road_ai


BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_FILE = Path(os.getenv("ROADGUARD_DATABASE", roadguard.DATABASE_FILE))
UPLOAD_DIR = Path(os.getenv("ROADGUARD_UPLOAD_DIR", roadguard.UPLOAD_DIR))
ANNOTATED_DIR = Path(os.getenv("ROADGUARD_ANNOTATED_DIR", road_ai.ANNOTATED_DIR))
MAX_UPLOAD_BYTES = roadguard.MAX_PHOTO_BYTES
ALLOWED_ORIGINS = [value.strip() for value in os.getenv(
    "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if value.strip()]

app = FastAPI(title="RoadGuard AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

_model_lock = threading.Lock()
_model_session = None
_model_version = None
_model_error = None


def get_model():
    global _model_session, _model_version, _model_error
    if _model_session is None and _model_error is None:
        with _model_lock:
            if _model_session is None and _model_error is None:
                try:
                    _model_session, _model_version, _ = road_ai.create_cpu_session(
                        Path(os.getenv("ROADGUARD_MODEL", road_ai.MODEL_FILE)))
                except road_ai.InferenceError as error:
                    _model_error = str(error)
    return _model_session, _model_version, _model_error


@lru_cache(maxsize=1)
def get_district_data():
    import json
    raw = json.loads(roadguard.DATA_FILE.read_text(encoding="utf-8"))
    return roadguard.audit_district_features(raw)


def report_public(report: dict) -> dict:
    result = {key: report.get(key) for key in (
        "report_id", "submitted_at", "road_name", "location", "district", "district_id",
        "problem_type", "severity", "description", "latitude", "longitude", "status",
    )}
    result["has_photo"] = bool(report.get("photo_path"))
    return result


def find_report(report_id: str):
    with roadguard.database_connection(DATABASE_FILE) as connection:
        row = connection.execute(
            "SELECT * FROM road_reports WHERE report_id = ?", (report_id,)
        ).fetchone()
    return None if row is None else dict(row)


def save_ai_error(report_id: str, message: str):
    """Record a real inference failure through the same AI history table."""
    return road_ai.save_detection_result(report_id, {
        "model_version": "unavailable",
        "detections": [],
        "inference_status": "error",
        "processing_duration_ms": None,
        "annotated_image": None,
        "error_message": message,
    }, DATABASE_FILE, annotated_dir=ANNOTATED_DIR)


def clean_text(value: str, label: str, maximum: int, required: bool = True) -> str:
    value = (value or "").strip()
    if required and not value:
        raise HTTPException(422, f"{label} is required.")
    if len(value) > maximum:
        raise HTTPException(422, f"{label} must be {maximum} characters or fewer.")
    return value


@app.on_event("startup")
def initialize():
    roadguard.initialize_database(DATABASE_FILE)


@app.get("/api/health")
def health():
    model_path = Path(os.getenv("ROADGUARD_MODEL", road_ai.MODEL_FILE))
    return {
        "status": "ok",
        "database": DATABASE_FILE.exists(),
        "model_file": model_path.is_file(),
        "python": __import__("sys").version.split()[0],
    }


@app.get("/api/districts")
def districts():
    try:
        prepared, diagnostics = get_district_data()
    except (OSError, ValueError) as error:
        raise HTTPException(503, f"District map data is unavailable: {error}") from error
    return {"geojson": prepared, "registry": roadguard.DISTRICT_REGISTRY,
            "diagnostics": diagnostics}


@app.get("/api/reports")
def reports(
    district_id: str | None = None,
    problem_type: str | None = None,
    severity: str | None = None,
    status: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    filters = {key: value for key, value in {
        "district_id": district_id, "problem_type": problem_type,
        "severity": severity, "status": status,
    }.items() if value}
    return {"reports": [report_public(item) for item in
            roadguard.fetch_reports(DATABASE_FILE, filters)[:limit]],
            "total": roadguard.count_reports(filters, DATABASE_FILE)}


@app.get("/api/dashboard")
def dashboard():
    total, open_count, verified, categories = roadguard.fetch_dashboard_stats(DATABASE_FILE)
    recent = roadguard.fetch_reports(DATABASE_FILE)[:5]
    return {"total_reports": total, "open_reports": open_count,
            "verified_reports": verified, "categories": categories,
            "recent_reports": [report_public(item) for item in recent]}


@app.get("/api/reports/{report_id}")
def report_detail(report_id: str):
    if len(report_id) > 64:
        raise HTTPException(404, "Report not found.")
    record = find_report(report_id)
    if record is None:
        raise HTTPException(404, "Report not found.")
    return {"report": report_public(record),
            "status_history": roadguard.fetch_report_status_history(report_id, DATABASE_FILE)}


@app.get("/api/reports/{report_id}/photo")
def report_photo(report_id: str):
    record = find_report(report_id)
    if record is None:
        raise HTTPException(404, "Report not found.")
    if not record.get("photo_path"):
        raise HTTPException(404, "No report photo is attached.")
    try:
        path = Path(record["photo_path"])
        if not path.is_absolute():
            path = BASE_DIR / path
        path = path.resolve()
        allowed_roots = [BASE_DIR.resolve(), UPLOAD_DIR.resolve()]
        if not any(path == root or root in path.parents for root in allowed_roots):
            raise road_ai.InferenceError("The saved photo path points outside its allowed directory.")
    except road_ai.InferenceError as error:
        raise HTTPException(404, "The saved photo is unavailable.") from error
    if not path.is_file():
        raise HTTPException(404, "The saved photo is unavailable.")
    return FileResponse(path)


@app.post("/api/reports", status_code=201)
async def create_report(
    road_name: Annotated[str, Form()],
    district_id: Annotated[str, Form()],
    problem_type: Annotated[str, Form()],
    severity: Annotated[str, Form()],
    description: Annotated[str, Form()],
    location: Annotated[str, Form()] = "",
    latitude: Annotated[float | None, Form()] = None,
    longitude: Annotated[float | None, Form()] = None,
    submission_token: Annotated[str | None, Form()] = None,
    photo: UploadFile | None = File(default=None),
):
    road_name = clean_text(road_name, "Road name", 160)
    description = clean_text(description, "Description", 2000)
    location = clean_text(location, "Location", 200)
    district = roadguard.DISTRICT_BY_ID.get(district_id)
    if district is None:
        raise HTTPException(422, "Choose a district from the Tamil Nadu registry.")
    if problem_type not in roadguard.PROBLEM_TYPES:
        raise HTTPException(422, "Choose a supported road problem type.")
    if severity not in roadguard.SEVERITIES:
        raise HTTPException(422, "Choose a supported severity.")
    if (latitude is None) != (longitude is None):
        raise HTTPException(422, "Provide both latitude and longitude, or neither.")
    if latitude is not None and not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise HTTPException(422, "Coordinates are outside valid geographic ranges.")
    if latitude is not None:
        try:
            matched = roadguard.match_district(longitude, latitude, get_district_data()[0])
        except (OSError, ValueError) as error:
            raise HTTPException(503, f"District coordinate validation is unavailable: {error}") from error
        if matched["status"] == "matched" and matched["district_id"] != district_id:
            raise HTTPException(422, f"These coordinates match {matched['district_name']}; choose that district or correct the coordinates.")
    if submission_token and (len(submission_token) > 128 or not re.fullmatch(r"[\w.-]+", submission_token)):
        raise HTTPException(422, "Submission token is invalid.")

    photo_bytes = extension = None
    if photo is not None:
        payload = await photo.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Photo must be 5 MB or smaller.")
        try:
            photo_bytes, extension = roadguard.inspect_image_upload(io.BytesIO(payload))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    try:
        report_id = roadguard.save_report({
            "road_name": road_name, "location": location, "district_id": district_id,
            "district": district["name"], "problem_type": problem_type, "severity": severity,
            "description": description, "latitude": latitude, "longitude": longitude,
        }, database_path=DATABASE_FILE, photo_bytes=photo_bytes, photo_extension=extension,
            upload_dir=UPLOAD_DIR, submission_token=submission_token)
    except (ValueError, OSError) as error:
        raise HTTPException(422, str(error)) from error

    ai = {"status": "not_requested", "message": "No photo was provided."}
    if photo_bytes is not None:
        session, version, model_error = get_model()
        if model_error or session is None:
            message = model_error or "Model unavailable."
            try:
                detection_id = save_ai_error(report_id, message)
                ai = {"status": "unavailable", "message": message, "detection_id": detection_id}
            except Exception as error:
                ai = {"status": "unavailable", "message": f"{message} Analysis error could not be recorded: {error}"}
        else:
            try:
                with Image.open(io.BytesIO(photo_bytes)) as image:
                    inference = road_ai.run_inference(image.convert("RGB"), session, version)
                detection_id = road_ai.save_detection_result(
                    report_id, inference, DATABASE_FILE, annotated_dir=ANNOTATED_DIR)
                ai = {"status": inference["inference_status"],
                      "detection_id": detection_id,
                      "detections": [{key: item[key] for key in
                                      ("class_name", "confidence", "box")}
                                     for item in inference["detections"]],
                      "model_version": inference["model_version"],
                      "processing_duration_ms": inference["processing_duration_ms"]}
            except Exception as error:  # Report remains saved; disclose the actual analysis failure.
                message = f"Analysis failed: {error}"
                try:
                    detection_id = save_ai_error(report_id, message)
                    ai = {"status": "error", "message": message, "detection_id": detection_id}
                except Exception as persistence_error:
                    ai = {"status": "error", "message": f"{message} Could not record AI error: {persistence_error}"}
    record = next(item for item in roadguard.fetch_reports(DATABASE_FILE)
                  if item["report_id"] == report_id)
    return {"report": report_public(record), "ai_analysis": ai}

