"""CPU ONNX inference helpers for the RoadGuard report-photo workflow.

Preprocessing and decoding follow the upstream road-hazard browser reference.
"""

import ast
import hashlib
import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "yolo11n.onnx"
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
NMS_IOU_THRESHOLD = 0.45
LOW_CONFIDENCE_THRESHOLD = 0.05
CLASS_NAMES = (
    "D00 longitudinal crack",
    "D10 transverse crack",
    "D20 alligator crack",
    "D40 pothole",
)
CLASS_COLORS = ("#0284c7", "#9333ea", "#16a34a", "#dc2626")


class InferenceError(RuntimeError):
    """The model or image could not be used for inference."""


@dataclass(frozen=True)
class Letterbox:
    tensor: np.ndarray
    scale: float
    pad_x: int
    pad_y: int
    original_width: int
    original_height: int


def load_image(path):
    """Open and validate a report image, applying EXIF orientation."""
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise InferenceError(f"The saved report photo is not a valid image: {error}") from error
    if image.width < 1 or image.height < 1:
        raise InferenceError("The saved report photo has invalid dimensions.")
    return image


def preprocess_image(image, size=INPUT_SIZE):
    """Match web/index.html: rounded resize, centered 114 padding, RGB CHW / 255."""
    if not isinstance(image, Image.Image):
        raise InferenceError("Inference requires a decoded image.")
    image = image.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise InferenceError("The image has invalid dimensions.")
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = math.floor((size - resized_width) / 2)
    pad_y = math.floor((size - resized_height) / 2)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))
    pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
    tensor = np.transpose(pixels, (2, 0, 1))[np.newaxis, ...].copy()
    return Letterbox(tensor, scale, pad_x, pad_y, width, height)


def _box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def decode_output(output, letterbox, confidence_threshold=DEFAULT_CONFIDENCE,
                  iou_threshold=NMS_IOU_THRESHOLD):
    """Decode upstream [1, 4 + classes, candidates] YOLO output and class-wise NMS."""
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1:
        raise InferenceError(f"Unexpected model output shape {values.shape}; expected [1, 8, N].")
    if values.shape[1:] != (4 + len(CLASS_NAMES), 8400):
        raise InferenceError(f"Unexpected model output shape {values.shape}; expected [1, 8, 8400].")
    if not np.isfinite(values).all():
        raise InferenceError("The model output contains non-finite values.")
    values = values[0]
    candidates = []
    count = values.shape[1]
    for index in range(count):
        class_scores = values[4:, index]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < confidence_threshold:
            continue
        center_x, center_y, width, height = (float(values[row, index]) for row in range(4))
        x1 = (center_x - width / 2 - letterbox.pad_x) / letterbox.scale
        y1 = (center_y - height / 2 - letterbox.pad_y) / letterbox.scale
        x2 = (center_x + width / 2 - letterbox.pad_x) / letterbox.scale
        y2 = (center_y + height / 2 - letterbox.pad_y) / letterbox.scale
        box = [
            max(0.0, x1), max(0.0, y1),
            min(float(letterbox.original_width), x2),
            min(float(letterbox.original_height), y2),
        ]
        candidates.append({"box": box, "class_id": class_id, "class_name": CLASS_NAMES[class_id],
                           "confidence": score})
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in candidates:
        if not any(previous["class_id"] == candidate["class_id"]
                   and _box_iou(previous["box"], candidate["box"]) > iou_threshold
                   for previous in kept):
            kept.append(candidate)
    return kept


def model_version(model_path=MODEL_FILE):
    path = Path(model_path)
    if not path.is_file():
        raise InferenceError(f"ONNX model file is missing: {path}")
    if path.stat().st_size < 8:
        raise InferenceError("ONNX model file is empty or truncated.")
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return f"yolo11n.onnx:{digest.hexdigest()[:12]}"


def validate_class_metadata(metadata):
    """Verify optional YOLO names metadata, accepting JSON or Python map notation."""
    labels = (metadata or {}).get("names") or (metadata or {}).get("class_names")
    if not labels:
        return
    try:
        if isinstance(labels, str):
            try:
                parsed = json.loads(labels)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(labels)
        else:
            parsed = labels
        if isinstance(parsed, dict):
            parsed = [parsed.get(index, parsed.get(str(index))) for index in range(len(CLASS_NAMES))]
        normalized = [str(label).casefold().replace("_", " ") for label in parsed]
        expected = [name.split(" ", 1)[1].casefold() for name in CLASS_NAMES]
        codes = [name.split(" ", 1)[0].casefold() for name in CLASS_NAMES]
        if len(normalized) != len(CLASS_NAMES) or any(
            expected_name not in actual_name and codes[index] not in actual_name
            for index, (expected_name, actual_name) in enumerate(zip(expected, normalized))
        ):
            raise InferenceError(f"Model class metadata does not match expected class order: {labels}")
    except (TypeError, ValueError, SyntaxError, KeyError) as error:
        if isinstance(error, InferenceError):
            raise
        raise InferenceError(f"Could not parse model class metadata: {labels}") from error


def create_cpu_session(model_path=MODEL_FILE):
    """Load and verify the upstream ONNX interface using CPUExecutionProvider only."""
    version = model_version(model_path)
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise InferenceError(
            "CPU ONNX Runtime is not installed. Run `python -m pip install -r requirements.txt`."
        ) from error
    try:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0
        session = ort.InferenceSession(str(model_path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise InferenceError("ONNX Runtime did not select CPUExecutionProvider.")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise InferenceError("Expected one image input and one detection output.")
        input_meta, output_meta = inputs[0], outputs[0]
        if input_meta.name != "images" or output_meta.name != "output0":
            raise InferenceError(
                f"Unsupported ONNX tensor names {input_meta.name!r} / {output_meta.name!r}; "
                "expected upstream names 'images' / 'output0'."
            )
        if input_meta.type != "tensor(float)" or output_meta.type != "tensor(float)":
            raise InferenceError(
                f"Expected float32 model input and output; found {input_meta.type} / {output_meta.type}."
            )
        input_shape = input_meta.shape
        if (len(input_shape) != 4 or input_shape[0] not in (1, "batch", None)
                or input_shape[1:] != [3, INPUT_SIZE, INPUT_SIZE]):
            raise InferenceError(f"Unsupported model input shape {input_shape}; expected [1,3,640,640].")
        shape = output_meta.shape
        if len(shape) != 3 or shape[0] not in (1, "batch", None) or shape[1:] != [8, 8400]:
            raise InferenceError(f"Unsupported model output shape {shape}; expected [1,8,8400].")
        metadata = session.get_modelmeta().custom_metadata_map or {}
        validate_class_metadata(metadata)
        return session, version, metadata
    except InferenceError:
        raise
    except Exception as error:
        raise InferenceError(f"ONNX model could not be loaded or is incompatible: {error}") from error


def annotate_image(image, detections):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, round(image.width / 40)))
    except OSError:
        font = ImageFont.load_default()
    line_width = max(2, round(image.width / 280))
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        color = CLASS_COLORS[detection["class_id"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{detection['class_name']} {detection['confidence']:.0%}"
        text_box = draw.textbbox((x1, y1), label, font=font)
        text_y = max(0, y1 - (text_box[3] - text_box[1]) - 5)
        draw.rectangle((x1, text_y, x1 + text_box[2] - text_box[0] + 8,
                        text_y + text_box[3] - text_box[1] + 5), fill=color)
        draw.text((x1 + 4, text_y + 2), label, fill="black", font=font)
    return annotated


def run_inference(image, session, model_version_value="unknown",
                  confidence_threshold=DEFAULT_CONFIDENCE):
    start = time.perf_counter()
    letterboxed = preprocess_image(image)
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    try:
        outputs = session.run([output_meta.name], {input_meta.name: letterboxed.tensor})
    except Exception as error:
        raise InferenceError(f"ONNX inference failed: {error}") from error
    detections = decode_output(outputs[0], letterboxed, confidence_threshold)
    duration_ms = (time.perf_counter() - start) * 1000
    best_score = float(np.max(np.asarray(outputs[0])[0, 4:, :]))
    if detections:
        status = "detections"
    elif best_score >= LOW_CONFIDENCE_THRESHOLD:
        status = "low_confidence"
        # Keep lower score candidates visible and persisted, clearly marked as below
        # the default display threshold. The upstream UI permits 0.05 as its minimum.
        detections = [item for item in decode_output(
            outputs[0], letterboxed, LOW_CONFIDENCE_THRESHOLD
        ) if item["confidence"] < confidence_threshold]
    else:
        status = "no_detections"
    return {
        "model_version": model_version_value,
        "detections": detections,
        "inference_status": status,
        "processing_duration_ms": duration_ms,
        "annotated_image": annotate_image(image, detections),
        "error_message": None,
    }


def ensure_detection_table(connection):
    """Add the AI detections table without modifying existing report data."""
    connection.execute(
        """CREATE TABLE IF NOT EXISTS ai_detections (
            detection_id TEXT PRIMARY KEY,
            report_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            bounding_boxes TEXT NOT NULL,
            classes TEXT NOT NULL,
            confidence_scores TEXT NOT NULL,
            inference_status TEXT NOT NULL,
            processing_duration_ms REAL,
            timestamp TEXT NOT NULL,
            annotated_image_path TEXT,
            error_message TEXT,
            FOREIGN KEY(report_id) REFERENCES road_reports(report_id)
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_detections_report_timestamp "
        "ON ai_detections(report_id, timestamp DESC)"
    )


def save_detection_result(report_id, result, database_path, annotated_dir=ANNOTATED_DIR):
    """Write annotation separately and persist one inference result transactionally."""
    detection_id = str(uuid.uuid4())
    detections = result.get("detections", [])
    annotated_path = None
    image = result.get("annotated_image")
    if image is not None:
        annotated_dir = Path(annotated_dir)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        destination = annotated_dir / f"{detection_id}.jpg"
        image.save(destination, format="JPEG", quality=92)
        annotated_path = str(destination.relative_to(BASE_DIR)) if destination.is_relative_to(BASE_DIR) else str(destination)
    try:
        connection = sqlite3.connect(database_path, timeout=10)
        try:
            with connection:
                ensure_detection_table(connection)
                connection.execute(
                    """INSERT INTO ai_detections (
                        detection_id, report_id, model_version, bounding_boxes, classes,
                        confidence_scores, inference_status, processing_duration_ms,
                        timestamp, annotated_image_path, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        detection_id, report_id, result.get("model_version", "unknown"),
                        json.dumps([item.get("box") for item in detections]),
                        json.dumps([item.get("class_name") for item in detections]),
                        json.dumps([item.get("confidence") for item in detections]),
                        result["inference_status"], result.get("processing_duration_ms"),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"), annotated_path,
                        result.get("error_message"),
                    ),
                )
                evidence_table = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='report_evidence'"
                ).fetchone()
                if evidence_table and annotated_path:
                    connection.execute(
                        """INSERT OR IGNORE INTO report_evidence (
                            evidence_id, report_id, image_type, image_path, original_filename,
                            media_type, recorded_actor, created_at
                        ) VALUES (?, ?, 'AI Annotation', ?, '', 'image/jpeg', 'AI analysis', ?)""",
                        (f"ai-annotation-{detection_id}", report_id, annotated_path,
                         datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    )
        finally:
            connection.close()
    except Exception:
        if annotated_path:
            Path(annotated_dir, f"{detection_id}.jpg").unlink(missing_ok=True)
        raise
    return detection_id


def fetch_detection_history(report_id, database_path):
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            ensure_detection_table(connection)
            rows = connection.execute(
                "SELECT * FROM ai_detections WHERE report_id = ? ORDER BY timestamp DESC",
                (report_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def path_under_project(relative_path):
    path = Path(relative_path)
    resolved = path.resolve() if path.is_absolute() else (BASE_DIR / path).resolve()
    if not resolved.is_relative_to(BASE_DIR):
        raise InferenceError("The saved photo path points outside the project directory.")
    return resolved

