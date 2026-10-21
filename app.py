"""Tamil Nadu road map and road-problem reporting prototype."""

import csv
import html
import io
import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import folium
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_folium import st_folium
import road_ai


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "tamil_nadu_districts.geojson"
DISTRICT_REGISTRY_FILE = DATA_DIR / "tamil_nadu_district_registry.json"
DATABASE_FILE = DATA_DIR / "road_reports.sqlite"
UPLOAD_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
MAX_PHOTO_BYTES = 5 * 1024 * 1024

PROBLEM_TYPES = (
    "Pothole", "Crack", "Broken Road", "Other",
)
SEVERITIES = ("Low", "Medium", "High", "Critical")
PENDING_VERIFICATION_STATUS = "Repair Completed \u2014 Pending Verification"
REPORT_STATUSES = (
    "Submitted",
    "Pending Validation",
    "Validated",
    "Assigned",
    "Repair Scheduled",
    "Repair In Progress",
    PENDING_VERIFICATION_STATUS,
    "Verified and Closed",
    "Rejected",
    "Reopened",
)
STATUS_TRANSITIONS = {
    "Submitted": ("Pending Validation", "Rejected"),
    "Pending Validation": ("Validated", "Rejected"),
    "Validated": ("Assigned", "Rejected"),
    "Assigned": ("Repair Scheduled",),
    "Repair Scheduled": ("Repair In Progress",),
    "Repair In Progress": (PENDING_VERIFICATION_STATUS,),
    PENDING_VERIFICATION_STATUS: ("Verified and Closed", "Repair In Progress", "Reopened"),
    "Verified and Closed": ("Reopened",),
    "Rejected": ("Reopened",),
    "Reopened": ("Pending Validation", "Rejected"),
}
FILTER_STATUSES = REPORT_STATUSES
LOCAL_ACTOR = "Local app operator (unauthenticated)"


def load_district_registry(registry_path=DISTRICT_REGISTRY_FILE):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    districts = registry.get("districts", [])
    validate_district_registry(districts)
    return registry


def normalize_district_name(name):
    return "".join(character for character in str(name).casefold() if character.isalnum())


def validate_district_registry(districts):
    ids = set()
    names = set()
    for entry in districts:
        district_id = entry.get("id")
        name = entry.get("name")
        if not district_id or not name:
            raise ValueError("Every district registry entry needs an id and displayed name.")
        if district_id in ids:
            raise ValueError(f"Duplicate canonical district id: {district_id}")
        ids.add(district_id)
        for value in [name, *entry.get("aliases", [])]:
            normalized = normalize_district_name(value)
            if normalized in names:
                raise ValueError(f"Duplicate canonical district name or alias: {value}")
            names.add(normalized)
    return True


DISTRICT_REGISTRY = load_district_registry()
DISTRICTS = DISTRICT_REGISTRY["districts"]
DISTRICT_BY_ID = {entry["id"]: entry for entry in DISTRICTS}
DISTRICT_NAME_TO_ENTRY = {
    normalize_district_name(value): entry
    for entry in DISTRICTS
    for value in [entry["name"], *entry.get("aliases", [])]
}
DISTRICT_NAMES = tuple(entry["name"] for entry in DISTRICTS)


def coordinate_bounds(value, bounds):
    """Collect GeoJSON longitude/latitude pairs into a lat/lon bounding box."""
    if (isinstance(value, list) and len(value) >= 2
            and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float))):
        longitude, latitude = value[:2]
        bounds[0] = min(bounds[0], latitude)
        bounds[1] = min(bounds[1], longitude)
        bounds[2] = max(bounds[2], latitude)
        bounds[3] = max(bounds[3], longitude)
    elif isinstance(value, list):
        for item in value:
            coordinate_bounds(item, bounds)


def district_entry_for_name(name):
    return DISTRICT_NAME_TO_ENTRY.get(normalize_district_name(name))


def _orientation(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, point, epsilon=1e-10):
    return (
        min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
        and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
        and abs(_orientation(a, b, point)) <= epsilon
    )


def _segments_intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    o1, o2, o3, o4 = _orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 1e-10 and o2 < -1e-10) or (o1 < -1e-10 and o2 > 1e-10)) and (
            (o3 > 1e-10 and o4 < -1e-10) or (o3 < -1e-10 and o4 > 1e-10)):
        return True
    return (
        (abs(o1) <= 1e-10 and _on_segment(a, b, c))
        or (abs(o2) <= 1e-10 and _on_segment(a, b, d))
        or (abs(o3) <= 1e-10 and _on_segment(c, d, a))
        or (abs(o4) <= 1e-10 and _on_segment(c, d, b))
    )


def _ring_is_valid(ring):
    if not isinstance(ring, list) or len(ring) < 4:
        return False, "a linear ring must contain at least four positions"
    points = []
    for position in ring:
        if (not isinstance(position, (list, tuple)) or len(position) < 2
                or not all(isinstance(value, (int, float)) and math.isfinite(value)
                           for value in position[:2])):
            return False, "a ring contains an invalid coordinate"
        longitude, latitude = position[:2]
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            return False, "a coordinate is outside longitude/latitude bounds"
        points.append((longitude, latitude))
    if points[0] != points[-1]:
        return False, "a linear ring is not closed"
    if len(set(points[:-1])) < 3:
        return False, "a ring must contain at least three distinct positions"
    area = sum(
        points[index][0] * points[index + 1][1]
        - points[index + 1][0] * points[index][1]
        for index in range(len(points) - 1)
    )
    if abs(area) <= 1e-12:
        return False, "a ring has zero area"

    # Sweep by segment minimum longitude, comparing only active segments whose
    # bounding boxes overlap. Adjacent segments are allowed to share endpoints.
    segments = []
    segment_count = len(points) - 1
    for index in range(segment_count):
        a, b = points[index], points[index + 1]
        segments.append((min(a[0], b[0]), max(a[0], b[0]),
                         min(a[1], b[1]), max(a[1], b[1]), index, a, b))
    active = []
    for current in sorted(segments):
        current_min_x, current_max_x, current_min_y, current_max_y, current_index, a, b = current
        active = [segment for segment in active if segment[1] + 1e-10 >= current_min_x]
        for other in active:
            _, _, other_min_y, other_max_y, other_index, c, d = other
            if (current_index - other_index) % segment_count in (1, segment_count - 1):
                continue
            if current_min_y <= other_max_y + 1e-10 and other_min_y <= current_max_y + 1e-10:
                if _segments_intersect(a, b, c, d):
                    return False, "a ring self-intersects"
        active.append(current)
    return True, None


def _rings_intersect(first_ring, second_ring):
    first_segments = list(zip(first_ring, first_ring[1:]))
    second_segments = list(zip(second_ring, second_ring[1:]))
    for a, b in first_segments:
        first_min_x, first_max_x = sorted((a[0], b[0]))
        first_min_y, first_max_y = sorted((a[1], b[1]))
        for c, d in second_segments:
            if (max(c[0], d[0]) < first_min_x or min(c[0], d[0]) > first_max_x
                    or max(c[1], d[1]) < first_min_y or min(c[1], d[1]) > first_max_y):
                continue
            if _segments_intersect(a, b, c, d):
                return True
    return False


def validate_district_geometry(geometry):
    """Validate GeoJSON polygon structure and self-intersecting linear rings."""
    if not isinstance(geometry, dict):
        return False, "missing geometry"
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type not in ("Polygon", "MultiPolygon"):
        return False, "geometry must be a non-empty Polygon or MultiPolygon"
    polygons = [coordinates] if geometry_type == "Polygon" else coordinates
    if not isinstance(polygons, list) or not polygons:
        return False, "geometry must be a non-empty Polygon or MultiPolygon"
    for polygon in polygons:
        if not isinstance(polygon, list) or not polygon:
            return False, "a polygon has no exterior ring"
        for ring in polygon:
            valid, message = _ring_is_valid(ring)
            if not valid:
                return False, message
        exterior = polygon[0]
        for hole_index, hole in enumerate(polygon[1:]):
            if (_point_ring_relation(hole[0][0], hole[0][1], exterior) != "inside"
                    or _rings_intersect(exterior, hole)):
                return False, "an interior ring is outside or intersects its exterior ring"
            for other_hole in polygon[1:hole_index + 1]:
                if (_rings_intersect(hole, other_hole)
                        or _point_ring_relation(hole[0][0], hole[0][1], other_hole) != "outside"
                        or _point_ring_relation(other_hole[0][0], other_hole[0][1], hole) != "outside"):
                    return False, "interior rings overlap"
    if geometry_type == "MultiPolygon":
        for index, polygon in enumerate(polygons):
            for other in polygons[index + 1:]:
                if (_rings_intersect(polygon[0], other[0])
                        or _point_polygon_relation(polygon[0][0][0], polygon[0][0][1], other) != "outside"
                        or _point_polygon_relation(other[0][0][0], other[0][0][1], polygon) != "outside"):
                    return False, "multipolygon parts overlap or touch"
    return True, None


def _point_ring_relation(longitude, latitude, ring):
    point = (longitude, latitude)
    inside = False
    for first, second in zip(ring, ring[1:]):
        a, b = tuple(first[:2]), tuple(second[:2])
        if _on_segment(a, b, point, epsilon=1e-9):
            return "boundary"
        if (a[1] > latitude) != (b[1] > latitude):
            crossing_x = (b[0] - a[0]) * (latitude - a[1]) / (b[1] - a[1]) + a[0]
            if longitude < crossing_x:
                inside = not inside
    return "inside" if inside else "outside"


def _point_polygon_relation(longitude, latitude, polygon):
    exterior = _point_ring_relation(longitude, latitude, polygon[0])
    if exterior != "inside":
        return exterior
    for hole in polygon[1:]:
        hole_relation = _point_ring_relation(longitude, latitude, hole)
        if hole_relation == "boundary":
            return "boundary"
        if hole_relation == "inside":
            return "outside"
    return "inside"


def match_district(longitude, latitude, districts):
    """Return an unambiguous feature match; boundary and uncovered points stay unmatched."""
    if (not isinstance(longitude, (int, float)) or not isinstance(latitude, (int, float))
            or not math.isfinite(longitude) or not math.isfinite(latitude)
            or not -180 <= longitude <= 180 or not -90 <= latitude <= 90):
        return {"status": "invalid_coordinate", "district_id": None, "district_name": None}
    interiors, boundaries = [], []
    for feature in districts.get("features", []):
        properties = feature.get("properties") or {}
        district_id = properties.get("district_id")
        geometry = feature.get("geometry")
        valid = properties.get("_geometry_valid")
        if valid is None:
            valid, _ = validate_district_geometry(geometry)
        if not valid or not district_id:
            continue
        geometry_type = geometry["type"]
        polygons = [geometry["coordinates"]] if geometry_type == "Polygon" else geometry["coordinates"]
        relations = [_point_polygon_relation(longitude, latitude, polygon) for polygon in polygons]
        if "inside" in relations:
            interiors.append((district_id, properties.get("district_name") or properties.get("shapeName")))
        elif "boundary" in relations:
            boundaries.append((district_id, properties.get("district_name") or properties.get("shapeName")))
    if len(interiors) == 1 and not boundaries:
        district_id, district_name = interiors[0]
        return {"status": "matched", "district_id": district_id, "district_name": district_name}
    if interiors or boundaries:
        return {"status": "boundary_or_overlap", "district_id": None, "district_name": None}
    return {"status": "no_valid_coverage", "district_id": None, "district_name": None}


@contextmanager
def database_connection(database_path=DATABASE_FILE):
    """Open a transaction-scoped SQLite connection and always close it."""
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def ensure_phase4_tables(connection):
    """Apply additive authority, assignment, repair, evidence and verification schema."""
    connection.execute("""CREATE TABLE IF NOT EXISTS authorities (
        authority_id TEXT PRIMARY KEY,
        authority_name TEXT NOT NULL,
        authority_type TEXT NOT NULL,
        jurisdiction TEXT NOT NULL DEFAULT '',
        district_ids TEXT NOT NULL DEFAULT '[]',
        road_categories TEXT NOT NULL DEFAULT '[]',
        routing_rules TEXT NOT NULL DEFAULT '',
        contact_details TEXT NOT NULL DEFAULT '',
        active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_authorities_active ON authorities(active, authority_name)")
    connection.execute("""CREATE TABLE IF NOT EXISTS assignments (
        assignment_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        authority_id TEXT NOT NULL,
        assigned_team TEXT NOT NULL,
        assigned_officer TEXT NOT NULL DEFAULT '',
        assigned_actor TEXT NOT NULL,
        assigned_at TEXT NOT NULL,
        due_at TEXT,
        acceptance_status TEXT NOT NULL DEFAULT 'Awaiting response',
        acceptance_actor TEXT,
        acceptance_at TEXT,
        acceptance_reference TEXT,
        assignment_status TEXT NOT NULL DEFAULT 'Active',
        manual_routing_reason TEXT NOT NULL,
        FOREIGN KEY(report_id) REFERENCES road_reports(report_id),
        FOREIGN KEY(authority_id) REFERENCES authorities(authority_id)
    )""")
    authority_columns = {row[1] for row in connection.execute("PRAGMA table_info(authorities)")}
    if "road_categories" not in authority_columns:
        connection.execute("ALTER TABLE authorities ADD COLUMN road_categories TEXT NOT NULL DEFAULT '[]'")
    if "routing_rules" not in authority_columns:
        connection.execute("ALTER TABLE authorities ADD COLUMN routing_rules TEXT NOT NULL DEFAULT ''")
    assignment_columns = {row[1] for row in connection.execute("PRAGMA table_info(assignments)")}
    if "assigned_officer" not in assignment_columns:
        connection.execute("ALTER TABLE assignments ADD COLUMN assigned_officer TEXT NOT NULL DEFAULT ''")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_assignments_report ON assignments(report_id, assigned_at DESC)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_assignments_due ON assignments(assignment_status, due_at)")
    connection.execute("""CREATE TABLE IF NOT EXISTS assignment_history (
        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
        assignment_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        actor TEXT NOT NULL,
        reason TEXT NOT NULL,
        changed_at TEXT NOT NULL,
        FOREIGN KEY(assignment_id) REFERENCES assignments(assignment_id)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_assignment_history_assignment ON assignment_history(assignment_id, history_id)")
    connection.execute("""CREATE TABLE IF NOT EXISTS manual_review_queue (
        queue_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        review_reason TEXT NOT NULL,
        queue_status TEXT NOT NULL DEFAULT 'Open',
        created_at TEXT NOT NULL,
        resolved_at TEXT,
        resolved_by TEXT,
        resolution TEXT,
        FOREIGN KEY(report_id) REFERENCES road_reports(report_id)
    )""")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_manual_review_open_report ON manual_review_queue(report_id) WHERE queue_status = 'Open'")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_manual_review_status ON manual_review_queue(queue_status, created_at)")
    connection.execute("""CREATE TABLE IF NOT EXISTS repair_records (
        repair_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        assignment_id TEXT,
        repair_status TEXT NOT NULL,
        repair_method TEXT NOT NULL DEFAULT '',
        materials TEXT NOT NULL DEFAULT '',
        repair_notes TEXT NOT NULL DEFAULT '',
        scheduled_at TEXT,
        started_at TEXT,
        completed_at TEXT,
        recorded_actor TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(report_id) REFERENCES road_reports(report_id),
        FOREIGN KEY(assignment_id) REFERENCES assignments(assignment_id)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_repair_records_report ON repair_records(report_id, created_at DESC)")
    connection.execute("""CREATE TABLE IF NOT EXISTS report_evidence (
        evidence_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        assignment_id TEXT,
        repair_id TEXT,
        image_type TEXT NOT NULL CHECK(image_type IN ('Original Evidence','AI Annotation','Repair Before','Repair After')),
        image_path TEXT NOT NULL UNIQUE,
        original_filename TEXT NOT NULL DEFAULT '',
        media_type TEXT NOT NULL DEFAULT 'image/jpeg',
        recorded_actor TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(report_id) REFERENCES road_reports(report_id),
        FOREIGN KEY(assignment_id) REFERENCES assignments(assignment_id),
        FOREIGN KEY(repair_id) REFERENCES repair_records(repair_id)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_report_evidence_report_type ON report_evidence(report_id, image_type, created_at)")
    connection.execute("""CREATE TABLE IF NOT EXISTS verifications (
        verification_id TEXT PRIMARY KEY,
        report_id TEXT NOT NULL,
        repair_id TEXT NOT NULL,
        reviewer_actor TEXT NOT NULL,
        decision TEXT NOT NULL CHECK(decision IN ('Approved','Rejected','More Evidence')),
        comments TEXT NOT NULL,
        evidence_references TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        FOREIGN KEY(report_id) REFERENCES road_reports(report_id),
        FOREIGN KEY(repair_id) REFERENCES repair_records(repair_id)
    )""")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_verifications_report ON verifications(report_id, created_at DESC)")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Preserve existing originals and AI annotations as evidence references; paths and files stay put.
    connection.execute("""INSERT OR IGNORE INTO report_evidence (
        evidence_id, report_id, image_type, image_path, original_filename, media_type,
        recorded_actor, created_at
    ) SELECT 'legacy-original-' || report_id, report_id, 'Original Evidence', photo_path,
        '', 'image/jpeg', 'System migration', submitted_at
      FROM road_reports WHERE photo_path IS NOT NULL AND TRIM(photo_path) <> ''""")
    connection.execute("""INSERT OR IGNORE INTO report_evidence (
        evidence_id, report_id, image_type, image_path, original_filename, media_type,
        recorded_actor, created_at
    ) SELECT 'ai-annotation-' || detection_id, report_id, 'AI Annotation', annotated_image_path,
        '', 'image/jpeg', 'System migration', timestamp
      FROM ai_detections WHERE annotated_image_path IS NOT NULL AND TRIM(annotated_image_path) <> ''""")

    # Reports with no existing routing record enter the manual queue. No authority
    # is inferred from districts; a human must verify road ownership and scope.
    for row in connection.execute("""SELECT report_id, submitted_at FROM road_reports r
        WHERE r.status NOT IN ('Verified and Closed','Rejected')
          AND NOT EXISTS (SELECT 1 FROM manual_review_queue q WHERE q.report_id = r.report_id)""").fetchall():
        connection.execute("""INSERT INTO manual_review_queue (
            queue_id, report_id, review_reason, queue_status, created_at
        ) VALUES (?, ?, ?, 'Open', ?)""",
        (str(uuid.uuid4()), row["report_id"],
         "Authority and road ownership are not verified; confirm routing manually.",
         row["submitted_at"] or now))


def initialize_database(database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS road_reports (
                report_id TEXT PRIMARY KEY,
                submitted_at TEXT NOT NULL,
                road_name TEXT NOT NULL,
                location TEXT NOT NULL DEFAULT '',
                district TEXT NOT NULL,
                district_id TEXT,
                problem_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                photo_path TEXT,
                status TEXT NOT NULL DEFAULT 'Submitted',
                submission_token TEXT
            )"""
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(road_reports)")}
        if "location" not in columns:
            connection.execute(
                "ALTER TABLE road_reports ADD COLUMN location TEXT NOT NULL DEFAULT ''"
            )
        if "district_id" not in columns:
            connection.execute("ALTER TABLE road_reports ADD COLUMN district_id TEXT")
        if "submission_token" not in columns:
            connection.execute("ALTER TABLE road_reports ADD COLUMN submission_token TEXT")
        connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_road_reports_submission_token "
            "ON road_reports(submission_token) WHERE submission_token IS NOT NULL"
        )
        for row in connection.execute(
                "SELECT report_id, district FROM road_reports WHERE district_id IS NULL"):
            entry = district_entry_for_name(row["district"])
            if entry:
                connection.execute(
                    "UPDATE road_reports SET district_id = ? WHERE report_id = ? AND district_id IS NULL",
                    (entry["id"], row["report_id"]),
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_road_reports_filters "
            "ON road_reports(district, problem_type, severity, status)"
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS report_status_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                verification_evidence TEXT,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(report_id) REFERENCES road_reports(report_id)
            )"""
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_report_status_history_report "
            "ON report_status_history(report_id, history_id)"
        )
        road_ai.ensure_detection_table(connection)
        ensure_phase4_tables(connection)

        # Map only known legacy labels. In particular, Resolved means repair was
        # reported complete, not independently verified and closed.
        legacy_status_map = {
            "Pending": "Pending Validation",
            "Pending Review": "Pending Validation",
            "Resolved": PENDING_VERIFICATION_STATUS,
        }
        reports = connection.execute(
            "SELECT report_id, status FROM road_reports ORDER BY submitted_at, report_id"
        ).fetchall()
        for report in reports:
            report_id, old_status = report["report_id"], report["status"]
            if old_status == "Repair Completed \ufffd Pending Verification":
                old_status = PENDING_VERIFICATION_STATUS
                connection.execute("UPDATE road_reports SET status=? WHERE report_id=?", (old_status, report_id))
                connection.execute("UPDATE report_status_history SET previous_status=? WHERE report_id=? AND previous_status=?",
                                   (old_status, report_id, "Repair Completed \ufffd Pending Verification"))
                connection.execute("UPDATE report_status_history SET new_status=? WHERE report_id=? AND new_status=?",
                                   (old_status, report_id, "Repair Completed \ufffd Pending Verification"))
            has_history = connection.execute(
                "SELECT 1 FROM report_status_history WHERE report_id = ? LIMIT 1",
                (report_id,),
            ).fetchone()
            if has_history:
                continue
            new_status = legacy_status_map.get(old_status, old_status)
            if new_status != old_status:
                connection.execute(
                    "UPDATE road_reports SET status = ? WHERE report_id = ?",
                    (new_status, report_id),
                )
                reason = f"Legacy status migration: {old_status} mapped to {new_status}."
                previous_status = old_status
            else:
                reason = "Existing status retained during lifecycle migration."
                previous_status = None
            connection.execute(
                """INSERT INTO report_status_history (
                    report_id, previous_status, new_status, actor, reason,
                    verification_evidence, changed_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?)""",
                (report_id, previous_status, new_status, "System migration", reason,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )


def inspect_image_upload(uploaded_file):
    """Validate an uploaded image by size and file signature, not its filename."""
    if uploaded_file is None:
        return None, None
    image_bytes = uploaded_file.getvalue()
    if not image_bytes:
        raise ValueError("The uploaded photo is empty.")
    if len(image_bytes) > MAX_PHOTO_BYTES:
        raise ValueError("Photo must be 5 MB or smaller.")
    if image_bytes.startswith(b"\xff\xd8\xff"):
        extension = ".jpg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = ".png"
    elif (len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF"
          and image_bytes[8:12] == b"WEBP"):
        extension = ".webp"
    else:
        raise ValueError("Upload a valid JPEG, PNG, or WebP image.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
            expected_format = {".jpg": "JPEG", ".png": "PNG", ".webp": "WEBP"}[extension]
            if image.format != expected_format:
                raise ValueError("The photo content does not match its file format.")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("The uploaded file is not a valid image.") from error
    return image_bytes, extension


def valid_saved_repair_photo(path):
    """Return true only when a saved repair-evidence path resolves to a valid image."""
    if not path:
        return False
    try:
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path
        image_bytes = image_path.read_bytes()
        inspect_image_upload(io.BytesIO(image_bytes))
    except (OSError, ValueError, TypeError):
        return False
    return True


def save_report(report, database_path=DATABASE_FILE, photo_bytes=None, photo_extension=None,
                upload_dir=UPLOAD_DIR, submission_token=None):
    """Persist a report and its optional photo; return its ID after commit."""
    district_id = report.get("district_id")
    entry = DISTRICT_BY_ID.get(district_id) if district_id else district_entry_for_name(report.get("district"))
    if entry is None:
        raise ValueError("Choose a district from the canonical Tamil Nadu district registry.")
    supplied_name = report.get("district")
    if supplied_name and district_entry_for_name(supplied_name) != entry:
        raise ValueError("The district name and canonical district identifier do not match.")
    submission_token = str(submission_token).strip() if submission_token is not None else None
    if submission_token == "":
        submission_token = None
    if submission_token and len(submission_token) > 128:
        raise ValueError("The report submission token is invalid.")
    if submission_token:
        with database_connection(database_path) as connection:
            existing = connection.execute(
                "SELECT report_id FROM road_reports WHERE submission_token=?", (submission_token,)
            ).fetchone()
        if existing:
            return existing["report_id"]
    report_id = str(uuid.uuid4())
    photo_path = None
    destination = None
    if photo_bytes is not None:
        if photo_extension not in (".jpg", ".png", ".webp"):
            raise ValueError("Unsupported photo format.")
        upload_dir = Path(upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"{report_id}{photo_extension}"
        photo_path = str(destination.relative_to(BASE_DIR)) if destination.is_relative_to(BASE_DIR) else str(destination)
        destination.write_bytes(photo_bytes)

    try:
        with database_connection(database_path) as connection:
            connection.execute(
                """INSERT INTO road_reports (
                report_id, submitted_at, road_name, location, district, district_id, problem_type,
                    severity, description, latitude, longitude, photo_path, status, submission_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    report["road_name"], report.get("location", ""), entry["name"], entry["id"], report["problem_type"],
                    report["severity"], report["description"], report.get("latitude"),
                    report.get("longitude"), photo_path, "Submitted", submission_token,
                ),
            )
            connection.execute(
                """INSERT INTO report_status_history (
                    report_id, previous_status, new_status, actor, reason,
                    verification_evidence, changed_at
                ) VALUES (?, NULL, ?, ?, ?, NULL, ?)""",
                (report_id, "Submitted", LOCAL_ACTOR, "Report submitted",
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            connection.execute(
                """INSERT INTO manual_review_queue (
                    queue_id, report_id, review_reason, queue_status, created_at
                ) VALUES (?, ?, ?, 'Open', ?)""",
                (str(uuid.uuid4()), report_id,
                 "Authority and road ownership are not verified; confirm routing manually.",
                 datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
    except sqlite3.IntegrityError:
        if destination is not None:
            destination.unlink(missing_ok=True)
        if submission_token:
            with database_connection(database_path) as connection:
                existing = connection.execute(
                    "SELECT report_id FROM road_reports WHERE submission_token=?", (submission_token,)
                ).fetchone()
            if existing:
                return existing["report_id"]
        raise
    except Exception:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    return report_id


ASSIGNMENT_WORK_STATUSES = ("Assigned", "Repair Scheduled")
OVERDUE_STATUSES = (*ASSIGNMENT_WORK_STATUSES, "Repair In Progress", PENDING_VERIFICATION_STATUS)


def _report_filter_sql(filters=None):
    """Build shared, parameterized filters for history rows and dashboard counts."""
    filters = filters or {}
    clauses, values = [], []
    for field in ("district_id", "district", "problem_type", "severity", "status"):
        selected = filters.get(field)
        if selected and selected != "All":
            if field == "district_id" and str(selected).startswith("legacy:"):
                clauses.append("r.district_id IS NULL AND r.district = ?")
                values.append(str(selected)[len("legacy:"):])
            else:
                clauses.append(f"r.{field} = ?")
                values.append(selected)
    statuses = filters.get("statuses")
    if statuses:
        clauses.append("r.status IN (" + ",".join("?" for _ in statuses) + ")")
        values.extend(statuses)
    if filters.get("submitted_from"):
        clauses.append("date(r.submitted_at) >= date(?)")
        values.append(str(filters["submitted_from"]))
    if filters.get("submitted_to"):
        clauses.append("date(r.submitted_at) <= date(?)")
        values.append(str(filters["submitted_to"]))

    authority_id = filters.get("authority_id")
    if authority_id == "__unassigned__":
        clauses.append("NOT EXISTS (SELECT 1 FROM assignments af WHERE af.report_id=r.report_id AND af.assignment_status='Active')")
    elif authority_id and authority_id != "All":
        clauses.append("EXISTS (SELECT 1 FROM assignments af WHERE af.report_id=r.report_id AND af.authority_id=?)")
        values.append(authority_id)

    outcome = filters.get("ai_outcome")
    latest_ai = "SELECT ad.inference_status FROM ai_detections ad WHERE ad.report_id=r.report_id ORDER BY ad.timestamp DESC, ad.rowid DESC LIMIT 1"
    if outcome == "none":
        clauses.append(f"NOT EXISTS ({latest_ai})")
    elif outcome == "detections":
        clauses.append(f"({latest_ai}) = 'detections'")
    elif outcome == "no_detections":
        clauses.append(f"({latest_ai}) = 'no_detections'")
    elif outcome == "low_confidence":
        clauses.append(f"({latest_ai}) = 'low_confidence'")
    elif outcome == "error":
        clauses.append(f"({latest_ai}) = 'error'")

    bucket = filters.get("workflow_bucket")
    active = "EXISTS (SELECT 1 FROM assignments aw WHERE aw.report_id=r.report_id AND aw.assignment_status='Active')"
    if bucket == "open":
        clauses.append("r.status NOT IN ('Verified and Closed','Rejected')")
    elif bucket == "unassigned":
        clauses.extend([
            "r.status IN ('Submitted','Pending Validation','Validated','Assigned','Repair Scheduled')",
            f"NOT {active}",
        ])
    elif bucket == "assigned":
        clauses.extend(["r.status IN ('Assigned','Repair Scheduled')", active])
    elif bucket == "repair_in_progress":
        clauses.append("r.status='Repair In Progress'")
    elif bucket == "awaiting_verification":
        clauses.append("r.status=?")
        values.append(PENDING_VERIFICATION_STATUS)
    elif bucket == "verified":
        clauses.append("r.status='Verified and Closed'")
    elif bucket == "reopened":
        clauses.append("r.status='Reopened'")
    elif bucket == "manual_review":
        clauses.append("EXISTS (SELECT 1 FROM manual_review_queue mq WHERE mq.report_id=r.report_id AND mq.queue_status='Open')")
    elif bucket == "overdue":
        clauses.extend([
            f"EXISTS (SELECT 1 FROM assignments ao WHERE ao.report_id=r.report_id AND ao.assignment_status='Active' AND ao.due_at IS NOT NULL AND datetime(ao.due_at) < datetime(?))",
            "r.status IN (" + ",".join("?" for _ in OVERDUE_STATUSES) + ")",
        ])
        values.append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
        values.extend(OVERDUE_STATUSES)
    return clauses, values


def fetch_reports(database_path=DATABASE_FILE, filters=None):
    clauses, values = _report_filter_sql(filters)
    query = "SELECT r.* FROM road_reports r"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY r.submitted_at DESC"
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query, values).fetchall()]


def count_reports(filters=None, database_path=DATABASE_FILE):
    clauses, values = _report_filter_sql(filters)
    query = "SELECT COUNT(*) FROM road_reports r"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    with database_connection(database_path) as connection:
        return int(connection.execute(query, values).fetchone()[0])


def fetch_dashboard_stats(database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        total = connection.execute("SELECT COUNT(*) FROM road_reports").fetchone()[0]
        pending = count_reports({"workflow_bucket": "open"}, database_path)
        resolved = count_reports({"workflow_bucket": "verified"}, database_path)
        categories = connection.execute(
            "SELECT problem_type, COUNT(*) AS count FROM road_reports "
            "GROUP BY problem_type ORDER BY problem_type"
        ).fetchall()
    return total, pending, resolved, {row["problem_type"]: row["count"] for row in categories}


def available_problem_types(database_path=DATABASE_FILE):
    """Include current form options and legacy categories from saved reports."""
    try:
        with database_connection(database_path) as connection:
            saved_types = [row[0] for row in connection.execute(
                "SELECT DISTINCT problem_type FROM road_reports WHERE problem_type IS NOT NULL"
            )]
    except sqlite3.Error:
        return list(PROBLEM_TYPES)
    return list(PROBLEM_TYPES) + sorted(set(saved_types) - set(PROBLEM_TYPES))


class InvalidStatusTransition(ValueError):
    """Raised when a requested report status change is not permitted."""


def permitted_next_statuses(current_status):
    return STATUS_TRANSITIONS.get(current_status, ())


def transition_report_status(report_id, new_status, reason, actor=LOCAL_ACTOR,
                             verification_evidence=None, database_path=DATABASE_FILE):
    reason = (reason or "").strip()
    actor = (actor or "").strip()
    verification_evidence = (verification_evidence or "").strip() or None
    if not reason:
        raise ValueError("A reason is required for every status change.")
    if not actor:
        raise ValueError("An acting user or actor is required.")
    if new_status not in REPORT_STATUSES:
        raise ValueError("Choose one of the canonical report statuses.")
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _transition_report_status_in_transaction(
            connection, report_id, new_status, reason, actor, verification_evidence
        )
    return True


def _transition_report_status_in_transaction(connection, report_id, new_status,
                                             reason, actor, verification_evidence=None):
    """Apply lifecycle rules and append history inside the caller's transaction."""
    reason = (reason or "").strip()
    actor = (actor or "").strip()
    verification_evidence = (verification_evidence or "").strip() or None
    if not reason:
        raise ValueError("A reason is required for every status change.")
    if not actor:
        raise ValueError("An acting user or actor is required.")
    if new_status not in REPORT_STATUSES:
        raise ValueError("Choose one of the canonical report statuses.")
    report = connection.execute(
        "SELECT status FROM road_reports WHERE report_id = ?", (report_id,)
    ).fetchone()
    if report is None:
        raise ValueError("The report was not found.")
    current_status = report["status"]
    allowed = permitted_next_statuses(current_status)
    if new_status not in allowed:
        next_steps = ", ".join(allowed) if allowed else "none (the current legacy status is unsupported)"
        raise InvalidStatusTransition(
            f"Cannot move report from {current_status!r} to {new_status!r}. "
            f"Permitted next step(s): {next_steps}."
        )

    if new_status == "Assigned":
        assignment = connection.execute(
            "SELECT 1 FROM assignments WHERE report_id = ? AND assignment_status = 'Active' LIMIT 1",
            (report_id,),
        ).fetchone()
        if assignment is None:
            raise InvalidStatusTransition("An authority assignment must be recorded before moving to Assigned.")
    elif new_status == "Repair Scheduled":
        repair = connection.execute(
            "SELECT 1 FROM repair_records WHERE report_id = ? AND repair_status = 'Scheduled' LIMIT 1",
            (report_id,),
        ).fetchone()
        if repair is None:
            raise InvalidStatusTransition("Record the repair schedule before moving to Repair Scheduled.")
    elif new_status == "Repair In Progress":
        repair = connection.execute(
            "SELECT 1 FROM repair_records WHERE report_id = ? AND repair_status = 'In Progress' LIMIT 1",
            (report_id,),
        ).fetchone()
        if repair is None:
            raise InvalidStatusTransition("Record repair work in progress before changing this status.")
    elif new_status == PENDING_VERIFICATION_STATUS:
        repair = connection.execute(
            """SELECT repair_id, repair_notes FROM repair_records WHERE report_id = ?
               AND repair_status = 'Completed' ORDER BY completed_at DESC, rowid DESC LIMIT 1""",
            (report_id,),
        ).fetchone()
        if repair is None or not (repair["repair_notes"] or "").strip():
            raise InvalidStatusTransition("Completed repair notes are required before verification.")
        after_photo = connection.execute(
            "SELECT image_path FROM report_evidence WHERE report_id = ? AND repair_id = ? AND image_type = 'Repair After' LIMIT 1",
            (report_id, repair["repair_id"]),
        ).fetchone()
        if after_photo is None or not valid_saved_repair_photo(after_photo["image_path"]):
            raise InvalidStatusTransition("A valid saved repair-after photo is required before verification.")
    elif new_status == "Verified and Closed":
        if not verification_evidence:
            raise InvalidStatusTransition("Only an explicit successful verification decision can close a report.")
        latest_verification = connection.execute(
            """SELECT decision, repair_id FROM verifications WHERE report_id = ?
               ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (report_id,),
        ).fetchone()
        if latest_verification is None or latest_verification["decision"] != "Approved":
            raise InvalidStatusTransition("A successful verification decision is required before closure.")
        completed = connection.execute(
            """SELECT r.repair_id FROM repair_records r WHERE r.report_id = ?
               AND r.repair_status = 'Completed' AND TRIM(r.repair_notes) <> ''
               ORDER BY r.completed_at DESC, r.rowid DESC LIMIT 1""",
            (report_id,),
        ).fetchone()
        after_photo = None if completed is None else connection.execute(
            "SELECT image_path FROM report_evidence WHERE report_id = ? AND repair_id = ? AND image_type = 'Repair After' LIMIT 1",
            (report_id, completed["repair_id"]),
        ).fetchone()
        if (completed is None or latest_verification["repair_id"] != completed["repair_id"]
                or after_photo is None or not valid_saved_repair_photo(after_photo["image_path"])):
            raise InvalidStatusTransition("Repair completion notes and after-photo evidence are required before closure.")

    cursor = connection.execute(
        "UPDATE road_reports SET status = ? WHERE report_id = ? AND status = ?",
        (new_status, report_id, current_status),
    )
    if cursor.rowcount != 1:
        raise InvalidStatusTransition("The report changed concurrently; reload and try again.")
    connection.execute(
        """INSERT INTO report_status_history (
            report_id, previous_status, new_status, actor, reason,
            verification_evidence, changed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (report_id, current_status, new_status, actor, reason, verification_evidence,
         datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def fetch_report_status_history(report_id, database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM report_status_history WHERE report_id = ? ORDER BY history_id",
            (report_id,),
        ).fetchall()]


def _iso_datetime(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def save_authority(authority_name, authority_type, jurisdiction, district_ids=(),
                   contact_details="", active=True, authority_id=None,
                   database_path=DATABASE_FILE, road_categories=(), routing_rules=""):
    authority_name = (authority_name or "").strip()
    authority_type = (authority_type or "").strip()
    if not authority_name or not authority_type:
        raise ValueError("Authority name and type are required.")
    district_ids = list(dict.fromkeys(district_ids or []))
    road_categories = list(dict.fromkeys(
        str(category).strip() for category in (road_categories or []) if str(category).strip()
    ))
    unknown = [district_id for district_id in district_ids if district_id not in DISTRICT_BY_ID]
    if unknown:
        raise ValueError("Configured district scope contains an unknown registry identifier.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if authority_id is None:
        authority_id = str(uuid.uuid4())
        with database_connection(database_path) as connection:
            connection.execute(
                """INSERT INTO authorities (
                    authority_id, authority_name, authority_type, jurisdiction,
                    district_ids, road_categories, routing_rules, contact_details, active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (authority_id, authority_name, authority_type, (jurisdiction or "").strip(),
                 json.dumps(district_ids), json.dumps(road_categories), (routing_rules or "").strip(),
                 (contact_details or "").strip(),
                 int(bool(active)), now, now),
            )
    else:
        with database_connection(database_path) as connection:
            cursor = connection.execute(
                """UPDATE authorities SET authority_name=?, authority_type=?, jurisdiction=?,
                   district_ids=?, road_categories=?, routing_rules=?, contact_details=?, active=?, updated_at=? WHERE authority_id=?""",
                (authority_name, authority_type, (jurisdiction or "").strip(), json.dumps(district_ids),
                 json.dumps(road_categories), (routing_rules or "").strip(), (contact_details or "").strip(),
                 int(bool(active)), now, authority_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("The authority entry was not found.")
    return authority_id


def fetch_authorities(database_path=DATABASE_FILE, active_only=False):
    query = "SELECT * FROM authorities"
    if active_only:
        query += " WHERE active = 1"
    query += " ORDER BY authority_name COLLATE NOCASE"
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query).fetchall()]


def fetch_manual_review_queue(database_path=DATABASE_FILE, open_only=True):
    query = """SELECT q.*, r.road_name, r.location, r.district, r.district_id,
        r.problem_type, r.severity, r.status AS report_status
        FROM manual_review_queue q JOIN road_reports r USING(report_id)"""
    if open_only:
        query += " WHERE q.queue_status = 'Open'"
    query += " ORDER BY q.created_at, q.queue_id"
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query).fetchall()]


def _resolve_manual_review_in_transaction(connection, report_id, actor, resolution):
    connection.execute(
        """UPDATE manual_review_queue SET queue_status='Resolved', resolved_at=?,
           resolved_by=?, resolution=? WHERE report_id=? AND queue_status='Open'""",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), actor, resolution, report_id),
    )


def resolve_manual_review(report_id, resolution, actor=LOCAL_ACTOR, database_path=DATABASE_FILE):
    resolution = (resolution or "").strip()
    if not resolution:
        raise ValueError("A manual review resolution note is required.")
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        _resolve_manual_review_in_transaction(connection, report_id, actor, resolution)


def assign_report(report_id, authority_id, assigned_team, due_at, routing_reason,
                  actor=LOCAL_ACTOR, database_path=DATABASE_FILE, assigned_officer=""):
    assigned_team = (assigned_team or "").strip()
    assigned_officer = (assigned_officer or "").strip()
    routing_reason = (routing_reason or "").strip()
    actor = (actor or "").strip()
    if not assigned_team or not routing_reason or not actor:
        raise ValueError("Assigned team, routing reason and actor metadata are required.")
    due_at = _iso_datetime(due_at)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    assignment_id = str(uuid.uuid4())
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        report = connection.execute(
            "SELECT status, district_id, district FROM road_reports WHERE report_id=?", (report_id,)
        ).fetchone()
        authority = connection.execute(
            "SELECT * FROM authorities WHERE authority_id=? AND active=1", (authority_id,)
        ).fetchone()
        if report is None:
            raise ValueError("The report was not found.")
        if authority is None:
            raise ValueError("Choose an active configured authority.")
        if report["status"] not in ("Validated", "Assigned"):
            raise InvalidStatusTransition("A report must be Validated or already Assigned before assigning an authority.")
        try:
            configured_district_ids = json.loads(authority["district_ids"] or "[]")
        except json.JSONDecodeError:
            configured_district_ids = []
        district_hint_matches = report["district_id"] in configured_district_ids
        routing_reason = (
            "MANUAL ROUTING: district configuration is a hint only; road ownership is unverified. "
            + ("Configured district hint matches. " if district_hint_matches else "No configured district hint match. ")
            + routing_reason
        )
        prior = connection.execute(
            "SELECT assignment_id FROM assignments WHERE report_id=? AND assignment_status='Active'",
            (report_id,),
        ).fetchall()
        for old in prior:
            connection.execute(
                "UPDATE assignments SET assignment_status='Superseded' WHERE assignment_id=?",
                (old["assignment_id"],),
            )
            connection.execute(
                "INSERT INTO assignment_history (assignment_id,event_type,actor,reason,changed_at) VALUES (?,?,?,?,?)",
                (old["assignment_id"], "Superseded", actor, "A replacement assignment was recorded.", now),
            )
        connection.execute(
            """INSERT INTO assignments (
                assignment_id,report_id,authority_id,assigned_team,assigned_officer,assigned_actor,assigned_at,due_at,
                acceptance_status,assignment_status,manual_routing_reason
            ) VALUES (?,?,?,?,?,?,?,?,'Awaiting response','Active',?)""",
            (assignment_id, report_id, authority_id, assigned_team, assigned_officer, actor, now, due_at, routing_reason),
        )
        connection.execute(
            "INSERT INTO assignment_history (assignment_id,event_type,actor,reason,changed_at) VALUES (?,?,?,?,?)",
            (assignment_id, "Assigned", actor, routing_reason, now),
        )
        if report["status"] == "Validated":
            _transition_report_status_in_transaction(
                connection, report_id, "Assigned", f"Assigned to {authority['authority_name']} / {assigned_team}.", actor
            )
    return assignment_id


def record_assignment_response(assignment_id, acceptance_status, reference,
                               actor=LOCAL_ACTOR, database_path=DATABASE_FILE):
    if acceptance_status not in ("Acceptance recorded", "Decline recorded"):
        raise ValueError("Choose an explicit recorded acceptance or decline.")
    reference = (reference or "").strip()
    actor = (actor or "").strip()
    if not reference or not actor:
        raise ValueError("A response reference/note and actor metadata are required.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT assignment_id FROM assignments WHERE assignment_id=? AND assignment_status='Active'",
            (assignment_id,),
        ).fetchone()
        if row is None:
            raise ValueError("The active assignment was not found.")
        connection.execute(
            """UPDATE assignments SET acceptance_status=?, acceptance_actor=?, acceptance_at=?,
               acceptance_reference=?, assignment_status=CASE WHEN ?='Decline recorded' THEN 'Declined' ELSE assignment_status END
               WHERE assignment_id=?""",
            (acceptance_status, actor, now, reference, acceptance_status, assignment_id),
        )
        connection.execute(
            "INSERT INTO assignment_history (assignment_id,event_type,actor,reason,changed_at) VALUES (?,?,?,?,?)",
            (assignment_id, acceptance_status, actor, reference, now),
        )


def fetch_assignment_history(assignment_id, database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM assignment_history WHERE assignment_id=? ORDER BY history_id", (assignment_id,)
        ).fetchall()]


def fetch_assignments(report_id=None, database_path=DATABASE_FILE):
    query = """SELECT a.*, h.authority_name, h.authority_type FROM assignments a
        JOIN authorities h USING(authority_id)"""
    values = []
    if report_id:
        query += " WHERE a.report_id=?"
        values.append(report_id)
    query += " ORDER BY a.assigned_at DESC"
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query, values).fetchall()]


def fetch_repair_records(report_id, database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM repair_records WHERE report_id=? ORDER BY created_at DESC, rowid DESC", (report_id,)
        ).fetchall()]


def fetch_report_evidence(report_id, database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM report_evidence WHERE report_id=? ORDER BY created_at, evidence_id", (report_id,)
        ).fetchall()]


def fetch_verifications(report_id, database_path=DATABASE_FILE):
    with database_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(
            "SELECT * FROM verifications WHERE report_id=? ORDER BY created_at DESC, rowid DESC", (report_id,)
        ).fetchall()]


def _active_assignment_id(connection, report_id):
    row = connection.execute(
        "SELECT assignment_id FROM assignments WHERE report_id=? AND assignment_status='Active' ORDER BY assigned_at DESC LIMIT 1",
        (report_id,),
    ).fetchone()
    return row["assignment_id"] if row else None


def schedule_repair(report_id, scheduled_at, repair_method, materials, notes,
                    actor=LOCAL_ACTOR, database_path=DATABASE_FILE):
    scheduled_at = _iso_datetime(scheduled_at)
    repair_method = (repair_method or "").strip()
    materials = (materials or "").strip()
    notes = (notes or "").strip()
    actor = (actor or "").strip()
    if not scheduled_at or not notes or not actor:
        raise ValueError("A scheduled time, repair notes and actor metadata are required.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repair_id = str(uuid.uuid4())
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        report = connection.execute("SELECT status FROM road_reports WHERE report_id=?", (report_id,)).fetchone()
        if report is None:
            raise ValueError("The report was not found.")
        if report["status"] != "Assigned":
            raise InvalidStatusTransition("Repair scheduling requires the report to be Assigned.")
        assignment_id = _active_assignment_id(connection, report_id)
        if assignment_id is None:
            raise ValueError("A current authority assignment is required before scheduling repair.")
        connection.execute(
            """INSERT INTO repair_records (
                repair_id,report_id,assignment_id,repair_status,repair_method,materials,repair_notes,
                scheduled_at,recorded_actor,created_at,updated_at
            ) VALUES (?,?,?,'Scheduled',?,?,?,?,?,?,?)""",
            (repair_id, report_id, assignment_id, repair_method, materials, notes,
             scheduled_at, actor, now, now),
        )
        _transition_report_status_in_transaction(
            connection, report_id, "Repair Scheduled", "Repair work scheduled.", actor
        )
    return repair_id


def start_repair(report_id, repair_method="", materials="", notes="",
                 actor=LOCAL_ACTOR, database_path=DATABASE_FILE):
    actor = (actor or "").strip()
    notes = (notes or "").strip()
    if not actor or not notes:
        raise ValueError("Work-start notes and actor metadata are required.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        report = connection.execute("SELECT status FROM road_reports WHERE report_id=?", (report_id,)).fetchone()
        if report is None:
            raise ValueError("The report was not found.")
        if report["status"] not in ("Repair Scheduled", "Repair In Progress"):
            raise InvalidStatusTransition("Work can start only after scheduling or following a failed verification.")
        assignment_id = _active_assignment_id(connection, report_id)
        latest = connection.execute(
            "SELECT * FROM repair_records WHERE report_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (report_id,)
        ).fetchone()
        if report["status"] == "Repair Scheduled" and latest and latest["repair_status"] == "Scheduled":
            repair_id = latest["repair_id"]
            connection.execute(
                """UPDATE repair_records SET repair_status='In Progress', repair_method=?, materials=?,
                   repair_notes=?, started_at=?, recorded_actor=?, updated_at=? WHERE repair_id=?""",
                ((repair_method or latest["repair_method"]).strip(), (materials or latest["materials"]).strip(),
                 notes, now, actor, now, repair_id),
            )
            _transition_report_status_in_transaction(
                connection, report_id, "Repair In Progress", "Repair work started.", actor
            )
        elif report["status"] == "Repair In Progress" and (latest is None or latest["repair_status"] == "Completed"):
            if assignment_id is None:
                raise ValueError("A current assignment is required to record rework.")
            repair_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO repair_records (
                    repair_id,report_id,assignment_id,repair_status,repair_method,materials,repair_notes,
                    started_at,recorded_actor,created_at,updated_at
                ) VALUES (?,?,?,'In Progress',?,?,?,?,?,?,?)""",
                (repair_id, report_id, assignment_id, (repair_method or "").strip(), (materials or "").strip(),
                 notes, now, actor, now, now),
            )
        else:
            raise InvalidStatusTransition("The latest repair record is not ready to start or resume.")
    return repair_id


def save_repair_before_photo(report_id, uploaded_file, actor=LOCAL_ACTOR,
                             database_path=DATABASE_FILE, evidence_dir=EVIDENCE_DIR):
    return save_report_evidence(report_id, "Repair Before", uploaded_file, actor,
                                 database_path, evidence_dir)


def save_report_evidence(report_id, image_type, uploaded_file, actor=LOCAL_ACTOR,
                         database_path=DATABASE_FILE, evidence_dir=EVIDENCE_DIR):
    if image_type not in ("Repair Before", "Repair After"):
        raise ValueError("Only repair-before or repair-after photos can be uploaded here.")
    image_bytes, extension = inspect_image_upload(uploaded_file)
    if image_bytes is None:
        raise ValueError("Choose a valid photo file.")
    actor = (actor or "").strip()
    if not actor:
        raise ValueError("Actor metadata is required.")
    evidence_id = str(uuid.uuid4())
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    destination = evidence_dir / f"{evidence_id}{extension}"
    with destination.open("xb") as evidence_file:
        evidence_file.write(image_bytes)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with database_connection(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            report = connection.execute("SELECT status FROM road_reports WHERE report_id=?", (report_id,)).fetchone()
            if report is None:
                raise ValueError("The report was not found.")
            if report["status"] not in ("Repair Scheduled", "Repair In Progress", PENDING_VERIFICATION_STATUS):
                raise InvalidStatusTransition("Repair photos can be added only during repair or review.")
            repair = connection.execute(
                "SELECT repair_id,assignment_id FROM repair_records WHERE report_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (report_id,),
            ).fetchone()
            if repair is None:
                raise ValueError("Create a repair record before attaching repair evidence.")
            relative_path = str(destination.relative_to(BASE_DIR)) if destination.is_relative_to(BASE_DIR) else str(destination)
            connection.execute(
                """INSERT INTO report_evidence (
                    evidence_id,report_id,assignment_id,repair_id,image_type,image_path,
                    original_filename,media_type,recorded_actor,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (evidence_id, report_id, repair["assignment_id"], repair["repair_id"], image_type,
                 relative_path, getattr(uploaded_file, "name", ""),
                 {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[extension], actor, now),
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return evidence_id


def complete_repair(report_id, repair_method, materials, notes, after_photo,
                    actor=LOCAL_ACTOR, database_path=DATABASE_FILE, evidence_dir=EVIDENCE_DIR):
    repair_method = (repair_method or "").strip()
    materials = (materials or "").strip()
    notes = (notes or "").strip()
    actor = (actor or "").strip()
    if not notes or not actor:
        raise ValueError("Completion notes and actor metadata are required.")
    image_bytes, extension = inspect_image_upload(after_photo)
    if image_bytes is None:
        raise ValueError("A repair-after photo is required before requesting verification.")
    evidence_id = str(uuid.uuid4())
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    destination = evidence_dir / f"{evidence_id}{extension}"
    with destination.open("xb") as evidence_file:
        evidence_file.write(image_bytes)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        with database_connection(database_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            report = connection.execute("SELECT status FROM road_reports WHERE report_id=?", (report_id,)).fetchone()
            if report is None or report["status"] != "Repair In Progress":
                raise InvalidStatusTransition("Repair completion is allowed only while work is in progress.")
            repair = connection.execute(
                "SELECT * FROM repair_records WHERE report_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1", (report_id,)
            ).fetchone()
            if repair is None or repair["repair_status"] != "In Progress":
                raise InvalidStatusTransition("A current in-progress repair record is required.")
            relative_path = str(destination.relative_to(BASE_DIR)) if destination.is_relative_to(BASE_DIR) else str(destination)
            connection.execute(
                """INSERT INTO report_evidence (
                    evidence_id,report_id,assignment_id,repair_id,image_type,image_path,
                    original_filename,media_type,recorded_actor,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (evidence_id, report_id, repair["assignment_id"], repair["repair_id"], "Repair After",
                 relative_path, getattr(after_photo, "name", ""),
                 {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}[extension], actor, now),
            )
            connection.execute(
                """UPDATE repair_records SET repair_status='Completed', repair_method=?, materials=?,
                   repair_notes=?, completed_at=?, recorded_actor=?, updated_at=? WHERE repair_id=?""",
                (repair_method, materials, notes, now, actor, now, repair["repair_id"]),
            )
            _transition_report_status_in_transaction(
                connection, report_id, PENDING_VERIFICATION_STATUS,
                "Repair completion recorded with notes and repair-after evidence.", actor,
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return repair["repair_id"], evidence_id


def record_verification_decision(report_id, decision, comments, reviewer=LOCAL_ACTOR,
                                 database_path=DATABASE_FILE):
    if decision not in ("Approved", "Rejected", "More Evidence"):
        raise ValueError("Choose Approved, Rejected, or More Evidence.")
    comments = (comments or "").strip()
    reviewer = (reviewer or "").strip()
    if not comments or not reviewer:
        raise ValueError("Verification comments and reviewer actor metadata are required.")
    verification_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database_connection(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        report = connection.execute("SELECT status FROM road_reports WHERE report_id=?", (report_id,)).fetchone()
        if report is None or report["status"] != PENDING_VERIFICATION_STATUS:
            raise InvalidStatusTransition("Verification decisions require a repair awaiting verification.")
        repair = connection.execute(
            """SELECT * FROM repair_records WHERE report_id=? AND repair_status='Completed'
               ORDER BY completed_at DESC, rowid DESC LIMIT 1""", (report_id,)
        ).fetchone()
        if repair is None or not (repair["repair_notes"] or "").strip():
            raise ValueError("A completed repair record with notes is required for verification.")
        after = connection.execute(
            """SELECT evidence_id, image_path FROM report_evidence WHERE report_id=? AND repair_id=?
               AND image_type='Repair After' ORDER BY created_at DESC""",
            (report_id, repair["repair_id"]),
        ).fetchall()
        if not after:
            raise ValueError("A repair-after photo is required before verification.")
        if not any(valid_saved_repair_photo(item["image_path"]) for item in after):
            raise ValueError("A valid saved repair-after photo is required before verification.")
        before = connection.execute(
            """SELECT evidence_id FROM report_evidence WHERE report_id=? AND repair_id=?
               AND image_type='Repair Before' ORDER BY created_at""",
            (report_id, repair["repair_id"]),
        ).fetchall()
        evidence_ids = [row["evidence_id"] for row in before + after]
        connection.execute(
            """INSERT INTO verifications (
                verification_id,report_id,repair_id,reviewer_actor,decision,comments,
                evidence_references,created_at
            ) VALUES (?,?,?,?,?,?,?,?)""",
            (verification_id, report_id, repair["repair_id"], reviewer, decision, comments,
             json.dumps(evidence_ids), now),
        )
        if decision == "Approved":
            _transition_report_status_in_transaction(
                connection, report_id, "Verified and Closed",
                f"Repair approved by explicit verification decision {verification_id}.",
                reviewer, f"Verification {verification_id}: {comments}",
            )
            connection.execute(
                "UPDATE assignments SET assignment_status='Completed' WHERE report_id=? AND assignment_status='Active'",
                (report_id,),
            )
        else:
            assignment_id = repair["assignment_id"] or _active_assignment_id(connection, report_id)
            rework_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO repair_records (
                    repair_id,report_id,assignment_id,repair_status,repair_notes,started_at,
                    recorded_actor,created_at,updated_at
                ) VALUES (?,?,?,'In Progress',?,?,?,?,?)""",
                (rework_id, report_id, assignment_id, f"Verification {decision}: {comments}",
                 now, reviewer, now, now),
            )
            _transition_report_status_in_transaction(
                connection, report_id, "Repair In Progress",
                f"Verification decision {decision}; repair work returned for follow-up.", reviewer,
            )
    return verification_id


def fetch_phase4_dashboard_stats(database_path=DATABASE_FILE):
    """Return disjoint workflow counts; manual review is a separate, overlapping queue."""
    buckets = {
        "unassigned_reports": "unassigned", "assigned_cases": "assigned",
        "repairs_in_progress": "repair_in_progress", "awaiting_verification": "awaiting_verification",
        "verified_cases": "verified", "reopened_cases": "reopened",
        "manual_authority_review": "manual_review", "overdue_jobs": "overdue",
    }
    return {key: count_reports({"workflow_bucket": bucket}, database_path) for key, bucket in buckets.items()}


def _elapsed_seconds(start, finish):
    try:
        start_dt = _parse_utc_datetime(start)
        finish_dt = _parse_utc_datetime(finish)
        seconds = (finish_dt - start_dt).total_seconds()
        return seconds if seconds >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_utc_datetime(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_dashboard_derived_metrics(database_path=DATABASE_FILE):
    """Averages use valid paired timestamps; failure rate counts non-approved decisions."""
    with database_connection(database_path) as connection:
        assignment_rows = connection.execute("""
            SELECT r.report_id, r.submitted_at, a.assigned_at
            FROM road_reports r JOIN assignments a USING(report_id)
            WHERE a.assigned_at IS NOT NULL
        """).fetchall()
        repair_rows = connection.execute("""
            SELECT started_at, completed_at FROM repair_records
            WHERE started_at IS NOT NULL AND completed_at IS NOT NULL
        """).fetchall()
        decisions = connection.execute(
            "SELECT decision, COUNT(*) AS n FROM verifications GROUP BY decision"
        ).fetchall()
    earliest_assignment = {}
    for row in assignment_rows:
        try:
            assigned_time = _parse_utc_datetime(row["assigned_at"])
        except (TypeError, ValueError, OverflowError):
            continue
        current = earliest_assignment.get(row["report_id"])
        if current is None or assigned_time < current[1]:
            earliest_assignment[row["report_id"]] = (row["submitted_at"], assigned_time.isoformat())
    assignment_durations = [
        duration for submitted, assigned in earliest_assignment.values()
        if (duration := _elapsed_seconds(submitted, assigned)) is not None
    ]
    repair_durations = [
        duration for row in repair_rows
        if (duration := _elapsed_seconds(row["started_at"], row["completed_at"])) is not None
    ]
    decision_counts = {row["decision"]: int(row["n"]) for row in decisions}
    total_decisions = sum(decision_counts.values())
    not_approved = decision_counts.get("Rejected", 0) + decision_counts.get("More Evidence", 0)
    return {
        "avg_submission_to_assignment_seconds": sum(assignment_durations) / len(assignment_durations) if assignment_durations else None,
        "submission_to_assignment_n": len(assignment_durations),
        "avg_repair_completion_seconds": sum(repair_durations) / len(repair_durations) if repair_durations else None,
        "repair_completion_n": len(repair_durations),
        "verification_failure_rate": not_approved / total_decisions if total_decisions else None,
        "verification_decision_n": total_decisions,
    }


def render_authority_page(database_ready):
    st.subheader("Authority Setup")
    st.warning(
        "Authority names, contacts, and district scope are local prototype configuration only. "
        "No government directory or road-ownership source is verified, and these entries do not imply acceptance."
    )
    if not database_ready:
        st.error("Authority configuration is unavailable because the database could not be opened.")
        return
    authorities = fetch_authorities()
    ids = ["new", *[item["authority_id"] for item in authorities]]
    selected = st.selectbox(
        "Create or maintain an entry", ids,
        format_func=lambda value: "Add a prototype authority" if value == "new" else next(
            item["authority_name"] + (" (inactive)" if not item["active"] else "")
            for item in authorities if item["authority_id"] == value
        ),
    )
    current = next((item for item in authorities if item["authority_id"] == selected), None)
    try:
        selected_districts = json.loads(current["district_ids"]) if current else []
    except json.JSONDecodeError:
        selected_districts = []
    try:
        selected_road_categories = json.loads(current["road_categories"]) if current else []
    except (json.JSONDecodeError, TypeError):
        selected_road_categories = []
    with st.form("authority-entry"):
        name = st.text_input("Authority name *", value=current["authority_name"] if current else "")
        authority_type = st.text_input("Authority type *", value=current["authority_type"] if current else "")
        jurisdiction = st.text_area(
            "Configured jurisdiction / routing note (unverified)",
            value=current["jurisdiction"] if current else "",
            help="Describe only the scope entered by your team. The app will not treat it as verified road ownership.",
        )
        district_ids = st.multiselect(
            "Optional configured district scope", [entry["id"] for entry in DISTRICTS],
            default=[item for item in selected_districts if item in DISTRICT_BY_ID],
            format_func=lambda value: DISTRICT_BY_ID[value]["name"],
        )
        road_categories_text = st.text_input(
            "Configured road categories (comma-separated, optional)",
            value=", ".join(selected_road_categories),
            help="Local routing hints only; no road ownership or official jurisdiction is inferred.",
        )
        routing_rules = st.text_area(
            "Explicit local routing rule / eligibility note",
            value=current["routing_rules"] if current else "",
            help="Describe the operator-configured conditions. The app stores this rule but does not verify ownership or automatically assign cases.",
        )
        contact = st.text_area(
            "Contact details (optional; enter only details you have verified)",
            value=current["contact_details"] if current else "",
        )
        active = st.checkbox("Active for manual assignment", value=bool(current["active"]) if current else True)
        submitted = st.form_submit_button("Save authority entry", type="primary")
    if submitted:
        try:
            authority_id = save_authority(
                name, authority_type, jurisdiction, district_ids, contact, active,
                authority_id=current["authority_id"] if current else None,
                road_categories=[value.strip() for value in road_categories_text.split(",") if value.strip()],
                routing_rules=routing_rules,
            )
        except (sqlite3.Error, ValueError) as error:
            st.error(f"Authority entry could not be saved: {error}")
        else:
            st.session_state["ops_message"] = f"Saved prototype authority entry {authority_id}."
            st.rerun()
    if authorities:
        st.markdown("#### Configured entries")
        st.dataframe([{
            "Authority": item["authority_name"],
            "Type": item["authority_type"],
            "Configured district IDs": ", ".join(
                DISTRICT_BY_ID[district_id]["name"]
                for district_id in json.loads(item["district_ids"] or "[]")
                if district_id in DISTRICT_BY_ID
            ) or "No configured district scope",
            "Configured road categories": ", ".join(json.loads(item["road_categories"] or "[]")) or "Not configured",
            "Local routing rule": item["routing_rules"],
            "Jurisdiction note": item["jurisdiction"],
            "Active": bool(item["active"]),
        } for item in authorities], hide_index=True, width="stretch")


def _display_evidence(items, image_type):
    matches = [item for item in items if item["image_type"] == image_type]
    if not matches:
        st.caption(f"No {image_type.lower()} image has been recorded.")
        return
    for item in matches:
        try:
            image_path = road_ai.path_under_project(item["image_path"])
            st.image(image_path.read_bytes(), caption=f"{image_type} · {item['created_at']}", use_container_width=True)
        except (road_ai.InferenceError, OSError) as error:
            st.warning(f"{image_type} image unavailable: {error}")


def _render_assignment_section(record):
    st.markdown("#### Authority assignment")
    authorities = fetch_authorities(active_only=True)
    assignments = fetch_assignments(record["report_id"])
    if assignments:
        st.dataframe([{
            "Authority": item["authority_name"], "Type": item["authority_type"],
            "Officer": item["assigned_officer"] or "Not recorded",
            "Team": item["assigned_team"], "Assigned": item["assigned_at"],
            "Due": item["due_at"] or "Not set", "Assignment": item["assignment_status"],
            "Acceptance": item["acceptance_status"],
        } for item in assignments], hide_index=True, width="stretch")
        for item in assignments:
            with st.expander(f"Assignment history · {item['authority_name']} · {item['assignment_id'][:8]}"):
                st.caption("Acceptance values are manually recorded prototype metadata, not a verified authority response.")
                for event in fetch_assignment_history(item["assignment_id"]):
                    st.caption(f"{event['changed_at']} · {event['event_type']} · {event['actor']} · {event['reason']}")
                if item["assignment_status"] == "Active":
                    with st.form(f"assignment-response-{item['assignment_id']}"):
                        response = st.selectbox(
                            "Manually recorded response", ["Acceptance recorded", "Decline recorded"],
                            key=f"assignment-response-type-{item['assignment_id']}",
                        )
                        reference = st.text_input(
                            "Response source/reference *", key=f"assignment-response-reference-{item['assignment_id']}"
                        )
                        response_submitted = st.form_submit_button("Record response metadata")
                    if response_submitted:
                        try:
                            record_assignment_response(item["assignment_id"], response, reference)
                        except (sqlite3.Error, ValueError) as error:
                            st.error(f"Response could not be recorded: {error}")
                        else:
                            st.session_state["ops_message"] = "Assignment response metadata saved."
                            st.rerun()
    if record["status"] not in ("Validated", "Assigned"):
        return
    if not authorities:
        st.info("No active prototype authority entries exist. Add one under Authority Setup before manual assignment.")
        return
    if record.get("district_id"):
        st.caption(f"Reported district: {DISTRICT_BY_ID.get(record['district_id'], {}).get('name', record['district'])}")
    st.caption("The app does not infer authority ownership. An operator must choose and document each routing decision.")
    with st.form(f"assign-report-{record['report_id']}"):
        authority_id = st.selectbox(
            "Prototype authority (manual choice)", [item["authority_id"] for item in authorities],
            format_func=lambda value: next(item["authority_name"] for item in authorities if item["authority_id"] == value),
        )
        selected_authority = next(item for item in authorities if item["authority_id"] == authority_id)
        try:
            configured_ids = json.loads(selected_authority["district_ids"] or "[]")
        except json.JSONDecodeError:
            configured_ids = []
        if record.get("district_id") in configured_ids:
            st.info("The district hint matches, but this does not identify road ownership. The manual review queue remains open until resolved explicitly.")
        else:
            st.warning("No district hint match. Assignment will remain in the manual review queue for explicit review.")
        try:
            configured_categories = json.loads(selected_authority["road_categories"] or "[]")
        except (json.JSONDecodeError, TypeError):
            configured_categories = []
        st.caption("Configured road-category hints: " + (", ".join(configured_categories) or "none"))
        team = st.text_input("Assigned team label *")
        officer = st.text_input("Assigned officer (optional; prototype metadata)")
        due_date = st.date_input("Due date (optional)", value=None)
        routing_reason = st.text_area("Manual routing reason / review note *")
        assign_clicked = st.form_submit_button("Record manual assignment", type="primary")
    if assign_clicked:
        try:
            assignment_id = assign_report(
                record["report_id"], authority_id, team, due_date, routing_reason,
                assigned_officer=officer,
            )
        except (sqlite3.Error, ValueError) as error:
            st.error(f"Assignment could not be recorded: {error}")
        else:
            st.session_state["ops_message"] = f"Assignment recorded ({assignment_id}). Authority acceptance remains unrecorded."
            st.rerun()


def _render_repair_section(record):
    st.markdown("#### Repair tracking")
    repairs = fetch_repair_records(record["report_id"])
    if repairs:
        st.dataframe([{
            "Repair ID": item["repair_id"], "Status": item["repair_status"],
            "Method": item["repair_method"], "Materials": item["materials"],
            "Notes": item["repair_notes"], "Scheduled": item["scheduled_at"],
            "Started": item["started_at"], "Completed": item["completed_at"],
            "Actor": item["recorded_actor"],
        } for item in repairs], hide_index=True, width="stretch")
    latest = repairs[0] if repairs else None
    status = record["status"]
    if status == "Assigned":
        with st.form(f"schedule-repair-{record['report_id']}"):
            date = st.date_input("Scheduled repair date")
            method = st.text_input("Planned repair method")
            materials = st.text_input("Planned materials")
            notes = st.text_area("Scheduling notes *")
            submitted = st.form_submit_button("Record repair schedule", type="primary")
        if submitted:
            try:
                repair_id = schedule_repair(record["report_id"], date, method, materials, notes)
            except (sqlite3.Error, ValueError) as error:
                st.error(f"Repair schedule could not be recorded: {error}")
            else:
                st.session_state["ops_message"] = f"Repair schedule recorded ({repair_id})."
                st.rerun()
    elif status == "Repair Scheduled":
        with st.form(f"start-repair-{record['report_id']}"):
            method = st.text_input("Repair method")
            materials = st.text_input("Materials used so far")
            notes = st.text_area("Work-start notes *")
            submitted = st.form_submit_button("Record work started", type="primary")
        if submitted:
            try:
                repair_id = start_repair(record["report_id"], method, materials, notes)
            except (sqlite3.Error, ValueError) as error:
                st.error(f"Work start could not be recorded: {error}")
            else:
                st.session_state["ops_message"] = f"Repair work started ({repair_id})."
                st.rerun()
    elif status == "Repair In Progress":
        if latest and latest["repair_status"] == "Completed":
            st.caption("Verification requested rework. Record a new repair attempt below.")
            with st.form(f"restart-repair-{record['report_id']}"):
                method = st.text_input("Rework method")
                materials = st.text_input("Rework materials")
                notes = st.text_area("Rework-start notes *")
                submitted = st.form_submit_button("Record rework started", type="primary")
            if submitted:
                try:
                    repair_id = start_repair(record["report_id"], method, materials, notes)
                except (sqlite3.Error, ValueError) as error:
                    st.error(f"Rework could not be recorded: {error}")
                else:
                    st.session_state["ops_message"] = f"Rework started ({repair_id})."
                    st.rerun()
        active_repair = fetch_repair_records(record["report_id"])
        active_repair = active_repair[0] if active_repair else None
        if active_repair and active_repair["repair_status"] == "In Progress":
            before_upload = st.file_uploader(
                "Optional repair-before photo (JPEG, PNG, WebP; max 5 MB)",
                type=["jpg", "jpeg", "png", "webp"], key=f"before-{active_repair['repair_id']}",
            )
            if before_upload and st.button("Save repair-before photo", key=f"save-before-{active_repair['repair_id']}"):
                try:
                    evidence_id = save_repair_before_photo(record["report_id"], before_upload)
                except (OSError, sqlite3.Error, ValueError) as error:
                    st.error(f"Repair-before photo could not be saved: {error}")
                else:
                    st.session_state["ops_message"] = f"Repair-before evidence saved ({evidence_id})."
                    st.rerun()
            with st.form(f"complete-repair-{active_repair['repair_id']}"):
                method = st.text_input("Completed repair method", value=active_repair["repair_method"])
                materials = st.text_input("Materials used", value=active_repair["materials"])
                notes = st.text_area("Repair completion notes *", value=active_repair["repair_notes"])
                after_photo = st.file_uploader(
                    "Repair-after photo required for verification (JPEG, PNG, WebP; max 5 MB)",
                    type=["jpg", "jpeg", "png", "webp"], key=f"after-{active_repair['repair_id']}",
                )
                submitted = st.form_submit_button("Submit repair for verification", type="primary")
            if submitted:
                try:
                    repair_id, evidence_id = complete_repair(
                        record["report_id"], method, materials, notes, after_photo,
                    )
                except (OSError, sqlite3.Error, ValueError) as error:
                    st.error(f"Repair cannot be submitted for verification: {error}")
                else:
                    st.session_state["ops_message"] = (
                        f"Repair {repair_id} submitted for verification with after-photo evidence {evidence_id}."
                    )
                    st.rerun()


def _render_verification_section(record):
    st.markdown("#### Repair evidence and verification")
    st.write({
        "Road": record["road_name"], "Location": record.get("location") or "Not provided",
        "District": record["district"], "Damage type": record["problem_type"],
        "Severity": record["severity"], "Description": record["description"],
        "Coordinates": (
            f"{record['latitude']}, {record['longitude']}"
            if record.get("latitude") is not None and record.get("longitude") is not None
            else "Not recorded"
        ),
    })
    evidence = fetch_report_evidence(record["report_id"])
    repairs = fetch_repair_records(record["report_id"])
    repair_id = repairs[0]["repair_id"] if repairs else None
    scoped = [item for item in evidence if item.get("repair_id") == repair_id]
    before_column, after_column = st.columns(2)
    with before_column:
        st.markdown("**Repair before**")
        _display_evidence(scoped, "Repair Before")
    with after_column:
        st.markdown("**Repair after**")
        _display_evidence(scoped, "Repair After")
    if repairs:
        st.markdown("**Repair notes**")
        st.write(repairs[0]["repair_notes"] or "No repair notes recorded.")
    st.markdown("#### Verification decision")
    blockers = []
    if record["status"] != PENDING_VERIFICATION_STATUS:
        blockers.append(f"Report status is {record['status']}; it must be awaiting verification.")
    latest_repair = repairs[0] if repairs else None
    if latest_repair is None or latest_repair["repair_status"] != "Completed":
        blockers.append("A completed repair record is required.")
    elif not (latest_repair.get("repair_notes") or "").strip():
        blockers.append("Repair completion notes are required.")
    after_photos = [
        item for item in scoped
        if item["image_type"] == "Repair After" and item.get("repair_id") == repair_id
    ]
    if not after_photos or not any(valid_saved_repair_photo(item["image_path"]) for item in after_photos):
        blockers.append("A valid saved repair-after photo is required.")
    decision_ready = not blockers
    if blockers:
        st.info("Verification is unavailable: " + " ".join(blockers))
    st.caption(
        "Reviewer identity is recorded as unauthenticated local prototype metadata. "
        "Uploading a photo does not approve or close a report."
    )
    decision_labels = {
        "Approve repair": "Approved",
        "Reject repair and return to repair": "Rejected",
        "Request additional evidence": "More Evidence",
    }
    selected_label = st.radio(
        "Decision", list(decision_labels), horizontal=True, disabled=not decision_ready,
        key=f"verification-decision-{record['report_id']}",
    )
    comments = st.text_area(
        "Reviewer comments / reason *", disabled=not decision_ready,
        key=f"verification-comments-{record['report_id']}",
    )
    submitted = st.button(
        "Record verification decision", type="primary",
        disabled=not decision_ready or not (comments or "").strip(),
        key=f"submit-verification-{record['report_id']}",
    )
    if decision_ready and submitted:
        try:
            verification_id = record_verification_decision(
                record["report_id"], decision_labels[selected_label], comments,
            )
        except (sqlite3.Error, ValueError) as error:
            st.error(f"Verification decision could not be recorded: {error}")
        else:
            decision_value = decision_labels[selected_label]
            st.session_state["ops_message"] = (
                f"Verification decision {decision_value} recorded ({verification_id}). "
                + ("The case is closed." if decision_value == "Approved"
                   else "The case returned to Repair In Progress.")
            )
            st.rerun()
    verifications = fetch_verifications(record["report_id"])
    if verifications:
        st.markdown("**Verification history**")
        for item in verifications:
            st.caption(f"{item['created_at']} · {item['decision']} · {item['reviewer_actor']} · {item['comments']}")


def render_operations_page(database_ready):
    st.subheader("Assignments, Repairs and Verification")
    st.warning(
        "No verified road-ownership or authority directory is configured. Authority matching is not automatic; "
        "open cases remain in the manual routing queue until an operator records a review."
    )
    if not database_ready:
        st.error("Operations are unavailable because the database could not be opened.")
        return
    queue = fetch_manual_review_queue()
    st.markdown("#### Manual routing review queue")
    if queue:
        st.dataframe([{
            "Report": item["report_id"], "Road": item["road_name"], "District": item["district"],
            "Damage": item["problem_type"], "Status": item["report_status"], "Reason": item["review_reason"],
        } for item in queue], hide_index=True, width="stretch")
    else:
        st.info("No open manual routing reviews.")
    records = fetch_reports()
    if not records:
        st.info("There are no saved reports to manage.")
        return
    record_by_id = {item["report_id"]: item for item in records}
    report_id = st.selectbox(
        "Report detail", list(record_by_id),
        format_func=lambda value: (
            f"{record_by_id[value]['problem_type']} · {record_by_id[value]['road_name']} · "
            f"{record_by_id[value]['district']} · {record_by_id[value]['report_id'][:8]}"
        ),
    )
    record = record_by_id[report_id]
    tabs = st.tabs(["Assignment", "Repair", "Verification & Evidence", "Status History"])
    with tabs[0]:
        _render_assignment_section(record)
        open_queue = next((item for item in queue if item["report_id"] == report_id), None)
        if open_queue:
            st.caption("District and road-category entries are routing hints only. Record a manual review outcome before clearing this queue item.")
            resolution = st.text_area(
                "Manual review outcome / routing note *", max_chars=1000,
                key=f"resolve-route-note-{report_id}",
            )
            if st.button("Resolve manual routing review", key=f"resolve-route-{report_id}"):
                try:
                    resolve_manual_review(report_id, resolution)
                except (sqlite3.Error, ValueError) as error:
                    st.error(f"Manual review could not be resolved: {error}")
                else:
                    st.session_state["ops_message"] = "Manual review resolution recorded by the local unauthenticated operator."
                    st.rerun()
    with tabs[1]:
        _render_repair_section(record)
    with tabs[2]:
        _render_verification_section(record)
        st.markdown("**Original and AI evidence**")
        evidence = fetch_report_evidence(report_id)
        original_column, annotation_column = st.columns(2)
        with original_column:
            st.markdown("Original report photo")
            _display_evidence(evidence, "Original Evidence")
        with annotation_column:
            st.markdown("AI annotated copies")
            _display_evidence(evidence, "AI Annotation")
    with tabs[3]:
        for event in fetch_report_status_history(report_id):
            st.caption(
                f"{event['changed_at']} · {event['previous_status'] or '(initial)'} → "
                f"{event['new_status']} · {event['actor']} · {event['reason']}"
            )


def render_report_ai_analysis(record):
    """Show and run model analysis against an existing saved evidence photo."""
    st.markdown("**AI photo analysis (advisory; it does not change report status)**")
    saved_photo = record.get("photo_path")
    if not saved_photo:
        st.info("This report has no saved photo to analyze.")
        return
    try:
        photo_path = road_ai.path_under_project(saved_photo)
        original_image = road_ai.load_image(photo_path)
    except (road_ai.InferenceError, OSError) as error:
        st.error(f"The saved report photo cannot be opened: {error}")
        return
    st.image(original_image, caption="Original report photo", use_container_width=True)
    if st.button("Run AI analysis", key=f"run-ai-{record['report_id']}"):
        with st.spinner("Checking model and analyzing the saved photo on CPU…"):
            try:
                session, version, _metadata = road_ai.create_cpu_session()
                result = road_ai.run_inference(original_image, session, version)
            except road_ai.InferenceError as error:
                try:
                    version = road_ai.model_version()
                except road_ai.InferenceError:
                    version = "unavailable"
                result = {
                    "model_version": version,
                    "detections": [],
                    "inference_status": "error",
                    "processing_duration_ms": None,
                    "annotated_image": None,
                    "error_message": str(error),
                }
            try:
                road_ai.save_detection_result(record["report_id"], result, DATABASE_FILE)
            except (sqlite3.Error, OSError, ValueError) as error:
                st.error(f"Analysis finished, but its result could not be saved: {error}")
            else:
                st.session_state["ai_analysis_message"] = (
                    "AI analysis saved. No report lifecycle status was changed."
                    if result["inference_status"] != "error"
                    else "AI analysis could not run; the failure was recorded. No report status was changed."
                )
                st.rerun()

    try:
        analyses = road_ai.fetch_detection_history(record["report_id"], DATABASE_FILE)
    except sqlite3.Error as error:
        st.error(f"Could not load AI analysis history: {error}")
        return
    if not analyses:
        st.caption("No AI analysis has been run for this report.")
        return
    st.markdown("**Saved AI analysis history**")
    for analysis in analyses:
        with st.container(border=True):
            st.caption(
                f"{analysis['timestamp']} · {analysis['inference_status']} · "
                f"Model {analysis['model_version']}"
            )
            if analysis.get("processing_duration_ms") is not None:
                st.caption(f"Processing time: {analysis['processing_duration_ms']:.1f} ms")
            if analysis.get("error_message"):
                st.error(analysis["error_message"])
            try:
                detections = json.loads(analysis["classes"])
                confidences = json.loads(analysis["confidence_scores"])
                boxes = json.loads(analysis["bounding_boxes"])
            except (TypeError, json.JSONDecodeError):
                detections, confidences, boxes = [], [], []
                st.warning("Saved detection details could not be decoded.")
            if detections:
                if analysis["inference_status"] == "low_confidence":
                    st.warning("Candidates below the 0.25 display threshold are shown for review; these are not confirmed damage.")
                st.dataframe([
                    {"Class": name, "Confidence": score, "Bounding box (x1,y1,x2,y2)": box}
                    for name, score, box in zip(detections, confidences, boxes)
                ], hide_index=True, width="stretch")
            elif analysis["inference_status"] == "no_detections":
                st.info("No detections were returned at the configured 0.25 confidence threshold. This does not reject the report.")
            annotated_path = analysis.get("annotated_image_path")
            if annotated_path:
                try:
                    annotated = road_ai.path_under_project(annotated_path)
                    st.image(annotated.read_bytes(), caption="Annotated copy (original evidence is unchanged)",
                             use_container_width=True)
                except (road_ai.InferenceError, OSError) as error:
                    st.warning(f"The saved annotated image is unavailable: {error}")


def reports_csv(reports):
    output = io.StringIO(newline="")
    fields = [
        "report_id", "submitted_at", "road_name", "location", "district", "district_id", "problem_type", "severity",
        "description", "latitude", "longitude", "photo_path", "status",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(reports)
    return output.getvalue().encode("utf-8-sig")


def point_in_districts(longitude, latitude, districts):
    """Compatibility wrapper; boundary and ambiguous points do not count as inside."""
    return match_district(longitude, latitude, districts)["status"] == "matched"


def audit_district_features(data, registry=DISTRICTS):
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("District GeoJSON must be a FeatureCollection.")
    registry_by_name = {
        normalize_district_name(value): entry
        for entry in registry
        for value in [entry["name"], *entry.get("aliases", [])]
    }
    candidates, invalid, unmatched = [], [], []
    ids_by_feature = {}
    for index, feature in enumerate(data["features"]):
        properties = feature.get("properties") or {}
        source_name = properties.get("shapeName") or properties.get("district_name")
        entry = registry_by_name.get(normalize_district_name(source_name)) if source_name else None
        if not entry:
            unmatched.append({"index": index, "name": source_name})
            continue
        valid, reason = validate_district_geometry(feature.get("geometry"))
        if not valid:
            invalid.append({"index": index, "name": source_name, "reason": reason})
            continue
        prepared = dict(feature)
        prepared_properties = dict(properties)
        prepared_properties["district_id"] = entry["id"]
        prepared_properties["district_name"] = entry["name"]
        prepared_properties["_geometry_valid"] = True
        prepared["properties"] = prepared_properties
        candidates.append(prepared)
        ids_by_feature.setdefault(entry["id"], []).append(prepared)

    duplicates = [district_id for district_id, entries in ids_by_feature.items() if len(entries) > 1]
    duplicate_ids = set(duplicates)
    valid_features = [
        feature for feature in candidates
        if feature["properties"]["district_id"] not in duplicate_ids
    ]
    present_ids = {feature["properties"]["district_id"] for feature in valid_features}
    missing = [entry for entry in registry if entry["id"] not in present_ids]
    diagnostics = {
        "valid_polygon_count": len(valid_features),
        "invalid_features": invalid,
        "duplicate_district_ids": duplicates,
        "unmatched_features": unmatched,
        "missing_districts": missing,
    }
    result = dict(data)
    result["features"] = valid_features
    result["roadguard_diagnostics"] = diagnostics
    return result, diagnostics


@st.cache_data(show_spinner=False)
def load_districts():
    with DATA_FILE.open(encoding="utf-8") as geojson_file:
        data = json.load(geojson_file)
    if data.get("type") != "FeatureCollection" or not data.get("features"):
        raise ValueError("The district GeoJSON is empty or invalid.")
    prepared, _ = audit_district_features(data)
    return prepared


def build_folium_map(districts, bounds, reports):
    road_map = folium.Map(
        location=[10.7905, 78.7047], zoom_start=7, tiles="OpenStreetMap",
        control_scale=True, prefer_canvas=True,
    )
    folium.GeoJson(
        districts,
        name="Tamil Nadu districts",
        style_function=lambda _feature: {
            "fillColor": "#dbeafe", "color": "#1d4ed8", "weight": 1.4, "fillOpacity": 0.38,
        },
        highlight_function=lambda _feature: {
            "fillColor": "#fbbf24", "color": "#b45309", "weight": 2.5, "fillOpacity": 0.62,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["district_name"], aliases=["District:"], sticky=True,
            style=("background-color: white; color: #111827; font-family: sans-serif; "
                   "font-size: 13px; padding: 7px;"),
        ),
        popup=folium.GeoJsonPopup(
            fields=["district_name"], aliases=["District:"], labels=True,
            style="font-family: sans-serif; font-size: 13px;",
        ),
    ).add_to(road_map)

    for report in reports:
        latitude, longitude = report.get("latitude"), report.get("longitude")
        if latitude is None or longitude is None:
            continue
        try:
            latitude, longitude = float(latitude), float(longitude)
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(latitude) or not math.isfinite(longitude)
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            continue
        popup_text = "<br>".join(
            html.escape(str(report.get(field) or ""))
            for field in ("problem_type", "severity", "status", "road_name", "location", "district")
        )
        folium.CircleMarker(
            location=[latitude, longitude], radius=7, color="#b91c1c",
            fill=True, fill_color="#ef4444", fill_opacity=0.9,
            tooltip=f"Saved report: {html.escape(str(report['problem_type']))}",
            popup=folium.Popup(popup_text, max_width=280),
        ).add_to(road_map)

    selected_location = st.session_state.get("selected_location")
    if selected_location:
        folium.Marker(
            [selected_location["lat"], selected_location["lng"]],
            tooltip="Selected report location",
            icon=folium.Icon(color="blue", icon="info-sign"),
        ).add_to(road_map)
    folium.LayerControl(collapsed=True).add_to(road_map)
    road_map.fit_bounds(
        [[bounds[0], bounds[1]], [bounds[2], bounds[3]]], padding=(22, 22), max_zoom=7
    )
    return road_map


def render_map(districts, bounds, reports, height=650, map_key="road-map"):
    return st_folium(
        build_folium_map(districts, bounds, reports),
        use_container_width=True,
        height=height,
        returned_objects=["last_clicked"],
        key=map_key,
    )


def sync_map_selection(map_result, districts):
    clicked = (map_result or {}).get("last_clicked")
    if not clicked:
        return
    try:
        latitude = float(clicked["lat"])
        longitude = float(clicked["lng"])
    except (KeyError, TypeError, ValueError):
        return
    signature = (round(latitude, 7), round(longitude, 7))
    if st.session_state.get("last_map_click") == signature:
        return
    st.session_state["last_map_click"] = signature
    match = match_district(longitude, latitude, districts)
    if match["status"] == "matched":
        st.session_state["selected_location"] = {"lat": latitude, "lng": longitude}
        st.session_state["selected_district_id"] = match["district_id"]
        st.session_state["location_error"] = None
        st.rerun()
    else:
        st.session_state["selected_location"] = {"lat": latitude, "lng": longitude}
        st.session_state.pop("selected_district_id", None)
        if match["status"] == "boundary_or_overlap":
            st.session_state["location_error"] = (
                "This point falls on a district boundary or overlapping polygons; no district was assigned. "
                "Choose an interior point or select a district manually."
            )
        else:
            st.session_state["location_error"] = (
                "No valid local district polygon covers this point. No district was assigned; "
                "choose a district manually."
            )
        st.rerun()


def render_map_page(districts, bounds, reports, map_version=0):
    st.subheader("Tamil Nadu Road Map")
    st.caption(
        "Explore OpenStreetMap streets and highways. Click inside a district to choose an "
        "approximate report location. Saved reports with coordinates appear as markers."
    )
    filters = st.columns(3)
    diagnostics = districts.get("roadguard_diagnostics", {})
    missing_names = [entry["name"] for entry in diagnostics.get("missing_districts", [])]
    if missing_names:
        st.warning(
            f"Local boundaries available for {diagnostics.get('valid_polygon_count', 0)} of "
            f"{len(DISTRICTS)} districts. Missing boundary polygons: {', '.join(missing_names)}."
        )
    district_labels = {entry["id"]: entry["name"] for entry in DISTRICTS}
    district_filter = filters[0].selectbox(
        "District", ["All", *district_labels],
        format_func=lambda value: "All" if value == "All" else district_labels[value],
        key="map_filter_district",
    )
    damage_filter = filters[1].selectbox(
        "Damage type", ["All", *available_problem_types()], key="map_filter_damage"
    )
    severity_filter = filters[2].selectbox(
        "Severity", ["All", *SEVERITIES], key="map_filter_severity"
    )
    visible_reports = [
        report for report in reports
        if (district_filter == "All" or report.get("district_id") == district_filter)
        and (damage_filter == "All" or report.get("problem_type") == damage_filter)
        and (severity_filter == "All" or report.get("severity") == severity_filter)
    ]
    st.caption(f"Showing {len(visible_reports)} saved report marker(s) matching these filters.")
    map_result = render_map(
        districts, bounds, visible_reports, height=600, map_key=f"road-map-{map_version}"
    )
    sync_map_selection(map_result, districts)
    selected_location = st.session_state.get("selected_location")
    if selected_location:
        st.info(
            f"Selected approximate location: {selected_location['lat']:.6f}, "
            f"{selected_location['lng']:.6f} - review these coordinates before including them in a report."
        )
        selected_district_id = st.session_state.get("selected_district_id")
        if selected_district_id:
            st.caption("Unambiguous polygon match: " + DISTRICT_BY_ID[selected_district_id]["name"])
        if st.button("Clear selected location", key="clear_map_location"):
            st.session_state.pop("selected_location", None)
            st.session_state.pop("selected_district_id", None)
            st.session_state.pop("last_map_click", None)
            st.session_state["map_widget_version"] = st.session_state.get("map_widget_version", 0) + 1
            st.rerun()
    if st.session_state.get("location_error"):
        st.warning(st.session_state.pop("location_error"))
    st.caption(
        "OpenStreetMap tiles need an internet connection. If tiles are unavailable, "
        "district boundaries and report markers remain visible."
    )


def render_report_page(districts, database_ready):
    st.subheader("Report Road Damage")
    st.write("Submit a road observation for review. This prototype does not forward reports to an authority.")
    selected_location = st.session_state.get("selected_location")
    if selected_location:
        st.info(
            f"Map-selected approximate coordinates: {selected_location['lat']:.6f}, "
            f"{selected_location['lng']:.6f}"
        )
    else:
        st.caption("Coordinates are optional. Select a point on the Map, or enter both coordinates manually.")
    selected_district_id = st.session_state.get("selected_district_id")
    submission_token = st.session_state.setdefault("report_submission_token", uuid.uuid4().hex)
    submission_complete = st.session_state.get("report_submission_complete_token") == submission_token
    if submission_complete:
        st.success(st.session_state.get("report_submission_success", "Report submitted successfully."))
        if st.button("Start another report", key="start_another_report"):
            st.session_state["report_submission_token"] = uuid.uuid4().hex
            st.session_state.pop("report_submission_complete_token", None)
            st.session_state.pop("report_submission_success", None)
            for key in (
                "report_road_name", "report_location", "report_district_id",
                "report_problem_type", "report_severity", "report_description", "report_photo",
            ):
                st.session_state.pop(key, None)
            st.rerun()

    with st.form("road_problem_form", clear_on_submit=False):
        road_name = st.text_input("Road or street name *", max_chars=160, key="report_road_name")
        location = st.text_input("Location / nearest landmark *", max_chars=200, key="report_location")
        district_ids = [entry["id"] for entry in DISTRICTS]
        district_id = st.selectbox(
            "District in Tamil Nadu *", district_ids,
            index=district_ids.index(selected_district_id) if selected_district_id in district_ids else 0,
            format_func=lambda value: DISTRICT_BY_ID[value]["name"],
            key="report_district_id",
        )
        problem_type = st.selectbox("Damage type *", PROBLEM_TYPES, key="report_problem_type")
        severity = st.selectbox("Severity *", SEVERITIES, key="report_severity")
        description = st.text_area("Description of the problem *", max_chars=3000, key="report_description")
        coordinate_key = (
            f"{selected_location['lat']:.6f}-{selected_location['lng']:.6f}"
            if selected_location else "manual"
        )
        latitude_text = st.text_input(
            "Latitude (optional)",
            value=f"{selected_location['lat']:.6f}" if selected_location else "",
            key=f"report-latitude-{coordinate_key}",
        )
        longitude_text = st.text_input(
            "Longitude (optional)",
            value=f"{selected_location['lng']:.6f}" if selected_location else "",
            key=f"report-longitude-{coordinate_key}",
        )
        photo = st.file_uploader(
            "Optional photo evidence (JPEG, PNG, WebP; max 5 MB)",
            type=["jpg", "jpeg", "png", "webp"],
            key="report_photo",
        )
        submitted = st.form_submit_button(
            "Submit Report", type="primary",
            disabled=not database_ready or submission_complete,
        )

    if submitted:
        missing = []
        if not road_name.strip():
            missing.append("road or street name")
        if not location.strip():
            missing.append("location or nearest landmark")
        if not description.strip():
            missing.append("description")
        if missing:
            st.error("Please provide the required " + " and ".join(missing) + ".")
            return
        try:
            photo_bytes, photo_extension = inspect_image_upload(photo)
        except ValueError as error:
            st.error(str(error))
            return

        latitude_value = latitude_text.strip()
        longitude_value = longitude_text.strip()
        if bool(latitude_value) != bool(longitude_value):
            st.error("Enter both latitude and longitude, or leave both blank.")
            return
        latitude = longitude = None
        if latitude_value:
            try:
                latitude, longitude = float(latitude_value), float(longitude_value)
            except ValueError:
                st.error("Latitude and longitude must be valid numbers.")
                return
            if (not math.isfinite(latitude) or not math.isfinite(longitude)
                    or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
                st.error("Latitude or longitude is outside the valid geographic range.")
                return
            coordinate_match = match_district(longitude, latitude, districts)
            if coordinate_match["status"] == "matched":
                if coordinate_match["district_id"] != district_id:
                    st.error(
                        f"These coordinates match {coordinate_match['district_name']}; "
                        "select that district or correct the coordinates."
                    )
                    return
            elif coordinate_match["status"] == "boundary_or_overlap":
                st.warning(
                    "The coordinates are on a boundary or ambiguous. The selected district "
                    "will remain a manual user choice; no automatic district match is recorded."
                )
            else:
                st.warning(
                    "No valid local polygon can verify these coordinates. The selected district "
                    "will remain a manual user choice; no automatic district match is recorded."
                )

        report = {
            "road_name": road_name.strip(),
            "location": location.strip(),
            "district": DISTRICT_BY_ID[district_id]["name"],
            "district_id": district_id,
            "problem_type": problem_type,
            "severity": severity,
            "description": description.strip(),
            "latitude": latitude,
            "longitude": longitude,
        }
        try:
            report_id = save_report(
                report, photo_bytes=photo_bytes, photo_extension=photo_extension,
                submission_token=submission_token,
            )
        except (OSError, sqlite3.Error, ValueError) as error:
            st.error(f"The report could not be saved. Please try again. Details: {error}")
            return
        message = f"Report submitted successfully. Reference ID: {report_id}"
        st.session_state["report_submission_complete_token"] = submission_token
        st.session_state["report_submission_success"] = message
        st.rerun()


def render_history_page(districts, database_ready):
    st.subheader("Report History")
    if not database_ready:
        st.error("Report history is unavailable because the database could not be opened.")
        return
    preset = st.session_state.get("dashboard_history_filters", {})
    columns = st.columns(4)
    try:
        all_reports = fetch_reports()
        authorities = fetch_authorities(active_only=False)
    except sqlite3.Error as error:
        st.error(f"Could not load district choices: {error}")
        return
    district_labels = {entry["id"]: entry["name"] for entry in DISTRICTS}
    legacy_values = sorted({
        report.get("district") for report in all_reports
        if report.get("district") and not report.get("district_id")
    })
    for legacy_name in legacy_values:
        district_labels[f"legacy:{legacy_name}"] = f"Legacy / unmatched: {legacy_name}"
    district_filter = columns[0].selectbox(
        "District", ["All", *district_labels],
        format_func=lambda value: "All" if value == "All" else district_labels[value],
        index=(["All", *district_labels].index(preset.get("district_id", "All"))
               if preset.get("district_id", "All") in ["All", *district_labels] else 0),
        key="filter_district",
    )
    type_filter = columns[1].selectbox(
        "Damage type", ["All", *available_problem_types()],
        index=(["All", *available_problem_types()].index(preset.get("problem_type", "All"))
               if preset.get("problem_type", "All") in ["All", *available_problem_types()] else 0),
        key="filter_type"
    )
    severity_options = ["All", *SEVERITIES]
    status_options = ["All", *FILTER_STATUSES]
    severity_filter = columns[2].selectbox(
        "Severity", severity_options,
        index=severity_options.index(preset.get("severity", "All")) if preset.get("severity", "All") in severity_options else 0,
        key="filter_severity",
    )
    status_filter = columns[3].selectbox(
        "Status", status_options,
        index=status_options.index(preset.get("status", "All")) if preset.get("status", "All") in status_options else 0,
        key="filter_status",
    )
    filter_cols = st.columns(4)
    submitted_from = filter_cols[0].date_input("Submitted from", value=None, key="filter_submitted_from")
    submitted_to = filter_cols[1].date_input("Submitted through", value=None, key="filter_submitted_to")
    authority_options = ["All", "__unassigned__", *[item["authority_id"] for item in authorities]]
    authority_labels = {"All": "All", "__unassigned__": "No active assignment"}
    authority_labels.update({item["authority_id"]: item["authority_name"] for item in authorities})
    selected_authority = filter_cols[2].selectbox(
        "Authority", authority_options,
        format_func=lambda value: authority_labels.get(value, value),
        index=authority_options.index(preset.get("authority_id", "All"))
        if preset.get("authority_id", "All") in authority_options else 0,
        key="filter_authority",
    )
    ai_options = ["All", "none", "detections", "no_detections", "low_confidence", "error"]
    ai_labels = {
        "All": "All AI outcomes", "none": "No AI analysis", "detections": "Detection candidates",
        "no_detections": "No detections", "low_confidence": "Low confidence", "error": "Analysis error",
    }
    selected_ai = filter_cols[3].selectbox(
        "Latest AI outcome", ai_options, format_func=lambda value: ai_labels[value],
        index=ai_options.index(preset.get("ai_outcome", "All"))
        if preset.get("ai_outcome", "All") in ai_options else 0,
        key="filter_ai_outcome",
    )
    workflow_options = [
        "All", "open", "unassigned", "assigned", "repair_in_progress",
        "awaiting_verification", "verified", "reopened", "overdue", "manual_review",
    ]
    workflow_labels = {
        "All": "All workflow groups", "open": "Open", "unassigned": "Unassigned",
        "assigned": "Assigned or scheduled", "repair_in_progress": "Repair in progress",
        "awaiting_verification": "Awaiting verification", "verified": "Verified and closed",
        "reopened": "Reopened", "overdue": "Overdue jobs", "manual_review": "Manual authority review",
    }
    workflow_filter = st.selectbox(
        "Dashboard grouping", workflow_options,
        format_func=lambda value: workflow_labels[value],
        index=workflow_options.index(preset.get("workflow_bucket", "All"))
        if preset.get("workflow_bucket", "All") in workflow_options else 0,
        key="filter_workflow_bucket",
        help="Workflow grouping is an additional database filter; queue review may overlap report status.",
    )
    if submitted_from and submitted_to and submitted_from > submitted_to:
        st.error("The start date must be on or before the end date.")
        return
    try:
        records = fetch_reports(filters={
            "district_id": district_filter, "problem_type": type_filter,
            "severity": severity_filter, "status": status_filter,
            "submitted_from": submitted_from.isoformat() if submitted_from else None,
            "submitted_to": submitted_to.isoformat() if submitted_to else None,
            "authority_id": selected_authority, "ai_outcome": selected_ai,
            "workflow_bucket": workflow_filter,
        })
    except sqlite3.Error as error:
        st.error(f"Could not load reports: {error}")
        return

    st.download_button(
        "Export filtered reports as CSV",
        data=reports_csv(records),
        file_name="road_reports.csv",
        mime="text/csv",
        disabled=not records,
    )
    if records:
        display_fields = [
            "report_id", "submitted_at", "road_name", "location", "district", "district_id", "problem_type",
            "severity", "description", "latitude", "longitude", "status",
        ]
        st.dataframe([{key: record.get(key) for key in display_fields} for record in records],
                     width="stretch", hide_index=True)
        st.caption(f"Showing {len(records)} matching report(s). Photo evidence is stored in data/uploads.")
        st.markdown("#### Report details and status history")
        for record in records:
            title = f"{record['problem_type']} | {record['road_name']} | {record['report_id'][:8]}"
            with st.expander(title):
                st.caption(
                    f"{record['district']} | {record.get('location') or 'Location not provided'} | "
                    f"Current status: {record['status']}"
                )
                try:
                    history = fetch_report_status_history(record["report_id"])
                except sqlite3.Error as error:
                    st.error(f"Could not load status history: {error}")
                    history = []
                if history:
                    st.markdown("**Status history**")
                    for event in history:
                        previous = event["previous_status"] or "(initial)"
                        st.caption(
                            f"{event['changed_at']} · {previous} → {event['new_status']} · "
                            f"{event['actor']} · {event['reason']}"
                        )
                        if event.get("verification_evidence"):
                            st.caption("Verification reference: " + event["verification_evidence"])

                render_report_ai_analysis(record)

                next_statuses = permitted_next_statuses(record["status"])
                if not next_statuses:
                    st.info("No permitted next status is configured for this legacy status.")
                    continue
                selected_status = st.selectbox(
                    "Permitted next status", next_statuses,
                    key=f"status-{record['report_id']}",
                )
                reason = st.text_area(
                    "Reason for this status change *", max_chars=1000,
                    key=f"status-reason-{record['report_id']}",
                )
                verification_evidence = None
                if selected_status == "Verified and Closed":
                    st.warning(
                        "A report photo alone cannot close a report. Enter an independent "
                        "verification reference or note."
                    )
                    verification_evidence = st.text_area(
                        "Independent verification evidence *", max_chars=1000,
                        key=f"verification-evidence-{record['report_id']}",
                    )
                st.caption(
                    f"Recorded actor: {LOCAL_ACTOR}. This prototype does not authenticate users."
                )
                if st.button("Record status change", key=f"update-status-{record['report_id']}"):
                    try:
                        transition_report_status(
                            record["report_id"], selected_status, reason,
                            verification_evidence=verification_evidence,
                        )
                    except (sqlite3.Error, ValueError) as error:
                        st.error(f"Could not update report status: {error}")
                    else:
                        st.session_state["status_update_message"] = (
                            f"Report {record['report_id']} updated to {selected_status}."
                        )
                        st.rerun()
    else:
        st.info("No saved reports match these filters.")


def _open_dashboard_records(filters):
    """Seed history controls before its widgets are created on this rerun."""
    for key in (
        "filter_district", "filter_type", "filter_severity", "filter_status",
        "filter_submitted_from", "filter_submitted_to", "filter_authority",
        "filter_ai_outcome", "filter_workflow_bucket",
    ):
        st.session_state.pop(key, None)
    st.session_state["dashboard_history_filters"] = dict(filters)
    st.session_state["page"] = "Report History"


def _dashboard_metric(column, label, value, key, filters):
    column.metric(label, value)
    if column.button(
        f"{value} · View records", key=f"dashboard-count-{key}",
        help=f"Open matching {label.lower()}",
    ):
        _open_dashboard_records(filters)


def main():
    st.set_page_config(page_title="Tamil Nadu Road Map", layout="wide")
    st.title("Tamil Nadu Road Map")
    if "status_update_message" in st.session_state:
        st.success(st.session_state.pop("status_update_message"))
    if "ai_analysis_message" in st.session_state:
        st.success(st.session_state.pop("ai_analysis_message"))

    try:
        initialize_database()
        database_ready = True
        db_error = None
    except (OSError, sqlite3.Error) as error:
        database_ready = False
        db_error = error

    if database_ready:
        try:
            total_reports, open_reports, closed_reports, issue_counts = fetch_dashboard_stats()
            metric_cols = st.columns(3)
            _dashboard_metric(metric_cols[0], "Total reports", total_reports, "total", {})
            _dashboard_metric(metric_cols[1], "Open reports", open_reports, "open", {"workflow_bucket": "open"})
            _dashboard_metric(metric_cols[2], "Verified and closed", closed_reports, "verified", {"workflow_bucket": "verified"})
            category_summary = " | ".join(f"{name}: {count}" for name, count in issue_counts.items())
            st.caption("Issue categories: " + (category_summary or "No reports submitted yet"))
            if issue_counts:
                category_cols = st.columns(min(4, len(issue_counts)))
                for index, (name, count) in enumerate(issue_counts.items()):
                    column = category_cols[index % len(category_cols)]
                    if column.button(f"{name}: {count}", key=f"dashboard-category-{index}"):
                        _open_dashboard_records({"problem_type": name})
            phase4 = fetch_phase4_dashboard_stats()
            phase4_cols = st.columns(4)
            metric_defs = [
                ("Unassigned reports", "unassigned_reports", "unassigned"),
                ("Assigned or scheduled", "assigned_cases", "assigned"),
                ("Repairs in progress", "repairs_in_progress", "repair_in_progress"),
                ("Awaiting verification", "awaiting_verification", "awaiting_verification"),
            ]
            for column, (label, key, bucket) in zip(phase4_cols, metric_defs):
                _dashboard_metric(column, label, phase4[key], key, {"workflow_bucket": bucket})
            phase4_cols = st.columns(4)
            metric_defs = [
                ("Verified cases", "verified_cases", "verified"),
                ("Reopened cases", "reopened_cases", "reopened"),
                ("Overdue jobs", "overdue_jobs", "overdue"),
                ("Manual authority review", "manual_authority_review", "manual_review"),
            ]
            for column, (label, key, bucket) in zip(phase4_cols, metric_defs):
                _dashboard_metric(column, label, phase4[key], key, {"workflow_bucket": bucket})
            st.caption(
                "Open = all statuses except Verified and Closed or Rejected. Unassigned counts Submitted, "
                "Pending Validation, Validated, Assigned, or Repair Scheduled reports without an active "
                "assignment. Assigned includes Assigned or Repair Scheduled with an active assignment. "
                "Repair in progress, awaiting verification, verified, and reopened are status buckets. "
                "These lifecycle buckets do not overlap; manual authority review is a separate open-queue "
                "count and can overlap any lifecycle bucket. Overdue counts distinct reports with an active "
                "assignment due before the current UTC time while status is Assigned, Repair Scheduled, "
                "Repair In Progress, or awaiting verification."
            )
            derived = fetch_dashboard_derived_metrics()
            derived_cols = st.columns(3)
            assignment_avg = derived["avg_submission_to_assignment_seconds"]
            repair_avg = derived["avg_repair_completion_seconds"]
            failure_rate = derived["verification_failure_rate"]
            derived_cols[0].metric(
                "Avg submission to assignment",
                f"{assignment_avg / 3600:.1f} h" if assignment_avg is not None else "Unavailable",
            )
            derived_cols[0].caption(f"Valid timestamp pairs: {derived['submission_to_assignment_n']}")
            derived_cols[1].metric(
                "Avg repair completion",
                f"{repair_avg / 3600:.1f} h" if repair_avg is not None else "Unavailable",
            )
            derived_cols[1].caption(f"Repairs with valid start and completion: {derived['repair_completion_n']}")
            derived_cols[2].metric(
                "Verification failure rate",
                f"{failure_rate:.1%}" if failure_rate is not None else "Unavailable",
            )
            derived_cols[2].caption(
                f"Rejected or More Evidence decisions / all recorded decisions "
                f"(n={derived['verification_decision_n']})"
            )
        except sqlite3.Error as error:
            database_ready = False
            db_error = error

    if not database_ready:
        st.error(f"Report database is unavailable: {db_error}")

    try:
        districts = load_districts()
        bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        for feature in districts["features"]:
            coordinate_bounds(feature.get("geometry", {}).get("coordinates", []), bounds)
        if not all(value not in (float("inf"), float("-inf")) for value in bounds):
            raise ValueError("District coordinates are missing from the GeoJSON.")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        districts = None
        bounds = None
        st.error(f"Tamil Nadu district data could not be loaded: {error}")

    if districts:
        if "page" not in st.session_state:
            st.session_state["page"] = "Map"
        page = st.sidebar.radio(
            "Navigation", ["Map", "Report Road Damage", "Report History", "Authority Setup", "Operations"],
            key="page",
        )
        if "ops_message" in st.session_state:
            st.success(st.session_state.pop("ops_message"))
        try:
            saved_reports = fetch_reports() if database_ready else []
        except sqlite3.Error as error:
            saved_reports = []
            st.error(f"Saved report locations are unavailable: {error}")

        if page == "Map":
            widget_version = st.session_state.get("map_widget_version", 0)
            try:
                render_map_page(districts, bounds, saved_reports, map_version=widget_version)
            except Exception as error:
                st.error(f"The map could not be rendered: {error}")
        elif page == "Report Road Damage":
            render_report_page(districts, database_ready)
        elif page == "Report History":
            render_history_page(districts, database_ready)
        elif page == "Authority Setup":
            render_authority_page(database_ready)
        else:
            render_operations_page(database_ready)


if __name__ == "__main__":
    main()
