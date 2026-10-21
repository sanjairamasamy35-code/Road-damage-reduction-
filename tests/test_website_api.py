"""FastAPI integration tests against isolated SQLite files and real project data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import main as api


class WebsiteApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db = root / "api-test.sqlite"
        self.uploads = root / "uploads"
        self.client_context = patch.multiple(
            api, DATABASE_FILE=self.db, UPLOAD_DIR=self.uploads,
            ANNOTATED_DIR=root / "annotated",
        )
        self.client_context.start()
        self.client = TestClient(api.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.client_context.stop()
        self.temp.cleanup()

    def submit(self, **overrides):
        fields = {
            "road_name": "Test Road", "district_id": "TN-DIST-CHENNAI",
            "problem_type": "Pothole", "severity": "Medium",
            "description": "A pothole near the junction.", "location": "Near the test junction",
        }
        fields.update(overrides)
        return self.client.post("/api/reports", data=fields)

    def test_health_and_real_district_boundaries(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)
        response = self.client.get("/api/districts")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["diagnostics"]["valid_polygon_count"], 34)
        self.assertEqual(len(response.json()["diagnostics"]["missing_districts"]), 4)

    def test_empty_dashboard_and_filtered_history_are_truthful(self):
        self.assertEqual(self.client.get("/api/dashboard").json()["total_reports"], 0)
        result = self.client.get("/api/reports?district_id=TN-DIST-CHENNAI").json()
        self.assertEqual(result, {"reports": [], "total": 0})

    def test_report_submission_persists_and_can_be_tracked(self):
        response = self.submit()
        self.assertEqual(response.status_code, 201, response.text)
        result = response.json()
        report_id = result["report"]["report_id"]
        self.assertEqual(result["report"]["status"], "Submitted")
        self.assertEqual(result["ai_analysis"]["status"], "not_requested")
        detail = self.client.get(f"/api/reports/{report_id}").json()
        self.assertEqual(detail["report"]["report_id"], report_id)
        self.assertEqual(detail["status_history"][0]["new_status"], "Submitted")
        self.assertEqual(self.client.get("/api/dashboard").json()["total_reports"], 1)

    def test_unknown_report_and_invalid_fields_are_rejected(self):
        self.assertEqual(self.client.get("/api/reports/not-a-real-id").status_code, 404)
        self.assertEqual(self.submit(district_id="unknown").status_code, 422)
        self.assertEqual(self.submit(severity="Extreme").status_code, 422)
        self.assertEqual(self.submit(latitude="12.5").status_code, 422)

    def test_upload_requires_a_valid_image_signature_and_content(self):
        response = self.client.post("/api/reports", data={
            "road_name": "Test Road", "district_id": "TN-DIST-CHENNAI",
            "problem_type": "Pothole", "severity": "Low", "description": "Test photo.",
            "location": "Near the test junction",
        }, files={"photo": ("damage.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
