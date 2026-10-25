"""
tests/test_phase10_api_experimental.py

Tests FastAPI integration for Phase 10:
1. Preserves HTTP 503 on /predict/passive-aggression
2. Successfully serves /predict/passive-aggression/experimental
3. Enforces label_provenance: LLM_SILVER
4. Exposes clear governance warnings
"""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app


class TestPhase10APIExperimental(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_cm.__exit__(None, None, None)

    def test_original_passive_aggression_endpoint_remains_503(self):
        """Verify strict gate preservation on primary endpoint."""
        response = self.client.post("/predict/passive-aggression")
        self.assertEqual(response.status_code, 503)
        data = response.json()
        self.assertEqual(data["status"], "unavailable")
        self.assertEqual(data["benchmark_status"], "BLOCKED_PENDING_HUMAN_ANNOTATION")

    def test_experimental_passive_aggression_endpoint_serves_predictions(self):
        """Verify exploratory silver model serves valid predictions."""
        payload = {"text": "Oh, what a brilliant contribution to society."}
        response = self.client.post("/predict/passive-aggression/experimental", json=payload)
        self.assertEqual(response.status_code, 200, f"Failed with {response.status_code}: {response.text}")
        data = response.json()
        self.assertEqual(data["task"], "passive_aggression_detection")
        self.assertEqual(data["model_id"], "pa_silver_tfidf_v1")
        self.assertEqual(data["label_provenance"], "LLM_SILVER")
        self.assertEqual(data["evaluation_status"], "LLM_SILVER_DIAGNOSTIC_EVALUATION_AVAILABLE")
        self.assertIn("predicted_label", data)
        self.assertIn(data["predicted_label"], ["passive_aggressive", "non_passive_aggressive"])
        self.assertIn("class_scores", data)
        self.assertTrue(any("BLOCKED_PENDING_HUMAN_ANNOTATION" in w for w in data["warnings"]))

    def test_experimental_endpoint_validation_error_on_empty_text(self):
        """Verify input validation rejecting empty strings."""
        payload = {"text": "   "}
        response = self.client.post("/predict/passive-aggression/experimental", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
