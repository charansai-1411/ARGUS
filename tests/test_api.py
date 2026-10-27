"""Demo API: serves demo data, records exactly one officer review per case, can be reset."""

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as main  # noqa: E402

CASE = {"case_id": "ARGUS-DEMO-001", "created_at": "2026-01-01T00:00:00+00:00", "seed_transfer": 1,
        "status": "pending_review", "entry_accounts": ["1_A"], "exit_accounts": ["2_C"], "money_trail": ["1_A", "2_B", "2_C"],
        "indicators": {"accounts": 3, "banks": 2, "transfers": 2, "cross_bank_share": 0.5, "amount_entered": 100.0,
                       "amount_to_exits": 98.0, "duration_minutes": 10, "median_pass_through_minutes": 5.0,
                       "dormant_accounts": 1, "shared_owner_groups": {}, "max_fan_out": 1, "max_fan_in": 1,
                       "has_cycle": False, "payment_formats": {"ACH": 2}},
        "accounts": [], "timeline": [], "freeze_recommendation": [], "review": None, "narrative": None}


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "demo_data.json"
    data.write_text(json.dumps({"case": CASE, "str_markdown": "# STR", "meta": {}}))
    monkeypatch.setattr(main, "DATA", data)
    monkeypatch.setattr(main, "REVIEWS", tmp_path / "reviews.json")
    return TestClient(main.app)


def test_demo_and_review_flow(client):
    assert client.get("/api/health").json()["demo_data"]
    assert client.get("/api/demo").json()["case"]["status"] == "pending_review"

    r = client.post("/api/review", json={"officer": "Officer A", "decision": "approve", "note": "ok"})
    assert r.status_code == 200
    body = r.json()
    assert body["case"]["status"] == "approved" and "approved by Officer A" in body["str_markdown"]
    assert client.post("/api/review", json={"officer": "B", "decision": "reject"}).status_code == 409

    assert client.delete("/api/review").json()["ok"]
    assert client.get("/api/demo").json()["case"]["status"] == "pending_review"


def test_review_validation(client):
    assert client.post("/api/review", json={"officer": "A", "decision": "freeze_now"}).status_code == 422
    assert client.post("/api/review", json={"officer": "", "decision": "approve"}).status_code == 422


def test_missing_demo_data(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA", tmp_path / "nope.json")
    assert TestClient(main.app).get("/api/demo").status_code == 503
