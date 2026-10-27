"""ARGUS demo backend.

    uvicorn api.main:app --port 8000      (from the repo root)

Serves the precomputed demo (api/demo_data.json, built by scripts/build_demo.py), records the
officer's review of the demo case, and serves the built frontend (frontend/dist) at /.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from argus.agent.report import render_str
from argus.agent.review import DECISIONS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "api" / "demo_data.json"
REVIEWS = ROOT / "outputs" / "demo_reviews.json"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="ARGUS demo API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


class Review(BaseModel):
    officer: str = Field(min_length=1, max_length=80)
    decision: str
    note: str = Field(default="", max_length=500)


def _load_demo() -> dict:
    if not DATA.exists():
        raise HTTPException(503, "demo data missing: run `python scripts/build_demo.py` first")
    return json.loads(DATA.read_text(encoding="utf-8"))


def _reviews() -> dict:
    return json.loads(REVIEWS.read_text(encoding="utf-8")) if REVIEWS.exists() else {}


def _with_review(demo: dict) -> dict:
    case = demo["case"]
    review = _reviews().get(case["case_id"])
    if review:
        case["review"], case["status"] = review, review["decision"]
        demo["str_markdown"] = render_str(case, case.get("narrative"))
    return demo


@app.get("/api/health")
def health():
    return {"ok": True, "demo_data": DATA.exists()}


@app.get("/api/demo")
def demo():
    return _with_review(_load_demo())


@app.post("/api/review")
def review(r: Review):
    if r.decision not in DECISIONS:
        raise HTTPException(422, f"decision must be one of {sorted(DECISIONS)}")
    case_id = _load_demo()["case"]["case_id"]
    reviews = _reviews()
    if case_id in reviews:
        raise HTTPException(409, f"{case_id} already {reviews[case_id]['decision']}; reset the demo to review again")
    reviews[case_id] = {"officer": r.officer.strip(), "decision": DECISIONS[r.decision], "note": r.note.strip(),
                        "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    return _with_review(_load_demo())


@app.delete("/api/review")
def reset_review():
    """Clear the demo case's review so the demo can be run again."""
    reviews = _reviews()
    reviews.pop(_load_demo()["case"]["case_id"], None)
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    REVIEWS.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    return {"ok": True}


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
