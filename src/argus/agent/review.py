"""Human-in-the-loop: an officer approves or rejects each case. ARGUS never freezes anything itself."""

import json
from datetime import datetime, timezone
from pathlib import Path

DECISIONS = {"approve": "approved", "reject": "rejected", "escalate": "escalated"}


def record_review(case_path: Path, officer: str, decision: str, note: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    if not officer.strip():
        raise ValueError("officer name is required")
    case = json.loads(Path(case_path).read_text())
    if case.get("review"):
        raise ValueError(f"case {case['case_id']} was already reviewed: {case['review']['decision']}")
    case["review"] = {"officer": officer.strip(), "decision": DECISIONS[decision], "note": note,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    case["status"] = DECISIONS[decision]
    Path(case_path).write_text(json.dumps(case, indent=2, default=str))
    return case
