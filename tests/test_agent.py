"""Investigation agent: ring reconstruction, STR rendering, grounding check, officer review."""

import json

import numpy as np
import pytest

from argus.agent.investigate import Investigator, reconstruction_quality
from argus.agent.llm import make_tools
from argus.agent.report import grounding_check, render_str
from argus.agent.review import record_review
from argus.data.ibm import load_accounts, load_transactions
from argus.twin.attacks import mutate
from argus.twin.protocol import Experiment
from argus.twin.rings import inject_rings
from argus.twin.world import make_world


@pytest.fixture
def twin(raw_dir):
    world = make_world(load_transactions(raw_dir, "T"), load_accounts(raw_dir, "T"), n_banks=5, end="2022-09-20")
    rings, acc = inject_rings(world, 8, seed=4)
    tx, accounts, split = Experiment(world, rings, acc, seed=1).version()
    scores = tx["is_laundering"].to_numpy(dtype=float)       # an oracle detector
    return tx, accounts, scores


def test_reconstructs_ring_with_oracle_scores(twin):
    tx, accounts, scores = twin
    inv = Investigator(tx, accounts, scores, threshold=0.5, window_hours=12)
    ring = tx[tx["ring_id"] == 0]
    case = inv.build_case(int(ring["tx_id"].iloc[0]))
    truth = set(ring["src"]) | set(ring["dst"])
    q = reconstruction_quality(case, truth, ring.sort_values("ts")["src"].iloc[0])
    assert q["recall"] == 1.0 and q["entry_correct"]
    assert case["status"] == "pending_review" and case["review"] is None
    assert all(f["role"] != "entry" for f in case["freeze_recommendation"])  # never freeze the victim
    assert case["money_trail"][0] in case["entry_accounts"]


def test_str_and_grounding(twin):
    tx, accounts, scores = twin
    case = Investigator(tx, accounts, scores, 0.5).build_case(int(tx.loc[tx["ring_id"] == 1, "tx_id"].iloc[0]))
    md = render_str(case)
    assert "requires officer approval" in md and case["case_id"] in md
    amount = case["indicators"]["amount_entered"]
    assert grounding_check(f"The ring moved {amount:,.2f} across banks.", case) == []
    assert grounding_check("The ring moved 123,456,789.00 in total.", case) == ["123,456,789.00"]


def test_llm_tools_expose_case_facts_only(twin):
    tx, accounts, scores = twin
    case = Investigator(tx, accounts, scores, 0.5).build_case(int(tx.loc[tx["ring_id"] == 2, "tx_id"].iloc[0]))
    overview, timeline, profile, freeze = make_tools(case)
    assert overview()["case_id"] == case["case_id"]
    assert len(timeline()) == len(case["timeline"])
    assert "error" in profile("not_an_account")
    assert freeze() == case["freeze_recommendation"]


def test_officer_review(twin, tmp_path):
    tx, accounts, scores = twin
    case = Investigator(tx, accounts, scores, 0.5).build_case(int(tx.loc[tx["ring_id"] == 3, "tx_id"].iloc[0]))
    path = tmp_path / "case.json"
    path.write_text(json.dumps(case, default=str))
    reviewed = record_review(path, "Officer A", "approve", "freeze confirmed")
    assert reviewed["status"] == "approved" and reviewed["review"]["officer"] == "Officer A"
    with pytest.raises(ValueError):
        record_review(path, "Officer B", "reject")   # one decision per case


def test_playbook_must_be_a_list(raw_dir):
    world = make_world(load_transactions(raw_dir, "T"), load_accounts(raw_dir, "T"), n_banks=5, end="2022-09-20")
    rings, acc = inject_rings(world, 4, seed=0)
    with pytest.raises(TypeError):
        mutate(world, rings, acc, "split", seed=0)
    with pytest.raises(ValueError):
        mutate(world, rings, acc, ["teleport"], seed=0)


class FakeGroq:
    """Scripted stand-in for groq.Groq: replays a list of assistant replies."""

    def __init__(self, replies):
        from types import SimpleNamespace as NS
        self.replies, self.seen = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, model, messages, tools, tool_choice, temperature):
        from types import SimpleNamespace as NS
        self.seen.append(messages[-1])
        reply = self.replies.pop(0)
        if isinstance(reply, tuple):          # (tool name, json args) -> tool call
            call = NS(id=f"c{len(self.seen)}", function=NS(name=reply[0], arguments=reply[1]))
            msg = NS(content=None, tool_calls=[call])
        else:
            msg = NS(content=reply, tool_calls=None)
        return NS(choices=[NS(message=msg)])


def _case(twin, ring_id=0):
    tx, accounts, scores = twin
    return Investigator(tx, accounts, scores, 0.5).build_case(int(tx.loc[tx["ring_id"] == ring_id, "tx_id"].iloc[0]))


def test_groq_tool_loop_accepts_grounded_answer(twin):
    from argus.agent.llm import run_agent
    case = _case(twin)
    amt = case["indicators"]["amount_entered"]
    acct = case["accounts"][0]["account"]
    client = FakeGroq([("get_case_overview", "{}"), ("profile_account", json.dumps({"account_id": acct})),
                       f"Funds of {amt:,.2f} entered via {acct}."])
    out = run_agent(client, "fake-model", case)
    assert out["grounded"] and out["tool_calls"] == ["get_case_overview", "profile_account"]
    assert client.seen[1]["role"] == "tool" and case["case_id"] in client.seen[1]["content"]


def test_groq_tool_loop_repairs_then_rejects_hallucination(twin):
    from argus.agent.llm import run_agent
    case = _case(twin)
    amt = case["indicators"]["amount_entered"]
    fixed = run_agent(FakeGroq(["It moved 987,654.32 dollars.", f"It moved {amt:,.2f}."]), "m", case)
    assert fixed["grounded"] and fixed["repaired"]
    bad = run_agent(FakeGroq(["It moved 987,654.32.", "Still 987,654.32."]), "m", case)
    assert not bad["grounded"] and bad["unsupported_numbers"] == ["987,654.32"]


def test_tool_schemas_match_functions(twin):
    from argus.agent.llm import tool_schemas
    schemas = {s["function"]["name"]: s["function"] for s in tool_schemas(make_tools(_case(twin)))}
    assert schemas["profile_account"]["parameters"]["required"] == ["account_id"]
    assert schemas["get_timeline"]["parameters"]["required"] == []


def test_grounding_checks_decimals_and_percentages(twin):
    case = _case(twin)
    share = case["indicators"]["cross_bank_share"]
    assert grounding_check(f"{share * 100:.0f}% of transfers crossed banks.", case) == []
    assert grounding_check("A score above 0.37 on each leg.", case) == ["0.37"] or 0.37 in [
        e["score"] for e in case["timeline"]]
    assert grounding_check(f"Moved via {case['indicators']['banks']} banks in 2022.", case) == []  # year is fine
    assert grounding_check("It moved 1,234.55 across banks.", case) == ["1,234.55"]  # amount tail is not a count


def test_clean_strips_quotes_and_bold():
    from argus.agent.llm import clean
    assert clean('Money entered via **"220_800117FF0"** as "19,773.44 USD".') == \
        "Money entered via 220_800117FF0 as 19,773.44 USD."


def test_count_claims_must_match_indicators(twin):
    from argus.agent.report import count_claims_check
    case = _case(twin)
    ind = case["indicators"]
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    assert count_claims_check(f"{ind['banks']} banks and {words[ind['dormant_accounts']]} dormant accounts", case) == []
    wrong = ind["dormant_accounts"] + 3
    assert count_claims_check(f"{words[wrong]} dormant accounts", case) == [f"{words[wrong]} dormant accounts"]
    assert count_claims_check(f"across {ind['banks'] + 4} banks", case) == [f"{ind['banks'] + 4} banks"]


def test_rate_limit_is_retried():
    from types import SimpleNamespace as NS
    from argus.agent.llm import _create

    class RateLimited(Exception):
        status_code = 429

    calls, waits = [], []

    def create(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise RateLimited("Rate limit reached. Please try again in 255ms.")
        return "ok"

    client = NS(chat=NS(completions=NS(create=create)))
    assert _create(client, sleep=waits.append, model="m") == "ok"
    assert len(calls) == 3 and len(waits) == 2 and all(w >= 2.0 for w in waits)
