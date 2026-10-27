"""Groq investigation agent: writes the STR narrative by calling ARGUS's investigation tools.

Groq's API is OpenAI-compatible; the agent runs an explicit tool loop (model asks for a tool,
we execute it on the case file, send the result back) until the model answers in text.
Its narrative is checked by `grounding_check` — any number not present in the case facts is
rejected and the model is asked once to fix it; otherwise the template summary is used.

Key: GROQ_API_KEY environment variable, or a line `GROQ_API_KEY=...` in the repo's .env
(gitignored). Free key: https://console.groq.com/keys
"""

import json
import os
import re
import time
from pathlib import Path

from argus.agent.report import grounding_check
from argus.config import ROOT

SYSTEM = """You are an anti-money-laundering investigation assistant for a bank compliance officer.
Use the provided tools to inspect the case. Then write the SUMMARY section of a Suspicious
Transaction Report: 4-7 sentences, plain professional English, describing how money entered,
moved and exited, the banks involved, timing, and the red flags.
Rules: quote only numbers and account ids returned by the tools; do not speculate about identities;
do not state that accounts are frozen — freezing is a recommendation awaiting officer approval.
Style: one plain-text paragraph, no markdown, no bold, no quotation marks around values. Write
amounts with thousands separators and two decimals (e.g. 19,773.44 USD), counts as digits
(e.g. 6 banks), and shares as percentages (e.g. 89% cross-bank). Reply with the paragraph only."""


def load_env(path: Path = ROOT / ".env") -> None:
    """Set variables from a KEY=VALUE .env file without overriding the real environment."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_tools(case: dict):
    """Tool functions bound to one case file."""
    accounts = {p["account"]: p for p in case["accounts"]}

    def get_case_overview() -> dict:
        """Ring-level facts: entry and exit accounts, money trail, indicators (banks, amounts, timing)."""
        return {k: case[k] for k in ("case_id", "entry_accounts", "exit_accounts", "money_trail", "indicators")}

    def get_timeline() -> list:
        """Every transfer in the ring in time order, with minute offset, banks, amount and format."""
        return case["timeline"]

    def profile_account(account_id: str) -> dict:
        """Profile of one ring account: bank, owner, prior activity, amounts in/out, role, score."""
        return accounts.get(account_id, {"error": f"{account_id} is not part of this case"})

    def get_freeze_recommendation() -> list:
        """Accounts ARGUS recommends the officer consider freezing, with reasons."""
        return case["freeze_recommendation"]

    return [get_case_overview, get_timeline, profile_account, get_freeze_recommendation]


def tool_schemas(tools) -> list:
    """OpenAI/Groq function-calling schemas from the tool functions' names, docstrings and args."""
    out = []
    for fn in tools:
        args = [a for a in fn.__code__.co_varnames[:fn.__code__.co_argcount]]
        out.append({"type": "function", "function": {
            "name": fn.__name__, "description": fn.__doc__,
            "parameters": {"type": "object", "properties": {a: {"type": "string"} for a in args},
                           "required": args}}})
    return out


def _create(client, retries: int = 4, sleep=time.sleep, **kw):
    """chat.completions.create with backoff on rate limits (Groq free tier: tokens per minute)."""
    for attempt in range(retries + 1):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or attempt == retries:
                raise
            hint = re.search(r"try again in ([\d.]+)(ms|s)", str(e))
            wait = (float(hint.group(1)) / (1000 if hint.group(2) == "ms" else 1)) if hint else 0
            sleep(min(30.0, max(wait + 0.5, 2.0 * (attempt + 1))))


def clean(text: str) -> str:
    """Strip markdown emphasis and quotation marks the model sometimes wraps around values."""
    text = re.sub(r"\*\*|__|[\"“”]", "", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def run_agent(client, model: str, case: dict, max_steps: int = 10, temperature: float = 0.2) -> dict:
    """Tool loop. `client` is a groq.Groq (or any OpenAI-compatible client)."""
    tools = {fn.__name__: fn for fn in make_tools(case)}
    schemas = tool_schemas(tools.values())
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Investigate case {case['case_id']} and write the STR summary."}]
    calls, repaired, text = [], False, ""
    for _ in range(max_steps):
        msg = _create(client, model=model, messages=messages, tools=schemas,
                      tool_choice="auto", temperature=temperature).choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                            for c in msg.tool_calls]})
            for c in msg.tool_calls:
                fn = tools.get(c.function.name)
                try:
                    result = fn(**json.loads(c.function.arguments or "{}")) if fn else {"error": "unknown tool"}
                except Exception as e:  # bad arguments from the model
                    result = {"error": f"{type(e).__name__}: {e}"}
                calls.append(c.function.name)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(result, default=str)})
            continue
        text = clean(msg.content or "")
        bad = grounding_check(text, case)
        if bad and not repaired:      # one repair attempt with explicit feedback
            repaired = True
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"These numbers are not in the case data: {', '.join(bad)}. "
                                                     "Rewrite the summary using only figures returned by the tools."}]
            continue
        return {"narrative": text, "grounded": not bad, "unsupported_numbers": bad, "tool_calls": calls,
                "repaired": repaired, "model": model}
    return {"narrative": text, "grounded": False, "unsupported_numbers": ["no final answer"], "tool_calls": calls,
            "repaired": repaired, "model": model}


def narrative_or_none(case: dict, model: str, log=print):
    """LLM narrative if a key is set and the draft is grounded, else None (template summary is used)."""
    load_env()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        log("no GROQ_API_KEY set: using the template summary")
        return None, None
    try:
        from groq import Groq
        out = run_agent(Groq(api_key=key), model, case)
    except Exception as e:  # network / quota / model errors must not block the case file
        log(f"LLM narrative failed ({type(e).__name__}: {e}); using the template summary")
        return None, {"error": str(e)}
    if not out["grounded"]:
        log(f"LLM narrative rejected, unsupported numbers: {out['unsupported_numbers']}")
        return None, out
    log(f"LLM narrative accepted ({len(out['tool_calls'])} tool calls{', after one repair' if out['repaired'] else ''})")
    return out["narrative"], out
