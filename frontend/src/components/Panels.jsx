import { useEffect, useState } from "react";
import { marked } from "marked";
import { money, num, pct, shortTime } from "../format.js";

const roleOf = (demo, acct) => demo.case.accounts.find((a) => a.account === acct)?.role || "legit";

export function AlertPanel({ demo }) {
  const a = demo.alert;
  return (
    <>
      <h2>Beat 1 — The alert fires</h2>
      <p className="lede">A customer was just scammed. Their bank sees one ordinary outgoing transfer. ARGUS scores it.</p>
      <div className="card alert reveal">
        <div className="muted">Transfer #{a.tx_id} · {shortTime(a.time)} · {a.format}</div>
        <div className="big">{money(a.amount, a.currency)}</div>
        <div className="mono" style={{ marginTop: 8 }}>
          <span className="chip entry">{a.from}</span> → <span className="chip mule">{a.to}</span>
        </div>
        <div style={{ marginTop: 10 }}>Suspicion score <b>{num(a.score, 2)}</b> <span className="muted">(threshold {num(demo.meta.threshold_naive, 2)})</span></div>
      </div>
      <p className="lede" style={{ marginTop: 16 }}>
        On its own, this looks like any other payment. The question is where the money goes next — across banks that never talk to each other.
      </p>
    </>
  );
}

export function NetworkPanel({ demo }) {
  const ind = demo.case.indicators;
  return (
    <>
      <h2>Beat 2 — The network reveals itself</h2>
      <p className="lede">Each bank saw only its own hop. ARGUS sees the whole {demo.meta.typology.replace("_", "-")} ring.</p>
      <div className="kpis reveal">
        <div className="kpi"><div className="v">{ind.accounts}</div><div className="l">accounts</div></div>
        <div className="kpi"><div className="v">{ind.banks}</div><div className="l">banks</div></div>
        <div className="kpi"><div className="v">{ind.duration_minutes} min</div><div className="l">entry to cash-out</div></div>
        <div className="kpi"><div className="v">{pct(ind.cross_bank_share)}</div><div className="l">cross-bank transfers</div></div>
        <div className="kpi"><div className="v">{money(ind.amount_entered).replace(" USD", "")}</div><div className="l">USD entered</div></div>
        <div className="kpi"><div className="v">{ind.dormant_accounts}</div><div className="l">dormant accounts woke up</div></div>
      </div>
      <h3>What each bank saw</h3>
      <p className="lede">Bank by bank, every account received money and sent it on — normal in isolation. Only the joined-up graph shows the shape.</p>
    </>
  );
}

const STEPS = ["Tracing the money trail", "Profiling every account", "Building the timeline", "Checking red flags", "Drafting the Suspicious Transaction Report"];

export function InvestigatePanel({ demo, onReview, onReset }) {
  const [shown, setShown] = useState(0);
  const [officer, setOfficer] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const c = demo.case;
  useEffect(() => {
    setShown(0);
    const ts = STEPS.map((_, i) => setTimeout(() => setShown(i + 1), 500 + i * 650));
    return () => ts.forEach(clearTimeout);
  }, [demo.case.case_id]);

  const decide = async (decision) => {
    setBusy(true); setErr("");
    try { await onReview({ officer, decision, note }); } catch (e) { setErr(String(e.message || e)); }
    setBusy(false);
  };
  const safeMd = demo.str_markdown.replace(/</g, "&lt;");

  return (
    <>
      <h2>Beat 3 — The agent investigates</h2>
      <p className="lede">Every fact below is computed from the transactions. The language model only writes the summary paragraph, and every number it writes is checked.</p>
      <ul className="step-log">
        {STEPS.map((s, i) => (
          <li key={s} className={i < shown ? "reveal" : ""} style={{ opacity: i < shown ? 1 : 0.25 }}>
            <span className={i < shown ? "tick" : "muted"}>{i < shown ? "✓" : "…"}</span> {s}
            {i === shown && <span className="typing" />}
          </li>
        ))}
      </ul>
      {shown >= 1 && (<><h3>Money trail</h3>
        <div className="trail reveal">{c.money_trail.map((a, i) => (
          <span key={a}>{i > 0 && <span className="muted"> → </span>}<span className={`chip ${roleOf(demo, a)}`}>{a}</span></span>))}
        </div></>)}
      {shown >= 4 && (<><h3>Red flags</h3><ul className="flags reveal">{demo.red_flags.map((f) => <li key={f}>{f}</li>)}</ul></>)}
      {shown >= 5 && (<>
        <h3>Draft STR {c.status !== "pending_review" && <span className={`status ${c.status}`}>{c.status}</span>}</h3>
        <div className="str reveal" dangerouslySetInnerHTML={{ __html: marked.parse(safeMd) }} />
        {c.review ? (
          <div className="card" style={{ marginTop: 12 }}>
            <b>{c.review.decision}</b> by {c.review.officer} <span className="muted">· {shortTime(c.review.at)}</span>
            {c.review.note && <div className="muted">{c.review.note}</div>}
            <div style={{ marginTop: 10 }}><button className="btn" onClick={onReset}>Reset review (re-run demo)</button></div>
          </div>
        ) : (
          <div className="review">
            <div className="muted">Freezing is a recommendation. An officer must decide.</div>
            <input placeholder="Officer name" value={officer} onChange={(e) => setOfficer(e.target.value)} />
            <textarea rows={2} placeholder="Note (optional)" value={note} onChange={(e) => setNote(e.target.value)} />
            <div className="row">
              <button className="btn primary" disabled={!officer.trim() || busy} onClick={() => decide("approve")}>Approve freeze</button>
              <button className="btn" disabled={!officer.trim() || busy} onClick={() => decide("escalate")}>Escalate</button>
              <button className="btn danger" disabled={!officer.trim() || busy} onClick={() => decide("reject")}>Reject</button>
            </div>
            {err && <div style={{ color: "var(--danger)" }}>{err}</div>}
          </div>
        )}
      </>)}
    </>
  );
}

export function AdaptPanel({ demo, model, setModel }) {
  const ad = demo.adapt;
  const r = ad[model];
  const missed = r.transfers - r.flagged;
  const verdict = r.ring_recall >= 1 ? ["catch", "RING CAUGHT"]
    : r.ring_recall >= 0.75 ? ["catch", `RING CAUGHT — ${missed} hop${missed > 1 ? "s" : ""} missed`]
    : r.ring_recall > 0 ? ["partial", "RING BROKEN UP — MOSTLY MISSED"] : ["miss", "RING MISSED"];
  const cover = ad.edges.filter((e) => e.kind === "camouflage").length;
  return (
    <>
      <h2>Beat 4 — The criminal adapts</h2>
      <p className="lede">
        The launderers learn. The same ring runs again, but now its mules make {cover} ordinary-looking payments in total to and from
        legitimate accounts (<b>camouflage</b>, dotted lines) to blend into normal traffic.
      </p>
      <div className="toggle">
        <button className={model === "naive" ? "on" : ""} onClick={() => setModel("naive")}>Naive model</button>
        <button className={model === "hardened" ? "on" : ""} onClick={() => setModel("hardened")}>Hardened model</button>
      </div>
      <div className="card reveal" key={model} style={{ marginTop: 14 }}>
        <div className={`verdict ${verdict[0]}`}>{verdict[1]}</div>
        <div style={{ marginTop: 6 }}>Flagged <b>{r.flagged}</b> of <b>{r.transfers}</b> laundering transfers ({pct(r.ring_recall)}).</div>
        <div className="muted" style={{ marginTop: 6 }}>
          {model === "naive"
            ? (r.flagged ? "Trained only on yesterday's tactics. The camouflage hides part of the trail (dashed = missed), so the ring breaks into fragments."
                         : "Trained only on yesterday's tactics. The camouflage hides the ring (dashed = missed).")
            : `Retrained on rings mutated by the attacker's playbook. Same ring, caught again (green = flagged)${missed ? "; the flagged hops still expose the mule accounts" : ""}.`}
        </div>
      </div>
      <h3>Before the attack</h3>
      <p className="lede">The naive model flagged {ad.clean_naive.flagged} of {ad.clean_naive.transfers} transfers of this ring on the clean data. One tactic change took that to {ad.naive.flagged}.</p>
      <p className="lede">This is the measurement ARGUS adds: how much accuracy an adaptive attacker removes (D) and how much retraining wins back (R).</p>
    </>
  );
}

function Bars({ rows, max = 1, color = "#4cc3ff", fmt = (v) => num(v, 2) }) {
  return (
    <div className="bars">
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span>{r.label}</span>
          <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(1, (100 * (r.value ?? 0)) / max)}%`, background: r.color || color }} /></div>
          <b>{fmt(r.value)}</b>
        </div>
      ))}
    </div>
  );
}

export function ResultsPanel({ demo }) {
  const R = demo.results;
  const hi = R.hi_small;
  const colors = { Rules: "#5b6f8a", LightGBM: "#ffb547", "GNN (GINE)": "#3ddc97" };
  const rob = R.robustness || {};
  const ops = Object.keys(rob.lightgbm?.single || {}).filter((o) => (rob.lightgbm.single[o].D > 0.05) || (rob.gnn?.single?.[o]?.D > 0.05));
  const agent = R.agent || {};
  return (
    <>
      <h2>The result</h2>
      <p className="lede">Rule-based checks vs a tabular ML model vs the ARGUS graph model, on 5 million real-format transactions (IBM HI-Small, held-out last 20% by time).</p>
      <h3>PR-AUC (higher is better)</h3>
      <Bars rows={hi.map((r) => ({ label: r.model, value: r.pr_auc, color: colors[r.model] }))} />
      <h3>F1 on laundering transfers</h3>
      <Bars rows={hi.map((r) => ({ label: r.model, value: r.f1, color: colors[r.model] }))} />
      {ops.length > 0 && (<>
        <h3>Accuracy lost to each attack — D (lower is better)</h3>
        <div className="bars">{ops.map((o) => (
          <div className="pair" key={o}>
            <Bars rows={[{ label: `${o} · LightGBM`, value: rob.lightgbm.single[o].D, color: colors.LightGBM },
                         ...(rob.gnn ? [{ label: `${o} · GNN`, value: rob.gnn.single[o].D, color: colors["GNN (GINE)"] }] : [])]} max={0.6} />
          </div>))}
        </div>
        {rob.gnn?.adaptive?.length > 0 && (
          <p className="lede" style={{ marginTop: 10 }}>
            Against an attacker who adapts {rob.gnn.adaptive.length} times, F1 after retraining ends at{" "}
            <b>{num(rob.gnn.adaptive.at(-1).retrained, 3)}</b> for the GNN vs <b>{num(rob.lightgbm.adaptive.at(-1).retrained, 3)}</b> for LightGBM.
          </p>)}
      </>)}
      {Object.keys(agent).length > 0 && (<>
        <h3>Agent ring reconstruction (Jaccard vs ground truth)</h3>
        <Bars rows={Object.entries(agent).map(([k, v]) => ({ label: k, value: v.mean_jaccard, color: "#b48cff" }))} />
      </>)}
    </>
  );
}
