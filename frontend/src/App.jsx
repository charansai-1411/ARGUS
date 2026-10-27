import { useCallback, useEffect, useState } from "react";
import Graph from "./components/Graph.jsx";
import { AdaptPanel, AlertPanel, InvestigatePanel, NetworkPanel, ResultsPanel } from "./components/Panels.jsx";
import { money, num } from "./format.js";

const BEATS = ["The alert fires", "The network reveals", "The agent investigates", "The criminal adapts", "Results"];

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json();
}

export default function App() {
  const [demo, setDemo] = useState(null);
  const [err, setErr] = useState("");
  const [beat, setBeat] = useState(0);
  const [model, setModel] = useState("naive");
  const [selected, setSelected] = useState(null);

  const load = useCallback(() => api("/api/demo").then(setDemo).catch((e) => setErr(e.message)), []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.closest("input, textarea")) return;
      if (e.key === "ArrowRight") setBeat((b) => Math.min(BEATS.length - 1, b + 1));
      if (e.key === "ArrowLeft") setBeat((b) => Math.max(0, b - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => { setSelected(null); if (beat === 3) setModel("naive"); }, [beat]);

  if (err) return <div className="error">Could not load the demo: {err}</div>;
  if (!demo) return <div className="error" style={{ color: "var(--muted)" }}>Loading ARGUS…</div>;

  const review = async (body) => setDemo(await api("/api/review", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }));
  const reset = async () => { await api("/api/review", { method: "DELETE" }); load(); };

  const adapt = beat === 3;
  const graph = adapt ? demo.adapt : demo.clean;
  const mode = ["alert", "reveal", "full", "adapt", "full"][beat];
  const threshold = model === "hardened" ? demo.meta.threshold_hardened : demo.meta.threshold_naive;

  return (
    <div className="app">
      <header className="header">
        <div className="brand"><h1>ARGUS</h1><span>One bank sees a hop. Argus sees the network.</span></div>
        <nav className="steps">
          {BEATS.map((b, i) => (
            <button key={b} className={`step ${i === beat ? "active" : i < beat ? "done" : ""}`} onClick={() => setBeat(i)}>
              <b>{i < 4 ? i + 1 : "★"}</b>{b}
            </button>
          ))}
        </nav>
        <div className="nav">
          <button disabled={beat === 0} onClick={() => setBeat(beat - 1)}>←</button>
          <button disabled={beat === BEATS.length - 1} onClick={() => setBeat(beat + 1)}>→</button>
        </div>
      </header>
      <main className="main">
        <section className="stage">
          <Graph graph={graph} mode={mode} model={adapt ? model : "naive"} threshold={threshold}
                 alertTx={demo.alert.tx_id} onSelect={setSelected} />
          {selected && (
            <div className="banner"><span /><div className="card" style={{ maxWidth: 300 }}>
              <div className="mono">{selected.id}</div>
              <div className="muted">Bank {selected.bank} · {selected.role === "entry" ? "victim" : selected.role}</div>
              <div>Max score {num(selected[`score_${adapt ? model : "naive"}`], 2)}</div>
            </div></div>)}
          <div className="legend">
            <span><i style={{ background: "var(--victim)" }} />victim</span>
            <span><i style={{ background: "var(--mule)" }} />mule</span>
            <span><i style={{ background: "var(--exit)" }} />cash-out</span>
            <span><i style={{ background: "var(--legit)" }} />legitimate</span>
            {adapt && <span>dotted = camouflage · green = flagged · dashed = missed</span>}
            <span className="muted">← → keys to step · {demo.meta.detector.toUpperCase()} · alert {money(demo.alert.amount)}</span>
          </div>
        </section>
        <aside className="panel">
          {beat === 0 && <AlertPanel demo={demo} />}
          {beat === 1 && <NetworkPanel demo={demo} />}
          {beat === 2 && <InvestigatePanel demo={demo} onReview={review} onReset={reset} />}
          {beat === 3 && <AdaptPanel demo={demo} model={model} setModel={setModel} />}
          {beat === 4 && <ResultsPanel demo={demo} />}
        </aside>
      </main>
    </div>
  );
}
