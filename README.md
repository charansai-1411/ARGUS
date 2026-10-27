# ARGUS — Graph Intelligence for Anti-Money-Laundering

> One bank sees a hop. ARGUS sees the network.

Major Project (B.E. AI & DS, CBIT, AY 2026–27, Batch 120) — Y. Charan Sai, B. Ram Charan, C. Ganesh Kumar.
Guide: Ms. V. Vasuki Rohini Devi.

ARGUS detects cross-bank mule-account rings with a heterogeneous temporal GNN, filters false
positives with LightGBM, drafts STRs with a GenAI agent for a human officer, and — the core
contribution — stress-tests the detector inside a multi-bank digital twin against adaptive,
camouflaged launderers, measuring degradation **D** and recovery **R**.

## Layout

```
configs/            experiment configs (YAML)
data/raw/           IBM AML files go here (not committed)
data/processed/     parquet caches (not committed)
src/argus/
  data/             IBM AML loaders, pattern (ring) parser, chronological split
  features/         past-only tabular features
  models/           rule baseline, LightGBM            (Phase 1)
  graph/            graph construction + GNNs           (Phase 2)
  twin/             ring injector, mutation operators   (Phase 3)
  agent/            trail-tracing tools + STR drafting  (Phase 4)
  eval/             PR-AUC, F1, Ring Recall, D / R
scripts/            runnable entry points
api/  frontend/     FastAPI backend, React + Cytoscape demo (Phase 5)
docs/               decks, briefs, roadmap
tests/              smoke tests on synthetic data
```

## Setup (Windows, everything on D:)

```powershell
cd D:\ARGUS
python -m venv .venv
.venv\Scripts\activate
$env:PIP_CACHE_DIR = "D:\ARGUS\.cache\pip"
pip install -r requirements.txt
pip install -e .
pytest
```

GNN training (Phase 2) needs `requirements-gnn.txt`; run it on Kaggle/Colab GPU.

## Data

Download from Kaggle: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
and put these in `data/raw/`:

- `HI-Small_Trans.csv` — transactions (~5M rows)
- `HI-Small_Patterns.txt` — injected laundering attempts (used for Ring Recall)

## Run Phase 1 baselines

```powershell
python scripts/run_baselines.py --limit 500000   # quick check
python scripts/run_baselines.py                  # full HI-Small
```

Results go to `outputs/baselines_HI-Small.json`.

## Phase 2 — heterogeneous GNN

Graph (`src/argus/graph/build.py`): **account**, **entity** (owner, from `HI-Small_accounts.csv`)
and **bank** nodes; transfers as feature-rich edges in both directions, plus owns / held-at links.
Task: classify each transfer. Model (`src/argus/graph/model.py`): HeteroConv with GINE / GATv2 /
SAGE on transfers, SAGE on ownership and bank links, scoring `[h_src, h_dst, edge_features]`.

No future leakage: the model trains on the train-period graph, is selected on the train+val graph
and is tested on the full graph.

Train on a GPU with [`notebooks/train_gnn_kaggle.ipynb`](notebooks/train_gnn_kaggle.ipynb)
(Kaggle or Colab). Local quick check (CPU, needs `requirements-gnn.txt` + `pyg_lib`):

```powershell
python scripts/train_gnn.py --limit 300000 --epochs 1
```

| Model (HI-Small test) | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Rules | 0.002 | 0.006 | 0.003 | 0.038 |
| LightGBM | 0.296 | 0.367 | 0.505 | 0.289 |
| GNN (GINE) | _run on GPU_ | | | |

## Evaluation rules

- **Chronological split only** (60/20/20 by time) — random splits leak future hops of a ring.
- **No accuracy.** Report PR-AUC, minority-class F1 / precision / recall, Ring Recall.
- Decision threshold is chosen on validation, applied unchanged to test.
- Robustness: `D = F1_baseline − F1_attack`, `R = F1_retrained − F1_attack`.

## Roadmap

| Phase | Weeks | Deliverable |
|---|---|---|
| 0 Lock scope | 1 | Docs fixed, metrics spec, repo |
| 1 Data + baselines | 2–3 | Rules + LightGBM numbers on HI-Small |
| 2 GNN detector | 4–6 | GCN / SAGE / GAT (+temporal), hybrid with LightGBM, ring-level output |
| 3 Digital twin + attacks | 7–9 | Ring injector, mutations, multi-round D/R curves |
| 4 Agent + STR | 10–11 | Gemini function-calling agent, STR draft, approve step |
| 5 Demo | 12–13 | FastAPI + React/Cytoscape, 4-beat demo |
| 6 Write-up | 14–16 | Gap matrix, results, report, viva |
