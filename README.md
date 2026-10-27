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

| Model (HI-Small test) | PR-AUC | F1 | Precision | Recall | Ring recall @1 | Ring recall @0.5 |
|---|---|---|---|---|---|---|
| Rules | 0.002 | 0.006 | 0.003 | 0.038 | 0.005 | — |
| LightGBM | 0.296 | 0.367 | 0.505 | 0.289 | 0.042 | 0.370 |
| **GNN (GINE), time-respecting** | **0.726** | **0.729** | **0.861** | **0.632** | **0.604** | **0.901** |
| GNN (GINE), with look-ahead ⚠️ (superseded) | 0.734 | 0.730 | 0.826 | 0.654 | 0.651 | 0.906 |

**Reported result:** the time-respecting GINE run (Kaggle T4, 10 epochs, best val PR-AUC 0.612
at epoch 9, ~110 min). Each scored transfer only sees transfers at or before its own
timestamp (`gnn.temporal: true`, enforced by `tests/test_graph.py`).

The earlier look-ahead run let a transfer's neighbourhood include *later* transfers (no labels
leaked). Removing it cost only ~0.01 PR-AUC, so the GNN's gain over LightGBM is not a
look-ahead artefact.

## Phase 3 — digital twin and adversarial robustness

`src/argus/twin/`: a multi-bank world built from real IBM background traffic (the 8 busiest
banks, Sep 1–10, IBM's own laundering removed) with **injected mule rings whose every transfer is
known** — chain, fan-out, scatter-gather and cycle typologies, mules recruited from dormant real
accounts or freshly opened, often sharing a herder entity.

The attacker's playbook (`attacks.py`):

| Operator | Tactic |
|---|---|
| `separate_owners` | fresh mules get distinct owners (hides the shared herder) |
| `add_hops` | every transfer routed through an extra mule at another bank |
| `split` | smurfing: transfers split into 2–4 smaller ones |
| `delay` | ring stretched from minutes to hours (beats velocity windows) |
| `mimic` | payment formats/currencies copied from normal traffic (feature camouflage) |
| `camouflage` | mules transact normally with legitimate accounts (relation camouflage) |

Protocol (`protocol.py`): train on the clean twin → attack only the test-period rings →
**D = F1_before − F1_attack** → retrain on train/val rings mutated the same way →
**R = F1_retrained − F1_attack**. The adaptive attacker probes the current detector on
validation rings each round and adds the operator that lowers its recall most.

```powershell
python scripts/run_twin.py --detector lightgbm        # ~1.5 min on CPU
python scripts/run_twin.py --detector gnn --epochs 6  # CPU works on the twin; GPU faster
```

**LightGBM on the twin** (400 rings; clean test F1 0.974):

| Attack alone | F1 attacked | F1 retrained | D | R |
|---|---|---|---|---|
| camouflage | 0.493 | 0.823 | **+0.481** | +0.330 |
| split | 0.535 | 0.976 | +0.439 | +0.441 |
| mimic | 0.667 | 0.960 | +0.308 | +0.294 |
| delay | 0.963 | 0.965 | +0.011 | +0.002 |
| add_hops | 0.979 | 0.998 | −0.005 | +0.019 |
| separate_owners | 0.974 | 0.974 | 0.000 | 0.000 |

Adaptive attacker, 4 rounds: split → mimic → delay → camouflage. F1 after each retrain:
0.976 → 0.964 → 0.937 → **0.834** — each round the detector heals less (camouflage is only
partly recoverable by retraining a tabular model).

**GNN (GINE, time-respecting) vs LightGBM on the same twin** (GNN: 6 epochs per fit, CPU, 34 min):

| Attack alone | LightGBM D | GNN D | LightGBM F1 retrained | GNN F1 retrained |
|---|---|---|---|---|
| clean F1 | 0.974 | 0.971 | — | — |
| camouflage | +0.481 | **+0.436** | 0.823 | **0.848** |
| split | +0.439 | **+0.038** | 0.976 | 0.986 |
| mimic | +0.308 | **+0.102** | 0.960 | 0.965 |
| delay | +0.011 | 0.000 | 0.965 | 0.967 |
| add_hops | −0.005 | −0.019 | 0.998 | 1.000 |
| separate_owners | 0.000 | 0.000 | 0.974 | 0.969 |

Adaptive attacker vs GNN, 4 rounds: mimic → split → camouflage → delay. F1 after each retrain:
0.965 → 0.980 → 0.897 → **0.878** (LightGBM ended at 0.834). Clean ring recall @1: GNN 0.883 vs
LightGBM 0.825.

**Findings.** (1) The graph view largely neutralises structuring: smurfing costs LightGBM 0.44 F1
but the GNN only 0.04, and feature mimicry costs a third as much. (2) Relation camouflage — mules
padding their neighbourhoods with legitimate counterparties — is the attack neither model shrugs
off (D ≈ 0.44–0.48) and the one retraining only partly repairs; it is the open problem ARGUS
exposes, and the motivation for camouflage-aware aggregation (CARE-GNN-style neighbour filtering)
as the next defence. (3) Under a 4-round adaptive attacker the GNN keeps a 4.4-point F1 lead.
Single seed so far — repeat with several seeds before quoting error bars.

## Phase 4 — investigation agent and STR drafting

`src/argus/agent/`: from one flagged transfer, the investigator rebuilds the ring (accounts linked
by suspicious transfers in a ±48 h window), finds the entry (victim) and exits (cash-out), traces the
money trail, profiles every account (owner, sibling accounts, dormancy, amounts) and computes red
flags (cross-bank hops, pass-through time, fan-in/out, cycles, shared owners). **All facts are
computed deterministically**; the STR is rendered from them (`report.py`).

The optional LLM agent (`llm.py`) runs on **Groq** (free tier, `openai/gpt-oss-120b` by
default). It writes only the summary paragraph, by calling the case tools in a function-calling
loop. Every number it writes must match a case fact (`grounding_check`); it gets one repair
attempt, then the draft is rejected and the template summary is used.

Setup: `pip install -r requirements-agent.txt`, get a free key at <https://console.groq.com/keys>,
then either set `GROQ_API_KEY` or create `D:\ARGUS\.env` (gitignored) containing
`GROQ_API_KEY=gsk_...`. Never commit the key. ARGUS never freezes accounts: each case starts `pending_review` and an officer records
approve / reject / escalate (`review.py`).

```powershell
python scripts/investigate.py cases                     # case files + STRs in outputs/cases/
python scripts/investigate.py cases --llm               # Groq LLM summary
python scripts/investigate.py cases --attack camouflage # agent under attack
python scripts/investigate.py review outputs/cases/<id>.json --officer "Name" --decision approve
```

**Ring reconstruction vs twin ground truth** (103 test rings, LightGBM scores):

| Twin | Rings detected | Account precision | Account recall | Jaccard | Victim found |
|---|---|---|---|---|---|
| clean | 103/103 | 0.908 | 0.989 | 0.899 | 95.1% |
| split attack | 103/103 | 0.891 | 0.992 | 0.885 | 96.1% |
| camouflage attack | 103/103 | 0.743 | 0.894 | **0.678** | 72.8% |

Camouflage again is the hard case: cover transactions pull legitimate accounts into the
reconstructed ring (precision drops), which is why freeze lists stay recommendations for an officer.

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
| 4 Agent + STR | 10–11 | Groq function-calling agent, grounded STR draft, approve step |
| 5 Demo | 12–13 | FastAPI + React/Cytoscape, 4-beat demo |
| 6 Write-up | 14–16 | Gap matrix, results, report, viva |
