# FastHalluCheck

**Rapid Evaluation of Pretrained Hallucination Detectors on QA Benchmarks**

A systematic evaluation of hallucination detection methods across two benchmarks — HaluEval QA (general domain) and PHANTOM (financial long-context QA) — with threshold optimization, supervised fine-tuning, ensemble methods, and a deployable 3-tier decision policy.

## Team

- **Vaishnav Ajai** — Dartmouth College, Thayer School of Engineering
- **Kritesh Singh** — Dartmouth College, Thayer School of Engineering
- **Prajwal Prasad** — Dartmouth College, Thayer School of Engineering

## Research Question

How effective are pretrained hallucination detection models at identifying unsupported answers in question-answering settings, and how can detection be improved through threshold optimization, supervised fine-tuning, and ensemble methods?

## Results Summary

### In-Domain (HaluEval QA, 1,000 evaluation pairs)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| HHEM zero-shot (t=0.50) | 0.7320 | 0.8816 | 0.5360 | 0.6667 | 0.7169 |
| HHEM CV-optimized (t=0.57) | 0.7340 | 0.8462 | 0.5720 | 0.6826 | 0.7169 |
| Fine-tuned DeBERTa-v3-small | 0.9650 | 0.9895 | 0.9400 | **0.9641** | 0.9764 |
| Ensemble (avg) | 0.9820 | 0.9918 | 0.9720 | **0.9818** | 0.9849 |

### Cross-Domain (PHANTOM Financial QA, 500 pairs)

| Model | HaluEval F1 | PHANTOM F1 | F1 Drop |
|-------|-------------|------------|---------|
| HHEM optimized | 0.6826 | 0.6790 | -0.004 |
| Fine-tuned DeBERTa-v3-small | 0.9641 | 0.5863 | -0.378 |

**Key finding:** Task-specific training matters more than model scale. Cross-domain evaluation reveals significant degradation, confirming domain adaptation is essential.

---

## Architecture

The project follows a layered architecture with clear separation of concerns:

```
hallucination-project/
├── app.py                          # Streamlit dashboard entry point
├── run_all.py                      # Pipeline runner (routes to CLI)
├── requirements.txt                # Python dependencies
│
├── src/                            # New layered architecture
│   ├── data/                       # DATA LAYER
│   │   ├── loaders.py              # Dataset loaders (HaluEval, PHANTOM)
│   │   ├── normalization.py        # Normalize raw data to standard format
│   │   └── store.py                # Unified CSV/JSON read/write
│   │
│   ├── services/                   # BUSINESS LOGIC LAYER
│   │   ├── metrics.py              # Classification metrics (single source of truth)
│   │   ├── inference.py            # HHEM + DeBERTa model inference
│   │   ├── threshold.py            # Threshold sweep + CV optimization
│   │   ├── analysis.py             # Error analysis + hallucination typing
│   │   ├── ensemble.py             # Ensemble prediction methods
│   │   └── policy.py               # 3-tier ANSWER/CAVEAT/ABSTAIN policy
│   │
│   ├── display/                    # DISPLAY LAYER (static output)
│   │   ├── figures.py              # All matplotlib/seaborn plot functions
│   │   ├── tables.py               # Markdown table formatters
│   │   └── report_generator.py     # Assemble final report from metrics
│   │
│   ├── dashboard/                  # DISPLAY LAYER (interactive)
│   │   ├── state.py                # Centralized Streamlit session state
│   │   ├── components.py           # Reusable UI components (cards, charts, tables)
│   │   ├── config.py               # Colors, labels, file paths
│   │   └── pages/
│   │       ├── overview.py         # Project summary + key metrics
│   │       ├── explorer.py         # Browse/filter/search predictions
│   │       ├── threshold_tuner.py  # Interactive threshold slider
│   │       ├── error_browser.py    # FP/FN browser with detail view
│   │       ├── model_comparison.py # Side-by-side model evaluation
│   │       └── cross_domain.py     # HaluEval vs PHANTOM analysis
│   │
│   └── cli/                        # CLI LAYER
│       └── commands.py             # Unified CLI: run, evaluate, analyze, report, compare, dashboard
│
├── legacy/                         # Original flat scripts (Phases 1-2)
│   ├── README.md                   # Legacy documentation
│   ├── src/                        # 17 original pipeline scripts
│   ├── run_all.py                  # Original pipeline runner
│   └── setup.py                    # Original setup script
│
├── data/                           # Normalized datasets (generated)
├── results/                        # Predictions + metrics (generated)
├── figures/                        # Static visualizations (generated)
├── models/                         # Fine-tuned model checkpoints
└── report/                         # Generated NeurIPS-style report
```

### Layer Responsibilities

| Layer | Directory | Responsibility | Depends On |
|-------|-----------|---------------|------------|
| **Data** | `src/data/` | Load datasets, read/write CSV/JSON | File system |
| **Services** | `src/services/` | All computation — metrics, inference, analysis | Data layer |
| **Display** | `src/display/` | Static figures + markdown report | Services layer |
| **Dashboard** | `src/dashboard/` | Interactive Streamlit UI | Services + Display |
| **CLI** | `src/cli/` | Command-line interface | Services + Display |

No layer reaches down more than one level. Dashboard and CLI both call Services; they never touch Data directly.

---

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended: RTX 4070+ with 8GB VRAM)
- ~2GB disk space for model weights and datasets

### Setup

```bash
# Clone and install
git clone <repo-url>
cd hallucination-project
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline (uses legacy scripts for data generation + inference)
python run_all.py
```

### Launch the Dashboard

```bash
streamlit run app.py
```

This opens an interactive web dashboard with 6 pages:
- **Overview** — Key metrics, confusion matrix, score distribution
- **Data Explorer** — Browse predictions with filters and search
- **Threshold Tuner** — Adjust threshold, see metrics update live
- **Error Browser** — Explore false positives and negatives
- **Model Comparison** — Side-by-side interactive Plotly charts
- **Cross-Domain Analysis** — HaluEval vs PHANTOM comparison

### CLI Commands

```bash
python -m src.cli.commands run          # Run full pipeline
python -m src.cli.commands evaluate     # Compute metrics from predictions
python -m src.cli.commands analyze      # Run error analysis
python -m src.cli.commands report       # Generate markdown report
python -m src.cli.commands compare      # Compare all models (table output)
python -m src.cli.commands dashboard    # Launch Streamlit dashboard
```

---

## Software Design Patterns

### 1. Separation of Concerns
Each layer has a single responsibility. Business logic never touches the UI; the UI never reads files directly.

### 2. State Management
`src/dashboard/state.py` provides centralized session state with lazy loading, `@st.cache_data` for expensive operations, and derived state invalidation when inputs change (e.g., threshold slider).

### 3. Data Flow
```
User Action → SessionState → Services Layer → Display Layer → Streamlit UI
```

### 4. APIs & Integrations
Services expose a clean functional API:
```python
from src.services.metrics import compute_metrics, compute_metrics_at_threshold
from src.services.inference import predict_hhem, predict_finetuned
from src.services.analysis import get_errors, categorize_hallucination
```

### 5. Error Handling
Graceful degradation throughout — missing files show informative messages, CUDA OOM triggers batch size reduction, model loading failures fall through to alternatives.

### 6. UX Patterns
Consistent color palette (blue=faithful, red=hallucinated), interactive Plotly charts with hover tooltips, `st.metric` cards with delta indicators, paginated data tables, and keyboard-navigable controls.

---

## Models Used

| Model | Params | Role | Source |
|-------|--------|------|--------|
| Vectara HHEM v2 | 86M | Zero-shot hallucination detector | `vectara/hallucination_evaluation_model` |
| DeBERTa-v3-small | 44M | Fine-tuned binary classifier | `microsoft/deberta-v3-small` |

## Datasets

| Dataset | Source | Domain | N pairs | Avg Context |
|---------|--------|--------|---------|-------------|
| HaluEval QA | `pminervini/HaluEval` | General (Wikipedia) | 1,000 | 334 chars |
| PHANTOM 10K | `seyled/Phantom_Hallucination_Detection` | Financial (SEC) | 500 | 64,204 chars |

## Hardware

- **GPU:** NVIDIA RTX 4070 Laptop (8GB VRAM)
- **RAM:** 16GB
- **OS:** Windows 11
- **Python:** 3.13

## License

This project is for academic use (Dartmouth College coursework).
