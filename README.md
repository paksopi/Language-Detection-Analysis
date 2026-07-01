# Language Detection Engine Evaluation

A proof-of-concept comparing language-detection engines — **lingua**, **langdetect**,
**pycld2**, **fasttext**, and **langid** — on short-text EN / MS / ID (English / Malay /
Indonesian) classification, and combining them into a weighted voting ensemble.

## Layout

All code lives under `src/`, grouped by purpose:

| Folder | Contents |
|---|---|
| `src/voting/` | Core voting-ensemble logic and evaluation scripts (accuracy, kappa, AUC, reproducibility, two-stage voting) |
| `src/voting/scenario2/` | A second voting scenario variant (`voting_s2*.py`), built on `src/voting/core.py` |
| `src/benchmark/` | Single-engine benchmark script producing confusion matrices and ROC curves |
| `src/examples/` | Earlier standalone demo/comparison scripts (lingua vs. langdetect, iterative versions) — kept for history, not part of the main pipeline |
| `models/` | Pretrained language-ID model binaries (`lid.176.ftz`, `openlid-v3.bin`) |
| `data/` | Test-case text files used as evaluation input |
| `results/` | Generated artifacts: `logs/`, `logs_scenario2/`, `calibration/`, `confusion_matrix/`, `roc_curve/` — all reproducible by re-running the scripts above |
| `reports/` | Written analysis of results — see [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) for the current canonical report. Superseded drafts/earlier runs live in `reports/archive/`. |
| `docs/reference/` | Background reference material from a separate, unrelated project — **not** documentation of this repo. See [`docs/reference/README.md`](docs/reference/README.md). |
| `tests/` | Unit tests (pytest) for the pure, model-free functions in `src/voting/core.py` |

## Setup

Requires **Python 3.12** — `pycld2`, `langid`, and `statsmodels` don't yet have prebuilt
wheels for newer Python versions. `requirements.txt` installs `fasttext-wheel` rather than
the official `fasttext` package, since `fasttext` has no prebuilt Windows wheel and fails
to compile from source on newer MSVC/pybind11 toolchains — `fasttext-wheel` is a drop-in,
prebuilt-wheel fork; the import name is still `import fasttext`.

```bash
python3.12 -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e .             # registers voting/benchmark as importable packages
```

## Usage

Run the main benchmark:

```bash
python src/benchmark/benchmark.py [path/to/test_case.txt]
```

Run the voting ensemble evaluation:

```bash
python src/voting/voting.py
```

Scripts write logs/plots into `results/`, keyed by test case, and default to
`data/test_case_7_enmyid.txt` if no input file is given.

## Tests

```bash
python -m pytest tests/
```
