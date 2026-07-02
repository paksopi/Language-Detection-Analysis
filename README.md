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
| `reports/` | Written analysis of results — see [`reports/language_detection_ensemble_evaluation.md`](reports/language_detection_ensemble_evaluation.md) for the current canonical report. Superseded drafts/earlier runs live in `reports/archive/`. |
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
python src/voting/voting_main.py
```

Scripts write logs/plots into `results/`, keyed by test case, and default to
`data/test_case_7_enmyid.txt` if no input file is given.

## Tests

```bash
python -m pytest tests/
```

## References

### Language detection libraries evaluated

- [lingua-py](https://github.com/pemistahl/lingua-py) — natural language detection library, tuned for short and mixed-language text
- [langdetect](https://github.com/Mimino666/langdetect) — Python port of Google's language-detection library
- [pycld2](https://github.com/aboSamoor/pycld2) — Python bindings for Google's Compact Language Detector 2 (CLD2)
- [fastText language identification](https://github.com/facebookresearch/fastText/blob/main/docs/language-identification.md) — pretrained fastText models for language ID (176 languages)
- [langid.py](https://github.com/saffsd/langid.py) — stand-alone language identification system
- [OpenLID](https://github.com/laurieburchell/open-lid-dataset) — fastText-based language ID model covering 201 languages, from Burchell et al., ["An Open Dataset and Model for Language Identification"](https://arxiv.org/abs/2305.13820) (2023); model weights on [Hugging Face](https://huggingface.co/laurievb/OpenLID)

### Evaluation methods and metrics

- [Confusion matrix (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [ROC curve (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) and [ROC AUC score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- [Cohen's kappa score (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) — inter-rater agreement metric
- [Voting classifiers / ensembles (scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) — reference for the weighted-voting ensemble approach used here
- [McNemar's test (mlxtend)](https://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/) — paired significance test used to check whether one model/ensemble is genuinely more accurate than another on the same test set
- [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) — confidence interval for per-language accuracy, more reliable than a normal approximation at small/imbalanced sample sizes
- [Expected Calibration Error (ECE) — "On Calibration of Modern Neural Networks"](https://arxiv.org/abs/1706.04599) — metric for whether a model's confidence scores match its real-world accuracy
- [FLORES-200](https://huggingface.co/datasets/facebook/flores) — multilingual benchmark dataset whose language/script labels were used to map model outputs to ISO codes
- [BCP 47 / IETF language tags (RFC 5646)](https://www.rfc-editor.org/info/bcp47) — standard used for the language codes (e.g. `en`, `ms`, `id`) that all models are scored against
