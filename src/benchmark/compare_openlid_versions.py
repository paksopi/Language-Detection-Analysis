"""
Compare laurievb/OpenLID (v1, 201 languages) vs HPLT/OpenLID-v3 on speed,
memory, and accuracy. Downloads whichever model file is missing from
models/, then runs both through the same test cases.
"""

import os
import sys
import time
import re
import urllib.request
from pathlib import Path
from collections import defaultdict

import regex
import fasttext
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================================================================
# 1. SETUP & MODEL DOWNLOAD
# ==============================================================================
ROOT     = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
LOG_DIR  = ROOT / "results" / "logs" / "openlid_comparison"
CM_DIR   = ROOT / "results" / "confusion_matrix"
DS_DIR   = ROOT / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "v1": {
        "path": MODELS_DIR / "openlid-v1.bin",
        "url":  "https://huggingface.co/laurievb/OpenLID/resolve/main/model.bin",
        "name": "openlid-v1 (laurievb/OpenLID)",
    },
    "v3": {
        "path": MODELS_DIR / "openlid-v3.bin",
        "url":  "https://huggingface.co/HPLT/OpenLID-v3/resolve/main/openlid-v3.bin",
        "name": "openlid-v3 (HPLT/OpenLID-v3)",
    },
}


def download_with_progress(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        mb  = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  downloading {dest.name}: {mb:8.1f} / {total_mb:8.1f} MB ({pct:5.1f}%)", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_report)
    print()
    tmp.rename(dest)


for spec in MODEL_SPECS.values():
    if spec["path"].exists():
        print(f"Found {spec['path'].name} ({spec['path'].stat().st_size / (1024*1024):.1f} MB) — skipping download.")
    else:
        print(f"{spec['path'].name} not found. Downloading from {spec['url']} ...")
        download_with_progress(spec["url"], spec["path"])

print()

TEST_CASE_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else DS_DIR / "test_case_7_enmyid.txt"
if not TEST_CASE_FILE.exists():
    print(f"Error: could not find {TEST_CASE_FILE!r}")
    sys.exit(1)

test_cases = []
with open(TEST_CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            label, text = line.split("|", 1)
            test_cases.append({"expected": label.strip().upper(), "text": text.strip()})

if not test_cases:
    print(f"Error: {TEST_CASE_FILE!r} had no usable lines.")
    sys.exit(1)

EXPECTED_TO_ISO = {"EN": "en", "MY": "ms", "ID": "id"}
LANGUAGE_ORDER  = ["EN", "MY", "ID"]
print(f"Loaded {len(test_cases)} test cases from {TEST_CASE_FILE!r}\n")


# ==============================================================================
# 2. LABEL MAPPING & PREPROCESSING (shared by both OpenLID versions)
# ==============================================================================
OPENLID_TO_ISO = {
    "eng_Latn": "en",
    "zsm_Latn": "ms",
    "msa_Latn": "ms",
    "ind_Latn": "id",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "tam_Taml": "ta",
}

def openlid_label_to_iso(label):
    code = label.replace("__label__", "")
    return OPENLID_TO_ISO.get(code, "unknown")

_NONWORD = regex.compile(r"[^\p{Word}\p{Zs}]|\d")
_SPACES  = regex.compile(r"\s\s+")

def preprocess_openlid(text):
    text = text.strip().replace('\n', ' ').lower()
    text = _SPACES.sub(" ", text)
    text = _NONWORD.sub("", text)
    return text


# ==============================================================================
# 3. LOAD MODELS
# ==============================================================================
print("Loading both OpenLID versions...")
models = {key: fasttext.load_model(str(spec["path"])) for key, spec in MODEL_SPECS.items()}
for key, spec in MODEL_SPECS.items():
    n_labels = len(models[key].get_labels())
    print(f"  {spec['name']}: {n_labels} labels, {spec['path'].stat().st_size / (1024*1024*1024):.2f} GB on disk")
print()


# ==============================================================================
# 4. SPEED BENCHMARK
# ==============================================================================
print("=" * 100)
print("SPEED (ms per call, warmed up, averaged over repeats)")
print("=" * 100)

REPEATS = 50
texts = [case["text"] for case in test_cases]

for key, spec in MODEL_SPECS.items():
    models[key].predict(preprocess_openlid("hello world"), k=1)  # warm-up

for key, spec in MODEL_SPECS.items():
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        for text in texts:
            models[key].predict(preprocess_openlid(text), k=1)
    elapsed_ms = (time.perf_counter() - t0) / (REPEATS * len(texts)) * 1000
    print(f"  {spec['name']:<32}: {elapsed_ms:.4f} ms/call")

print()


# ==============================================================================
# 5. ACCURACY BENCHMARK
# ==============================================================================
print("=" * 100)
print("ACCURACY BY LANGUAGE (strict exact-match scoring)")
print("=" * 100)

lang_stats = defaultdict(lambda: {"total": 0, "v1": 0, "v3": 0})
y_true = []
preds  = {"v1": [], "v3": []}

for case in test_cases:
    text, expected_lbl = case["text"], case["expected"]
    if expected_lbl not in EXPECTED_TO_ISO:
        continue
    expected_iso = EXPECTED_TO_ISO[expected_lbl]
    processed = preprocess_openlid(text)

    y_true.append(expected_iso)
    for key in MODEL_SPECS:
        try:
            labels, _ = models[key].predict(processed, k=1)
            pred_iso = openlid_label_to_iso(labels[0])
        except Exception:
            pred_iso = "unknown"
        preds[key].append(pred_iso)
        lang_stats[expected_lbl][key] += int(pred_iso == expected_iso)
    lang_stats[expected_lbl]["total"] += 1

col_hdr = f"  {'LANG':<5} | {'n':>4} | {'openlid-v1':>11} | {'openlid-v3':>11}"
print(col_hdr)
print("-" * len(col_hdr))
for lang in LANGUAGE_ORDER:
    if lang not in lang_stats:
        continue
    s = lang_stats[lang]
    n = s["total"]
    print(f"  {lang:<5} | {n:>4} | {s['v1']/n*100:>10.1f}% | {s['v3']/n*100:>10.1f}%")

overall_total = sum(s["total"] for s in lang_stats.values())
print("-" * len(col_hdr))
overall_v1 = sum(s["v1"] for s in lang_stats.values()) / overall_total * 100
overall_v3 = sum(s["v3"] for s in lang_stats.values()) / overall_total * 100
print(f"  {'ALL':<5} | {overall_total:>4} | {overall_v1:>10.1f}% | {overall_v3:>10.1f}%")
print()


# ==============================================================================
# 6. CONFUSION MATRICES
# ==============================================================================
CM_LABELS = ['en', 'ms', 'id']
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, key in zip(axes, MODEL_SPECS):
    cm   = confusion_matrix(y_true, preds[key], labels=CM_LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CM_LABELS)
    disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(MODEL_SPECS[key]["name"], fontsize=11)
plt.suptitle("Confusion Matrices — OpenLID v1 vs v3", fontsize=13, y=1.02)
plt.tight_layout()

def next_path(directory: Path, stem: str, suffix: str) -> Path:
    n = 1
    while (directory / f"{stem}_{n}{suffix}").exists():
        n += 1
    return directory / f"{stem}_{n}{suffix}"

cm_path = next_path(CM_DIR, "confusion_matrix_openlid_v1_vs_v3", ".png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f" -> Exported '{cm_path}'")

print(f"\nDone. Model files: {MODEL_SPECS['v1']['path']}, {MODEL_SPECS['v3']['path']}")
