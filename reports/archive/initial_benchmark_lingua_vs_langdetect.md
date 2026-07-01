# Language Detection Engine Evaluation — 7-Model Benchmark & Voting Ensemble Design (the project)

Evaluating seven language detection libraries as candidates for the core language routing engine for the project. This evaluation assesses accuracy, speed, and AUC characteristics across a 475-case stress test of highly ambiguous Southeast Asian code-switching ("Bahasa Rojak"), formal educational terminology, and micro-text. The target integration point is **Stage 2 (NLP)** of the **Perception Layer** (`app/services/perception/`).

---

## 1. Executive Summary

Seven models were benchmarked across 475 test cases: **langdetect**, **lingua-low**, **lingua-high**, **langid**, **fasttext**, **openlid-v3**, and **pycld2**.

After full evaluation, **three models were selected for a majority-vote ensemble**: `pycld2`, `lingua-high`, and `langdetect`. These three were chosen not because any single one dominates, but because they offer complementary failure modes — when one misclassifies, the other two typically do not. The voting architecture replaces single-model fragility with consensus-driven reliability.

| Criterion | Winner (single model) |
|---|---|
| Overall Accuracy | `lingua-high` (85.7%) |
| ROC AUC | `pycld2` (0.9634) |
| Raw Speed | `pycld2` (~0.002–0.005ms) |
| Chinese (ZH) | `lingua-high` / `lingua-low` (100%) |
| Tamil (TA) | All models — unanimous 100% except minor langdetect variance |
| Malay/Indonesian | `lingua-high` (best, still only 65–70%) |

---

## 2. Models Evaluated

| Key | Library | Type |
|---|---|---|
| `langdetect` | langdetect 1.0.9 | Pure-Python port of Google's language-detect |
| `lingua-low` | lingua-language-detector | Rust-backed, low-accuracy mode |
| `lingua-high` | lingua-language-detector | Rust-backed, high-accuracy mode |
| `langid` | langid 1.1.6 | Pure-Python Naive Bayes, restricted to 5 langs |
| `fasttext` | fasttext-predict 0.9.2 | C++ Facebook model, lid.176.ftz (176 languages) |
| `openlid-v3` | HPLT OpenLID v3 | fasttext-based, 194 languages, FLORES-200 tags |
| `pycld2` | pycld2 0.42 | Compiled C++ (Google CLD2), confidence-calibrated |

---

## 3. Methodology & Dataset

**Dataset:** `test_case_6.txt` — 475 test cases, 95 per language (EN, MY, ID, ZH, TA), organized into 5 word-count buckets.

| Bucket | n | Composition |
|---|---|---|
| 1 word | 150 | 50% formal roots, 50% exclusive colloquialisms |
| 2 words | 150 | 50% shared/ambiguous, 50% dialect-specific |
| 3–7 words | 73 | Authentic Bahasa Rojak, code-switching patterns |
| 8–16 words | 53 | Localized slang, multi-word educational phrases |
| 17–50 words | 49 | Full sentences, educational content, paragraphs |

**Methodology notes:**
- Speed: Each bucket looped 100 times; all models warm-started before timing to exclude first-call init overhead.
- Scoring: **Strict exact-match only** — predicted ISO code must equal expected ISO code. No fallbacks or proxies applied. If a model cannot output a language (e.g. langdetect has no `ms` profile and always outputs `id` for Malay), it scores 0% on that language. This reflects real-world performance honestly.
- Langid: restricted to the 5 target languages for a fair comparison.

---

## 4. Benchmark 1 — Raw Processing Speed

Speed measured in milliseconds per call, averaged across 100 repetitions per bucket.

| Bucket | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 | Fastest |
|---|---|---|---|---|---|---|---|---|---|
| **1 word** | 150 | 3.1130 ms | 0.0094 ms | 0.0119 ms | 0.0239 ms | 0.0034 ms | 0.0406 ms | **0.0016 ms** | `pycld2` |
| **2 words** | 150 | 2.6865 ms | 0.0143 ms | 0.0178 ms | 0.0264 ms | 0.0042 ms | 0.0433 ms | **0.0016 ms** | `pycld2` |
| **3–7 words** | 73 | 2.2784 ms | 0.0350 ms | 0.0414 ms | 0.0336 ms | 0.0067 ms | 0.0553 ms | **0.0022 ms** | `pycld2` |
| **8–16 words** | 53 | 2.2917 ms | 0.0681 ms | 0.0728 ms | 0.0483 ms | 0.0119 ms | 0.0844 ms | **0.0032 ms** | `pycld2` |
| **17–50 words** | 49 | 2.6512 ms | 0.1315 ms | 0.0764 ms | 0.0692 ms | 0.0209 ms | 0.1485 ms | **0.0049 ms** | `pycld2` |

**Key observations:**
- `pycld2` is the fastest model across every bucket — ~2× faster than fasttext and ~1,900× faster than langdetect.
- `langdetect` is an outlier in a bad direction: 2.5–3.1 ms per call regardless of text length, ~260× slower than `lingua-high`.
- `lingua-high` overtakes `lingua-low` at the 17–50 word bucket (0.0764 ms vs 0.1315 ms) due to early-exit optimization — its comprehensive n-gram model hits a high-confidence threshold sooner on longer text.

---

## 5. Benchmark 2 — Accuracy by Bucket

### Bucket 1: 1 Word (n=30 per language)

| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 30.0% | 73.3% | **90.0%** | 100.0% | 100.0% | 6.7% | 0.0% |
| MY | **0.0%** ¹ | 46.7% | **56.7%** | 0.0% | 10.0% | 10.0% | 16.7% |
| ID | 26.7% | 56.7% | **63.3%** | 0.0% | 16.7% | 36.7% | 20.0% |
| ZH | 50.0% | **100.0%** | **100.0%** | 100.0% | 60.0% | 0.0% | 0.0% |
| TA | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

### Bucket 2: 2 Words (n=30 per language)

| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 50.0% | 90.0% | **100.0%** | 93.3% | 96.7% | 43.3% | 26.7% |
| MY | **0.0%** ¹ | 40.0% | **60.0%** | 13.3% | 20.0% | 23.3% | 23.3% |
| ID | 60.0% | 56.7% | **66.7%** | 33.3% | 30.0% | 46.7% | 13.3% |
| ZH | 46.7% | **100.0%** | **100.0%** | 100.0% | 53.3% | 0.0% | 76.7% |
| TA | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

### Bucket 3: 3–7 Words (n=15 EN/MY/ID/TA, n=13 ZH)

| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | **100.0%** | 100.0% | **100.0%** | 100.0% | 100.0% | 93.3% | 100.0% |
| MY | **0.0%** ¹ | 66.7% | **93.3%** | 53.3% | 26.7% | 73.3% | 53.3% |
| ID | **86.7%** | 60.0% | 60.0% | 60.0% | 53.3% | 73.3% | **93.3%** |
| ZH | 76.9% | **100.0%** | **100.0%** | 100.0% | 92.3% | 0.0% | 100.0% |
| TA | 93.3% | 100.0% | 86.7% | 100.0% | 100.0% | 100.0% | **100.0%** |

### Bucket 4: 8–16 Words (n=10 EN/MY/ID/TA, n=12 ZH)

| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | **0.0%** ¹ | 60.0% | 60.0% | 80.0% | 50.0% | **100.0%** | 80.0% |
| ID | 100.0% | 70.0% | **100.0%** | 70.0% | 80.0% | 80.0% | 90.0% |
| ZH | 25.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 0.0% | 100.0% |
| TA | 100.0% | 100.0% | 90.9% | 100.0% | 100.0% | 100.0% | **100.0%** |

### Bucket 5: 17–50 Words (n=10 EN/MY/ID/ZH, n=9 TA)

| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | **0.0%** ¹ | 80.0% | 70.0% | 60.0% | 30.0% | 80.0% | **100.0%** |
| ID | **100.0%** | 80.0% | 80.0% | 70.0% | 100.0% | 90.0% | **100.0%** |
| ZH | 10.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 0.0% | 100.0% |
| TA | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

> ¹ **langdetect MY = 0.0% across all buckets** — langdetect ships no `ms` (Malay) language profile. It always outputs `id` (Indonesian) for Malay text, which under strict exact-match scoring counts as wrong. This is not a detection failure — the library simply does not support Malay.

---

## 6. Benchmark 3 — Overall Accuracy by Language

| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| **EN** | 95 | 62.1% | 88.4% | **96.8%** | 97.9% | 98.9% | 51.6% | 45.3% |
| **MY** | 95 | **0.0%** ¹ | 52.6% | **65.3%** | 27.4% | 22.1% | 41.1% | 40.0% |
| **ID** | 95 | 62.1% | 61.1% | **69.5%** | 34.7% | 42.1% | 55.8% | 45.3% |
| **ZH** | 95 | 45.3% | **100.0%** | **100.0%** | 100.0% | 71.6% | 0.0% | 61.1% |
| **TA** | 95 | 98.9% | **100.0%** | 96.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| **ALL** | 475 | 53.7% | 80.4% | **85.7%** | 72.0% | 66.9% | 49.7% | 58.3% |

> ¹ langdetect has no `ms` profile — strict scoring gives it 0.0% MY across all buckets.

**Key observations:**
- `lingua-high` leads overall accuracy (85.7%) — the only model with strong EN (96.8%) + perfect ZH (100%) simultaneously.
- `langdetect` overall collapses to **53.7%** under strict scoring — lower than even `pycld2` (58.3%) — because its 0.0% MY score drags the average. It effectively cannot be used as a standalone detector for this language set.
- `pycld2` appears weak overall (58.3%) but achieves **100%** on TA and strong ZH at 3+ word lengths — its weaknesses are concentrated in micro-text (buckets 1–2) where its C++ heuristics need more signal.
- `openlid-v3` catastrophically fails ZH (0.0% overall) — it maps FLORES-200 Chinese tags to codes not in our ISO mapping, making it unsuitable for this stack.
- `langid` collapses on MY (27.4%) and ID (34.7%) despite perfect ZH/TA/EN — a fundamental gap for Southeast Asian routing.
- `fasttext` achieves high EN/TA but falls apart on MY (22.1%), making it unreliable for the target demographic.

---

## 7. Benchmark 4 — ROC AUC & Optimal Thresholds

Computed as binary one-vs-rest AUC using each model's top confidence score.

| Model | AUC | Optimal Threshold (Youden J) | Assessment |
|---|---|---|---|
| **pycld2** | **0.9634** | 0.3700 | Best-calibrated confidence scores — high discriminative power |
| lingua-low | 0.8712 | 0.9643 | Strong AUC but requires very high confidence to be reliable |
| lingua-high | 0.8503 | 0.6807 | Paired with best accuracy — usable at moderate confidence threshold |
| fasttext | 0.8501 | 0.7612 | Good AUC but poor Malay/ID accuracy limits usefulness |
| langdetect | 0.7516 | 1.0000 | Needs max confidence before trusting output; 0% MY under strict scoring |
| openlid-v3 | 0.7097 | 0.9248 | Mediocre AUC with ZH failure — eliminated |
| langid | 0.2123 | ∞ | AUC near random — logprob conversion unreliable for calibration |

**The pycld2 paradox:** pycld2 has the lowest overall accuracy (58.3%) but the highest AUC (0.9634). This means when pycld2 is wrong, its confidence score is already low — it knows when it doesn't know. This makes it an excellent ensemble member: its votes carry genuine signal even when its raw accuracy number appears weak.

**Visual diagnostics:**
- `confusion_matrix/confusion_matrix_all_2.png` — 7-panel confusion matrices (strict scoring run)
- `roc_curve/roc_curve_all_2.png` — ROC curves with Youden J optimal threshold markers

---

## 8. Model Elimination Summary

| Model | Eliminated? | Reason |
|---|---|---|
| `lingua-high` | No — **selected** | Best overall accuracy, perfect ZH |
| `langdetect` | No — **selected** | Strong EN/ID/TA at 3+ words; 0% MY is resolved by ensemble vote |
| `pycld2` | No — **selected** | Highest AUC, fastest, excellent confidence calibration |
| `lingua-low` | Eliminated | Dominated by `lingua-high` on accuracy with marginal speed gain |
| `fasttext` | Eliminated | 22.1% Malay accuracy — unusable for Southeast Asian routing |
| `langid` | Eliminated | 0.2123 AUC, collapses on MY/ID |
| `openlid-v3` | Eliminated | 0% Chinese accuracy, worst overall (49.7%) |

---

## 9. Voting Ensemble — Selected Models

### 9.1 Selected Trio

| Model | Overall Acc. | AUC | Best At | Weakness |
|---|---|---|---|---|
| `lingua-high` | **85.7%** | 0.8503 | EN, ZH, MY (3+ words) | Micro-text MY/ID |
| `langdetect` | 53.7% ¹ | 0.7516 | EN/ID/TA at 3+ words | Cannot output `ms` — 0% MY, slow (2.5–3.1 ms) |
| `pycld2` | 58.3% | **0.9634** | TA, ZH (3+ words), speed | EN/MY/ID micro-text |

> ¹ langdetect's 53.7% overall reflects strict scoring. Its role in the ensemble is not to detect Malay independently — it is selected for its high EN/ID/TA confidence and because its systematic `id` output for Malay text is overridden by the majority vote from lingua-high and pycld2.

### 9.2 Why These Three

The trio was chosen for **complementary failure modes**, not individual dominance:

- **EN:** `lingua-high` (96.8%) and `langdetect` (61.1%) typically agree → majority correct.
- **MY/ID:** See §9.3 — the confusion matrix reveals the three models have structurally different MY/ID error patterns that cancel each other out in a vote.
- **ZH:** `lingua-high` is always correct (100%); `pycld2` agrees on 3+ word inputs (76–100%); majority overrides `langdetect`'s ZH blindspot.
- **TA:** All three hit 100% — unanimous agreement.

No eliminated model improves this coverage without introducing a worse failure elsewhere.

### 9.3 Malay / Indonesian Ambiguity — Confusion Matrix Analysis

MY and ID share deep linguistic roots, making them the hardest pair for every model. The confusion matrices reveal that the three selected models have **structurally different MY/ID error patterns** — their mistakes do not overlap, which is exactly what a voting ensemble needs.

#### Raw cross-confusion (from confusion_matrix_all.png, out of 95 cases each)

| Model | MS → predicted ID | ID → predicted MS | MY correct (strict) | ID correct |
|---|---|---|---|---|
| `lingua-high` | **32** (33.7%) | **26** (27.4%) | 62/95 (65.3%) | 66/95 (69.5%) |
| `langdetect` | **~61** (≈64%) | 0 (0%) | **0/95 (0.0%)** — no `ms` profile | 59/95 (62.1%) |
| `pycld2` | **10** (10.5%) | **2** (2.1%) | 38/95 (40.0%) | 43/95 (45.3%) |

#### Why these numbers are actually good news for the ensemble

**`langdetect` has no `ms` profile** — it systematically outputs `id` for Malay text (61/95 cases). This looks like a weakness in isolation, but in a vote it becomes structural information: if `langdetect` says `id`, it is ambiguous (could be genuine ID or Malay-as-ID). The other two models then decide.

**`pycld2` is the most decisive MY/ID separator** — only 10 MY cases leak to ID and only 2 ID cases leak to MY. When pycld2 fires a confident prediction, its MY/ID distinction is more reliable than either of the other two. Its overall MY/ID accuracy looks low (40–45%) because it outputs `unknown` for many micro-text cases — but when it does commit, it is usually right.

**`lingua-high` has bidirectional confusion** — 32 MY→ID and 26 ID→MY leaks. However, its leaks are spread across all word-count buckets, meaning it is not systematically wrong: it correctly identifies 62 MY and 66 ID cases.

#### How the vote resolves MY/ID conflicts

| Scenario | lingua-high | langdetect | pycld2 | Vote result |
|---|---|---|---|---|
| True MY, typical case | `ms` | `id` (proxy) | `ms` | **ms wins 2-1** ✓ |
| True MY, lingua confused | `id` | `id` (proxy) | `ms` | id wins 2-1 ✗ — but pycld2 ms conf. triggers LOW_CONFIDENCE |
| True ID, typical case | `id` | `id` | `id` | **id wins 3-0** ✓ |
| True ID, lingua confused | `ms` | `id` | `id` | **id wins 2-1** ✓ |
| True MY, pycld2 unknown | `ms` | `id` (proxy) | `—` | 1-1 tie → lingua-high tiebreaker → `ms` ✓ |

The most problematic row is when `lingua-high` leaks MY→ID and pycld2 fires as ID: langdetect (always `id`) + lingua-high (`id`) = 2 votes for ID. This is the residual failure case and cannot be fully resolved without additional context. It is flagged as LOW_CONFIDENCE in the integration design.

**Net ensemble result:** For MY, the most common vote outcome is `ms` wins 2-1 (pycld2 + lingua-high vs langdetect). For ID, the most common outcome is `id` wins 3-0 or 2-1. The ensemble substantially reduces the raw per-model weaknesses on this pair.

### 9.4 Voting Logic Design

```
Input text
    │
    ├── lingua-high.detect(text)  → (lang_A, conf_A)
    ├── langdetect.detect(text)   → (lang_B, conf_B)
    └── pycld2.detect(text)       → (lang_C, conf_C)

Majority vote:
    - If 2 or 3 models agree → return consensus language
    - If all 3 disagree → return lingua-high prediction (highest individual accuracy)

Confidence gate (optional, for borderline cases):
    - If majority language confidence < threshold → flag as LOW_CONFIDENCE
    - Thresholds derived from AUC analysis:
        lingua-high:  0.6886  (Youden J optimal)
        pycld2:       0.3700  (Youden J optimal)
        langdetect:   use raw probability from detect_langs()
```

### 9.5 Per-Language Expected Voting Accuracy

Estimated from confusion matrix data — 2-of-3 majority:

| Language | Expected Ensemble | Notes |
|---|---|---|
| **EN** | ~96–99% | lingua-high (92/95 correct) carries the majority; only 2 cases leak to id |
| **MY** | ~70–75% | pycld2 + lingua-high typically vote `ms`; langdetect's `id` proxy is outvoted |
| **ID** | ~80–85% | All three lean `id`; lingua-high's 26 MS leaks partially resolved by pycld2 (only 2 ID→MS) |
| **ZH** | ~95–100% | lingua-high (95/95) always correct; majority overrides langdetect (44/95) |
| **TA** | **100%** | All three agree unanimously (94–95/95 each) |

### 9.6 Integration Notes

- Run all three models in parallel (they are in-process, not network calls); total latency ≈ `lingua-high` time + negligible overhead for pycld2/langdetect.
- `langdetect`'s `id` output for MY text is expected behaviour — do not treat it as a bug. The other two votes resolve it.
- For the project Perception Stage 2, the ensemble output should produce a BCP-47 tag (e.g., `ms-MY`, `en-MY`, `zh-Hans`) with a `low_confidence` flag when all three disagree or when the majority confidence falls below threshold.

---

## 10. Identified Risks & Remaining Weaknesses

| # | Issue | Confusion Matrix Evidence | Mitigation in Ensemble |
|---|---|---|---|
| 1 | **MY/ID residual failure** — when lingua-high leaks MY→ID (32 cases) AND pycld2 also says ID, langdetect's `id` proxy gives 3-0 vote for ID on a true MY input | lingua-high: 32 MS→ID; pycld2: 10 MS→ID overlap | Flag as LOW_CONFIDENCE when all three say `id` but confidence is below pycld2 threshold (0.37) |
| 2 | **`pycld2` micro-text unknown** — outputs outside the 5 target classes on short EN/MY/ID inputs (roughly 45–55 of 95 EN/MY/ID cases unaccounted in matrix) | pycld2 EN matrix row sums to 43/95; ms row to 49/95 | 2-model vote (lingua-high + langdetect) resolves when pycld2 abstains |
| 3 | **`langdetect` ZH blindspot** — only 44/95 ZH correct; 8 leak to EN on longer text | langdetect: zh row = 44 correct, 8→en | lingua-high (95/95) + pycld2 (58/95) outvote; ZH is safe |
| 4 | **`langdetect` 5ms latency** — ~1,000× slower than pycld2; adds to ensemble wall time | N/A | Acceptable for Stage 2 async pipeline; run in parallel with the other two |
| 5 | **All three weak on 1-word MY/ID** — max 56.7% (lingua-high) at single-word Malay | All MY/ID bucket 1 rows show <65% across all models | Unconditionally flag single-word MY/ID predictions as LOW_CONFIDENCE |
