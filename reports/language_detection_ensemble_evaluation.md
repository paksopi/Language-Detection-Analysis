# Language Detection Engine — Final Consolidated Report

> **Dataset: `test_case_7_enmyid.txt` — 1,273 cases (EN 432 / MY 421 / ID 420), filtered to English/Malay/Indonesian only.** Chinese (ZH) and Tamil (TA) cases were removed from `test_case_7.txt` at the user's request, since MY/ID disambiguation — not ZH/TA, which every model already handled near-perfectly — is the actual problem this evaluation exists to solve. A full 5-language archived report is kept at `Language_Detection_Engine_Final_Report_test_case_7.md`, and the older `test_case_6` (475-case, 5-language) report at `Language_Detection_Engine_Final_Report_test_case_6.md`.

> **Naming convention:** `EN`/`MY`/`ID` are this report's dataset language labels, used in prose ("Malay (MY)") and as table/column headers. `en`/`ms`/`id` are the corresponding [BCP-47](https://www.rfc-editor.org/info/bcp47) ISO codes, used in inline code font when referring to a literal model output value (e.g. "the model outputs `ms`"). `ms` is never used as shorthand for "milliseconds" outside of explicit `X ms` latency figures — those are always written with a numeric prefix to avoid ambiguity with the Malay ISO code.

## TL;DR

- Current production (`langdetect` alone) scores 29.1% overall and 0.0% on Malay (MY). The best single proposed config — **S2 Weighted** (`lingua-high` + `openlid-v3` + `pycld2`) — reaches **70.8%** overall, **+41.7pp**, and is 8.6–115.5× faster per call.
- **But S2 Weighted's own MY accuracy is 43.7% — *worse* than `lingua-high` running alone (55.8%) and worse than the S1 Two-Stage Weighted alternative (56.5%).** The 70.8% headline is an ALL-language (EN+MY+ID blended) number bought partly by trading away MY accuracy for a large ID accuracy gain.
- **Recommendation:** ship **S2 Weighted** if ID accuracy and per-request latency matter most and low-confidence MY/ID predictions get flagged for downstream context-gathering (§13). Ship **S1 Two-Stage Weighted** instead if MY protection matters most — it matches/exceeds `lingua-high` alone on MY at a cost of only 0.5pp ALL accuracy. See the decision matrix in §9.
- The 70.8% figure is itself bucket-composition-dependent: 53% of the test set (677/1,273 cases) is single-word text, the bucket with the weakest, least-representative signal (§12). §12b shows what ALL/EN/MY/ID accuracy becomes under more realistic query-length distributions.

## Executive Summary

**Headline finding — and the trade-off that comes with it:** A representative production baseline currently runs language detection on **`langdetect` alone** — a single unvoted model that cannot detect Malay at all (0% MY accuracy) and scores only 29.1% overall accuracy on this dataset. Because language detection is a synchronous step that must complete *before* downstream AI processing receives any context (§10.1), every millisecond and every misdetection here has a direct, un-hidden cost to the user. The proposed 3-model weighted-voting ensemble (`lingua-high` + `openlid-v3` + `pycld2`, "**S2 Weighted**") reaches 70.8% overall accuracy — **+41.7 percentage points** — while also running 8.6–115.5× faster per call. **That headline number comes with a real, immediate trade-off: S2 Weighted's own Malay (MY) accuracy is 43.7%, which is *below* both `lingua-high` running alone (55.8%) and the S1 Two-Stage Weighted alternative (56.5%, §6).** On the ALL-language axis this is not a speed-for-accuracy trade — the proposed system beats what ships today on both accuracy and speed simultaneously. But on the MY axis specifically, the config with the best ALL accuracy is *not* the config with the best MY accuracy, and which one to ship depends on which error type is more costly downstream — see the decision matrix in §9.

This report evaluates seven language detection libraries across 1,273 English/Malay/Indonesian text cases (`test_case_7_enmyid.txt`, filtered from the 2,036-case `test_case_7.txt` to drop Chinese and Tamil) to design the core routing engine for a downstream NLP pipeline's language-detection stage. Initial benchmarking identified three models—`lingua-high`, `langdetect`, and `pycld2`—that exhibited complementary failure modes. We hypothesized that a majority-vote ensemble of these three would resolve their individual weaknesses, particularly the deep ambiguity between Malay (MY) and Indonesian (ID).

However, empirical voting results replicated the same structural failure seen in both prior evaluations: while voting improved ID accuracy, it degraded MY accuracy, dropping it from 55.8% to 51.3% under hard voting and to 32.3% under soft voting. McNemar's test confirmed the MY drop under hard voting is highly significant (p < 0.0001), and Cohen's kappa again showed `langdetect`'s vote on MY text is structurally adversarial (κ = −0.0191, 18.1% agreement with `lingua-high` — worse than chance). These per-language figures are identical to the full 5-language dataset's, since removing ZH/TA cases does not change how any model scores on EN/MY/ID text — only the blended "ALL" figures shift, because ZH/TA were the two languages every model handled almost perfectly, and their removal pulls the overall average down to reflect the genuinely hard part of the problem.

A simple leave-one-out ablation (removing `langdetect` entirely) again does **not** fully recover MY accuracy (53.4–53.9% vs. `lingua-high`'s individual 55.8%). Two-Stage Voting (weighted variant) is still the only configuration that fully restores MY protection (56.5%, exceeding `lingua-high` alone), while Scenario 2 (`openlid-v3`-based) Weighted Voting achieves the best overall accuracy (70.8%) and the best ID accuracy (76.0%), at the cost of materially worse MY accuracy (43.7%).

**Decision matrix — pick based on which error costs more downstream:**

| Choose... | If... | ALL | MY | ID | EN |
|---|---|---|---|---|---|
| **S2 Weighted** (`lingua-high`+`openlid-v3`+`pycld2`) | ID accuracy, per-request latency, and single-stage simplicity matter more than MY protection; MY/ID low-confidence flagging (§13) is in place downstream | **70.8%** | 43.7% | **76.0%** | 92.1% |
| **S1 Two-Stage Weighted** (`lingua-high`+`langdetect`+`pycld2`, 2-stage) | Protecting Malay-language users from misdetection matters more than squeezing out the last few points of ID/ALL accuracy | 70.3% | **56.5%** | 61.4% | 92.4% |

The two configs are close on ALL (70.8% vs. 70.3%, 0.5pp apart) but diverge sharply per-language: S2 trades 12.8pp of MY accuracy for 14.6pp of ID accuracy relative to Two-Stage. Neither config is a strict win — the choice depends on which language's errors are more costly to the downstream product.

Based on these findings — and per the user's explicit direction in the prior 5-language report — we recommend deploying **Scenario 2 Weighted Voting (`lingua-high` + `openlid-v3` + `pycld2`)** as the primary architecture, on the assumption that downstream low-confidence flagging (§13) adequately mitigates the MY gap. It wins on overall accuracy, ID accuracy, and per-request latency, and needs no Stage-1/Stage-2 routing logic. But this is a conditional recommendation, not a strict win: if MY protection turns out to matter more than the decision matrix above assumes, **S1 Two-Stage Weighted** (56.5% MY, 70.3% ALL) is the better default. The MY tradeoff (43.7% vs. 55.8% for `lingua-high` alone, vs. 56.5% for the Two-Stage fallback) should be monitored in production regardless of which config ships; see §10 and §13.

---

## Table of Contents
0. [TL;DR](#tldr)
1. [Benchmark Methodology & Individual Model Performance](#1-benchmark-methodology--individual-model-performance)
2. [Model Selection & Initial Ensemble Design Hypothesis](#2-model-selection--initial-ensemble-design-hypothesis)
3. [Initial Voting Results (The Failure)](#3-initial-voting-results-the-failure)
4. [Diagnosis of the Voting Failure](#4-diagnosis-of-the-voting-failure)
5. [Ablation Study](#5-ablation-study)
6. [Fix 1: Two-Stage Voting Architecture](#6-fix-1-two-stage-voting-architecture)
7. [Fix 2: Scenario 2 (Replacing langdetect with openlid-v3)](#7-fix-2-scenario-2-replacing-langdetect-with-openlid-v3)
8. [Train/Test Leakage Check](#8-traintest-leakage-check)
9. [Master Summary Table & Final Recommendation](#9-master-summary-table--final-recommendation)
10. [Scenario Comparison — Speed, Accuracy & Complexity](#10-scenario-comparison--speed-accuracy--complexity)
11. [Methodology Notes](#11-methodology-notes)
12. [Limitations and Threats to Validity](#12-limitations-and-threats-to-validity)
    - 12b. [Accuracy Under Realistic Query-Length Distributions](#12b-accuracy-under-realistic-query-length-distributions)
13. [Integration Notes for Downstream Deployment](#13-integration-notes-for-downstream-deployment)

---

## 1. Benchmark Methodology & Individual Model Performance

### 1.1 Methodology and Dataset
Seven models were benchmarked: `langdetect`, `lingua-low`, `lingua-high`, `langid`, `fasttext`, `openlid-v3`, and `pycld2`.
The test set (`test_case_7_enmyid.txt`) consists of 1,273 cases across three languages: EN (432), MY (421), ID (420) — the ZH/TA cases from `test_case_7.txt` were filtered out. Cases are structured into five word-count buckets:
* **1 word (677 cases)**
* **2 words (399 cases)**
* **3–7 words (94 cases):** Authentic Bahasa Rojak, code-switching patterns.
* **8–16 words (73 cases):** Localized slang, multi-word educational phrases.
* **17–50 words (30 cases):** Full sentences, educational content, paragraphs.

Scoring relied strictly on exact-match BCP-47 ISO codes. No fallbacks or proxies were applied. `lingua-high`'s candidate language set and `langid`'s language restriction were both narrowed to {en, ms, id} for this run (previously {en, ms, id, zh, ta}).

### 1.2 Raw Processing Speed
Measured in milliseconds per call, averaged across 100 warm-started repetitions per bucket.

| Bucket | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 | Fastest |
|---|---|---|---|---|---|---|---|---|---|
| **1 word** | 677 | 5.5216 ms | 0.0134 ms | 0.0188 ms | 0.0221 ms | 0.0040 ms | 0.0478 ms | **0.0017 ms** | `pycld2` |
| **2 words** | 399 | 4.5003 ms | 0.0199 ms | 0.0294 ms | 0.0227 ms | 0.0061 ms | 0.0477 ms | **0.0019 ms** | `pycld2` |
| **3–7 words** | 94 | 3.3995 ms | 0.0474 ms | 0.0740 ms | 0.0322 ms | 0.0085 ms | 0.0766 ms | **0.0026 ms** | `pycld2` |
| **8–16 words** | 73 | 2.2760 ms | 0.1084 ms | 0.1627 ms | 0.0561 ms | 0.0165 ms | 0.1204 ms | **0.0034 ms** | `pycld2` |
| **17–50 words** | 30 | 2.1027 ms | 0.2364 ms | 0.1553 ms | 0.0853 ms | 0.0246 ms | 0.2435 ms | **0.0063 ms** | `pycld2` |

`pycld2` is the fastest model across every bucket. Per-model latencies are somewhat higher across the board than the prior 5-language run (measured on the same machine, different session) — see the single-run-noise caveat in §12.

### 1.3 Accuracy by Bucket
*Note: Due to a lack of a Malay profile, `langdetect` still systematically predicts ID for MY cases, scoring 0.0% on MY under strict exact-match scoring.*

**Bucket 1: 1 Word (n=677)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 226 | 25.7% | 73.9% | **88.9%** | 99.1% | 88.9% | 6.6% | 4.0% |
| MY | 226 | **0.0%** | 39.8% | **47.3%** | 0.4% | 7.5% | 6.6% | 6.6% |
| ID | 225 | 25.8% | 45.8% | **49.3%** | 8.0% | 12.0% | 33.3% | 11.6% |

**Bucket 2: 2 Words (n=399)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 141 | 45.4% | 89.4% | **94.3%** | 97.2% | 92.9% | 22.0% | 29.8% |
| MY | 129 | **0.0%** | 41.1% | **61.2%** | 12.4% | 14.0% | 22.5% | 22.5% |
| ID | 129 | 50.4% | 57.4% | **62.0%** | 40.3% | 24.0% | 58.9% | 17.8% |

**Bucket 3: 3–7 Words (n=94)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 30 | 93.3% | 96.7% | 96.7% | **100.0%** | **100.0%** | 90.0% | 96.7% |
| MY | 32 | **0.0%** | 59.4% | **78.1%** | 50.0% | 37.5% | 78.1% | 56.2% |
| ID | 32 | 90.6% | 59.4% | 65.6% | 65.6% | 59.4% | 78.1% | **78.1%** |

**Bucket 4: 8–16 Words (n=73)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 25 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | 24 | **0.0%** | 66.7% | 70.8% | 50.0% | 50.0% | **87.5%** | 79.2% |
| ID | 24 | 100.0% | 70.8% | **95.8%** | 83.3% | 87.5% | 91.7% | 91.7% |

**Bucket 5: 17–50 Words (n=30)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 10 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | 10 | **0.0%** | 80.0% | 70.0% | 60.0% | 30.0% | 80.0% | **100.0%** |
| ID | 10 | **100.0%** | 80.0% | 80.0% | 70.0% | 100.0% | 90.0% | **100.0%** |

### 1.4 Overall Accuracy & Confidence Intervals (95% Wilson)

| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| **EN** | 432 | 42.8% [38.2-47.5] | 82.6% | **92.1% [89.2–94.3]** | 98.6% | 91.9% | 25.0% | 26.6% [22.7-31.0] |
| **MY** | 421 | **0.0% [0.0–0.9]** | 44.2% | **55.8% [51.0–60.5]** | 12.1% | 14.7% | 23.3% | 21.6% [17.9-25.8] |
| **ID** | 420 | 44.3% [39.6-49.1] | 52.6% | **57.9% [53.1–62.5]** | 28.1% | 25.7% | 49.3% | 25.2% [21.3-29.6] |
| **ALL** | 1273 | 29.1% [26.7-31.7] | 60.0% | **68.8% [66.2–71.3]** | 46.7% | 44.5% | 32.4% | 24.5% [22.2-26.9] |

Every model's ALL figure drops substantially versus the 5-language dataset (e.g. `lingua-high` 80.2% → 68.8%), because ZH and TA — where every model scored 95–100% — are gone. This is expected and, if anything, more representative: the blended accuracy now reflects only the genuinely contested EN/MY/ID space rather than being propped up by two languages nobody was struggling with.

### 1.5 ROC AUC Analysis
Computed as binary one-vs-rest AUC using each model's top confidence score.
* **`pycld2` (0.9746):** The most discriminative, consistent with both prior evaluations.
* **`lingua-high` (0.7417):** Notably lower than the 5-language figure (0.8569) — with ZH/TA removed, `lingua-high` no longer gets "easy" high-confidence correct calls on those languages padding its AUC.
* **`langdetect` (0.6586):** Similarly lower than the 5-language figure (0.7655), for the same reason.

---

## 2. Model Selection & Initial Ensemble Design Hypothesis

Based on the individual benchmarks, four models were eliminated:
* `lingua-low` (dominated by `lingua-high`).
* `langid` and `fasttext` (collapsed on MY and ID — 12.1%/14.7% and 28.1%/25.7% respectively).
* `openlid-v3` (32.4% overall in this benchmark script; note the same FLORES-200 mapping caveat as before — this script's local mapping differs from `voting/core.py`'s corrected one, though it matters less here since ZH is no longer in scope).

Three models were selected for a voting ensemble: **`lingua-high`**, **`langdetect`**, and **`pycld2`**. We hypothesized that this trio offered complementary failure modes that would cancel each other out in a majority vote.

**The Initial Hypothesis for MY/ID Resolution:**
Because `langdetect` lacks a Malay profile, it outputs Indonesian (`id`) for true Malay (MY) 100% of the time (421/421 cases). Originally, we hypothesized that this systematic `id` proxy output would be safely overridden by the majority consensus of `lingua-high` and `pycld2`. We assumed that when true Malay text was processed, `lingua-high` and `pycld2` would both vote `ms`, successfully winning the election 2-to-1 against `langdetect`.

---

## 3. Initial Voting Results (The Failure)

We tested three ensemble voting strategies on the trio (lingua-high + langdetect + pycld2):
* **Hard Vote:** Simple majority class vote.
* **Soft Vote:** Averaging the output probability/confidence vectors.
* **Weighted Vote:** AUC-weighted probability averaging.

Contrary to our hypothesis, the voting ensemble did not resolve the MY/ID ambiguity; it actively harmed MY detection — identical per-language behavior to the 5-language run.

| LANG | lingua-high (Baseline) | 3-model Hard | 3-model Soft | 3-model Weighted |
|---|---|---|---|---|
| EN | 92.1% | 92.4% | 91.7% | 91.7% |
| **MY** | **55.8%** | **51.3%** | **32.3%** | **37.1%** |
| ID | 57.9% | 63.1% | 74.3% | 73.1% |
| **ALL** | 68.8% | 69.1% | 66.3% | 67.5% |

While ID accuracy jumped significantly (+5.2 percentage points under hard voting), **MY accuracy dropped from 55.8% to 51.3% (hard) and 32.3% (soft)**. The ALL column is lower than the 5-language report's (which was 80.2/80.6/78.8/79.6) purely because ZH/TA — both near-100% for every strategy — are no longer averaged in.

---

## 4. Diagnosis of the Voting Failure

To understand why the hypothesis failed, we evaluated the results using multiple statistical frameworks.

### 4.1 Statistical Significance (McNemar's Test)
McNemar's test confirmed the drop in MY accuracy was not random noise:
* **MY drop:** `lingua-high` → hard vote (−4.5 pp) is **Highly Significant** (p < 0.0001, b=19, c=0 — every error introduced by hard voting flipped a correct `lingua-high` prediction, with zero reverse corrections).
* **MY drop:** `lingua-high` → soft vote (−23.5 pp) is **Highly Significant** (p < 0.0001).
* On the ALL scope (now EN+MY+ID only), `lingua-high` vs. hard vote is **not significant** (p = 0.6511) — with ZH/TA removed, the aggregate-level McNemar test no longer detects the MY-specific harm, because it is diluted across three languages instead of five. This is a useful illustration of why per-language McNemar tests (as done here) matter more than aggregate ones.

### 4.2 Inter-Model Agreement and Diversity (Cohen's Kappa)
Cohen's Kappa (κ) again exposed a critical structural problem. `lingua-high` and `langdetect` showed a **negative kappa (κ = −0.0191)** on MY — identical to the 5-language run, since this is computed only over MY cases. They agreed on only 18.1% of cases—less than chance.

### 4.3 Calibration Analysis (Expected Calibration Error - ECE)
`langdetect` again emerged as the worst-calibrated model (ECE = 0.1845, vs. 0.1282 on the 5-language set — the removal of ZH/TA, where `langdetect` had more correct high-confidence calls, makes its miscalibration on EN/MY/ID more visible). `lingua-high`'s ECE also rose (0.0431 vs. 0.0268), for the same reason: its very high accuracy on ZH/TA previously masked more mediocre calibration on the harder EN/MY/ID cases.

### 4.4 The Failed Assumption
As before, the assumption that `langdetect` would simply be outvoted failed because `pycld2` frequently abstains or is uncertain on short/micro-text (`pycld2`'s own MY accuracy is only 21.6% — well below `lingua-high`'s 55.8%). When `pycld2` is uncertain, hard voting triggers a 1-vs-1 tie, and because `langdetect` is guaranteed to vote `id`, any hesitation from `pycld2` allows the tied vote to default to `id`.

---

## 5. Ablation Study

To test whether `langdetect` alone was the source of the failure, we performed a leave-one-model-out ablation on the full 1,273-case dataset.

| LANG | lingua-high | 3m-hard (with LD) | no-LD hard | no-LD soft | no-LD weighted |
|---|---|---|---|---|---|
| EN | 92.1% | 92.4% | 92.4% | 92.4% | 92.4% |
| **MY** | **55.8%** | 51.3% | 53.4% | 53.4% | 53.9% |
| ID | 57.9% | 63.1% | 64.0% | 64.0% | 64.0% |
| **ALL** | 68.8% | 69.1% | **70.1%** | **70.1%** | **70.3%** |

Identical to the 5-language finding: removing `langdetect` improves MY accuracy over the 3-model hard vote (51.3% → 53.4–53.9%) but does **not** fully recover it to `lingua-high`'s individual level (55.8%). ID gains are preserved (64.0%). The simple ablation hypothesis is **not fully supported** — motivating Section 6.

---

## 6. Fix 1: Two-Stage Voting Architecture

Since simple ablation left a residual MY gap, we applied the Two-Stage Voting mechanism to safely integrate `langdetect`'s useful signal on English while removing its structural bias on the MY/ID axis.

**Design:**
* **Stage 1:** Coarse classification. `langdetect`'s `id` vote is mapped to a broad "MSID" (Malay/Indonesian) class. All three models hard-vote on {en, MSID}. If the outcome is not MSID, the decision is final.
* **Stage 2:** Fine MY/ID decision. Triggered only if Stage 1 yields MSID. This stage strictly uses `lingua-high` and `pycld2`.
    * *two_stage_agree:* If they agree, output the consensus. If they disagree, use the model with higher confidence.
    * *two_stage_weighted:* Score each label using per-class dev-fitted accuracy weights for `lingua-high` and `pycld2`.

**Results (full 1,273-case dataset):**
| LANG | lingua-high | 3m-hard | **2s-agree** | **2s-weighted** |
|---|---|---|---|---|
| EN | 92.1% | 92.4% | 92.4% | 92.4% |
| **MY** | 55.8% | 51.3% | 53.9% | **56.5%** |
| **ID** | 57.9% | 63.1% | **64.0%** | 61.4% |
| **ALL** | 68.8% | 69.1% | **70.3%** | **70.3%** |

Identical per-language numbers to the 5-language run: `two_stage_weighted` fully restores and slightly exceeds `lingua-high`'s individual MY accuracy (56.5% vs. 55.8%). McNemar testing confirmed `two_stage_weighted` vs. `lingua-high` on MY shows **no significant difference** (p = 0.7835). `two_stage_agree` achieves the best ID accuracy of the two-stage variants (64.0%) while still improving MY over the biased baseline (53.9% vs. 51.3%).

---

## 7. Fix 2: Scenario 2 (Replacing langdetect with openlid-v3)

We evaluated a new ensemble (Scenario 2) replacing `langdetect` with `openlid-v3` (`lingua-high` + `openlid-v3` + `pycld2`), using the corrected FLORES-200 ISO mapping (`voting/core.py`). `openlid-v3` natively supports Malay (MY), achieving 27.8% MY accuracy individually — identical to the 5-language run, since this is measured only on MY cases.

**Scenario 2 Results (full 1,273-case dataset):**
| LANG | S1 hard (langdetect) | S2 hard (openlid) | S2 soft | S2 weighted |
|---|---|---|---|---|
| EN | 92.4% | **92.4%** | 92.1% | 92.1% |
| **MY** | 51.3% | **53.0%** | 42.3% | 43.7% |
| ID | 63.1% | 64.3% | 76.2% | **76.0%** |
| **ALL** | 69.1% | 70.1% | 70.4% | **70.8%** |

McNemar testing confirmed S2's aggregate accuracy gain over S1 is significant for hard voting (+0.9pp, p=0.0376) and highly significant for soft/weighted (+4.1pp/+3.3pp, p<0.0001/p=0.0003) on the ALL scope — a larger relative jump than the 5-language run, since ALL no longer includes ZH/TA (where S1 and S2 already tied at 100%). On MY specifically, S2 soft/weighted significantly beat S1 soft/weighted (+10.0pp p<0.0001, +6.7pp p=0.0011) — but both remain well below `lingua-high`'s individual 55.8% and below the Two-Stage results in §6. Cohen's kappa between `lingua-high` and `openlid-v3` on MY is **0.2254** (vs. `langdetect`'s −0.0191), confirming `openlid-v3` is a genuinely diverse voter — it outputs `ms` for 117/421 true-MY cases, versus `langdetect`'s 0.

---

## 8. Train/Test Leakage Check

To verify that weighted-voting weights derived from the dataset's ROC AUC were not overfitting, we conducted a stratified 60/40 dev/test split (dev n=763, test n=510).

| Strategy | Held-Out Test Accuracy (n=510) | Note |
|---|---|---|
| weighted (global AUC) | 67.1% | weights from full 1,273 — leaked |
| weighted (dev AUC) | 65.5% | weights from dev set only — honest |

The gap (−1.57 pp) falls within standard sampling noise, confirming there is no material overfitting from computing weights globally — consistent with both prior evaluations.

---

## 9. Master Summary Table & Final Recommendation

All figures below are on the full 1,273-case (EN/MY/ID-only) dataset except where noted.

**No single strategy wins on every axis.** S2 Weighted has the best ALL and ID accuracy; S1 Two-Stage Weighted has the best MY accuracy and matches `lingua-high` run alone on MY. Read the table with that in mind before jumping to the recommendation below.

| Strategy | Ensemble | EN | MY | ID | ALL |
|---|---|---|---|---|---|
| lingua-high (individual) | — | 92.1% | 55.8% | 57.9% | 68.8% |
| S1 hard | ld+li+py | 92.4% | 51.3% | 63.1% | 69.1% |
| S1 weighted | ld+li+py | 91.7% | 37.1% | 73.1% | 67.5% |
| S1 two_stage_agree | ld+li+py (2-stage) | 92.4% | 53.9% | **64.0%** | 70.3% |
| S1 two_stage_weighted | ld+li+py (2-stage) | 92.4% | **56.5%** | 61.4% | 70.3% |
| S2 hard | ol+li+py | **92.4%** | 53.0% | 64.3% | 70.1% |
| **S2 weighted (RECOMMENDED, conditionally — see below)** | ol+li+py | 92.1% | 43.7% | **76.0%** | **70.8%** |

### Decision matrix

| Choose... | If... | ALL | MY | ID |
|---|---|---|---|---|
| **S2 Weighted** | ID accuracy, latency, and single-stage simplicity matter more than MY protection, and low-confidence MY/ID flagging (§13) is in place downstream | **70.8%** | 43.7% | **76.0%** |
| **S1 Two-Stage Weighted** | MY protection matters more than the last ~0.5pp of ALL accuracy — e.g. MY-heavy traffic, or MY misdetection is costlier downstream than ID misdetection | 70.3% | **56.5%** | 61.4% |

**Final Recommendation:** Deploy **Scenario 2 Weighted Voting (`lingua-high` + `openlid-v3` + `pycld2`)** as the primary architecture, *conditional on* downstream low-confidence flagging (§13) being in place to catch the MY gap described below. It achieves the best overall accuracy on this dataset (70.8%) and the best ID accuracy (76.0%), needs only a single-stage vote (no Stage-1/Stage-2 routing logic to build and maintain), and is 8–116× faster per request than any `langdetect`-based configuration (§10.2) — while using a genuinely diverse voter on the MY/ID axis (κ = 0.2254 vs. `langdetect`'s −0.0191, §7). More fundamentally, it already beats current production (`langdetect` alone) by +41.7pp overall accuracy while also being faster — see §10.1.

**The MY regression is real and is not a minor footnote:** S2 Weighted's MY accuracy (43.7%) sits *below* both `lingua-high` run alone (55.8%) and every S1 two-stage variant (best: 56.5%) — a 12.8pp gap versus the Two-Stage fallback. If MY protection matters more than the decision matrix above assumes for your traffic, **ship S1 Two-Stage Weighted instead** (56.5% MY, 70.3% ALL) — it is only 0.5pp behind S2 on ALL accuracy and requires no separate model artifact beyond what Scenario 1 already needs. Mitigate the MY gap the way §13 recommends for all micro-text — flag low-confidence MY/ID predictions (especially 1-word inputs) for downstream context-gathering rather than trusting the raw vote — and track MY-specific accuracy in production regardless of which config ships. The untested `lingua-high` + `openlid-v3`-only Stage 2 (§10.5) is worth building as a possible best-of-both option before reverting to `langdetect`, since `openlid-v3` is already the model in use.

See §10 for the full speed/accuracy/complexity comparison behind this recommendation — notably, `openlid-v3` requires shipping a 1.2 GB model artifact, which is the main cost of choosing Scenario 2. See §12b for how the 70.8%/56.5% ALL figures shift under more realistic query-length distributions than this dataset's synthetic bucket split.

---

## 10. Scenario Comparison — Speed, Accuracy & Complexity

Sections 3–9 focused on accuracy. This section pulls together the three axes that actually decide which architecture to ship: raw speed, accuracy, and deployment complexity — then gives a single conclusion.

### 10.1 Current Production Baseline vs. Proposed Ensemble

This is the headline comparison for this project, and it matters more than the Scenario 1 vs. Scenario 2 debate below: **language detection is not a background or parallel step.** In a typical deployment, it runs synchronously as an early NLP stage — strictly *before* any of the downstream AI reasoning stages (e.g. intent classification, entity extraction, strategy selection) that depend on knowing the input language. Detection has to finish and hand its result to the pipeline before the AI gets any context at all. Every millisecond spent here is added directly to the user-perceived response time — it is not hidden behind, or amortized by, the other pipeline stages.

A representative production baseline uses **`langdetect` alone** — a single model, no voting, no fallback, and no mitigation for its structural inability to detect Malay at all. Benchmarked in isolation (§1.4), that leaves real accuracy on the table:

| Metric | Current production (`langdetect` alone) | Proposed (Scenario 2 Weighted ensemble) | Change |
|---|---|---|---|
| Overall accuracy (ALL) | 29.1% | **70.8%** | **+41.7 pp** |
| EN accuracy | 42.8% | **92.1%** | +49.3 pp |
| MY accuracy | 0.0% (cannot detect Malay at all — structural) | 43.7% | +43.7 pp |
| ID accuracy | 44.3% | **76.0%** | +31.7 pp |
| Per-call latency, 1-word text | 5.5216 ms | 0.0478 ms (ensemble bottleneck) | **115.5× faster** |
| Per-call latency, 17–50 word text | 2.1027 ms | 0.2435 ms (ensemble bottleneck) | 8.6× faster |

This is not a speed-vs-accuracy tradeoff — moving from today's single unvoted model to the proposed 3-model weighted-voting ensemble is a win on **both axes simultaneously**, on every bucket size tested, before Scenario 1 vs. Scenario 2 or two-stage routing even enters the discussion. The sections below compare the two proposed *replacement* architectures against each other; both already beat current production decisively.

### 10.2 Raw Speed

**Per-model latency** (from §1.2): `langdetect` is consistently the slowest model — 2.10–5.52 ms/call vs. `openlid-v3`'s 0.05–0.24 ms/call. Because detection sits on the synchronous critical path before any AI call (§10.1), this gap is fully additive to end-to-end response time, not diluted by parallel or downstream work.

**Pipeline latency** (parallel execution, bottlenecked by the slowest of the three ensemble members):

| Bucket | S1 pipeline (li+ld+py, bottleneck=`langdetect`) | S2 pipeline (li+ol+py, bottleneck=slowest of li/ol) | S2 speed-up |
|---|---|---|---|
| 1 word | 5.5216 ms | 0.0478 ms | 115.5× |
| 2 words | 4.5003 ms | 0.0477 ms | 94.3× |
| 3–7 words | 3.3995 ms | 0.0766 ms | 44.4× |
| 8–16 words | 2.2760 ms | 0.1627 ms (`lingua-high` is the bottleneck here, not `openlid-v3`) | 14.0× |
| 17–50 words | 2.1027 ms | 0.2435 ms | 8.6× |

If the ensemble runs its three models in parallel per request, Scenario 2's steady-state per-request latency is **8.6–115.5× lower** than Scenario 1's. Note this run's absolute latencies are higher than the 5-language run's (measured in a separate session on the same machine) — see §12's noise caveat — but the *relative* speed-up remains dramatic either way.

**Cold-start / model load time** (one-time cost when the service process starts), re-measured for this run with `lingua-high` restricted to 3 languages:

| Model | Load + warmup time |
|---|---|
| `pycld2` | 0.004 s |
| `lingua-high` (3-lang) | **0.011 s** (down from 0.154 s with 5 languages — a smaller candidate set loads faster) |
| `langdetect` | 0.227 s |
| `openlid-v3` | 1.140 s |

| Scenario | Total cold-start time |
|---|---|
| S1 (lingua + langdetect + pycld2) | ~0.242 s |
| S2 (lingua + openlid-v3 + pycld2) | ~1.155 s |

S2 takes roughly **4.8× longer to become ready** at process startup, driven almost entirely by loading the 1.2 GB `openlid-v3.bin` model.

### 10.3 Accuracy

Summarized from §3, §6, §7, and §9 (full 1,273-case dataset unless noted):

| Metric | Best S1 config | Best S2 config |
|---|---|---|
| MY accuracy | **56.5%** (two_stage_weighted) | 53.0% (S2 hard) |
| ID accuracy | 64.0% (two_stage_agree) | **76.0%** (S2 weighted) |
| Overall (ALL) | 70.3% (two-stage) | **70.8%** (S2 weighted) |
| MY vs. `lingua-high` alone (55.8%) | Matches/exceeds — but only with two-stage routing | Always below |

These per-language figures are unchanged from the 5-language dataset; only the ALL column is lower across the board (since ZH/TA no longer inflate it).

### 10.4 Complexity

| Dimension | Scenario 1 (`langdetect`) | Scenario 2 (`openlid-v3`) |
|---|---|---|
| Model artifact size | ~2.3 MB (bundled in the pip package) | **1.2 GB** (`src/openlid-v3.bin`, shipped separately) |
| Deployment | `pip install langdetect` — no extra asset management | Requires versioning/distributing a gigabyte-scale binary — not a plain `pip install` |
| Cold-start load time | 0.227 s | 1.140 s |
| Preprocessing code | None beyond `seed=0` | Custom `preprocess_openlid()` + a FLORES-200 label→ISO map |
| MY/ID output space | Cannot express `ms` at all (structural bias, §2–4) | Can express `ms`, but is individually weak at choosing it correctly (27.8% accuracy, §7) |

Scenario 2 remains architecturally "cleaner" — no Stage-1/Stage-2 routing needed — but trades that for a much heavier, harder-to-deploy model artifact.

### 10.5 Conclusion — Which Is Preferable?

Restricting the dataset to EN/MY/ID doesn't change the underlying tradeoff from the 5-language evaluation — it just removes two languages that weren't part of the actual problem. We recommend **Scenario 2 Weighted Voting** as the primary architecture for a downstream NLP pipeline's language-detection stage. It wins on overall accuracy (70.8%), ID accuracy (76.0%), and per-request latency (8.6–115.5× lower than Scenario 1 once warm), and its architecture is simpler — a single-stage vote with no routing logic required.

The 1.2 GB model artifact and ~4.8× slower cold start are real, one-time operational costs, paid once at deployment/restart rather than per request — a reasonable trade for the steady-state latency win. The more consequential tradeoff is accuracy-side: Scenario 2's MY accuracy (43.7%) is materially below `lingua-high` alone (55.8%) and below every Scenario 1 two-stage variant (best: 56.5%). This should be mitigated operationally — flag low-confidence MY/ID predictions for downstream context-gathering (§13) — and revisited if production data shows the MY gap causing real harm. In that case, **Scenario 1 Two-Stage Weighted** (56.5% MY, 70.3% ALL) is the immediate fallback, and a `lingua-high` + `openlid-v3`-only Stage 2 (dropping `pycld2`, untested here but motivated by `openlid-v3`'s strong individual ID accuracy of 79.3% and genuine MY diversity from `lingua-high`, κ=0.2254, §7) is worth building as a possible best-of-both option before reintroducing `langdetect`.

---

## 11. Methodology Notes
Majority voting requires two assumptions to succeed:
1.  **Voter independence:** Errors must be uncorrelated.
2.  **Shared output space:** Every voter must be able to express any class label.

When `langdetect` was forced to vote on Malay text, it broke both assumptions simultaneously. Because it structurally could not output `ms`, it acted as an adversarial constant rather than an independent probabilistic judge. Voting ensembles must ensure all models possess a fully aligned class taxonomy before participating in hard or soft voting.

---

## 12. Limitations and Threats to Validity
* **Sample Size:** n=421 MY, n=420 ID, n=432 EN; n=1,273 overall. At n≈420, the 95% Wilson binomial confidence intervals are roughly ±5 percentage points near 50% and narrower near the extremes; at n=1,273 overall, the CI width is roughly ±2.5 percentage points.
* **Single Dataset Evaluation:** The evaluation tests a highly specific, adversarial text distribution (Bahasa Rojak, micro-text, educational terminology). Results may not generalize smoothly to standardized, long-form document classification.
* **Langdetect Non-Determinism:** `langdetect`'s underlying architecture uses non-deterministic initialization, potentially yielding minor variances across executions if seeds are unmanaged (all scripts here pin `seed=0`).
* **Micro-text limitations:** All models struggled on 1-word inputs for MY and ID (peaking at ~49% for `lingua-high`). The ensemble heavily relies on context to break the ambiguity, limiting its utility on isolated colloquial keywords. 1-word cases make up over half the dataset (677/1,273).
* **Aggregate ("ALL") figures are lower than the 5-language report's, by design:** removing ZH/TA — the two languages every model handled at 95–100% — pulls the blended accuracy down across the board (e.g. `lingua-high` ALL: 80.2% → 68.8%). This is not a regression in per-language performance (every EN/MY/ID number is identical to the 5-language run); it simply means ALL now reflects only the genuinely contested part of the problem. Don't compare ALL figures across the two reports without accounting for this.
* **Speed/complexity measurements are single-machine, single-run, and vary between sessions:** §10's cold-start and pipeline-latency figures were re-measured for this run and differ somewhat from the 5-language report's (e.g. `langdetect`'s 1-word latency: 3.68 ms there vs. 5.52 ms here). Treat the absolute numbers as directionally correct rather than precise SLA figures; the relative speed-up between scenarios is large enough (an order of magnitude or more) to be robust to this noise.

### 12b. Accuracy Under Realistic Query-Length Distributions

Every headline figure in this report (70.8% ALL for S2 Weighted, 56.5% MY for S1 Two-Stage Weighted, etc.) is a blend across this dataset's five word-count buckets, weighted by **this test set's own bucket composition** — 677/399/94/73/30 cases (53% single-word). That composition was chosen to stress-test EN/MY/ID ambiguity across text lengths (§1.1), not to mirror real production query-length traffic. Since 1-word inputs are also the bucket every model scores worst on (§12, "Micro-text limitations"), the ALL figure is materially sensitive to how much single-word traffic actually shows up in production.

[`src/benchmark/reweight_by_real_distribution.py`](../src/benchmark/reweight_by_real_distribution.py) recomputes ALL/EN/MY/ID accuracy as a weighted average of the same measured per-bucket accuracy, substituting a different bucket-mix histogram for the raw test-set counts. Below are the report's four key configurations (current production, `lingua-high` alone, S1 Two-Stage Weighted, S2 Weighted) reweighted under three illustrative distributions — **these histograms are illustrative, not measured from real LIAM traffic**; swap in a real query-length histogram via `--histogram` once one is available:

* **`chat_short`** — mostly 1–2 word lookups (45%/35%/15%/4%/1% across the five buckets)
* **`query_typical`** — mostly 3–16 word queries (5%/10%/45%/30%/10%)
* **`long_form`** — mostly 8–50 word content (2%/3%/15%/35%/45%)

| Distribution | Strategy | EN | MY | ID | ALL |
|---|---|---|---|---|---|
| test_set (published figures) | langdetect | 42.8% | 0.0% | 44.3% | 29.1% |
| test_set (published figures) | lingua-high | 92.1% | 55.8% | 57.9% | 68.8% |
| test_set (published figures) | S1 two_stage_weighted | 92.4% | 56.5% | 61.4% | 70.3% |
| test_set (published figures) | **S2 weighted** | 92.1% | 43.7% | **76.0%** | **70.8%** |
| chat_short | langdetect | 46.4% | 0.0% | 47.8% | 31.5% |
| chat_short | lingua-high | 92.5% | 58.0% | 58.4% | 69.8% |
| chat_short | S1 two_stage_weighted | 93.0% | 58.9% | 63.7% | 72.1% |
| chat_short | **S2 weighted** | 92.9% | 46.2% | **78.2%** | **72.6%** |
| query_typical | langdetect | 87.8% | 0.0% | 87.1% | 58.5% |
| query_typical | lingua-high | 97.4% | 71.9% | 74.9% | 81.6% |
| query_typical | S1 two_stage_weighted | 98.9% | **79.4%** | 87.7% | 88.8% |
| query_typical | **S2 weighted** | 98.9% | 79.0% | **96.5%** | **91.5%** |
| long_form | langdetect | 95.9% | 0.0% | 95.6% | 64.1% |
| long_form | lingua-high | 99.1% | 70.8% | 82.2% | 84.2% |
| long_form | S1 two_stage_weighted | 99.6% | **80.6%** | 87.5% | 89.3% |
| long_form | **S2 weighted** | 99.6% | 85.7% | **98.8%** | **94.7%** |

**Takeaway:** the 70.8%/56.5% headline figures are bucket-composition-dependent, and the direction of the dependency matters for the S2-vs-Two-Stage decision (§9):
* Under `query_typical` and `long_form` — arguably the more realistic shapes for a downstream NLP query, since 1-word isolated input is an edge case for most product surfaces — **the MY gap between S2 Weighted and S1 Two-Stage Weighted shrinks to 0.4pp (query_typical) or even reverses in S2's favor (long_form: 85.7% vs. 80.6%)**, while S2's ALL/ID lead widens. This weakens the case for defaulting to Two-Stage purely to protect MY, *if* production queries are mostly multi-word.
* Under `chat_short` — closer to this test set's own composition — the MY gap is smaller than the published 12.8pp (43.7% vs. 56.5%) but still real (46.2% vs. 58.9%, a 12.7pp gap), because single-word MY/ID disambiguation is where every strategy is weakest.
* **This does not change the recommendation in §9** — it should inform which branch of the decision matrix applies. If real LIAM query-length data shows queries are predominantly single-word or two-word (like this test set), the MY caveat in §9 applies at close to full published strength. If real queries are predominantly multi-word, the MY caveat is smaller than published, and S2 Weighted's case strengthens further. **This is exactly the kind of production data this report cannot supply on its own** — re-run `reweight_by_real_distribution.py --histogram <real_query_length_histogram.json>` once real query-length telemetry exists, rather than trusting either the published bucket mix or the illustrative distributions above as ground truth.

---

## 13. Integration Notes for Downstream Deployment
* **Placement:** Implement the ensemble as an early stage of the application's NLP/perception pipeline, ahead of any language-dependent processing.
* **Execution:** Language detection should be a synchronous, blocking step in that pipeline — it must complete before any later stages (intent classification, entity extraction, strategy selection, etc.) run, since those stages need the detected language as input context. Run the three models (`lingua-high`, `openlid-v3`, `pycld2`) in parallel within this step, as they are in-process memory calls. Ensure `openlid-v3.bin` (1.2 GB) is loaded once at process startup, not per-request — cold-start load time is ~1.1–1.5 s depending on OS file-cache state (§10.2).
* **Micro-Text & MY/ID Handling:** The models peak at ~47–56% accuracy for single-word MY/ID inputs, and Scenario 2's MY accuracy (43.7% overall) is a known weak point relative to `lingua-high` alone (55.8%, §10.3). Unconditionally flag any 1-word MY or ID predictions — and any MY prediction generally — with a `low_confidence` tag to trigger downstream context-gathering. Track MY-specific accuracy in production; if it proves too weak in practice, fall back to Scenario 1 Two-Stage Weighted (§9).
* **Output Mapping:** Ensure the resulting majority consensus maps cleanly to a BCP-47 tag (e.g., `ms-MY`, `id-ID`, `en-MY`) before passing it to subsequent processing layers. Since this evaluation excluded ZH/TA, confirm separately (e.g. from the 5-language archived report) that the deployed ensemble still handles those languages acceptably if actual production traffic includes them.
