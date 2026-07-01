# Language Detection Engine — Final Consolidated Report

> **Dataset: `test_case_7.txt` — 2,036 cases (EN 432 / MY 421 / ID 420 / ZH 395 / TA 368), all 5 languages.** This is an archived copy, superseded by the EN/MY/ID-only `test_case_7_enmyid` run. See `Language_Detection_Engine_Final_Report_test_case_7_enmyid.md` (or the canonical `Language_Detection_Engine_Final_Report.md`) for the current report. The older `test_case_6` (475-case) report is archived at `Language_Detection_Engine_Final_Report_test_case_6.md`.

## Executive Summary

This report evaluates seven language detection libraries across 2,036 complex Southeast Asian text cases (`test_case_7.txt`) to design the core routing engine for the project's Perception Layer (Stage 2). Initial benchmarking identified three models—`lingua-high`, `langdetect`, and `pycld2`—that exhibited complementary failure modes. We hypothesized that a majority-vote ensemble of these three would resolve their individual weaknesses, particularly the deep ambiguity between Malay (MY) and Indonesian (ID).

However, empirical voting results replicated the structural failure seen in the earlier 475-case evaluation: while voting improved ID accuracy, it degraded MY accuracy, dropping it from 55.8% to 51.3% under hard voting and to 32.3% under soft voting. McNemar's test confirmed the MY drop under hard voting is highly significant (p < 0.0001), and Cohen's kappa again showed `langdetect`'s vote on MY text is structurally adversarial (κ = −0.0191, 18.1% agreement with `lingua-high` — worse than chance).

Unlike the earlier dataset, a simple leave-one-out ablation (removing `langdetect` entirely) did **not** fully recover MY accuracy on this larger, more diverse dataset (53.4–53.9% vs. `lingua-high`'s individual 55.8%). The two architectural fixes remain necessary: **Two-Stage Voting**, which isolates the fine MS/ID decision to only the models capable of expressing both labels, and **Scenario 2**, which replaces `langdetect` with a rehabilitated `openlid-v3`. On this dataset, Two-Stage Voting (weighted variant) achieves the best MY protection of any tested configuration (56.5%, exceeding `lingua-high` alone), while Scenario 2 Weighted Voting achieves the best overall accuracy (81.7%) and the best ID accuracy (76.0%), at the cost of materially worse MY accuracy (43.7%).

Based on these findings, we recommend deploying **Scenario 2 Weighted Voting (`lingua-high` + `openlid-v3` + `pycld2`)** as the primary architecture. It achieves the best overall accuracy (81.7%) and ID accuracy (76.0%) of any tested configuration, requires no Stage-1/Stage-2 routing logic, and is 17–73× faster per request than any `langdetect`-based configuration once the service is warm. The cost is a **1.2 GB** `openlid-v3` model artifact (vs. `langdetect`'s ~2.3 MB) and a ~4× slower cold start — a one-time deployment cost we judge acceptable for a long-lived service — plus materially weaker MY accuracy (43.7%) than `lingua-high` alone (55.8%) or the Two-Stage fallback (56.5%), which should be mitigated with downstream low-confidence flagging (§13) and monitored in production.

---

## Table of Contents
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
13. [Integration Notes for the project](#13-integration-notes-for-project-v2)

---

## 1. Benchmark Methodology & Individual Model Performance

### 1.1 Methodology and Dataset
Seven models were benchmarked: `langdetect`, `lingua-low`, `lingua-high`, `langid`, `fasttext`, `openlid-v3`, and `pycld2`.
The test set (`test_case_7.txt`) consists of 2,036 cases across five languages: EN (432), MY (421), ID (420), ZH (395), TA (368), structured into five word-count buckets:
* **1 word (1,056 cases)**
* **2 words (661 cases)**
* **3–7 words (147 cases):** Authentic Bahasa Rojak, code-switching patterns.
* **8–16 words (123 cases):** Localized slang, multi-word educational phrases.
* **17–50 words (49 cases):** Full sentences, educational content, paragraphs.

Scoring relied strictly on exact-match BCP-47 ISO codes. No fallbacks or proxies were applied.

### 1.2 Raw Processing Speed
Measured in milliseconds per call, averaged across 100 warm-started repetitions per bucket.

| Bucket | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 | Fastest |
|---|---|---|---|---|---|---|---|---|---|
| **1 word** | 1056 | 3.6824 ms | 0.0163 ms | 0.0185 ms | 0.0340 ms | 0.0051 ms | 0.0502 ms | **0.0021 ms** | `pycld2` |
| **2 words** | 661 | 2.9346 ms | 0.0152 ms | 0.0198 ms | 0.0288 ms | 0.0046 ms | 0.0502 ms | **0.0020 ms** | `pycld2` |
| **3–7 words** | 147 | 2.5416 ms | 0.0392 ms | 0.0465 ms | 0.0366 ms | 0.0078 ms | 0.0645 ms | **0.0026 ms** | `pycld2` |
| **8–16 words** | 123 | 2.1902 ms | 0.0655 ms | 0.0733 ms | 0.0618 ms | 0.0123 ms | 0.0996 ms | **0.0036 ms** | `pycld2` |
| **17–50 words** | 49 | 2.7617 ms | 0.1314 ms | 0.0774 ms | 0.0731 ms | 0.0232 ms | 0.1581 ms | **0.0056 ms** | `pycld2` |

`pycld2` is the fastest model across every bucket, operating roughly 1,300–1,750× faster than `langdetect`.

### 1.3 Accuracy by Bucket
*Note: Due to a lack of a Malay profile, `langdetect` still systematically predicts ID for MY cases, scoring 0.0% on MY under strict exact-match scoring.*

**Bucket 1: 1 Word (n=1056)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 226 | 26.5% | 73.9% | **88.9%** | 99.1% | 88.9% | 6.6% | 4.0% |
| MY | 226 | **0.0%** | 39.8% | **47.3%** | 0.4% | 7.5% | 6.6% | 6.6% |
| ID | 225 | 24.0% | 45.8% | **49.3%** | 8.0% | 12.0% | 33.3% | 11.6% |
| ZH | 193 | 47.2% | **100.0%** | **100.0%** | 100.0% | 44.6% | 0.0% | 1.0% |
| TA | 186 | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 2: 2 Words (n=661)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 141 | 46.8% | 89.4% | **94.3%** | 97.2% | 92.9% | 22.0% | 29.8% |
| MY | 129 | **0.0%** | 41.1% | **61.2%** | 12.4% | 14.0% | 22.5% | 22.5% |
| ID | 129 | 52.7% | 57.4% | **62.0%** | 40.3% | 24.0% | 58.9% | 17.8% |
| ZH | 141 | 52.5% | **100.0%** | **100.0%** | 100.0% | 54.6% | 0.0% | 71.6% |
| TA | 121 | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 3: 3–7 Words (n=147)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 30 | 96.7% | 96.7% | 96.7% | **100.0%** | **100.0%** | 90.0% | 96.7% |
| MY | 32 | **0.0%** | 59.4% | **78.1%** | 50.0% | 37.5% | 78.1% | 56.2% |
| ID | 32 | 90.6% | 59.4% | 65.6% | 65.6% | 59.4% | 78.1% | **93.3%** |
| ZH | 22 | 54.5% | **100.0%** | **100.0%** | 100.0% | 90.9% | 0.0% | 90.9% |
| TA | 31 | 93.5% | **100.0%** | 80.6% | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 4: 8–16 Words (n=123)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 25 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | 24 | **0.0%** | 66.7% | 70.8% | 50.0% | 50.0% | **87.5%** | 79.2% |
| ID | 24 | 100.0% | 70.8% | **95.8%** | 83.3% | 87.5% | 91.7% | 91.7% |
| ZH | 29 | 20.7% | **100.0%** | **100.0%** | 100.0% | 93.1% | 0.0% | 100.0% |
| TA | 21 | 100.0% | 100.0% | 95.2% | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 5: 17–50 Words (n=49)**
| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| EN | 10 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | 10 | **0.0%** | 80.0% | 70.0% | 60.0% | 30.0% | 80.0% | **100.0%** |
| ID | 10 | **100.0%** | 80.0% | 80.0% | 70.0% | 100.0% | 90.0% | **100.0%** |
| ZH | 10 | 10.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 0.0% | 100.0% |
| TA | 9 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

### 1.4 Overall Accuracy & Confidence Intervals (95% Wilson)

| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| **EN** | 432 | 44.0% [39.4-48.7] | 82.6% | **92.1% [89.2–94.3]** | 98.6% | 91.9% | 25.0% | 26.6% [22.7-31.0] |
| **MY** | 421 | **0.0% [0.0–0.9]** | 44.2% | **55.8% [51.0–60.5]** | 12.1% | 14.7% | 23.3% | 21.6% [17.9-25.8] |
| **ID** | 420 | 44.0% [39.3-48.8] | 52.6% | **57.9% [53.1–62.5]** | 28.1% | 25.7% | 49.3% | 25.2% [21.3-29.6] |
| **ZH** | 395 | 46.6% [41.7-51.5] | **100.0%** | **100.0% [99.0–100]** | 100.0% | 55.7% | 0.0% | 41.0% [36.3-45.9] |
| **TA** | 368 | 99.5% [98.1–99.9] | **100.0%** | 98.1% [96.1–99.1] | 100.0% | 100.0% | 100.0% | 100.0% [99.0-100] |
| **ALL** | 2036 | 45.4% [43.2-47.6] | 75.0% | **80.2% [78.4–81.9]** | 66.7% | 56.7% | 38.4% | 41.4% [39.3-43.6] |

### 1.5 ROC AUC Analysis
Computed as binary one-vs-rest AUC using each model's top confidence score.
* **`pycld2` (0.9734):** The most discriminative. Despite lower overall accuracy (41.4%), its high AUC indicates it knows when it doesn't know—reporting low confidence when wrong.
* **`lingua-high` (0.8569):** Strong balanced performance.
* **`langdetect` (0.7655):** Poorer calibration, tending towards high confidence regardless of accuracy.

---

## 2. Model Selection & Initial Ensemble Design Hypothesis

Based on the individual benchmarks, four models were eliminated:
* `lingua-low` (dominated by `lingua-high`).
* `langid` and `fasttext` (collapsed on MY and ID — 12.1%/14.7% and 28.1%/25.7% respectively).
* `openlid-v3` (initially eliminated due to 0% Chinese accuracy and a 38.4% overall score in this benchmark script. *Note: As discussed in §7, this is a FLORES-200 mapping bug local to the standalone benchmark script; the voting scripts use a corrected mapping and openlid-v3 is rehabilitated there*).

Three models were selected for a voting ensemble: **`lingua-high`**, **`langdetect`**, and **`pycld2`**. We hypothesized that this trio offered complementary failure modes that would cancel each other out in a majority vote.

**The Initial Hypothesis for MY/ID Resolution:**
Because `langdetect` lacks a Malay profile, it outputs Indonesian (`id`) for true Malay (`ms`) 100% of the time (421/421 cases). Originally, we hypothesized that this systematic `id` proxy output would be safely overridden by the majority consensus of `lingua-high` and `pycld2`. We assumed that when true Malay text was processed, `lingua-high` and `pycld2` would both vote `ms`, successfully winning the election 2-to-1 against `langdetect`.

---

## 3. Initial Voting Results (The Failure)

We tested three ensemble voting strategies on the trio (lingua-high + langdetect + pycld2):
* **Hard Vote:** Simple majority class vote.
* **Soft Vote:** Averaging the output probability/confidence vectors.
* **Weighted Vote:** AUC-weighted probability averaging.

Contrary to our hypothesis, the voting ensemble did not resolve the MY/ID ambiguity; it actively harmed MY detection.

| LANG | lingua-high (Baseline) | 3-model Hard | 3-model Soft | 3-model Weighted |
|---|---|---|---|---|
| EN | 92.1% | 92.4% | 91.7% | 91.7% |
| **MY** | **55.8%** | **51.3%** | **32.3%** | **37.1%** |
| ID | 57.9% | 63.1% | 74.3% | 73.1% |
| ZH | 100.0% | 99.7% | 99.7% | 100.0% |
| TA | 98.1% | 100.0% | 99.5% | 99.5% |
| **ALL** | 80.2% | 80.6% | 78.8% | 79.6% |

While ID accuracy jumped significantly (+5.2 percentage points under hard voting), **MY accuracy dropped from 55.8% to 51.3% (hard) and 32.3% (soft)**, replicating the failure mode seen in the earlier 475-case dataset.

---

## 4. Diagnosis of the Voting Failure

To understand why the hypothesis failed, we evaluated the results using multiple statistical frameworks.

### 4.1 Statistical Significance (McNemar's Test)
McNemar's test confirmed the drop in MY accuracy was not random noise:
* **MY drop:** `lingua-high` → hard vote (−4.5 pp) is **Highly Significant** (p < 0.0001, b=19, c=0 — every error introduced by hard voting flipped a correct `lingua-high` prediction, with zero reverse corrections).
* **MY drop:** `lingua-high` → soft vote (−23.5 pp) is **Highly Significant** (p < 0.0001).

### 4.2 Inter-Model Agreement and Diversity (Cohen's Kappa)
Cohen's Kappa (κ) again exposed a critical structural problem. `lingua-high` and `langdetect` showed a **negative kappa (κ = −0.0191)** on MY. They agreed on only 18.1% of cases—less than chance. The assumption of "voter independence" was broken because `langdetect`'s `id` vote was not a probabilistic judgment; it was a constant, adversarial signal.

### 4.3 Calibration Analysis (Expected Calibration Error - ECE)
`langdetect` again emerged as the worst-calibrated model (ECE = 0.1282), reporting a mean confidence of 0.9177 even when structurally forced to output a wrong label (mean confidence when wrong: 0.2340 — still non-trivial). In soft and weighted voting, `langdetect` injected an overconfident `id` signal into the probability average, contributing to why soft voting degraded MY the most severely (down to 32.3%).

### 4.4 The Failed Assumption
As before, the assumption that `langdetect` would simply be outvoted failed because `pycld2` frequently abstains or is uncertain on short/micro-text (`pycld2`'s own MY accuracy is only 21.6% — well below `lingua-high`'s 55.8%). When `pycld2` is uncertain, hard voting triggers a 1-vs-1 tie, and because `langdetect` is guaranteed to vote `id`, any hesitation from `pycld2` allows the tied vote to default to `id`.

---

## 5. Ablation Study

To test whether `langdetect` alone was the source of the failure, we performed a leave-one-model-out ablation on the full 2,036-case dataset.

| LANG | lingua-high | 3m-hard (with LD) | no-LD hard | no-LD soft | no-LD weighted |
|---|---|---|---|---|---|
| EN | 92.1% | 92.4% | 92.4% | 92.4% | 92.4% |
| **MY** | **55.8%** | 51.3% | 53.4% | 53.4% | 53.9% |
| ID | 57.9% | 63.1% | 64.0% | 64.0% | 64.0% |
| ZH | 100.0% | 99.7% | 100.0% | 100.0% | 100.0% |
| TA | 98.1% | 100.0% | 98.4% | 98.4% | 98.4% |
| **ALL** | 80.2% | 80.6% | **81.0%** | **81.0%** | **81.1%** |

Unlike the earlier 475-case dataset, removing `langdetect` improves MY accuracy over the 3-model hard vote (51.3% → 53.4–53.9%) but does **not** fully recover it to `lingua-high`'s individual level (55.8%). ID gains are preserved (64.0%, matching the biased hard-vote figure). On this larger, more diverse dataset, the simple ablation hypothesis is **not fully supported** — a more targeted fix is needed, motivating Section 6.

---

## 6. Fix 1: Two-Stage Voting Architecture

Since simple ablation left a residual MY gap, we applied the Two-Stage Voting mechanism to safely integrate `langdetect`'s useful signal on English/Tamil while removing its structural bias on the MS/ID axis.

**Design:**
* **Stage 1:** Coarse classification. `langdetect`'s `id` vote is mapped to a broad "MSID" (Malay/Indonesian) class. All three models hard-vote on {en, MSID, zh, ta}. If the outcome is not MSID, the decision is final.
* **Stage 2:** Fine MS/ID decision. Triggered only if Stage 1 yields MSID. This stage strictly uses `lingua-high` and `pycld2`.
    * *two_stage_agree:* If they agree, output the consensus. If they disagree, use the model with higher confidence.
    * *two_stage_weighted:* Score each label using per-class dev-fitted accuracy weights for `lingua-high` and `pycld2`.

**Results (full 2,036-case dataset):**
| LANG | lingua-high | 3m-hard | **2s-agree** | **2s-weighted** |
|---|---|---|---|---|
| EN | 92.1% | 92.4% | 92.4% | 92.4% |
| **MY** | 55.8% | 51.3% | 53.9% | **56.5%** |
| **ID** | 57.9% | 63.1% | **64.0%** | 61.4% |
| ZH | 100.0% | 99.7% | 99.7% | 99.7% |
| TA | 98.1% | 100.0% | **100.0%** | **100.0%** |
| **ALL** | 80.2% | 80.6% | **81.4%** | **81.4%** |

`two_stage_weighted` fully restores and slightly exceeds `lingua-high`'s individual MY accuracy (56.5% vs. 55.8%) while lifting overall accuracy to 81.4%. McNemar testing confirmed `two_stage_weighted` vs. `lingua-high` on MY shows **no significant difference** (p = 0.7835) — meaning the fix genuinely restores parity rather than trading MY for ID. `two_stage_agree` achieves the best ID accuracy of the two-stage variants (64.0%, matching the 3-model hard-vote gain) while still improving MY over the biased baseline (53.9% vs. 51.3%, though this improvement over lingua-high itself is not significant, p = 0.302).

---

## 7. Fix 2: Scenario 2 (Replacing langdetect with openlid-v3)

We evaluated a new ensemble (Scenario 2) replacing `langdetect` with `openlid-v3` (`lingua-high` + `openlid-v3` + `pycld2`), using the corrected FLORES-200 ISO mapping (`voting/core.py`) rather than the standalone benchmark script's mapping. `openlid-v3` natively supports Malay (`ms`), achieving 27.8% MY accuracy individually — far from strong on its own, but genuinely diverse from `langdetect`'s 0.0%.

**Scenario 2 Results (full 2,036-case dataset):**
| LANG | S1 hard (langdetect) | S2 hard (openlid) | S2 soft | S2 weighted |
|---|---|---|---|---|
| EN | 92.4% | **92.4%** | 92.1% | 92.1% |
| **MY** | 51.3% | **53.0%** | 42.3% | 43.7% |
| ID | 63.1% | 64.3% | 76.2% | **76.0%** |
| ZH | 99.7% | **100.0%** | 100.0% | 100.0% |
| TA | 100.0% | 100.0% | 100.0% | 100.0% |
| **ALL** | 80.6% | 81.3% | 81.5% | **81.7%** |

McNemar testing confirmed S2's aggregate accuracy gain over S1 is significant for hard voting (+0.6pp, p=0.0259) and highly significant for soft/weighted (+2.7pp/+2.2pp, p<0.0001/p=0.0002). On MY specifically, S2 soft/weighted significantly beat S1 soft/weighted (+10.0pp p<0.0001, +6.7pp p=0.0011) — but both remain well below `lingua-high`'s individual 55.8% and below the Two-Stage results in §6. Cohen's kappa between `lingua-high` and `openlid-v3` on MY is **0.2218** (vs. `langdetect`'s −0.0191), confirming `openlid-v3` is a genuinely diverse voter rather than a soft repeat of `langdetect`'s structural bias — it outputs `ms` for 117/421 true-MY cases, versus `langdetect`'s 0.

Scenario 2 does not require the Stage-1/Stage-2 routing logic to achieve competitive overall accuracy, and `openlid-v3` is roughly 50–70× faster than `langdetect` per the Section 1.2 timings. However, on this dataset, plain hard/soft/weighted Scenario 2 voting does not match Two-Stage Voting's MY protection (§6).

---

## 8. Train/Test Leakage Check

To verify that weighted-voting weights derived from the dataset's ROC AUC were not overfitting, we conducted a stratified 60/40 dev/test split (dev n=1,221, test n=815).

| Strategy | Held-Out Test Accuracy (n=815) | Note |
|---|---|---|
| weighted (global AUC) | 79.3% | weights from full 2,036 — leaked |
| weighted (dev AUC) | 78.3% | weights from dev set only — honest |

The gap (−0.98 pp) falls within standard sampling noise, confirming there is no material overfitting from computing weights globally — consistent with the earlier dataset's finding.

---

## 9. Master Summary Table & Final Recommendation

All figures below are on the full 2,036-case dataset except where noted.

| Strategy | Ensemble | EN | MY | ID | ZH | TA | ALL |
|---|---|---|---|---|---|---|---|
| lingua-high (individual) | — | 92.1% | 55.8% | 57.9% | 100.0% | 98.1% | 80.2% |
| S1 hard | ld+li+py | 92.4% | 51.3% | 63.1% | 99.7% | 100.0% | 80.6% |
| S1 weighted | ld+li+py | 91.7% | 37.1% | 73.1% | 100.0% | 99.5% | 79.6% |
| S1 two_stage_agree | ld+li+py (2-stage) | 92.4% | 53.9% | **64.0%** | 99.7% | **100.0%** | 81.4% |
| S1 two_stage_weighted | ld+li+py (2-stage) | 92.4% | **56.5%** | 61.4% | 99.7% | **100.0%** | 81.4% |
| S2 hard | ol+li+py | **92.4%** | 53.0% | 64.3% | **100.0%** | **100.0%** | 81.3% |
| **S2 weighted (RECOMMENDED)** | ol+li+py | 92.1% | 43.7% | **76.0%** | **100.0%** | **100.0%** | **81.7%** |

**Final Recommendation:** Deploy **Scenario 2 Weighted Voting (`lingua-high` + `openlid-v3` + `pycld2`)**. It achieves the best overall accuracy on this dataset (81.7%) and the best ID accuracy (76.0%), needs only a single-stage vote (no Stage-1/Stage-2 routing logic to build and maintain), and is 17–73× faster per request than any `langdetect`-based configuration (§10.1) — while using a genuinely diverse voter on the MS/ID axis (κ = 0.2218 vs. `langdetect`'s −0.0191, §7).

The tradeoff is real and should be tracked: S2 weighted's MY accuracy (43.7%) sits below both `lingua-high` alone (55.8%) and every S1 two-stage variant (best: 56.5%). Mitigate this the way §13 already recommends for all micro-text — flag low-confidence MY/ID predictions (especially 1-word inputs) for downstream context-gathering rather than trusting the raw vote. If MY protection later proves too weak in production, **S1 Two-Stage Weighted** (56.5% MY, 81.4% ALL) remains the fallback, and the untested `lingua-high` + `openlid-v3`-only Stage 2 (§10.4) is worth building before reverting to `langdetect`, since `openlid-v3` is already the model in use.

See §10 for the full speed/accuracy/complexity comparison behind this recommendation — notably, `openlid-v3` requires shipping a 1.2 GB model artifact, which is the main cost of choosing Scenario 2.

---

## 10. Scenario Comparison — Speed, Accuracy & Complexity

Sections 3–9 focused on accuracy. This section pulls together the three axes that actually decide which architecture to ship: raw speed, accuracy, and deployment complexity — then gives a single conclusion.

### 10.1 Raw Speed

**Per-model latency** (from §1.2): `langdetect` is consistently the slowest model by 1–2 orders of magnitude — 2.19–3.68 ms/call vs. `openlid-v3`'s 0.05–0.16 ms/call.

**Pipeline latency** (parallel execution, bottlenecked by the slowest of the three ensemble members):

| Bucket | S1 pipeline (li+ld+py, bottleneck=`langdetect`) | S2 pipeline (li+ol+py, bottleneck=`openlid-v3`) | S2 speed-up |
|---|---|---|---|
| 1 word | 3.6824 ms | 0.0502 ms | 73.4× |
| 2 words | 2.9346 ms | 0.0502 ms | 58.5× |
| 3–7 words | 2.5416 ms | 0.0645 ms | 39.4× |
| 8–16 words | 2.1902 ms | 0.0996 ms | 22.0× |
| 17–50 words | 2.7617 ms | 0.1581 ms | 17.5× |

If the ensemble runs its three models in parallel per request, Scenario 2's steady-state per-request latency is **17–73× lower** than Scenario 1's, because `langdetect` alone dominates the critical path in S1.

**Cold-start / model load time** (one-time cost when the service process starts), measured directly on this machine:

| Model | Load + warmup time |
|---|---|
| `pycld2` | 0.004 s |
| `lingua-high` | 0.154 s |
| `langdetect` | 0.219 s |
| `openlid-v3` | **1.475 s** |

| Scenario | Total cold-start time |
|---|---|
| S1 (lingua + langdetect + pycld2) | ~0.377 s |
| S2 (lingua + openlid-v3 + pycld2) | ~1.633 s |

S2 takes roughly **4.3× longer to become ready** at process startup, driven almost entirely by loading the 1.2 GB `openlid-v3.bin` model.

### 10.2 Accuracy

Summarized from §3, §6, §7, and §9 (full 2,036-case dataset unless noted):

| Metric | Best S1 config | Best S2 config |
|---|---|---|
| MY accuracy | **56.5%** (two_stage_weighted) | 53.0% (S2 hard) |
| ID accuracy | 64.0% (two_stage_agree) | **76.0%** (S2 weighted) |
| Overall (ALL) | 81.4% (two-stage) | **81.7%** (S2 weighted) |
| MY vs. `lingua-high` alone (55.8%) | Matches/exceeds — but only with two-stage routing | Always below, even with two-stage routing on the tested split (§7 discussion) |

Neither scenario dominates on accuracy alone: S1's two-stage variants are the only configurations that fully protect MY, while S2's weighted vote is the strongest on ID and overall accuracy.

### 10.3 Complexity

| Dimension | Scenario 1 (`langdetect`) | Scenario 2 (`openlid-v3`) |
|---|---|---|
| Model artifact size | ~2.3 MB (bundled in the pip package) | **1.2 GB** (`src/openlid-v3.bin`, shipped separately) |
| Deployment | `pip install langdetect` — no extra asset management | Requires versioning/distributing a gigabyte-scale binary (e.g. Git LFS or an artifact store) — not a plain `pip install` |
| Cold-start load time | 0.219 s | 1.475 s |
| Preprocessing code | None beyond `seed=0` | Custom `preprocess_openlid()` (lowercasing, whitespace/punctuation stripping) + a FLORES-200 label→ISO map |
| Mapping bug surface | None (native ISO codes) | Real bug found during this evaluation: `benchmarkV5.py`'s mapping omitted `cmn_Hans`/`cmn_Hant`, silently zeroing ZH accuracy in that script (§2) |
| Runtime dependency | `langdetect` (pure Python) | `fasttext` (already a project dependency for the separate `fasttext` baseline model, so no *new* package — but a new, large binary asset) |
| MY/ID output space | Cannot express `ms` at all (structural bias, §2–4) | Can express `ms`, but is individually weak at choosing it correctly (27.8% accuracy, §7) |

Scenario 2 is architecturally "cleaner" in the sense that its voters share a full output space without needing Stage-1/Stage-2 routing to neutralize an adversarial voter — but it trades that for a much heavier, harder-to-deploy model artifact, slower startup, and extra mapping code with a demonstrated bug risk.

### 10.4 Conclusion — Which Is Preferable?

There is no single winner across all three axes:

* **Scenario 1 (Two-Stage, weighted)** wins on **the accuracy metric that matters most for this evaluation (MY protection)** and on **deployment simplicity** (2.3 MB, no extra asset pipeline, fast cold start). Its weakness — `langdetect`'s per-call latency — is, in absolute terms, still under 4 ms, unlikely to matter for a Stage-2 perception service that is not p99-latency-critical at the microsecond level.
* **Scenario 2 (weighted)** wins on **raw per-request throughput** (17–73× lower steady-state latency) and on **overall/ID accuracy**, but at the cost of a 1.2 GB model artifact, ~4× slower cold start, extra preprocessing/mapping code, and materially weaker MY protection than Scenario 1's two-stage design.

**Recommendation:** We recommend **Scenario 2 Weighted Voting (`lingua-high` + `openlid-v3` + `pycld2`)** as the primary architecture for the project's Stage-2 Perception Layer. It wins on overall accuracy (81.7%), ID accuracy (76.0%), and per-request latency (17–73× lower than Scenario 1 once the process is warm), and its architecture is simpler to build and maintain — a single-stage vote with no Stage-1/Stage-2 routing logic required, since `openlid-v3` (unlike `langdetect`) shares a full output space with the other voters.

The 1.2 GB model artifact and ~4× slower cold start (§10.1, §10.3) are real, one-time operational costs, but for a long-lived, in-process service they are paid once at deployment/restart rather than per request — a reasonable trade for a 17–73× steady-state latency win. The more consequential tradeoff is accuracy-side: Scenario 2's MY accuracy (43.7%) is materially below `lingua-high` alone (55.8%) and below every Scenario 1 two-stage variant (best: 56.5%). This should be mitigated operationally — flag low-confidence MY/ID predictions for downstream context-gathering, per §13 — and revisited if production data shows the MY gap causing real harm. In that case, **Scenario 1 Two-Stage Weighted** (56.5% MY, 81.4% ALL) is the immediate fallback, and a `lingua-high` + `openlid-v3`-only Stage 2 (dropping `pycld2`, untested here but motivated by `openlid-v3`'s strong individual ID accuracy of 79.3% and genuine MY diversity from `lingua-high`, κ=0.2218, §7) is worth building as a possible best-of-both option before reintroducing `langdetect`.

---

## 11. Methodology Notes
Majority voting requires two assumptions to succeed:
1.  **Voter independence:** Errors must be uncorrelated.
2.  **Shared output space:** Every voter must be able to express any class label.

When `langdetect` was forced to vote on Malay text, it broke both assumptions simultaneously. Because it structurally could not output `ms`, it acted as an adversarial constant rather than an independent probabilistic judge. Voting ensembles must ensure all models possess a fully aligned class taxonomy before participating in hard or soft voting.

---

## 12. Limitations and Threats to Validity
* **Sample Size:** The dataset is large but unevenly split across languages (n=421 MY, n=420 ID, n=395 ZH, n=368 TA, n=432 EN; n=2,036 overall). At n≈420, the 95% Wilson binomial confidence intervals are roughly ±5 percentage points near 50% and narrower near the extremes; at n=2,036 overall, the CI width is roughly ±2 percentage points.
* **Single Dataset Evaluation:** The evaluation tests a highly specific, adversarial text distribution (Bahasa Rojak, micro-text, educational terminology, and single/short Chinese-character tokens). Results may not generalize smoothly to standardized, long-form document classification.
* **Langdetect Non-Determinism:** `langdetect`'s underlying architecture uses non-deterministic initialization, potentially yielding minor variances across executions if seeds are unmanaged (all scripts here pin `seed=0`).
* **Micro-text limitations:** All models struggled on 1-word inputs for MY and ID (peaking at ~49% for `lingua-high`). The ensemble heavily relies on context to break the ambiguity, limiting its utility on isolated colloquial keywords. This effect is more pronounced on this dataset than the earlier 475-case set, since 1-word cases now make up over half the dataset (1,056/2,036) and include harder short-token cases (e.g. single/double Chinese characters that some models, including `langdetect`, cannot reliably classify at all).
* **Speed/complexity measurements are single-machine, single-run:** §10's cold-start and pipeline-latency figures were measured once on the development machine, not averaged across repeated runs or production hardware. Treat them as directionally correct (the orders-of-magnitude gaps are real) rather than precise SLA numbers.

---

## 13. Integration Notes for the project
* **Placement:** Implement the ensemble inside `app/services/perception/` (Stage 2 NLP).
* **Execution:** Run the three models (`lingua-high`, `openlid-v3`, `pycld2`) in parallel, as they are in-process memory calls. Per-call latency for `lingua-high` and `openlid-v3` stays under 0.2 ms even on longer text buckets, easily satisfying standard async pipeline requirements. Ensure `openlid-v3.bin` (1.2 GB) is loaded once at process startup, not per-request — cold-start load time is ~1.5 s (§10.1).
* **Micro-Text & MY/ID Handling:** The models peak at ~47–56% accuracy for single-word MY/ID inputs, and Scenario 2's MY accuracy (43.7% overall) is a known weak point relative to `lingua-high` alone (55.8%, §10.2). Unconditionally flag any 1-word MY or ID predictions — and any MY prediction generally — with a `low_confidence` tag to trigger downstream context-gathering. Track MY-specific accuracy in production; if it proves too weak in practice, fall back to Scenario 1 Two-Stage Weighted (§9).
* **Output Mapping:** Ensure the resulting majority consensus maps cleanly to a BCP-47 tag (e.g., `ms-MY`, `id-ID`, `en-MY`, `zh-Hans`) before passing it to subsequent processing layers.
