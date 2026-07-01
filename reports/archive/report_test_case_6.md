# Language Detection Engine — Final Consolidated Report

> **Dataset: `test_case_6.txt` — 475 cases (95/language).** This is an archived copy, superseded by the `test_case_7` run. See `Language_Detection_Engine_Final_Report_test_case_7.md` for the current report.

## Executive Summary

This report evaluates seven language detection libraries across 475 complex Southeast Asian text cases to design the core routing engine for the project's Perception Layer (Stage 2). Initial benchmarking identified three models—`lingua-high`, `langdetect`, and `pycld2`—that exhibited complementary failure modes. We hypothesized that a majority-vote ensemble of these three would resolve their individual weaknesses, particularly the deep ambiguity between Malay (MY) and Indonesian (ID).

However, empirical voting results revealed a structural failure: while voting improved ID accuracy, it significantly degraded MY accuracy, dropping it from 65.3% to 58.9% under hard voting. Rigorous diagnosis using McNemar's significance testing, Cohen's kappa, and calibration analysis proved this degradation was not random noise. The root cause was `langdetect`'s lack of a Malay profile; it deterministically cast an overconfident "proxy" vote for Indonesian on every Malay text, structurally biasing the election and overriding `lingua-high`'s correct predictions.

To resolve this, we designed and validated two architectural fixes. The first fix, a Two-Stage Voting mechanism, isolated the fine MS/ID decision to only models capable of expressing both labels, successfully restoring MY accuracy. The second fix (Scenario 2) replaced `langdetect` with a rehabilitated `openlid-v3`, which had initially been eliminated due to a FLORES-200 mapping bug. Once corrected, `openlid-v3` achieved a genuine three-way vote on the MS/ID axis without structural bias.

Based on these findings, we recommend deploying **Scenario 2 Hard Voting (`lingua-high` + `openlid-v3` + `pycld2`)** or **Scenario 1 Two-Stage Agree (`lingua-high` + `langdetect` + `pycld2`)**. Both achieve peak ensemble performance (~88.2–88.6% overall accuracy), gracefully handle Bahasa Rojak and micro-text, and ensure robust structural independence.

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
10. [Methodology Notes](#10-methodology-notes)
11. [Limitations and Threats to Validity](#11-limitations-and-threats-to-validity)
12. [Integration Notes for the project](#12-integration-notes-for-project-v2)

---

## 1. Benchmark Methodology & Individual Model Performance

### 1.1 Methodology and Dataset
Seven models were benchmarked: `langdetect`, `lingua-low`, `lingua-high`, `langid`, `fasttext`, `openlid-v3`, and `pycld2`.
The test set (`test_case_6.txt`) consists of 475 cases, evenly split with 95 cases per language (EN, MY, ID, ZH, TA), structured into five word-count buckets.
* **1 word (150 cases):** 50% formal roots, 50% exclusive colloquialisms.
* **2 words (150 cases):** 50% shared/ambiguous, 50% dialect-specific.
* **3–7 words (73 cases):** Authentic Bahasa Rojak, code-switching patterns.
* **8–16 words (53 cases):** Localized slang, multi-word educational phrases.
* **17–50 words (49 cases):** Full sentences, educational content, paragraphs.

Scoring relied strictly on exact-match BCP-47 ISO codes. No fallbacks or proxies were applied.

### 1.2 Raw Processing Speed
Measured in milliseconds per call, averaged across 100 warm-started repetitions per bucket.

| Bucket | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 | Fastest |
|---|---|---|---|---|---|---|---|---|---|
| **1 word** | 150 | 3.1130 ms | 0.0094 ms | 0.0119 ms | 0.0239 ms | 0.0034 ms | 0.0406 ms | **0.0016 ms** | `pycld2` |
| **2 words** | 150 | 2.6865 ms | 0.0143 ms | 0.0178 ms | 0.0264 ms | 0.0042 ms | 0.0433 ms | **0.0016 ms** | `pycld2` |
| **3–7 words** | 73 | 2.2784 ms | 0.0350 ms | 0.0414 ms | 0.0336 ms | 0.0067 ms | 0.0553 ms | **0.0022 ms** | `pycld2` |
| **8–16 words** | 53 | 2.2917 ms | 0.0681 ms | 0.0728 ms | 0.0483 ms | 0.0119 ms | 0.0844 ms | **0.0032 ms** | `pycld2` |
| **17–50 words** | 49 | 2.6512 ms | 0.1315 ms | 0.0764 ms | 0.0692 ms | 0.0209 ms | 0.1485 ms | **0.0049 ms** | `pycld2` |

`pycld2` is the fastest model across every bucket, operating roughly 1,900× faster than `langdetect`.

### 1.3 Accuracy by Bucket
*Note: Due to a lack of a Malay profile, `langdetect` systematically predicted ID for all 95 MY cases, scoring 0.0% on MY under strict exact-match scoring.*

**Bucket 1: 1 Word (n=30 per language)**
| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 30.0% | 73.3% | **90.0%** | 100.0% | 100.0% | 6.7% | 0.0% |
| MY | **0.0%** | 46.7% | **56.7%** | 0.0% | 10.0% | 10.0% | 16.7% |
| ID | 26.7% | 56.7% | **63.3%** | 0.0% | 16.7% | 36.7% | 20.0% |
| ZH | 50.0% | **100.0%** | **100.0%** | 100.0% | 60.0% | 0.0% | 0.0% |
| TA | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 2: 2 Words (n=30 per language)**
| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 50.0% | 90.0% | **100.0%** | 93.3% | 96.7% | 43.3% | 26.7% |
| MY | **0.0%** | 40.0% | **60.0%** | 13.3% | 20.0% | 23.3% | 23.3% |
| ID | 60.0% | 56.7% | **66.7%** | 33.3% | 30.0% | 46.7% | 13.3% |
| ZH | 46.7% | **100.0%** | **100.0%** | 100.0% | 53.3% | 0.0% | 76.7% |
| TA | 100.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

**Bucket 3: 3–7 Words (n=15 EN/MY/ID/TA, n=13 ZH)**
| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | **100.0%** | 100.0% | **100.0%** | 100.0% | 100.0% | 93.3% | 100.0% |
| MY | **0.0%** | 66.7% | **93.3%** | 53.3% | 26.7% | 73.3% | 53.3% |
| ID | **86.7%** | 60.0% | 60.0% | 60.0% | 53.3% | 73.3% | **93.3%** |
| ZH | 76.9% | **100.0%** | **100.0%** | 100.0% | 92.3% | 0.0% | 100.0% |
| TA | 93.3% | 100.0% | 86.7% | 100.0% | 100.0% | 100.0% | **100.0%** |

**Bucket 4: 8–16 Words (n=10 EN/MY/ID/TA, n=12 ZH)**
| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | **0.0%** | 60.0% | 60.0% | 80.0% | 50.0% | **100.0%** | 80.0% |
| ID | 100.0% | 70.0% | **100.0%** | 70.0% | 80.0% | 80.0% | 90.0% |
| ZH | 25.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 0.0% | 100.0% |
| TA | 100.0% | 100.0% | 90.9% | 100.0% | 100.0% | 100.0% | **100.0%** |

**Bucket 5: 17–50 Words (n=10 EN/MY/ID/ZH, n=9 TA)**
| LANG | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|
| EN | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |
| MY | **0.0%** | 80.0% | 70.0% | 60.0% | 30.0% | 80.0% | **100.0%** |
| ID | **100.0%** | 80.0% | 80.0% | 70.0% | 100.0% | 90.0% | **100.0%** |
| ZH | 10.0% | **100.0%** | **100.0%** | 100.0% | 100.0% | 0.0% | 100.0% |
| TA | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% | 100.0% | 100.0% |

### 1.4 Overall Accuracy & Confidence Intervals (95% Wilson)
The table below integrates the overall accuracy alongside the 95% Wilson binomial confidence intervals.

| LANG | n | langdetect | lingua-low | lingua-high | langid | fasttext | openlid-v3 | pycld2 |
|---|---|---|---|---|---|---|---|---|
| **EN** | 95 | 62.1% [52.1-71.1] | 88.4% | **96.8% [91.1–98.9]** | 97.9% | 98.9% | 51.6% | 45.3% [35.6-55.3] |
| **MY** | 95 | **0.0% [0.0–3.9]** | 52.6% | **65.3% [55.3–74.1]** | 27.4% | 22.1% | 41.1% | 40.0% [30.7-50.1] |
| **ID** | 95 | 62.1% [52.1-71.1] | 61.1% | **69.5% [59.6–77.8]** | 34.7% | 42.1% | 55.8% | 45.3% [35.6-55.3] |
| **ZH** | 95 | 45.3% [35.6-55.3] | **100.0%** | **100.0% [96.1–100]** | 100.0% | 71.6% | 0.0% | 61.1% [51.0-70.2] |
| **TA** | 95 | 98.9% [94.3–99.8] | **100.0%** | 96.8% [91.1–98.9] | 100.0% | 100.0% | 100.0% | 100.0% [96.1-100] |
| **ALL** | 475 | 53.7% [49.2-58.1] | 80.4% | **85.7% [82.2–88.5]** | 72.0% | 66.9% | 49.7% | 58.3% [53.8-62.7] |

### 1.5 ROC AUC Analysis
Computed as binary one-vs-rest AUC using each model's top confidence score.
* **`pycld2` (0.9634):** The most discriminative. Despite lower overall accuracy (58.3%), its high AUC indicates it knows when it doesn't know—reporting low confidence when wrong.
* **`lingua-high` (0.8503):** Strong balanced performance.
* **`langdetect` (0.7516):** Poorer calibration, tending towards high confidence regardless of accuracy.

---

## 2. Model Selection & Initial Ensemble Design Hypothesis

Based on the individual benchmarks, four models were eliminated:
* `lingua-low` (dominated by `lingua-high`).
* `langid` and `fasttext` (collapsed on MY and ID).
* `openlid-v3` (initially eliminated due to 0% Chinese accuracy and a 49.7% overall score. *Note: As discussed in §7, this was later discovered to be a FLORES-200 mapping bug, and the model was successfully rehabilitated*).

Three models were selected for a voting ensemble: **`lingua-high`**, **`langdetect`**, and **`pycld2`**. We hypothesized that this trio offered complementary failure modes that would cancel each other out in a majority vote.

**The Initial Hypothesis for MY/ID Resolution:**
Because `langdetect` lacks a Malay profile, it outputs Indonesian (`id`) for true Malay (`ms`) 100% of the time (95/95 cases). Originally, we hypothesized that this systematic `id` proxy output would be safely overridden by the majority consensus of `lingua-high` and `pycld2`. We assumed that when true Malay text was processed, `lingua-high` and `pycld2` would both vote `ms`, successfully winning the election 2-to-1 against `langdetect`.

---

## 3. Initial Voting Results (The Failure)

We tested three ensemble voting strategies on the trio (lingua-high + langdetect + pycld2):
* **Hard Vote:** Simple majority class vote.
* **Soft Vote:** Averaging the output probability/confidence vectors.
* **Weighted Vote:** AUC-weighted probability averaging.

Contrary to our hypothesis, the voting ensemble did not resolve the MY/ID ambiguity; it actively harmed MY detection.

| LANG | lingua-high (Baseline) | 3-model Hard | 3-model Soft | 3-model Weighted |
|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | 96.8% |
| **MY** | **65.3%** | **58.9%** | **42.1%** | **50.5%** |
| ID | 69.5% | 78.9% | 86.3% | 84.2% |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 96.8% | 100.0% | 98.9% | 98.9% |
| **ALL** | 85.7% | 86.9% | 84.8% | 86.1% |

While ID accuracy jumped significantly (+9.4 percentage points under hard voting), **MY accuracy plummeted from 65.3% to 58.9% (hard) and 42.1% (soft)**.

---

## 4. Diagnosis of the Voting Failure

To understand why the hypothesis failed, we evaluated the results using multiple statistical frameworks.

### 4.1 Statistical Significance (McNemar's Test)
McNemar's exact binomial test confirmed the drop in MY accuracy was not random noise:
* **MY drop:** `lingua-high` → hard vote (−6.4 pp) is **Significant** (p = 0.0312).
* **MY drop:** `lingua-high` → soft vote (−23.2 pp) is **Highly Significant** (p < 0.0001).

### 4.2 Inter-Model Agreement and Diversity (Cohen's Kappa)
Cohen's Kappa (κ) exposed a critical structural problem. `lingua-high` and `langdetect` showed a **negative kappa (κ = −0.053)** on MY. They agreed on only 18.9% of cases—less than chance. The assumption of "voter independence" was broken because `langdetect`'s `id` vote was not a probabilistic judgment; it was a constant, adversarial signal.

### 4.3 Calibration Analysis (Expected Calibration Error - ECE)
`langdetect` emerged as the worst-calibrated model (ECE = 0.1636), reporting a mean confidence of 0.927 even when structurally forced to output a wrong label. In soft and weighted voting, `langdetect` injected a massive, overconfident `id` signal into the probability average, explaining why soft voting degraded MY the most severely (down to 42.1%).

### 4.4 The Failed Assumption
Our initial assumption—that `langdetect` would simply be outvoted—failed because `pycld2` often abstains or outputs `unknown` on micro-text. When `pycld2` abstains or is uncertain, the hard vote triggers a 1-vs-1 tie. Because `langdetect` is guaranteed to vote `id`, any slight hesitation from `pycld2` allowed the tied vote to default to `id`. The 3-way election was structurally rigged as a 2-voter election plus a guaranteed `id` ballot.

---

## 5. Ablation Study

To prove `langdetect` was the source of the failure, we performed a leave-one-model-out ablation on the full dataset.

| LANG | lingua-high | 3m-hard (with LD) | no-LD hard | no-LD soft | no-LD weighted |
|---|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | 96.8% | 96.8% |
| **MY** | **65.3%** | 58.9% | **66.3%** | **66.3%** | **68.4%** |
| ID | 69.5% | 78.9% | 78.9% | 78.9% | 78.9% |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 96.8% | 100.0% | 96.8% | 96.8% | 96.8% |
| **ALL** | 85.7% | 86.9% | **87.8%** | **87.8%** | **88.2%** |

Removing `langdetect` recovered MY accuracy entirely (reaching 68.4% under weighted vote) without sacrificing the ID gains (holding steady at 78.9%).

---

## 6. Fix 1: Two-Stage Voting Architecture

Rather than removing `langdetect` and losing its robust signal on English and Tamil, we implemented a Two-Stage Voting mechanism to safely integrate it.

**Design:**
* **Stage 1:** Coarse classification. `langdetect`'s `id` vote is mapped to a broad "MSID" (Malay/Indonesian) class. All three models hard-vote on {en, MSID, zh, ta}. If the outcome is not MSID, the decision is final.
* **Stage 2:** Fine MS/ID decision. Triggered only if Stage 1 yields MSID. This stage strictly uses `lingua-high` and `pycld2`.
    * *two_stage_agree:* If they agree, output the consensus. If they disagree, use the model with higher confidence.

**Results:**
| LANG | lingua-high | 3m-hard | **2s-agree** | **2s-weighted** |
|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | 96.8% |
| **MY** | 65.3% | 58.9% | **67.4%** | 58.9% |
| **ID** | 69.5% | 78.9% | **78.9%** | **84.2%** |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 96.8% | 100.0% | **100.0%** | **100.0%** |
| **ALL** | 85.7% | 86.9% | **88.6%** | **88.0%** |

`two_stage_agree` fixed the MY/ID bias. It achieved **67.4% MY** (better than `lingua-high`) while holding **78.9% ID**. McNemar testing verified that the MY improvement over the 3-model hard vote (+8.5 pp) was highly significant (p = 0.0078).

---

## 7. Fix 2: Scenario 2 (Replacing langdetect with openlid-v3)

We discovered that `openlid-v3`'s initial 0% score on Chinese (ZH) was not a model failure, but a mapping bug: the FLORES-200 `cmn_Hans` tag was missing from our ISO mapper. Upon fixing the mapping, `openlid-v3` achieved 98.9% on ZH and 81.9% overall.

We evaluated a new ensemble (Scenario 2) replacing `langdetect` with `openlid-v3` (`lingua-high` + `openlid-v3` + `pycld2`). Crucially, `openlid-v3` natively supports Malay (`ms`), achieving 43.2% MY accuracy individually.

**Scenario 2 Results:**
| LANG | S1 hard (langdetect) | S2 hard (openlid) | S2 soft | S2 weighted |
|---|---|---|---|---|
| EN | 96.8% | **96.8%** | 95.8% | 95.8% |
| **MY** | 58.9% | **65.3%** | 56.8% | 57.9% |
| ID | 78.9% | 78.9% | 86.3% | **87.4%** |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 100.0% | 100.0% | 100.0% | 100.0% |
| **ALL** | 86.9% | **88.2%** | 87.8% | **88.2%** |

Scenario 2 Hard Voting enables a genuine 3-way competition without the need for two-stage workarounds. McNemar testing confirmed S2 Hard significantly outperformed S1 Hard on MY (+6.3 pp, p = 0.0312). `openlid-v3` is also ~75× faster than `langdetect` (0.04 ms/call vs 3.1 ms).

---

## 8. Train/Test Leakage Check
To verify that the weights used in weighted voting (derived from the dataset's ROC AUC) were not overfitting, we conducted a split validation (60% dev, 40% test).

| Strategy | Held-Out Test Accuracy (n=190) | Note |
|---|---|---|
| weighted (global AUC) | 85.8% | weights from full 475 — leaked |
| weighted (dev AUC) | 84.7% | weights from dev set only — honest |

The gap (−1.05 pp) falls within standard sampling noise, confirming there is no material overfitting from computing weights globally.

---

## 9. Master Summary Table & Final Recommendation

| Strategy | Ensemble | EN | MY | ID | ZH | TA | ALL |
|---|---|---|---|---|---|---|---|
| lingua-high (individual) | — | 96.8% | 65.3% | 69.5% | 100.0% | 96.8% | 85.7% |
| S1 hard | ld+li+py | 96.8% | 58.9% | 78.9% | 100.0% | 100.0% | 86.9% |
| S1 weighted | ld+li+py | 96.8% | 50.5% | 84.2% | 100.0% | 98.9% | 86.1% |
| **S1 two_stage_agree** | ld+li+py | 96.8% | **67.4%** | 78.9% | 100.0% | **100.0%** | **88.6%** |
| **S2 hard (RECOMMENDED)** | ol+li+py | **96.8%** | 65.3% | 78.9% | 100.0% | **100.0%** | **88.2%** |
| S2 weighted | ol+li+py | 95.8% | 57.9% | **87.4%** | 100.0% | **100.0%** | 88.2% |

**Final Recommendation:** Deploy **Scenario 2 Hard Voting (`lingua-high` + `openlid-v3` + `pycld2`)**. It provides the cleanest architectural solution. Because `openlid-v3` can express all required classes, a standard majority vote functions as intended, removing the need for a complex two-stage pipeline. It tied for best overall accuracy (88.2%) while improving pipeline latency. If `openlid-v3` dependencies pose issues, **S1 two_stage_agree** is an equally capable fallback.

---

## 10. Methodology Notes
Majority voting requires two assumptions to succeed:
1.  **Voter independence:** Errors must be uncorrelated.
2.  **Shared output space:** Every voter must be able to express any class label.

When `langdetect` was forced to vote on Malay text, it broke both assumptions simultaneously. Because it structurally could not output `ms`, it acted as an adversarial constant rather than an independent probabilistic judge. Voting ensembles must ensure all models possess a fully aligned class taxonomy before participating in hard or soft voting.

---

## 11. Limitations and Threats to Validity
* **Sample Size:** The dataset is heavily curated but small (n=95 per language, n=475 overall). At n=95, the 95% Wilson binomial confidence intervals are relatively wide (roughly ±10 percentage points near 50%, and ±5 percentage points near 95%).
* **Single Dataset Evaluation:** The evaluation tests a highly specific, adversarial text distribution (Bahasa Rojak, micro-text, educational terminology). Results may not generalize smoothly to standardized, long-form document classification.
* **Langdetect Non-Determinism:** `langdetect`'s underlying architecture uses non-deterministic initialization, potentially yielding minor variances across executions if seeds are unmanaged.
* **Micro-text limitations:** All models struggled on 1-word inputs for MY and ID (peaking at ~63%). The ensemble heavily relies on context to break the ambiguity, limiting its utility on isolated colloquial keywords.

---

## 12. Integration Notes for the project
* **Placement:** Implement the ensemble inside `app/services/perception/` (Stage 2 NLP).
* **Execution:** Run the three models (`lingua-high`, `openlid-v3`, `pycld2`) in parallel, as they are in-process memory calls. Total latency will be bottlenecked by `lingua-high` (~0.07 ms) and `openlid-v3` (~0.15 ms), which easily satisfies standard async pipeline requirements.
* **Micro-Text Handling:** The models peak at ~56–66% accuracy for single-word MY/ID inputs. Unconditionally flag any 1-word MY or ID predictions with a `low_confidence` tag to trigger downstream context-gathering.
* **Output Mapping:** Ensure the resulting majority consensus maps cleanly to a BCP-47 tag (e.g., `ms-MY`, `id-ID`, `en-MY`, `zh-Hans`) before passing it to subsequent processing layers.
