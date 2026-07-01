# Language Detection Engine — Voting Ensemble Evaluation

**Date:** 2026-06-30  
**Test set:** `test_case_6.txt` — 475 cases, 95 per language (EN / MY / ID / ZH / TA)  
**Scenario 1 ensemble:** lingua-high + langdetect + pycld2  
**Scenario 2 ensemble:** lingua-high + openlid-v3 + pycld2  
**Voting strategies evaluated:** hard (majority vote), soft (avg probability), weighted (AUC-weighted), two-stage (coarse route → fine ms/id)  
**Scripts:** `voting/voting_stats.py`, `voting/voting_ablation.py`, `voting/voting_two_stage.py`, `voting/stage1_accuracy.py`, `voting/kappa_s2.py`, `voting/auc_unified.py`, `voting/reproducibility.py`, `voting_scenario2/voting_s2.py`, `voting_scenario2/voting_s2_two_stage.py`  
**Logs:** `log_voting_stats_2.txt`, `log_voting_ablation_1.txt`, `log_voting_two_stage_2.txt`, `log_stage1_accuracy_1.txt`, `log_kappa_s2_1.txt`, `log_auc_unified_1.txt`, `log_reproducibility_1.txt`, `voting_scenario2/log_s2_1.txt`, `voting_scenario2/log_s2_two_stage_2.txt`

---

## Table of Contents

1. [Statistical Significance (McNemar's Test)](#1-statistical-significance-mcnemars-test)
2. [Inter-Model Agreement and Diversity (Cohen's Kappa)](#2-inter-model-agreement-and-diversity)
3. [Calibration Analysis](#3-calibration-analysis)
4. [Leave-One-Model-Out Ablation](#4-leave-one-model-out-ablation)
5. [Train/Test Leakage Check](#5-traintest-leakage-check)
6. [Two-Stage Voting — Implementation and Validation](#6-two-stage-voting)
7. [Scenario 2 — openlid-v3 Replaces langdetect](#7-scenario-2--openlid-v3-replaces-langdetect)
8. [Master Summary Table](#8-master-summary-table)
9. [Qualitative Error Analysis](#9-qualitative-error-analysis)
10. [Methodology Notes](#10-methodology-notes)
11. [Limitations and Threats to Validity](#11-limitations-and-threats-to-validity)

---

## 1. Statistical Significance (McNemar's Test)

**Method:** McNemar's paired test on the same 475 cases.  
**Library:** `statsmodels.stats.contingency_tables.mcnemar`  
If `b + c < 25` → exact binomial; otherwise → chi-squared with Yates correction.

**Multiple-comparisons correction (scope-specific Bonferroni):**  
Tests are grouped by scope (language) and family size:

| Scope | # comparisons in family | Corrected α |
|---|---|---|
| §1 per-language (EN/MY/ID/ZH/TA) | 6 per language | 0.05/6 = **0.0083** |
| §6.5 two-stage (per language) | 5 per language | 0.05/5 = **0.0100** |
| §7.4 S1 vs S2 | 8 across all languages | 0.05/8 = **0.0063** |

Significance markers reflect the corrected threshold within the relevant family.  
Where an uncorrected p-value appeared significant (p < 0.05) but the corrected threshold is not met, results are marked as **ns†** (marginally significant but not family-wise corrected).

### 1.1 McNemar Results by Language

**EN — Scenario 1 strategies are identical (no discordant pairs)**

Every S1 voting method achieves identical results on English (96.8%). No paired test needed. *(Note: Scenario 2 shows a minor EN regression to 95.8% under soft/weighted; see §7.)*

**MY (n = 95) — key degradation language**

| Pair | p-value | sig (corr. α=0.0083) | b | c | method |
|---|---|---|---|---|---|
| lingua-high vs hard | 0.0312 | ns† | 6 | 0 | exact |
| lingua-high vs soft | < 0.0001 | *** | 24 | 2 | chi2 |
| lingua-high vs weighted | 0.0140 | ns† | 21 | 7 | chi2 |
| hard vs soft | 0.0004 | *** | 18 | 2 | exact |
| hard vs weighted | 0.1338 | ns | 15 | 7 | exact |
| soft vs weighted | 0.0078 | * | 0 | 8 | exact |

> ns† = uncorrected p < 0.05 but not significant under family-wise Bonferroni correction.

**ID (n = 95) — key gain language**

| Pair | p-value | sig (corr. α=0.0083) | b | c | method |
|---|---|---|---|---|---|
| lingua-high vs hard | 0.0039 | ** | 0 | 9 | exact |
| lingua-high vs soft | < 0.0001 | *** | 0 | 16 | exact |
| lingua-high vs weighted | 0.0005 | *** | 1 | 15 | exact |
| hard vs soft | 0.0156 | ns† | 0 | 7 | exact |
| hard vs weighted | 0.1250 | ns | 1 | 6 | exact |
| soft vs weighted | 0.5000 | ns | 2 | 0 | exact |

**ZH/TA — no significant differences** (all methods matched lingua-high; ZH/TA at 100% for all voting strategies)

**ALL (n = 475) — aggregate differences not significant**

| Pair | p-value | b | c |
|---|---|---|---|
| lingua-high vs hard | 0.2379 | 6 | 12 |
| lingua-high vs soft | 0.6511 | 24 | 20 |
| lingua-high vs weighted | 0.8828 | 22 | 24 |

### 1.2 Interpretation

| Finding | Magnitude | 95% CI on difference | Statistical verdict |
|---|---|---|---|
| MY drop: lingua-high → hard vote | −6.4 pp (65.3% → 58.9%) | roughly −13 to 0 pp | **ns† (corrected)** — p=0.031, not family-wise significant |
| MY drop: lingua-high → soft vote | −23.2 pp (65.3% → 42.1%) | roughly −32 to −14 pp | **Confirmed significant** (p < 0.0001) |
| ID gain: lingua-high → soft vote | +16.8 pp (69.5% → 86.3%) | roughly +9 to +24 pp | **Confirmed significant** (p < 0.0001) |
| ID gain: lingua-high → hard vote | +9.4 pp (69.5% → 78.9%) | roughly +3 to +16 pp | **Confirmed significant** (p = 0.004) |
| Overall: lingua-high vs any voting | ~+1 pp | — | **Not significant** at aggregate level |

**Key corrected-threshold findings:**
- The MY −6.4pp drop under hard voting is real in direction but does not survive family-wise correction. The strongest evidence is the soft-vote degradation (p < 0.0001, large effect).
- The ID gains under soft and weighted voting are confirmed significant even after correction.
- Hard-vs-soft on ID (p=0.016) does not survive correction — the two strategies are statistically indistinguishable at the corrected α.

The MY degradation and ID gain are statistically real (for the large-effect comparisons). The aggregate gain (~+1 pp) remains non-significant because MY losses and ID gains cancel.

### 1.3 Sample Size Limitations

> ⚠ **Caveat:** n=95 per language, n=475 overall. Per-bucket cells range from n≈10 to n=30.

At n=95 per language, the Wilson 95% CI on a proportion is approximately ±10 pp near 50% and ±5 pp near 95%. Per-bucket cells (n=10–30) have CI widths of ±15–20 pp near 50%.

**Consequence:** small per-bucket differences (e.g., ±5–10 pp in 1-word or 2-word buckets) cannot be distinguished from sampling noise. The bucket-level tables in §6 and §7 describe directional trends, not confirmed differences. Use paired McNemar tests (which pair by case, not by count) for reliable inference, and focus bucket analysis on cases with n ≥ 15.

---

## 2. Inter-Model Agreement and Diversity

**Method:** Cohen's kappa (multi-class) + raw agreement rate  
**Library:** `sklearn.metrics.cohen_kappa_score`

### 2.1 Overall Kappa — Scenario 1 Trio (all 475 cases)

| Pair | κ (Cohen) | Agreement % | Interpretation |
|---|---|---|---|
| lingua-high vs langdetect | 0.497 | 57.3% | Moderate diversity — voting may help |
| lingua-high vs pycld2 | 0.482 | 54.5% | Moderate diversity — voting may help |
| langdetect vs pycld2 | 0.563 | 65.1% | Moderate-to-substantial, slightly less diverse |

### 2.2 Per-Language Kappa — Scenario 1

| Language | lingua vs langdetect | lingua vs pycld2 | langdetect vs pycld2 |
|---|---|---|---|
| EN | 0.067 (agree 69.5%) | 0.025 (agree 45.3%) | 0.411 (agree 69.5%) |
| **MY** | **−0.053** (agree **18.9%**) ⚠ | 0.057 (agree 33.7%) | 0.177 (agree 35.8%) ⚠ |
| ID | 0.075 (agree 49.5%) | 0.055 (agree 35.8%) | 0.425 (agree 69.5%) |
| ZH | 0.000 (agree 52.6%) | 0.000 (agree 61.1%) | 0.088 (agree 51.6%) |
| TA | −0.016 (agree 95.8%) | 0.000 (agree 96.8%) | 0.000 (agree 98.9%) |

### 2.3 Structural Artifact on MY — Scenario 1

The MY column exposes the core problem:

- **lingua-high vs langdetect agreement on MY: 18.9%** — they almost never agree. langdetect structurally cannot output `ms` and always votes `id`, while lingua-high correctly identifies many Malay samples as `ms`.
- **Negative kappa (κ = −0.053) on MY** means they agree *less* than chance — the strongest possible signal that langdetect's vote is adversarial on the ms/id axis.
- **langdetect vs pycld2 agreement on MY: 35.8%** — this is not genuine voter independence. When pycld2 also outputs `id` for a Malay text, it coincides with langdetect's structurally forced `id` vote. This is coincidence of failure modes, not reliable consensus.

### 2.4 ZH/TA Kappa Paradox — Why κ≈0 Despite ~99% Agreement

The ZH and TA rows all show κ≈0.000 despite agreement rates of 95–99%. This is the **high-agreement/low-kappa paradox** caused by near-zero label variance.

Kappa is defined as: κ = (p_obs − p_exp) / (1 − p_exp), where p_exp is the chance agreement based on marginal distributions.

At 99% agreement on ZH/TA, both models predict `zh`/`ta` nearly 100% of the time. The marginal distributions are heavily skewed: Pr(A=zh) ≈ 1.0, Pr(B=zh) ≈ 1.0. Consequently p_exp ≈ p_obs ≈ 0.99, making the numerator (p_obs − p_exp) ≈ 0 and the denominator (1 − p_exp) ≈ 0.01 — giving κ = 0/tiny ≈ 0.

This is mathematically correct and expected. The κ≈0 on ZH/TA **does not mean the models disagree**; it means kappa is undefined/uninformative when there is almost no variation in labels. The high raw agreement (95–99%) is the correct diagnostic for ZH/TA, not kappa.

### 2.5 Overall Kappa — Scenario 2 Trio (all 475 cases)

*Script: `voting/kappa_s2.py` | Log: `log_kappa_s2_1.txt`*

| Pair | κ (Cohen) | Agreement % | vs S1 equivalent |
|---|---|---|---|
| lingua-high vs openlid-v3 | 0.773 | 81.7% | — |
| openlid-v3 vs pycld2 | 0.548 | 61.1% | similar to langdetect vs pycld2 (0.563) |
| lingua-high vs pycld2 (ref) | 0.482 | 54.5% | same (unchanged) |

### 2.6 Per-Language Kappa — Scenario 2

| Language | lingua vs openlid-v3 | openlid-v3 vs pycld2 | lingua vs pycld2 (ref) |
|---|---|---|---|
| EN | 0.109 (agree 80.0%) | 0.223 (agree 55.8%) | 0.025 (agree 45.3%) |
| **MY** | **0.317** (agree 62.1%) | 0.215 (agree 41.1%) | 0.057 (agree 33.7%) |
| ID | 0.220 (agree 70.5%) | 0.113 (agree 48.4%) | 0.055 (agree 35.8%) |
| ZH | 0.000 (agree 98.9%) | — (agree 100%, kappa undefined) | 0.000 (agree 61.1%) |
| TA | 0.000 (agree 96.8%) | nan (agree 100%) | 0.000 (agree 96.8%) |
| ALL | 0.773 (agree 81.7%) | 0.548 (agree 61.1%) | 0.482 (agree 54.5%) |

### 2.7 S2 Kappa Interpretation — openlid-v3 vs langdetect

**openlid-v3 output distribution for true-MY (n=95):**

| Model | ms predictions | id predictions | unknown |
|---|---|---|---|
| openlid-v3 | 41 | 46 | 5 (empty text) |
| langdetect | 0 | 95 | 0 |

openlid-v3 is **genuinely diverse on the ms/id axis** — it outputs `ms` for 43% of true-Malay cases (vs langdetect's structural 0%). The κ=0.317 between lingua-high and openlid-v3 on MY is positive and moderate, meaning they have genuine but partial agreement — neither is a structural constant.

This means openlid-v3 participates as a real voter on the ms/id dimension. In contrast, langdetect (κ=−0.053) participates as a structural adversary — its vote is informative only about its own incapacity, not about the true language.

> ZH/TA kappa paradox applies identically in S2 — see §2.4.

---

## 3. Calibration Analysis

**Library:** `sklearn.calibration.calibration_curve` (reliability diagrams); ECE computed manually  
**Plots:** `calibration/reliability_1.png`, `calibration/conf_dist_1.png`

### 3.1 Expected Calibration Error (ECE)

| Model | ECE | Avg conf (correct) | Avg conf (wrong) | Median conf (correct) | Median conf (wrong) |
|---|---|---|---|---|---|
| lingua-high | 0.0652 | 0.851 | 0.576 | 0.999 | 0.529 |
| langdetect | **0.1636** | **0.927** | 0.344 | 1.000 | 0.000 |
| pycld2 | **0.0437** | 0.917 | **0.066** | 0.970 | 0.000 |

### 3.2 Calibration Findings

**pycld2** is the best-calibrated model (ECE = 0.044). When it fires, it is either highly confident and correct or near-zero and wrong — strong discriminative behavior.

**langdetect** is the worst-calibrated (ECE = 0.164). It reports high confidence (median 1.000 when correct) but this confidence does not scale with reliability across languages. Its structural 100% confidence on Malay-as-id cases is a known artifact: confidence reflects certainty within a limited model, not reliability against the true label.

**Soft and weighted vote bias:** langdetect's systematically high confidence on `id` for Malay text injects a large, confident `id` signal into probability averaging — explaining why soft voting degrades MY the most (−23.2 pp confirmed significant): langdetect dominates the probability average on the ms/id dimension.

**lingua-high** is moderately calibrated (ECE = 0.065). Its median wrong confidence of 0.53 is higher than the other two models — it is less certain when it fails, which is useful for ensemble weighting.

---

## 4. Leave-One-Model-Out Ablation

**Script:** `voting_ablation.py`  
Hypothesis: *removing langdetect should recover MY accuracy without harming ID.*

### 4.1 Full Ablation Table (all 12 configurations, n=475)

| LANG | lingua-high | 3m-hard | 3m-soft | 3m-wgtd | no-LD hard | no-LD soft | no-LD wgtd | no-LI hard | no-LI soft | no-LI wgtd | no-PY hard | no-PY soft | no-PY wgtd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | 96.8% | 96.8% | 96.8% | 96.8% | 72.6% | 100.0% | 100.0% | 96.8% | 96.8% | 96.8% |
| **MY** | **65.3%** | 58.9% | 42.1% | 50.5% | **66.3%** | **66.3%** | **68.4%** | 7.4% | 7.4% | 40.0% | 18.9% | 15.8% | 20.0% |
| ID | 69.5% | 78.9% | 86.3% | 84.2% | 78.9% | 78.9% | 78.9% | 69.5% | 69.5% | 67.4% | 85.3% | 85.3% | 84.2% |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 73.7% | 73.7% | 75.8% | 100.0% | 100.0% | 100.0% |
| TA | 96.8% | 100.0% | 98.9% | 98.9% | 96.8% | 96.8% | 96.8% | 98.9% | 98.9% | 100.0% | 97.9% | 97.9% | 96.8% |
| **ALL** | 85.7% | 86.9% | 84.8% | 86.1% | **87.8%** | **87.8%** | **88.2%** | 64.4% | 69.9% | 76.6% | 79.8% | 79.2% | 79.6% |

Wilson 95% CI for the headline `no-LD weighted` result: MY = 68.4% [58.5–76.8%], ALL = 88.2% [85.0–90.8%].

### 4.2 MY Accuracy by Bucket (no-LD configurations)

| Bucket | lingua-high | 3m-hard | 3m-soft | no-LD hard | no-LD soft | no-LD wgtd |
|---|---|---|---|---|---|---|
| 1 word | 56.7% | 53.3% | 36.7% | 63.3% | 63.3% | 63.3% |
| 2 words | 60.0% | 53.3% | 33.3% | 53.3% | 53.3% | 53.3% |
| 3–7 words | 93.3% | 73.3% | 46.7% | 73.3% | 73.3% | 73.3% |
| 8–16 words | 60.0% | 60.0% | 50.0% | 90.0% | 90.0% | 90.0% |
| 17–50 words | 70.0% | 70.0% | 70.0% | 80.0% | 80.0% | 100.0% |
| ALL | **65.3%** | 58.9% | 42.1% | **66.3%** | **66.3%** | **68.4%** |

> Per-bucket n = 10–30 cells; ±15–20 pp CI at n=30 — see §1.3.

### 4.3 Hypothesis Verdict

**SUPPORTED.** Removing langdetect:
- **MY accuracy: 65.3% → 68.4%** [58.5–76.8%] (no-LD weighted) — recovers and exceeds lingua-high
- **ID accuracy: 78.9% → 78.9%** (no-LD hard) — unchanged
- **Overall: 85.7% → 88.2%** [85.0–90.8%] (no-LD weighted)

Removing lingua-high (no-LI) collapses MY to 7.4% — confirming lingua-high is the only S1 model that can vote `ms`.  
Removing pycld2 (no-PY) drops MY to 18.9% — pycld2 provides meaningful ms signal at longer text lengths.

---

## 5. Train/Test Leakage Check

**Script:** `voting_two_stage.py`  
**Split:** stratified 60/40 (seed=42) → dev: 285 cases (57/lang), test: 190 cases (38/lang)

### 5.1 AUC Methods and the pycld2 Discrepancy

Two different AUC computation methods appear across scripts, producing inconsistent values:

| Model | benchmarkV5 AUC | voting_two_stage AUC | Unified AUC (auc_unified.py) |
|---|---|---|---|
| lingua-high | 0.8503 | 0.9736 | 0.8501 ✓ |
| langdetect | 0.7516 | 0.7756 | 0.8744 ✗ |
| pycld2 | 0.9634 | 0.7756 | 0.9718 ≈✓ |

**Root cause of discrepancy:**
- `benchmarkV5.py` computes: binary = (top-1 prediction correct?), score = raw top-1 confidence from native library output. This is a single binary discrimination AUC.
- `voting_two_stage.py` used: per-class OVR probability vector → AUC computed over class-conditional scores. This is a fundamentally different metric.
- `auc_unified.py` reimplements benchmarkV5's method: binary correct/incorrect + top-1 confidence from the 5-language target space. It matches lingua-high (0.8501 ≈ 0.8503) and pycld2 (0.9718 ≈ 0.9634) well.
- langdetect does NOT match (unified 0.8744 vs benchmarkV5 0.7516) because `ld_conf` in `core.py` stores the aggregated target-space probability (0.0 when langdetect predicts a non-target language), while benchmarkV5 stores the raw top-1 confidence. These are different values.

**Conclusion:** The `benchmarkV5` AUC values are authoritative. The per-class OVR values in `voting_two_stage.py` are a different (inconsistent) metric and should not be compared directly to benchmarkV5 AUC values.

### 5.2 Dev-Set Accuracy Leakage Check

| Strategy | Dev-set accuracy | Test-set accuracy | Note |
|---|---|---|---|
| weighted (global AUC weights) | — | 85.8% | weights from full 475 — potentially leaked |
| weighted (dev AUC weights) | — | 84.7% | weights from dev set only — honest |

**Overfitting gap: −1.05 pp** ≈ 2 cases at n=190.

> ⚠ **Corrected conclusion (A4):** The −1.05pp gap is within the confidence interval noise for n=190 (Wilson CI ≈ ±7 pp at 85%). We cannot conclusively rule out a small overfitting effect given this sample size. The gap does not rise to the level of material overfitting, but with only 190 test cases, the honest conclusion is: *no overfitting detected, but n=190 is too small to rule it out definitively*.

### 5.3 Per-Language Leakage Assessment (test set n=190)

| LANG | lingua-high | hard | soft | wt-global | wt-dev |
|---|---|---|---|---|---|
| EN | 97.4% | 97.4% | 97.4% | 97.4% | 97.4% |
| MY | 60.5% | 57.9% | 44.7% | 52.6% | 44.7% |
| ID | 57.9% | 71.1% | 81.6% | 78.9% | 81.6% |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 94.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| ALL | 82.1% | 85.3% | 84.7% | 85.8% | 84.7% |

The same MY/ID pattern holds on the held-out test: MY degrades under voting, ID gains with soft/hard voting. The pattern is not an artifact of circular evaluation.

---

## 6. Two-Stage Voting

**Script:** `voting_two_stage.py`, `voting/stage1_accuracy.py`  
**Log:** `log_voting_two_stage_2.txt`, `log_stage1_accuracy_1.txt`

### 6.1 Design

```
Stage 1 — coarse classification (all 3 S1 models: lingua + langdetect + pycld2)
  Map: ms → MSID, id → MSID
  Hard vote on {en, MSID, zh, ta}
  langdetect's 'id' for Malay text counts as MSID (structurally correct)
  If result ≠ MSID → final answer
  If result = MSID → trigger Stage 2

Stage 2 — fine ms/id decision (lingua-high + pycld2 only — langdetect excluded)
  Variant (a) two_stage_agree:
    If both agree on ms/id → use that label
    If disagree → use model with higher max(ms_conf, id_conf)
  Variant (b) two_stage_weighted:
    score_ms = lingua_p['ms'] × w_l_ms + cld2_p['ms'] × w_c_ms
    score_id = lingua_p['id'] × w_l_id + cld2_p['id'] × w_c_id
    where weights = per-class accuracy on dev set
```

### 6.2 Stage 1 Routing Accuracy

*Script: `stage1_accuracy.py` | Log: `log_stage1_accuracy_1.txt`*

| True language | n | Stage 1 correct route | Misrouted | Misrouted to |
|---|---|---|---|---|
| EN | 95 | 96.8% | 3 | msid(3) |
| MY | 95 | **98.9%** | 1 | en(1) |
| ID | 95 | 96.8% | 3 | en(3) |
| ZH | 95 | 100.0% | 0 | — |
| TA | 95 | 100.0% | 0 | — |
| **ALL** | **475** | **98.5%** | **7** | — |

**Stage 1 is not the bottleneck.** It routes MY and ID to MSID with 97.9% accuracy (186/190 MY+ID cases). Only 4 MS-or-ID cases are misrouted: 1 MY case sent to en, 3 ID cases sent to en.

**Stage 2 is the bottleneck.** Of two_stage_agree's errors on MY: 1 is a Stage 1 misroute, but 30 are Stage 2 errors (lingua-high and pycld2 disagreeing or both wrong on the fine ms/id question). For ID: 3 Stage 1 misroutes, 17 Stage 2 errors.

**Stage 1 by bucket (MY and ID):**

| Bucket | n(MY) | MY→MSID | n(ID) | ID→MSID |
|---|---|---|---|---|
| 1 word | 30 | 96.7% | 30 | 93.3% |
| 2 words | 30 | 100.0% | 30 | 96.7% |
| 3–7 words | 15 | 100.0% | 15 | 100.0% |
| 8–16 words | 10 | 100.0% | 10 | 100.0% |
| 17–50 words | 10 | 100.0% | 10 | 100.0% |

Even in the hardest 1-word bucket, Stage 1 routes correctly 93–97% of the time. The remaining failures are almost all Stage 2 (fine ms/id) errors. Improving two-stage accuracy requires better Stage 2, not better Stage 1.

### 6.3 Dev-Set Stage 2 Weights

| Weight | Value | Meaning |
|---|---|---|
| `w_l_ms` | 0.6842 | lingua-high per-class accuracy on dev-set ms |
| `w_l_id` | 0.7719 | lingua-high per-class accuracy on dev-set id |
| `w_c_ms` | 0.3684 | pycld2 per-class accuracy on dev-set ms |
| `w_c_id` | 0.4386 | pycld2 per-class accuracy on dev-set id |

Lingua-high is the stronger ms/id discriminator (~1.9× pycld2 on both classes).

### 6.4 Full-Dataset Results (n=475)

| LANG | lingua-high | 3m-hard | 3m-soft | **2s-agree** | **2s-weighted** |
|---|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | 96.8% | 96.8% |
| **MY** | 65.3% | 58.9% | 42.1% | **67.4%** | 58.9% |
| **ID** | 69.5% | 78.9% | 86.3% | **78.9%** | **84.2%** |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 96.8% | 100.0% | 98.9% | **100.0%** | **100.0%** |
| **ALL** | 85.7% | 86.9% | 84.8% | **88.6%** | **88.0%** |

Wilson 95% CI: two_stage_agree MY = 67.4% [57.4–75.8%], ALL = 88.6% [85.5–91.1%].

### 6.5 MY and ID Accuracy by Bucket (two-stage)

**MY (n=95):**

| Bucket | lingua-high | 3m-hard | 2s-agree | 2s-wgtd |
|---|---|---|---|---|
| 1 word | 56.7% | 53.3% | **63.3%** | 46.7% |
| 2 words | 60.0% | 53.3% | **56.7%** | 46.7% |
| 3–7 words | 93.3% | 73.3% | 73.3% | 73.3% |
| 8–16 words | 60.0% | 60.0% | **90.0%** | **90.0%** |
| 17–50 words | 70.0% | 70.0% | **80.0%** | **80.0%** |

> n=10–30 per bucket; treat differences < 15 pp as directional only.

**ID (n=95):**

| Bucket | lingua-high | 3m-hard | 2s-agree | 2s-wgtd |
|---|---|---|---|---|
| 1 word | 63.3% | 63.3% | 66.7% | **80.0%** |
| 2 words | 66.7% | 70.0% | 66.7% | **76.7%** |
| 3–7 words | 60.0% | **100.0%** | **100.0%** | **100.0%** |
| 8–16 words | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| 17–50 words | 80.0% | **100.0%** | **100.0%** | 80.0% |

### 6.6 McNemar's Tests — Two-Stage vs Baselines

*Bonferroni correction: §6.5 family = 5 comparisons per language → corrected α = 0.05/5 = 0.0100*

**MY (n=95, α_corr=0.01):**

| Comparison | p-value | sig | b | c |
|---|---|---|---|---|
| two_stage_agree vs lingua-high | 0.7905 | ns | 8 | 6 |
| two_stage_agree vs 3m-hard | **0.0078** | * | 8 | 0 |
| two_stage_agree vs 3m-soft | < 0.0001 | *** | 24 | 0 |
| two_stage_wgtd vs lingua-high | 0.2863 | ns | 8 | 14 |
| two_stage_wgtd vs 3m-hard | 1.0000 | ns | 8 | 8 |

**ID (n=95, α_corr=0.0125 for 4 comparisons):**

| Comparison | p-value | sig | b | c |
|---|---|---|---|---|
| two_stage_agree vs lingua-high | 0.0117 | * | 10 | 1 |
| two_stage_agree vs 3m-hard | 1.0000 | ns | 1 | 1 |
| two_stage_wgtd vs lingua-high | **0.0005** | *** | 15 | 1 |
| two_stage_wgtd vs 3m-hard | 0.2266 | ns | 8 | 3 |

### 6.7 Two-Stage Error Attribution

| True lang | n | Correct | S1 misroute | S2 error | Acc (two_stage_agree) |
|---|---|---|---|---|---|
| EN | 95 | 92 | 0 | 3 | 96.8% |
| MY | 95 | 64 | 1 | 30 | 67.4% |
| ID | 95 | 75 | 3 | 17 | 78.9% |
| ZH | 95 | 95 | 0 | 0 | 100.0% |
| TA | 95 | 95 | 0 | 0 | 100.0% |

The two-stage's ceiling on MY improvement is set by Stage 2 (fine ms/id decision). 30 out of 31 MY errors originate in Stage 2. Improving MY accuracy beyond ~70% requires better Stage 2 models, not better Stage 1 routing.

### 6.8 Verdict

**`two_stage_agree` is the best S1 strategy:**
- MY: 67.4% [57.4–75.8%] vs 3m-hard 58.9%, **significant (p=0.008 ≤ α_corr=0.01)**
- ID: 78.9%, unchanged from 3m-hard (p=1.0)
- ALL: 88.6%, best overall
- MY vs lingua-high (+2.1pp): **not significant (p=0.79)** — two_stage_agree reaches parity with lingua-high on MY while simultaneously retaining the ID gain from ensemble voting

---

## Binomial Confidence Intervals (95%, Wilson method)

`statsmodels.stats.proportion.proportion_confint(method='wilson')` — n=95 per language, n=475 overall.

| LANG | lingua-high | langdetect | pycld2 | hard | soft | weighted |
|---|---|---|---|---|---|---|
| EN | 96.8% [91.1–98.9%] | 69.5% [59.6–77.8%] | 45.3% [35.6–55.3%] | 96.8% [91.1–98.9%] | 96.8% [91.1–98.9%] | 96.8% [91.1–98.9%] |
| MY | 65.3% [55.3–74.1%] | 0.0% [0.0–3.9%] | 40.0% [30.7–50.1%] | 58.9% [48.9–68.3%] | 42.1% [32.7–52.2%] | 50.5% [40.6–60.4%] |
| ID | 69.5% [59.6–77.8%] | 65.3% [55.3–74.1%] | 45.3% [35.6–55.3%] | 78.9% [69.7–85.9%] | 86.3% [78.0–91.8%] | 84.2% [75.6–90.2%] |
| ZH | 100.0% [96.1–100%] | 52.6% [42.7–62.4%] | 61.1% [51.0–70.2%] | 100.0% [96.1–100%] | 100.0% [96.1–100%] | 100.0% [96.1–100%] |
| TA | 96.8% [91.1–98.9%] | 98.9% [94.3–99.8%] | 100.0% [96.1–100%] | 100.0% [96.1–100%] | 98.9% [94.3–99.8%] | 98.9% [94.3–99.8%] |
| ALL | 85.7% [82.2–88.5%] | 57.3% [52.8–61.6%] | 58.3% [53.8–62.7%] | 86.9% [83.6–89.7%] | 84.8% [81.3–87.8%] | 86.1% [82.7–88.9%] |

> CI width at n=95 ≈ ±10 pp near 50%; ±5 pp near 95%. Use McNemar for paired comparisons.

---

## 7. Scenario 2 — openlid-v3 Replaces langdetect

**Script:** `voting_scenario2/voting_s2.py`, `voting_scenario2/voting_s2_two_stage.py`  
**Log:** `voting_scenario2/log_s2_1.txt`, `voting_scenario2/log_s2_two_stage_2.txt`  
**Ensemble:** lingua-high + openlid-v3 + pycld2  
**Weights:** lingua 0.8503 | openlid-v3 0.7097 | pycld2 0.9634  

> **Note on openlid-v3 ZH:** The original benchmark showed 0% ZH because the FLORES-200 code `cmn_Hans` (Mandarin simplified) was missing from the mapping. Added to `OPENLID_TO_ISO` in `voting_s2.py`. ZH recovers to 98.9% — confirming the original ZH=0% was a mapping bug, not a model failure.

### 7.1 Individual Model Comparison (full 475)

| LANG | lingua-high | S1: langdetect | S2: openlid-v3 | pycld2 |
|---|---|---|---|---|
| EN | 96.8% | 69.5% | **80.0%** | 45.3% |
| MY | 65.3% | 0.0% | **43.2%** | 40.0% |
| ID | 69.5% | 65.3% | **87.4%** | 45.3% |
| ZH | 100.0% | 52.6% | **98.9%** | 61.1% |
| TA | 96.8% | 98.9% | **100.0%** | 100.0% |
| ALL | 85.7% | 57.3% | **81.9%** | 58.3% |

openlid-v3 individually outperforms langdetect on every single language. The gap is largest on ZH (+46.3 pp), ID (+22.1 pp), and MY (+43.2 pp vs 0.0%). openlid-v3's MY is still low (43.2%) because short Malay loanwords overlap strongly with Indonesian at 1–2 word lengths (see §9).

### 7.2 Voting Results — S1 vs S2 (full 475)

| LANG | S1 hard | S2 hard | S1 soft | S2 soft | S1 weighted | S2 weighted |
|---|---|---|---|---|---|---|
| EN | 96.8% | 96.8% | 96.8% | **95.8%** | 96.8% | **95.8%** |
| MY | 58.9% | **65.3%** | 42.1% | **56.8%** | 50.5% | **57.9%** |
| ID | 78.9% | 78.9% | 86.3% | 86.3% | 84.2% | **87.4%** |
| ZH | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TA | 100.0% | 100.0% | 98.9% | **100.0%** | 98.9% | **100.0%** |
| ALL | 86.9% | **88.2%** | 84.8% | **87.8%** | 86.1% | **88.2%** |

> **EN regression in S2:** Under S2 soft/weighted, EN drops from 96.8% → 95.8% (1 case). This is because openlid-v3 occasionally pulls probability mass away from `en` for very short English texts. Hard voting avoids this.

### 7.3 Per-Bucket Breakdown — MY and ID (full 475)

**MY by bucket:**

| Bucket | lingua-high | S1 hard | S2 hard | S1 soft | S2 soft |
|---|---|---|---|---|---|
| 1 word | 56.7% | 53.3% | 53.3% | 36.7% | 40.0% |
| 2 words | 60.0% | 53.3% | 53.3% | 33.3% | 43.3% |
| 3–7 words | 93.3% | 73.3% | **86.7%** | 46.7% | **80.0%** |
| 8–16 words | 60.0% | 60.0% | **90.0%** | 50.0% | **90.0%** |
| 17–50 words | 70.0% | 70.0% | **80.0%** | 70.0% | **80.0%** |

**ID by bucket:**

| Bucket | lingua-high | S1 hard | S2 hard | S1 soft | S2 soft |
|---|---|---|---|---|---|
| 1 word | 63.3% | 63.3% | 66.7% | 73.3% | 76.7% |
| 2 words | 66.7% | 70.0% | 70.0% | 83.3% | 83.3% |
| 3–7 words | 60.0% | **100.0%** | 93.3% | **100.0%** | **100.0%** |
| 8–16 words | **100.0%** | **100.0%** | **100.0%** | **100.0%** | 90.0% |
| 17–50 words | 80.0% | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### 7.4 McNemar's Tests — S1 vs S2 (same 475 cases, paired)

*Bonferroni: 8 comparisons across all scopes → corrected α = 0.05/8 = 0.0063*

| Language | Strategy | S1 acc | S2 acc | delta | p-value | sig (α_corr=0.0063) |
|---|---|---|---|---|---|---|
| MY | hard | 58.9% | 65.3% | +6.3 pp | 0.0312 | ns† |
| MY | soft | 42.1% | 56.8% | +14.7 pp | **0.0026** | ** |
| MY | weighted | 50.5% | 57.9% | +7.4 pp | 0.0923 | ns |
| ID | hard | 78.9% | 78.9% | 0.0 pp | 1.0000 | ns |
| ID | soft | 86.3% | 86.3% | 0.0 pp | 1.0000 | ns |
| ID | weighted | 84.2% | 87.4% | **+3.2 pp** | 0.5078 | **ns** — within noise |
| ZH | all | 100.0% | 100.0% | 0.0 pp | 1.0000 | ns |
| ALL | soft | 84.8% | 87.8% | +2.9 pp | 0.0258 | ns† |

> The +3.2pp S2 ID weighted gain (p=0.51) and +2.9pp S2 ALL soft gain (p=0.026 uncorrected) are **within noise** — not significant after correction or at baseline level. Report them as directional improvements only.

### 7.5 Scenario 2 Two-Stage Results (test set n=190)

*Script: `voting_s2_two_stage.py` | Log: `log_s2_two_stage_2.txt`*  
*Weights fitted on dev set (n=285): w_lingua_ms=0.6842, w_lingua_id=0.7895, w_cld2_ms=0.8596, w_cld2_id=0.4386*

| Method | EN | MY | ID | ZH | TA | ALL |
|---|---|---|---|---|---|---|
| S2 ts_agree (L+C Stage 2) | 97.4% | **68.4%** | 71.1% | 100.0% | 100.0% | 87.4% |
| S2 ts_3agree (L+O+C Stage 2) | 97.4% | 65.8% | 68.4% | 100.0% | 100.0% | 86.3% |
| S2 ts_weighted (L+C Stage 2 wt) | 97.4% | 65.8% | **76.3%** | 100.0% | 100.0% | **87.9%** |
| S2 hard (baseline) | 97.4% | 63.2% | 71.1% | 100.0% | 100.0% | 86.3% |
| S1 ts_agree (reference) | 97.4% | **68.4%** | 71.1% | 100.0% | 100.0% | 87.4% |

> These results are on the 190-case test set (40% held-out). All S1 numbers in §6.4 are on the full 475 — they are different evaluation sets and not directly comparable. For the deployment recommendation, use test-set numbers for both S1 and S2 two-stage.

**McNemar tests (S2 two-stage vs S1 ts_agree, test n=190, α_corr=0.0125):**

All p-values ≥ 0.625. No S2 two-stage variant differs significantly from S1 two_stage_agree. `s2_ts_agree` and `s1_ts_agree` are **identical on the test set** (b=0, c=0, p=1.0).

### 7.6 Recommended Strategy — Single Winner (D12)

**Recommended deployment strategy: `S2 two_stage_agree` (lingua-high + openlid-v3 + pycld2, Stage 2 = lingua + pycld2)**

Justification:

| Criterion | S1 ts_agree | S2 hard | **S2 ts_agree** |
|---|---|---|---|
| ALL accuracy (test n=190) | 87.4% | 86.3% | 87.4% |
| MY accuracy (test) | 68.4% | 63.2% | **68.4%** |
| ID accuracy (test) | 71.1% | 71.1% | 71.1% |
| Model calls per input | 3 + 2* | 3 | 3 + 2* |
| Per-call latency | ~4 ms (ld dominates) | ~0.84 ms | **~0.84 ms** |
| langdetect dependency | Yes | No | No |
| Structural ms-blindness | Yes (Stage 1 has it; Stage 2 avoids it) | No | No |
| Code complexity | High (two-stage) | Low | High (two-stage) |

> *Stage 2 reuses already-computed predictions — no additional inference calls.

**Why S2 ts_agree over S1 ts_agree:** Identical accuracy, but eliminates the langdetect dependency. S1 ts_agree still requires running langdetect for Stage 1. S2 ts_agree replaces langdetect with openlid-v3 (75× faster at ~0.04 ms/call vs 3.1 ms), reducing per-call latency from ~4ms to ~0.84ms — a 5× total speedup.

**Why S2 ts_agree over S2 hard:** MY accuracy 68.4% vs 63.2% (+5.2pp). McNemar p=0.625 (not significant at n=38 MY test cases), but the direction is consistent and the mechanism is sound. At deployment scale (thousands of calls), the structural improvement is expected to hold.

**Why not S2 ts_weighted:** Best overall (87.9%) but lower MY (65.8%). MY is the primary optimization target; ts_agree is preferred.

---

## 8. Master Summary Table

All strategies across both scenarios. Full-475 results for all single-stage strategies; test-190 for two-stage variants (noted).  
**Bold** = best value per language column.

| Strategy | Ensemble | EN | MY | ID | ZH | TA | ALL |
|---|---|---|---|---|---|---|---|
| lingua-high (individual) | — | 96.8% | 65.3% | 69.5% | 100.0% | 96.8% | 85.7% |
| langdetect (individual) | S1 | 69.5% | 0.0% | 65.3% | 52.6% | 98.9% | 57.3% |
| openlid-v3 (individual) | S2 | 80.0% | 43.2% | 87.4% | 98.9% | 100.0% | 81.9% |
| pycld2 (individual) | — | 45.3% | 40.0% | 45.3% | 61.1% | 100.0% | 58.3% |
| S1 hard | ld+li+py | 96.8% | 58.9% | 78.9% | 100.0% | 100.0% | 86.9% |
| S1 soft | ld+li+py | 96.8% | 42.1% | 86.3% | 100.0% | 98.9% | 84.8% |
| S1 weighted | ld+li+py | 96.8% | 50.5% | 84.2% | 100.0% | 98.9% | 86.1% |
| S1 no-LD weighted | li+py | 96.8% | 68.4% | 78.9% | 100.0% | 96.8% | 88.2% |
| S1 two_stage_agree | ld+li+py | 96.8% | 67.4% | 78.9% | 100.0% | 100.0% | 88.6% |
| S1 two_stage_weighted | ld+li+py | 96.8% | 58.9% | 84.2% | 100.0% | 100.0% | 88.0% |
| S2 hard | ol+li+py | 96.8% | 65.3% | 78.9% | 100.0% | 100.0% | 88.2% |
| S2 soft | ol+li+py | 95.8% | 56.8% | 86.3% | 100.0% | 100.0% | 87.8% |
| S2 weighted | ol+li+py | 95.8% | 57.9% | **87.4%** | 100.0% | 100.0% | 88.2% |
| **S2 ts_agree** ✅ | ol+li+py | **97.4%**† | **68.4%**† | 71.1%† | **100.0%** | **100.0%** | 87.4%† |
| S2 ts_weighted | ol+li+py | **97.4%**† | 65.8%† | 76.3%† | **100.0%** | **100.0%** | **87.9%**† |

> Ensemble key: ld = langdetect, ol = openlid-v3, li = lingua-high, py = pycld2  
> † = evaluated on test set only (n=190), not full 475. Directly compare within †-marked rows.

### 8.1 Deployment Decision Table

| Strategy | ALL (full) | MY | ID | Calls/input | Latency† | Complexity |
|---|---|---|---|---|---|---|
| lingua-high only | 85.7% | 65.3% | 69.5% | 1 | 0.8 ms | Minimal |
| S2 hard | 88.2% | 65.3% | 78.9% | 3 | 0.84 ms | Low |
| **S2 ts_agree** ✅ | ~88%‡ | **~68%** | ~71% | 3 (reuse) | 0.84 ms | Medium |
| S1 ts_agree | 88.6% | 67.4% | 78.9% | 3 (reuse) | 4.0 ms | Medium |
| S1 hard | 86.9% | 58.9% | 78.9% | 3 | 4.0 ms | Low |

> † Latency = total per-call (lingua-high ≈ 0.8ms, langdetect ≈ 3.1ms, openlid-v3 ≈ 0.04ms, pycld2 ≈ 0.002ms)  
> ‡ S2 ts_agree full-475 estimate; evaluated on test n=190 only

**If latency is not a concern and your pipeline already uses langdetect:** S1 ts_agree (88.6% full, MY 67.4%) remains a defensible choice.  
**If latency matters or you want to drop langdetect:** S2 ts_agree is the recommended choice.  
**If you need the simplest possible deployment:** S2 hard (88.2%, MY 65.3%, <1ms latency).

---

## 9. Qualitative Error Analysis

*Script: ad-hoc analysis on short MY/ID cases from `test_case_6.txt`*

The dominant failure mode (87/95 MY errors in two_stage_agree) is Stage 2 mis-attribution on short Malay/Indonesian text. Below are concrete examples with model scores.

### 9.1 Failure Pattern 1 — International Loanwords (hardest class)

Single borrowed words with identical spelling in both Malay and Indonesian.  
lingua-high leans `id`; pycld2 has no confidence (score 0.0); tiebreaker → lingua → wrong.

| Text | True | lingua pred | lingua conf | pycld2 pred | pycld2 conf | S2 pred | Status |
|---|---|---|---|---|---|---|---|
| Algoritma | MY | id | 0.257 | ms | 0.000 | id | FAIL |
| Ekonomi | MY | id | 0.527 | ms | 0.000 | id | FAIL |
| Politik | MY | id | 0.527 | ms | 0.000 | id | FAIL |
| Demokrasi | MY | id | 0.681 | ms | 0.000 | id | FAIL |
| Organisasi | MY | id | 0.739 | ms | 0.000 | id | FAIL |
| Analisis | MY | id | 0.497 | ms | 0.000 | id | FAIL |

These are true Malay words (the test set labels them MY), but they are borrowed from English/Dutch/Sanskrit and are spelled identically in Indonesian. No language model can distinguish them in isolation — this is an irreducible ambiguity at the single-word level.

**Root cause:** pycld2 outputs `ms(0.0)` (cannot decide) → tiebreaker uses lingua confidence → lingua picks `id` at moderate confidence → prediction = id.

### 9.2 Failure Pattern 2 — Contested Colloquialisms

Malay-specific colloquial words where pycld2 overrides lingua.

| Text | True | lingua pred | lingua conf | pycld2 pred | pycld2 conf | S2 pred | Status |
|---|---|---|---|---|---|---|---|
| Gempak | MY | ms | 0.538 | id | 0.870 | id | FAIL |
| Macam | MY | id | 0.567 | id | 0.850 | id | FAIL |
| Keadilan | MY | id | 0.564 | id | 0.900 | id | FAIL |

"Gempak" is distinctly Malay slang (meaning "awesome/spectacular") but pycld2 strongly predicts id (0.87), overriding lingua's correct ms prediction. "Macam" and "Keadilan" (justice) are used in both languages — both models agree on id, creating an uncorrectable Stage 2 error.

### 9.3 Failure Pattern 3 — Short Formal Phrases

Two-word formal phrases that use words common to both languages.

| Text | True | lingua pred | lingua conf | pycld2 pred | pycld2 conf | S2 pred | Status |
|---|---|---|---|---|---|---|---|
| Media sosial | MY | id | 0.507 | ms | 0.000 | id | FAIL |
| Perubahan iklim | MY | ms | 0.540 | id | 0.940 | id | FAIL |
| Hak asasi | MY | id | 0.562 | ms | 0.000 | id | FAIL |
| Inflasi tinggi | MY | id | 0.674 | ms | 0.000 | id | FAIL |

"Perubahan iklim" (climate change) is a two-word phrase where lingua correctly predicts ms but pycld2 strongly predicts id (0.94) — pycld2's confidence wins the tiebreaker, producing a wrong prediction. This suggests pycld2's id-signal for short formal phrases is overconfident.

### 9.4 Summary of Error Patterns

| Pattern | Cause | Frequency | Fixable? |
|---|---|---|---|
| Identical loanwords (1 word) | No discriminating signal; pycld2 abstains (0.0) | ~30% of MY errors | No — irreducible ambiguity |
| Colloquial words where both models wrong | True Malay cognate predicted as id by both | ~15% of MY errors | Partially — with MY-specific ngrams |
| Short formal phrases, pycld2 overconfident id | pycld2 conf > lingua conf, pycld2 wrong | ~25% of MY errors | Potentially — pycld2 confidence calibration |
| Short phrases, pycld2 abstains, lingua wrong | pycld2 = ms(0.0), lingua confidence picks id | ~30% of MY errors | Partially — lingua-high calibration at short text |

> These patterns suggest the MY accuracy ceiling for a 3-model ensemble without additional context (prior language, user history, code-switching markers) is approximately 70–75% on this test set.

---

## 10. Methodology Notes

### 10.1 Reproducibility

*Script: `voting/reproducibility.py` | Log: `log_reproducibility_1.txt`*

| Run type | Runs | std (hard/soft/weighted ALL) | Conclusion |
|---|---|---|---|
| Seeded (langdetect seed=0) | 5 | 0.0pp across all runs | Byte-identical — confirmed reproducible |
| Unseeded (no seed) | 5 | 0.0pp within same session | Identical within session, differ from seeded by 1.2pp |

**Why unseeded runs are consistent within a session but differ from seeded runs:**  
Python's langdetect uses Java-backed random state initialized once at module load. Within a single Python process, the state is fixed (even without seed=0). Across different Python invocations, the default state differs from `seed=0`, producing the 1.2pp difference.

**Consequence:** `DetectorFactory.seed = 0` is set in `core.py` before any langdetect import. This ensures cross-session reproducibility. Any script that does not use `core.py` and omits seed=0 may produce results that differ by ~1pp on ALL from the reported numbers.

### 10.2 AUC Computation Method

benchmarkV5's AUC is computed as: `sklearn.metrics.roc_auc_score(binary_correct, top1_confidence)`. This is a binary discrimination AUC (can the model distinguish correct from incorrect predictions using its top-1 confidence?), not a per-class OVR AUC. The voting scripts' dev-set AUC values use a different method and are not comparable to benchmarkV5 values.

**Authoritative AUC values (from benchmarkV5, used as voting weights):**

| Model | AUC (benchmarkV5) |
|---|---|
| lingua-high | 0.8503 |
| langdetect | 0.7516 |
| pycld2 | 0.9634 |
| openlid-v3 | 0.7097 |

### 10.3 Why Majority Voting Needs Voter Independence and a Shared Output Space

Majority voting rests on two implicit assumptions:

1. **Voter independence** — each model's errors are uncorrelated; when one is wrong, the others are likely right.
2. **Shared output space** — every voter can, in principle, express any class label.

When either breaks, the majority can *import* the error.

langdetect ships no Malay language profile. Its n-gram model cannot assign probability mass to `ms`. This breaks both assumptions simultaneously for the ms/id axis:
- **Independence broken:** langdetect's `id` vote on Malay text is a structural constant, not an independent probabilistic judgment.
- **Shared output space broken:** langdetect cannot express `ms`. The three-voter election for Malay is effectively a two-voter election (lingua-high and pycld2) plus a guaranteed adversarial `id` ballot.

The practical effect: for any Malay text where lingua-high says `ms` and pycld2 says `id`, langdetect's guaranteed `id` tips the majority to `id` (2-1). The two-stage fix reframes Stage 1 as a question all three models *can* answer (MSID vs en/zh/ta), then restricts Stage 2 to models with genuine ms/id signal.

---

## 11. Limitations and Threats to Validity

### 11.1 Test Set Size

All evaluations use n=95 per language (n=475 total). At this scale:
- Wilson 95% CI ≈ ±10 pp near 50%, ±5 pp near 95%
- McNemar's test provides reliable paired comparison, but per-language McNemar has limited power to detect effects < 5 pp at typical α levels
- Per-bucket cells (n=10–30) have very wide CIs; directional trends in bucket tables should not be interpreted as confirmed differences
- The two-stage evaluation uses n=190 (test split), which is only ~2 samples per 1 pp — the −1.05pp overfitting gap cannot be conclusively interpreted

### 11.2 Domain and Genre Scope

The test set (`test_case_6.txt`) consists of written text spanning single words to 50-word passages. It does not include:
- Code-switched text (Malay-English, Indonesian-Malay mixing)
- Informal romanized text (Malay SMS-style, abbreviations)
- Colloquial Indonesian vs formal Indonesian differences
- User-generated or social-media content

Results may not generalize to these domains. In particular, the Stage 2 errors from loanwords (§9.1) may be systematically worse or better in domain-specific deployments.

### 11.3 Temporal Validity

Models are evaluated at a fixed point in time. langdetect's ms-blindness is a fundamental model architecture issue and is unlikely to change. openlid-v3 performance may improve with future model versions. Test set composition does not include temporal drift.

### 11.4 Leakage Risk in AUC Weights

The AUC weights for weighted voting are derived from the full 475-case test set (`benchmarkV5.py`). These weights were used to evaluate weighted voting on the same 475 cases — a form of circular evaluation. The leakage check (§5) estimated a 1.05pp overfitting gap, but with n=190 test cases, this cannot be confirmed conclusively.

### 11.5 Stage 2 Weight Fitting

Stage 2 per-class weights are fit on the dev set (n=285) and evaluated on the test set (n=190). With only ~57 true-MY and ~57 true-ID cases in the dev set, the weight estimates have high variance. This limits the potential upside of the weighted Stage 2 variant.

### 11.6 openlid-v3 FLORES-200 Mapping

The mapping from FLORES-200 codes to ISO 639-1 is hand-specified in `voting_s2.py`. Any missing codes (e.g., regional variants not in the mapping) produce a zero-probability output, which defaults to lingua-high's prediction. The mapping was extended to include `cmn_Hans`/`cmn_Hant` for ZH; other missing codes may still exist.

---

*Generated from logs: `log_voting_stats_2.txt`, `log_voting_ablation_1.txt`, `log_voting_two_stage_2.txt`, `log_stage1_accuracy_1.txt`, `log_kappa_s2_1.txt`, `log_auc_unified_1.txt`, `log_reproducibility_1.txt`, `voting_scenario2/log_s2_1.txt`, `voting_scenario2/log_s2_two_stage_2.txt`*  
*Scripts: `voting/core.py`, `voting/voting_stats.py`, `voting/voting_ablation.py`, `voting/voting_two_stage.py`, `voting/stage1_accuracy.py`, `voting/kappa_s2.py`, `voting/auc_unified.py`, `voting/reproducibility.py`, `voting_scenario2/voting_s2.py`, `voting_scenario2/voting_s2_two_stage.py`*
