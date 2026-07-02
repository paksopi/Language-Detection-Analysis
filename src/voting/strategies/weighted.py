"""voting/strategies/weighted.py — AUC-weighted voting for a generic 3-model trio.

Weights below are from log_combined_3.txt (strict scoring) — same source as
voting.core.DEFAULT_WEIGHTS, which covers Scenario 1 (lingua-high + langdetect + pycld2).
S1_WEIGHTS is kept here too for side-by-side comparison scripts; S2_WEIGHTS is what
"S2 weighted" in the report (§7, §9) refers to.
"""

from voting.core import TARGET_LANGS

S1_WEIGHTS = {"lingua": 0.8503, "model2": 0.7516, "pycld2": 0.9634}
S2_WEIGHTS = {"lingua": 0.8503, "model2": 0.7097, "pycld2": 0.9634}


def weighted_vote_3(pa, pb, pc, weights):
    """AUC-weighted probability sum across three models; pick the max.

    `weights` must have keys "lingua", "model2", "pycld2" (see S1_WEIGHTS/S2_WEIGHTS above).
    """
    combined = {l: pa.get(l, 0.0) * weights["lingua"] +
                   pb.get(l, 0.0) * weights["model2"] +
                   pc.get(l, 0.0) * weights["pycld2"]
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)
