"""voting/strategies/two_stage.py — Two-stage voting decision logic for Scenario 2.

Stage 1: collapse ms/id -> MSID, hard vote on {en, MSID} across all 3 S2 models.
Stage 2 variants (triggered only if Stage 1 = MSID):
  s2_agree_lc: lingua + pycld2 agree on ms/id; confidence tiebreaker (mirrors S1's two_stage_agree).
  s2_3agree:   all 3 S2 models vote on ms/id.
  fit_weighted(dev_results): fits per-class dev-accuracy weights for lingua + pycld2, returns a
               decision function (mirrors S1's two_stage_weighted from voting/voting_two_stage.py).

Report cross-reference: these implement the "S2 ts_agree" / "S2 ts_3agree" / "S2 ts_weighted"
rows in §9's master table. `s1_ts_agree` bridges the comparison — it applies the same two-stage
shape to Scenario 1's (lingua-high + langdetect + pycld2) results so the two scenarios can be
McNemar-tested against each other on equal footing.
"""

from collections import Counter, defaultdict

MSID = "msid"


def collapse_msid(pred):
    return MSID if pred in ("ms", "id") else pred


def stage1_s2(r):
    """Majority vote (lingua+openlid+pycld2) on collapsed {en,MSID,zh,ta}; lingua ties."""
    p1 = collapse_msid(r["lingua_pred"])
    p2 = collapse_msid(r["openlid_pred"])
    p3 = collapse_msid(r["cld2_pred"])
    votes = Counter(p for p in [p1, p2, p3] if p != "unknown")
    if not votes:
        return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return p1 if p1 in winners else winners[0]


def s2_agree_lc(r):
    """lingua + pycld2 only — mirrors S1's two_stage_agree."""
    lp, cp = r["lingua_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    if l == c:
        return l
    lc = max(lp.get("ms", 0), lp.get("id", 0))
    cc = max(cp.get("ms", 0), cp.get("id", 0))
    return l if lc >= cc else c


def s2_3agree(r):
    """All 3 S2 models vote on ms/id."""
    lp, op, cp = r["lingua_p"], r["openlid_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    o = "ms" if op.get("ms", 0) >= op.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    votes = Counter([l, o, c])
    if votes["ms"] != votes["id"]:
        return max(votes, key=votes.get)
    confs = [(max(lp.get("ms", 0), lp.get("id", 0)), l),
             (max(op.get("ms", 0), op.get("id", 0)), o),
             (max(cp.get("ms", 0), cp.get("id", 0)), c)]
    return max(confs, key=lambda x: x[0])[1]


def fit_weighted(dev_results, log=print):
    """Per-class dev-accuracy weights for lingua + pycld2; returns a Stage-2 decision function."""
    acc = {m: {iso: [0, 0] for iso in ("ms", "id")} for m in ("l", "c")}
    for r in dev_results:
        exp = r["expected_iso"]
        if exp not in ("ms", "id"):
            continue
        lp, cp = r["lingua_p"], r["cld2_p"]
        for m, p in [("l", lp), ("c", cp)]:
            pred = "ms" if p.get("ms", 0) >= p.get("id", 0) else "id"
            acc[m][exp][1] += 1
            acc[m][exp][0] += int(pred == exp)
    w = {m: {iso: acc[m][iso][0] / max(acc[m][iso][1], 1) for iso in ("ms", "id")}
         for m in ("l", "c")}
    log(f"  [Dev-fitted weights] w_lingua_ms={w['l']['ms']:.4f}  w_lingua_id={w['l']['id']:.4f}")
    log(f"                       w_cld2_ms={w['c']['ms']:.4f}    w_cld2_id={w['c']['id']:.4f}")

    def _fn(r):
        lp, cp = r["lingua_p"], r["cld2_p"]
        ms_score = w["l"]["ms"] * lp.get("ms", 0) + w["c"]["ms"] * cp.get("ms", 0)
        id_score = w["l"]["id"] * lp.get("id", 0) + w["c"]["id"] * cp.get("id", 0)
        return "ms" if ms_score >= id_score else "id"

    return _fn


def s2_hard_pred(r):
    votes = Counter(p for p in [r["lingua_pred"], r["openlid_pred"], r["cld2_pred"]] if p != "unknown")
    if not votes:
        return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return r["lingua_pred"] if r["lingua_pred"] in winners else winners[0]


def s1_ts_agree(r):
    """Same two-stage shape as s2_agree_lc, but for S1 (lingua+langdetect+pycld2) results —
    used to compare Scenario 1 and Scenario 2 two-stage voting head to head."""
    p1 = collapse_msid(r["lingua_pred"])
    p2 = collapse_msid(r["ld_pred"])
    p3 = collapse_msid(r["cld2_pred"])
    votes = Counter(p for p in [p1, p2, p3] if p != "unknown")
    if not votes:
        return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    s1 = p1 if p1 in winners else winners[0]
    if s1 != MSID:
        return s1
    lp, cp = r["lingua_p"], r["cld2_p"]
    l = "ms" if lp.get("ms", 0) >= lp.get("id", 0) else "id"
    c = "ms" if cp.get("ms", 0) >= cp.get("id", 0) else "id"
    if l == c:
        return l
    lc = max(lp.get("ms", 0), lp.get("id", 0))
    cc = max(cp.get("ms", 0), cp.get("id", 0))
    return l if lc >= cc else c
