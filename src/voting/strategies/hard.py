"""voting/strategies/hard.py — hard (majority) voting for a generic 3-model trio."""

from collections import defaultdict


def hard_vote_3(p1, p2, p3, tiebreak_pred):
    """Majority vote across three predictions; ties broken by `tiebreak_pred`."""
    votes = defaultdict(int)
    for p in [p1, p2, p3]:
        if p and p != "unknown":
            votes[p] += 1
    if not votes:
        return "unknown"
    top_v = max(votes.values())
    winners = [l for l, v in votes.items() if v == top_v]
    return tiebreak_pred if tiebreak_pred in winners else winners[0]
