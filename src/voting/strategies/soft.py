"""voting/strategies/soft.py — soft (averaged-probability) voting for a generic 3-model trio."""

from voting.core import TARGET_LANGS


def soft_vote_3(pa, pb, pc):
    """Average each language's probability across three models; pick the max."""
    combined = {l: (pa.get(l, 0.0) + pb.get(l, 0.0) + pc.get(l, 0.0)) / 3
                for l in TARGET_LANGS}
    return max(combined, key=combined.get)
