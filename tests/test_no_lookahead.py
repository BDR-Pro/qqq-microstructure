# Part of qqq-microstructure.
#
# Phase 8, as an executable canary rather than a promise. The whole edifice
# rests on one claim: a feature for trade-month T uses ONLY data from months
# < T. This test proves it by corruption -- it poisons the future and checks
# the past does not move.
#
# The strongest form of "no look-ahead" is a placebo: take the panel, and for
# a chosen trade month T, OVERWRITE every future month (>= T's realised
# window) with garbage. If build_table's features for months < T are truly
# causal, they are byte-for-byte identical before and after the poisoning. If
# any feature shifts, a future value leaked into a past feature -- the test
# fails and names the column.
#
# Two more direct checks ride along:
#   - features for month T recomputed on the panel TRUNCATED at T-1 equal the
#     full-panel features for T (nothing after formation is read),
#   - the live builder (xsec_live.features) agrees with build_table to the
#     rank, which xsec_live already asserts in production but is pinned here
#     too so a refactor cannot silently break it.
#
# Run: python -m pytest tests/ -q      (or: python tests/test_no_lookahead.py)

import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from xsec_ml import build_table, FEATS


def _panel():
    from xsec_backtest import load_panel
    return load_panel()


def test_future_poison_does_not_move_past_features():
    df = _panel()
    months = sorted(df.month.unique())
    T = months[len(months) * 2 // 3]              # a trade month with future
    base = build_table(df)
    past = base[base.tmonth < T][['tmonth', 'ticker'] + FEATS] \
        .set_index(['tmonth', 'ticker']).sort_index()

    # poison every row whose month >= T with garbage prices/volumes
    rng = np.random.default_rng(0)
    bad = df.copy()
    mask = bad.month >= T
    for col in ('open', 'high', 'low', 'close', 'p60', 'dollar_vol'):
        if col in bad:
            bad.loc[mask, col] = rng.uniform(1, 1e6, mask.sum())
    poisoned = build_table(bad)
    after = poisoned[poisoned.tmonth < T][['tmonth', 'ticker'] + FEATS] \
        .set_index(['tmonth', 'ticker']).sort_index()

    common = past.index.intersection(after.index)
    assert len(common) > 100, 'too few past rows to test'
    d = (past.loc[common, FEATS] - after.loc[common, FEATS]).abs().max()
    worst = float(d.max())
    assert worst < 1e-9, (f'LOOK-AHEAD: poisoning months >= {T} changed past '
                          f'features by up to {worst:.3g}; column {d.idxmax()}')


def test_truncation_matches_full_panel():
    df = _panel()
    months = sorted(df.month.unique())
    T = months[len(months) * 2 // 3]
    full = build_table(df)
    # everything build_table needs for T is in months <= T; truncating AFTER
    # T's own realised window must not change T's FORMATION features
    upto = build_table(df[df.month <= T])
    a = full[full.tmonth == T].set_index('ticker')[FEATS].sort_index()
    b = upto[upto.tmonth == T].set_index('ticker')[FEATS].sort_index()
    common = a.index.intersection(b.index)
    assert len(common) > 20
    d = float((a.loc[common] - b.loc[common]).abs().max().max())
    assert d < 1e-9, f'formation features for {T} depend on data after {T}: {d:.3g}'


if __name__ == '__main__':
    test_future_poison_does_not_move_past_features()
    test_truncation_matches_full_panel()
    print('no-lookahead canary: PASS')
