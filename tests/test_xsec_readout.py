# Part of qqq-microstructure.
#
# xsec_readout.py: the non-binding readouts of the paper log. The trailing
# beta never sees the night it adjusts, and a residual removes the market's
# noise without changing what the tilt measures.

import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import xsec_readout as xr


def daily(n=200, beta=2.4, alpha=5.0, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(4, 70, n)
    q5 = alpha + beta * q + rng.normal(0, 60, n)
    return pd.DataFrame({'date': [f'2026{i:04d}' for i in range(n)],
                         'q5_on_bps': q5, 'qqq_on_bps': q})


def test_the_trailing_beta_uses_only_earlier_nights():
    d = daily()
    q5, q = d.q5_on_bps.to_numpy(), d.qqq_on_bps.to_numpy()
    b = xr.trailing_beta(q5, q)
    assert np.isnan(b[:xr.TRAIL_MIN]).all() and not np.isnan(b[xr.TRAIL_MIN])
    poked = q5.copy()
    poked[100] += 1e4                     # a shock on night 100 ...
    b2 = xr.trailing_beta(poked, q)
    assert b2[100] == b[100]              # ... cannot move night 100's beta
    assert b2[101] != b[101]


def test_residuals_cut_the_noise_and_estimate_the_alpha():
    r = xr.readout(daily(n=4000), '2026-10-14').set_index('readout')
    assert r.loc['resid_trailing', 'sd'] < r.loc['tilt', 'sd'] < \
        r.loc['raw', 'sd']
    assert abs(r.loc['resid_trailing', 'mean'] - 5.0) < 3.0
    assert r.loc['tilt', 'ref_edge'] == 4.9


def test_too_few_nights_reads_nothing():
    r = xr.readout(daily(n=1), 'x').set_index('readout')
    assert np.isnan(r.loc['raw', 't'])
    assert r.loc['resid_trailing', 'nights'] == 0
