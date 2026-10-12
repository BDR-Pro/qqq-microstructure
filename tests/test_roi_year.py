# Part of qqq-microstructure.
#
# roi_year.py answers "what does a year look like" for the traded book. Its
# parts must be exactly what they claim: the alpha/beta split recovers a
# planted truth, each world is the stated mean shift of a REAL series, the
# account arithmetic (deployed fraction, nightly cost) is exact, and the
# live-volatility stress widens the fan without moving its centre.

import os, sys, math
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import roi_year as ry


def _series(n=4000, alpha=6.0, beta=1.4, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2010-01-01', periods=n).strftime('%Y%m%d')
    qqq = pd.Series(rng.normal(4.8, 60, n), index=days)
    q5 = alpha + beta * qqq + pd.Series(rng.normal(0, 80, n), index=days)
    return q5, qqq


def test_alpha_beta_recovers_a_planted_split():
    q5, qqq = _series(n=40000)
    a, b = ry.alpha_beta(q5, qqq)
    assert abs(a - 6.0) < 1.0 and abs(b - 1.4) < 0.03


def test_worlds_are_the_stated_shifts_of_the_real_series():
    q5, qqq = _series()
    a, _ = ry.alpha_beta(q5, qqq)
    w = ry.worlds(q5, a)
    m = q5.mean()
    assert math.isclose(w['as backtested'].mean(), m - ry.CENS)
    assert math.isclose(w['half the edge'].mean(),
                        m - ry.CENS - 0.5 * (a - ry.CENS))
    assert math.isclose(w['no edge (beta only)'].mean(), m - a)
    assert abs(w['nothing (zero drift)'].mean()) < 1e-9
    rec = q5[q5.index >= ry.RECENT]
    assert len(w['recent era (2016+)']) == len(rec)
    assert math.isclose(w['recent era (2016+)'].mean(), rec.mean() - ry.CENS)


def test_account_arithmetic_is_exact_on_a_constant_night():
    r = np.full(500, 10.0)                  # 10 bps every night, no noise
    s = ry.fan(r, deployed=0.9, cost_bps=3.0, k=1.0, n=50,
               rng=np.random.default_rng(0))
    assert math.isclose(s['night'], 0.9 * 10 - 3.0)
    want = math.exp(ry.H * 6.0 / 1e4) - 1
    for q in ('p5', 'p50', 'p95'):
        assert math.isclose(s[q], want, rel_tol=1e-12)
    assert s['loss'] == 0 and s['dd50'] == 0


def test_the_volatility_stress_widens_the_fan_not_its_centre():
    q5, _ = _series()
    r = q5.values
    base = ry.fan(r, 1.0, 0.0, 1.0, 4000, np.random.default_rng(3))
    hot = ry.fan(r, 1.0, 0.0, 2.0, 4000, np.random.default_rng(3))
    assert math.isclose(base['night'], hot['night'])
    assert hot['p95'] - hot['p5'] > 1.7 * (base['p95'] - base['p5'])
    assert hot['loss'] > base['loss']
