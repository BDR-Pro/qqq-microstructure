# Part of qqq-microstructure.
#
# roi_year.py answers "what does a year look like" for the traded book, and
# an adversarial review (2026-10-12) found it overstating the good tail
# (compounding a fixed-notional book), adding free return under its
# volatility stress (holding the log mean), and using a one-factor alpha.
# These tests pin each fix, plus the double-buy guard in orders.py.

import os, sys, math
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import roi_year as ry
import orders


def _series(n=4000, alpha=6.0, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2010-01-01', periods=n).strftime('%Y%m%d')
    mkt = pd.Series(rng.normal(4.8, 60, n), index=days)
    spy = 0.8 * mkt + pd.Series(rng.normal(-1.0, 25, n), index=days)
    q5 = alpha + 0.7 * mkt + 0.7 * spy + pd.Series(rng.normal(0, 70, n),
                                                     index=days)
    return q5, pd.DataFrame({'mkt_on': mkt, 'spy_on': spy,
                             'mkt_cc': mkt + rng.normal(1.0, 90, n)})


def test_worlds_are_the_stated_shifts_of_the_real_series():
    q5, _ = _series()
    a = 6.0
    w = ry.worlds(q5, a)
    m = q5.mean()
    assert math.isclose(w['as backtested'].mean(), m - ry.CENS)
    assert math.isclose(w['half the edge'].mean(),
                        m - ry.CENS - 0.5 * (a - ry.CENS))
    assert math.isclose(w['no edge (beta only)'].mean(), m - a)
    assert abs(np.expm1(w['nothing (zero drift)'] / 1e4).mean()) < 1e-15
    rec = q5[q5.index >= ry.RECENT]
    assert math.isclose(w['recent era (2016+)'].mean(), rec.mean() - ry.CENS)


def test_the_stress_keeps_the_arithmetic_mean_and_widens():
    q5, _ = _series()
    r = q5.values
    for k in (1.5, 2.0):
        rk = ry.scaled(r, k)
        assert math.isclose(np.expm1(rk / 1e4).mean(), np.expm1(r / 1e4).mean(),
                            rel_tol=1e-12)
        assert math.isclose(rk.std(), k * r.std(), rel_tol=1e-9)


def test_a_fixed_notional_book_adds_and_a_held_one_compounds():
    r = np.full(500, 10.0)                       # 10 bps every night
    fixed = ry.fan(r, 0.9, 3.0, 50, np.random.default_rng(0))
    per = 0.9 * math.expm1(10 / 1e4) - 3.0 / 1e4
    assert math.isclose(fixed['night'], per * 1e4)
    assert math.isclose(fixed['p50'], ry.H * per, rel_tol=1e-12)
    held = ry.fan(r, 1.0, 0.0, 50, np.random.default_rng(0), compound=True)
    assert math.isclose(held['p50'], math.exp(ry.H * 10 / 1e4) - 1,
                        rel_tol=1e-12)
    assert fixed['loss'] == 0 and fixed['dd50'] == 0


def test_the_stress_never_improves_the_expected_night():
    q5, _ = _series()
    base = ry.fan(q5.values, 1.0, 0.0, 3000, np.random.default_rng(3))
    hot = ry.fan(q5.values, 1.0, 0.0, 3000, np.random.default_rng(3), k=2.0)
    assert math.isclose(hot['night'], base['night'], rel_tol=1e-9)
    assert hot['p95'] - hot['p5'] > 1.7 * (base['p95'] - base['p5'])
    assert hot['loss'] > base['loss']


def test_run_end_to_end_prints_every_world_and_both_benchmarks(capsys):
    q5, F = _series(n=5000)
    names = [f'S{i}' for i in range(11)]
    px = {n: 100.0 + 50 * i for i, n in enumerate(names)}
    px['QQQ'] = 750.0
    live = np.random.default_rng(9).normal(2, 200, 33)
    out = ry.run(q5, F, '999912', names, px, live, 25000, 'tiered', 800)
    text = capsys.readouterr().out
    for w in list(ry.worlds(q5, 6.0)) + ['QQQ, bought and held',
                                         'QQQ, overnight only']:
        assert w in text and w in out
    assert 'THE YEAR AFTER A GO VERDICT' in text
    assert "LIVE basket's volatility" in text
    assert out['nothing (zero drift)']['night'] < 0      # costs alone


def test_a_missing_price_is_refused_not_simulated(tmp_path):
    f = tmp_path / 'p.csv'
    f.write_text('A,10.0\nQQQ,700.0\n')
    try:
        ry.prices(['A', 'B', 'QQQ'], path=str(f))
        assert False, 'a missing price must stop the run'
    except SystemExit as ex:
        assert "'B'" in str(ex)
    assert ry.prices(['A', 'QQQ'], path=str(f)) == {'A': 10.0, 'QQQ': 700.0}


def test_the_close_leg_tops_up_and_never_doubles():
    rows = [('MU', 1000.0, 2, 2000.0), ('BRK.B', 400.0, 5, 2000.0),
            ('IREN', 35.0, 64, 2240.0)]
    held = {'MU': 2, 'BRK B': 3}                 # last night's sell failed
    got = {n: q for n, _, q, _ in orders.close_quantities(rows, held)}
    assert got == {'MU': 0, 'BRK.B': 2, 'IREN': 64}
