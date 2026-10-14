# Part of qqq-microstructure.
#
# roi_year.py answers "what does a year look like" for the traded book, and
# an adversarial review (2026-10-12) found it overstating the good tail
# (compounding a fixed-notional book), adding free return under its
# volatility stress (holding the log mean), and using a one-factor alpha.
# These tests pin each fix, plus the double-buy guard in orders.py.
#
# --hedge (2026-10-14) adds the QQQ-hedged book. Its tests pin that the
# long-only output did not move by a byte, that the hedged series, costs and
# carry are what the header says, and that the live stress uses the tilt.

import os, sys, math, io, hashlib, contextlib
import numpy as np, pandas as pd
import pytest

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


# --- the hedged book (--hedge) ---------------------------------------------

def _frozen_costs(monkeypatch):
    """Stand-ins for paper_bench's sizing and fees, so the byte pin below is
    roi_year's own output and does not move when a fee constant does."""
    def sizing(names, px, capital):
        per = capital / len(names)
        rows = []
        for n in names:
            q = int(per // px[n])
            rows.append(dict(sym=n, price=px[n], shares=q, notional=q * px[n]))
        return pd.DataFrame(rows), per
    monkeypatch.setattr(ry.pb, 'sizing', sizing)
    monkeypatch.setattr(ry.pb, 'night_cost', lambda plan, book: 0.70 * len(
        book) + 0.0001 * float(book.notional.sum()))


def _run_text(hedge=None, n=400, capital=25000, live=None):
    q5, F = _series(n=5000)
    names = [f'S{i}' for i in range(11)]
    px = {n_: 100.0 + 50 * i for i, n_ in enumerate(names)}
    px['QQQ'] = 750.0
    if live is None:
        live = np.random.default_rng(9).normal(2, 200, 33)
    kw = {} if hedge is None else dict(hedge=hedge)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = ry.run(q5, F, '999912', names, px, live, capital, 'tiered', n,
                     **kw)
    return buf.getvalue(), out


# sha256 of _run_text() under _frozen_costs, printed by roi_year.py as it
# stood before --hedge existed (commit f19df29), under numpy 2.4 and 2.5
LONG_ONLY_SHA = ('033cf46be52ccc460f07445992e0bb86'
                 'c3c613fde9600d5bad253194f0969441')


def test_hedge_zero_is_the_long_only_book_byte_for_byte(monkeypatch):
    _frozen_costs(monkeypatch)
    before, _ = _run_text()
    assert hashlib.sha256(before.encode()).hexdigest() == LONG_ONLY_SHA
    for h in (0, 0.0):
        assert _run_text(hedge=h)[0] == before
    assert 'QQQ leg' not in before and 'Reg T' not in before


def test_the_hedged_series_is_q5_minus_ratio_qqq_on_the_common_days():
    from factor import regress
    q5, F = _series(n=600)
    qqq = F.mkt_on.drop(F.index[[3, 10, 11]])           # nights QQQ lacks
    q5 = q5.drop(q5.index[[0, 50]]).iloc[::-1]          # and out of order
    y = ry.hedged(q5, qqq, 1.0)
    days = sorted(set(q5.index) & set(qqq.index))
    assert list(y.index) == days
    assert np.allclose(y.values, q5[days].values - qqq[days].values)
    assert np.allclose(ry.hedged(q5, qqq, 0.5).values,
                       q5[days].values - 0.5 * qqq[days].values)
    # QQQ overnight is a regressor, so the hedge leaves the alpha alone
    cols = ['mkt_on', 'spy_on']
    a = [regress(s[days], F.loc[days, cols])['alpha']
         for s in (q5, y)]
    assert math.isclose(a[0], a[1], rel_tol=1e-9)


def test_the_hedge_leg_is_one_short_sale_and_one_cover():
    pb = ry.pb
    q, c = ry.hedge_leg('tiered', 24_800.0, 750.0, day='20270104')
    assert q == 33                                       # 33.07, nearest
    assert math.isclose(c, pb.order_cost('tiered', 33, 750.0, 'SELL',
                                         '20270104') +
                        pb.order_cost('tiered', 33, 750.0, 'BUY', '20270104'))
    assert ry.hedge_leg('tiered', 25_200.0, 750.0)[0] == 34   # 33.6 -> up
    # the SEC fee (and the TAF outside its holiday) fall on the short sale
    sell = pb.order_cost('fixed', 33, 750.0, 'SELL', '20270104')
    buy = pb.order_cost('fixed', 33, 750.0, 'BUY', '20270104')
    assert sell > buy
    assert ry.hedge_leg('fixed', 0.0, 750.0) == (0, 0.0)  # nothing bought
    assert ry.hedge_leg('fixed', 1e4, None) == (0, 0.0)   # no price


def test_the_default_run_is_the_book_orders_py_trades():
    assert ry.args([]).hedge == orders.HEDGE[1] == 1.0
    assert ry.args(['--hedge', '0']).hedge == 0.0      # RESULTS 26's book


def test_the_hedge_matches_the_order_codes_own_helpers():
    import orders
    pb = ry.pb
    if not hasattr(orders, 'hedge_shares'):
        pytest.skip('orders.py has no hedge_shares')
    for notional in (1.0, 9_000.0, 24_800.0, 25_200.0, 247_147.3):
        assert ry.hedge_leg('tiered', notional, 746.04)[0] == \
            orders.hedge_shares(notional, 746.04)
    if hasattr(pb, 'hedge_cost'):
        for plan in ('fixed', 'tiered'):
            q, c = ry.hedge_leg(plan, 98_542.9, 746.04)
            assert math.isclose(c, pb.hedge_cost(plan, q, 746.04))


def test_the_short_carries_borrow_and_dividend_and_earns_only_above_100k():
    day = ry.DAYS / 360
    c = ry.carry(98_000.0, 100_000.0)
    assert c['credit'] == 0.0                     # IBKR pays 0 below $100k
    assert math.isclose(c['borrow'], 98_000 * 0.0040 * day)
    assert math.isclose(c['div'], 98_000 * 0.00414 / 252)
    c = ry.carry(250_000.0, 250_000.0)
    assert math.isclose(c['credit'], 150_000 * (3.88 - 1.25) / 100 * day)
    # NAV below $100k scales the rate down; nothing above the $1M tier
    assert math.isclose(ry.carry(150_000.0, 50_000.0)['credit'],
                        50_000 * 0.0263 * day * 0.5)
    assert math.isclose(ry.carry(3e6, 3e6)['credit'],
                        900_000 * 0.0263 * day)
    assert ry.carry(0.0, 1e5) == dict(borrow=0.0, credit=0.0, div=0.0)


def test_the_live_stress_uses_the_tilt_for_a_hedged_book():
    d = pd.read_csv(ry.pb.DAILY, dtype={'date': str})
    assert ry.live_series(d, 0).equals(d.q5_on_bps)
    assert ry.live_series(d, 1).equals(d.tilt_bps)
    assert np.allclose(ry.live_series(d, 1), d.q5_on_bps - d.qqq_on_bps)
    assert np.allclose(ry.live_series(d, 0.5),
                       d.q5_on_bps - 0.5 * d.qqq_on_bps)


def test_a_hedged_run_prices_both_legs_and_the_carry(capsys):
    q5, F = _series(n=5000)
    names = [f'S{i}' for i in range(11)]
    px = {n: 100.0 + 50 * i for i, n in enumerate(names)}
    px['QQQ'] = 750.0
    live = np.random.default_rng(9).normal(2, 150, 34)
    cap = 250_000
    out = ry.run(q5, F, '999912', names, px, live, cap, 'tiered', 600,
                 hedge=1.0)
    text = capsys.readouterr().out
    assert 'short 1x its notional in QQQ overnight' in text
    assert "LIVE hedged book's" in text and 'Reg T: long + short' in text
    for w in list(ry.worlds(q5, 6.0)) + ['QQQ, bought and held',
                                         'QQQ, overnight only']:
        assert w in out
    # the expected night is the hedged world's mean less every cost term
    book, _ = ry.pb.sizing(names, px, cap)
    dep = book.notional.sum() / cap
    h = ry.hedge_book('tiered', book, px, cap, 1.0)
    cost = (ry.pb.night_cost('tiered', book) + h['leg'] + h['borrow']
            - h['credit'] + h['div']) / cap * 1e4
    assert h['credit'] > 0                       # a $247k short earns some
    y = ry.hedged(q5, F.mkt_on, 1.0)
    want = (dep * np.expm1((y.values - ry.CENS) / 1e4).mean()) * 1e4 - cost
    assert math.isclose(out['as backtested']['night'], want, rel_tol=1e-9)
    assert out['nothing (zero drift)']['night'] < 0
    # the hedged book is calmer than the long one at the same capital
    lo = ry.run(q5, F, '999912', names, px, live, cap, 'tiered', 600)
    capsys.readouterr()
    spread = lambda s: s['p95'] - s['p5']
    assert spread(out['as backtested']) < spread(lo['as backtested'])


def test_a_hedge_over_reg_t_or_below_zero_is_refused():
    q5, F = _series(n=800)
    names = ['A', 'B']
    px = {'A': 10.0, 'B': 10.0, 'QQQ': 700.0}
    for h, why in ((1.5, 'Reg T'), (-1.0, '0 or more')):
        with pytest.raises(SystemExit) as ex:
            ry.run(q5, F, '999912', names, px, [], 10_000, 'tiered', 50,
                   hedge=h)
        assert why in str(ex.value)
