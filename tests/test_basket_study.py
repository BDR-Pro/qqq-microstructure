# Part of qqq-microstructure.
#
# basket_study.py on synthetic panels, no network: the pinned declaration,
# point-in-time universe and features (month T's data cannot move T's
# basket), trailing-only clustering, the cap, the cost arithmetic against
# paper_bench.order_cost, the night alignment and hedge, and the selection
# rule on hand-made tables.

import os, sys
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import basket_study as bs
import paper_bench as pb
from xsec_backtest import load_panel
from xsec_ml import FEATS


# ------------------------------------------------------------ synthetic panel

def bdays(end, start='2019-01-01'):
    return [d.strftime('%Y%m%d') for d in pd.bdate_range(start, end)]


def synth(tmp, n_names=70, end='2020-03-31', seed=0, edit=None):
    """A wide-panel-shaped directory of monthly parquet files. Names differ
    in overnight drift (so the RULE has something to rank) and dollar
    volume; `edit(df)` may alter the frame before it is written."""
    rng = np.random.default_rng(seed)
    days = bdays(end)
    rows = []
    for i in range(n_names):
        tk = f'N{i:03d}'
        drift = rng.normal(0, 8)
        px = 50.0 + i
        for d in days:
            o = px * np.exp((drift + rng.normal(0, 60)) / 1e4)
            c = o * np.exp(rng.normal(0, 80) / 1e4)
            rows.append(dict(ticker=tk, day=d, month=f'{d[:4]}-{d[4:6]}',
                             open=o, close=c, high=max(o, c) * 1.01,
                             low=min(o, c) * 0.99,
                             dollar_vol=1e9 / (i + 1) * rng.uniform(0.9, 1.1),
                             bars=390, p15=o, p30=o,
                             p60=o * np.exp(rng.normal(0, 20) / 1e4)))
            px = c
    df = pd.DataFrame(rows)
    if edit is not None:
        df = edit(df)
    os.makedirs(tmp, exist_ok=True)
    for m, g in df.groupby('month'):
        g.to_parquet(os.path.join(tmp, f'{m}.parquet'), index=False)
    return tmp


class Stub:
    """A booster stand-in: a fixed linear score of the ranked features."""
    def predict(self, X):
        X = np.asarray(X, float)
        return X @ np.linspace(1, -1, X.shape[1])


def table(path, n):
    df = bs.attach_ranks(load_panel(path))
    df['on_keep'] = bs.on_keep(df)
    t = bs.feature_table(df, n)
    t['s_MODEL'] = Stub().predict(t[FEATS])
    t['s_RULE'] = t.on12_raw
    return df, t


# ------------------------------------------------------------ the declaration

def test_the_declaration_is_pinned(tmp_path):
    assert bs.prereg_sha256() == bs.PREREG_SHA256
    src = open(bs.__file__).read()
    p = tmp_path / 'b.py'
    p.write_text(src.replace('cut at distance 0.4', 'cut at distance 0.5', 1))
    assert bs.prereg_sha256(str(p)) != bs.PREREG_SHA256
    # an edit below the rule does not move the pin
    p.write_text(src + '\n# a machinery comment\n')
    assert bs.prereg_sha256(str(p)) == bs.PREREG_SHA256


def test_grid_is_the_declared_grid():
    g = bs.grid()
    assert len(g) == 48 and len(set(g)) == 48
    assert bs.CURRENT in g and bs.RULE_BASE in g
    assert {c[0] for c in g} == {'top150', 'top300', 'top500', 'top1000'}
    assert {c[1] for c in g} == {11, 20, 40}
    assert {c[2] for c in g} == {'none', 'cap3'}


# ------------------------------------------------------------ point in time

def test_universe_reads_only_t_minus_1_ranks():
    prev = {'AAA': 1, 'SPY': 2, 'BBB': 3, 'CCC': 4}
    assert bs.universe(prev, 3) == {'AAA', 'BBB'}         # ETF out, rank cut
    assert bs.universe(prev, 4) == {'AAA', 'BBB', 'CCC'}


def test_month_t_data_cannot_move_month_t_basket(tmp_path):
    """Scramble everything in the last panel month T (volumes, prices) --
    T's scored names, features and scores must not change. Scrambling T-1's
    volumes must change T's universe (the test can fail)."""
    base = synth(str(tmp_path / 'a'))
    df, t = table(base, 60)
    T = sorted(df.month.unique())[-1]
    Tm1 = sorted(df.month.unique())[-2]

    def scramble(month, prices):
        def f(d):
            d = d.copy()
            m = d.month == month
            r = np.random.default_rng(9)
            d.loc[m, 'dollar_vol'] = d.loc[m, 'dollar_vol'].values * \
                r.uniform(0.01, 100, m.sum())
            if prices:
                for c in ('open', 'close', 'high', 'low', 'p60'):
                    d.loc[m, c] = d.loc[m, c].values * r.uniform(0.5, 1.5,
                                                                 m.sum())
            return d
        return f

    _, t2 = table(synth(str(tmp_path / 'b'), edit=scramble(T, True)), 60)
    a = t[t.tmonth == T].sort_values('ticker').reset_index(drop=True)
    b = t2[t2.tmonth == T].sort_values('ticker').reset_index(drop=True)
    assert len(a) >= 40
    pd.testing.assert_frame_equal(a, b)
    assert bs.ordering(a, 'MODEL') == bs.ordering(b, 'MODEL')
    # the trade month AFTER the panel (formable from T-1 = T's data) moved
    nx = bs.next_month(T)
    assert not t[t.tmonth == nx].reset_index(drop=True).equals(
        t2[t2.tmonth == nx].reset_index(drop=True))
    _, t3 = table(synth(str(tmp_path / 'c'), edit=scramble(Tm1, False)), 60)
    assert set(t3[t3.tmonth == T].ticker) != set(a.ticker)


def test_universe_panel_is_restricted_to_top_n_rows(tmp_path):
    """A name outside the top N in any of T-12 .. T-1 has no history in U_N's
    panel and cannot be scored; in a wider universe it can."""
    df, t = table(synth(str(tmp_path / 'a')), 60)
    T = sorted(df.month.unique())[-1]
    s60 = set(t[t.tmonth == T].ticker)
    _, t70 = table(synth(str(tmp_path / 'a')), 70)
    s70 = set(t70[t70.tmonth == T].ticker)
    assert s60 < s70
    hist = df[(df.month < T) & (df.month >= sorted(df.month.unique())[-13])]
    worst = hist.groupby('ticker').mrank.max()
    assert all(worst[s] <= 60 for s in s60)


# ------------------------------------------------------------ the cap

def test_pair_corr_matches_pandas_and_zeroes_thin_pairs():
    r = np.random.default_rng(1)
    X = r.normal(size=(63, 6))
    X[:, 1] += X[:, 0]
    X[r.uniform(size=X.shape) < 0.1] = np.nan
    X[:30, 5] = np.nan                         # only 33 days: below 40
    c = bs.pair_corr(X)
    ref = pd.DataFrame(X).corr(min_periods=40).fillna(0).to_numpy(copy=True)
    np.fill_diagonal(ref, 1)
    assert np.allclose(c, ref, atol=1e-10)
    assert (c[5, :5] == 0).all()


def test_clusters_cut_at_average_correlation_0_6():
    c = np.array([[1, .9, .65, .1],
                  [.9, 1, .65, .1],
                  [.65, .65, 1, .1],
                  [.1, .1, .1, 1]])
    lab = bs.clusters(c)
    assert lab[0] == lab[1] == lab[2] != lab[3]       # avg(.65,.65) >= .6
    c2 = c.copy()
    c2[2, :2] = c2[:2, 2] = [.65, .5]                 # average .575 < .6
    lab2 = bs.clusters(c2)
    assert lab2[0] == lab2[1] and lab2[2] != lab2[0]


def test_clusters_use_the_trailing_window_only():
    """Two independent themes before T; after T's first day every name moves
    together. The formation must not see it."""
    r = np.random.default_rng(2)
    days, p0 = 200, 150
    f1, f2 = r.normal(size=days), r.normal(size=days)
    A = np.empty((days, 8), np.float32)
    for j in range(8):
        A[:, j] = (f1 if j < 4 else f2) + 0.3 * r.normal(size=days)
    g = pd.DataFrame(dict(ticker=[f'T{j}' for j in range(8)],
                          s_MODEL=np.arange(8.0)[::-1],
                          s_RULE=np.arange(8.0)))
    tk = {f'T{j}': j for j in range(8)}
    rows = bs.window_rows(p0)
    assert rows[0] == p0 - 63 and rows[-1] == p0 - 1
    res, k = bs.form_month(g, A, rows, tk, ns=(5,), caps=('none', 'cap3'))
    assert k == 2
    B = A.copy()
    common = 50 * r.normal(size=days - p0)
    B[p0:] = common[:, None]                       # the shock after formation
    res2, k2 = bs.form_month(g, B, rows, tk, ns=(5,), caps=('none', 'cap3'))
    assert res2 == res and k2 == k
    # the cap binds on theme 1 (T0..T3 best by MODEL): 3 of them, then T4, T5
    assert res[(5, 'cap3', 'MODEL')][0] == ['T0', 'T1', 'T2', 'T4', 'T5']
    assert res[(5, 'none', 'MODEL')][0] == ['T0', 'T1', 'T2', 'T3', 'T4']
    # ...whereas a shock INSIDE the window does reach the formation
    C = A.copy()
    C[p0 - 63:p0] = (50 * r.normal(size=63))[:, None] + \
        0.01 * r.normal(size=(63, 8))
    _, k3 = bs.form_month(g, C, rows, tk, ns=(5,), caps=('cap3',))
    assert k3 == 1


def test_cap_logic():
    order = ['a1', 'a2', 'a3', 'a4', 'b1', 'a5', 'b2', 'c1']
    lab = {t: t[0] for t in order}
    assert bs.pick(order, 5, lab, 3) == ['a1', 'a2', 'a3', 'b1', 'b2']
    assert bs.pick(order, 6, lab, 3) == ['a1', 'a2', 'a3', 'b1', 'b2', 'c1']
    assert bs.pick(order, 9, lab, 3) == ['a1', 'a2', 'a3', 'b1', 'b2', 'c1']
    assert bs.pick(order, 5) == order[:5]
    assert bs.pick(order, 20) == order


def test_ordering_ties_go_to_the_ticker():
    g = pd.DataFrame(dict(ticker=['B', 'A', 'C'], s_MODEL=[1.0, 1.0, 2.0],
                          s_RULE=[0.0, 1.0, 2.0]))
    assert bs.ordering(g, 'MODEL') == ['C', 'A', 'B']
    assert bs.ordering(g, 'RULE') == ['C', 'A', 'B']


# ------------------------------------------------------------ costs

def test_vectorised_cost_is_paper_bench_order_cost():
    r = np.random.default_rng(3)
    q = np.r_[0, 1, 5, 99, 100, 101, 250, 5000, 60000,
              r.integers(0, 3000, 300)]
    px = np.r_[r.uniform(0.2, 900, len(q) - 3), 0.5, 1.2, 2000.0]
    for plan in ('tiered', 'fixed'):
        for side in ('BUY', 'SELL'):
            for day in ('20270104', '20261115', '20010312'):
                v = bs.order_cost_vec(plan, q, px, side, day)
                ref = [pb.order_cost(plan, int(a), float(b), side, day)
                       for a, b in zip(q, px)]
                assert np.allclose(v, ref, rtol=0, atol=1e-12), (plan, side)


# ------------------------------------------------------------ the nights

def tiny():
    """Two names and QQQ over four days spanning a month end. Night D is
    held with D's month's basket; its outcome is the row at d."""
    days = ['20200129', '20200130', '20200131', '20200203']
    px = {'AAA': [100., 101, 102, 103], 'BBB': [40., 41, 39, 42],
          'QQQ': [200., 201, 202, 199]}
    op = {'AAA': [None, 101.5, 101, 104], 'BBB': [None, 40.5, 41.5, 38],
          'QQQ': [None, 200.5, 201.5, 203]}
    rows = []
    for tk in ('AAA', 'BBB', 'QQQ'):
        for i, d in enumerate(days):
            o = op[tk][i] or px[tk][i]
            on = np.log(o / px[tk][i - 1]) * 1e4 if i else np.nan
            rows.append(dict(ticker=tk, day=d, month=f'{d[:4]}-{d[4:6]}',
                             open=o, close=px[tk][i], dollar_vol=1e8,
                             dpos=i, on_bps=on, on_keep=on))
    df = pd.DataFrame(rows).sort_values(['ticker', 'day']) \
        .reset_index(drop=True)
    # BBB's last night is masked (a 'split'), AAA's is not
    df.loc[(df.ticker == 'BBB') & (df.day == '20200203'), 'on_bps'] = np.nan
    months = ['2020-01', '2020-02']
    tk = {'AAA': 0, 'BBB': 1, 'QQQ': 2}
    return df, np.array(days), months, tk, px, op


def test_night_alignment_whole_shares_hedge_and_costs():
    df, cal, months, tk, px, op = tiny()
    N = bs.Nights(df, cal, months, tk)
    bk = pd.DataFrame(dict(tmonth='2020-01', ticker=['AAA', 'BBB'], nb=2))
    C = 10_000.0
    ev = bs.evaluate(N, bk, tk, capital=C)
    # January's nights: D = 0129, 0130, 0131 (the last one's outcome is
    # February's first row); 0203 has no next day
    assert list(ev.index) == ['20200129', '20200130', '20200131']
    assert list(ev.month) == ['2020-01'] * 3      # D's month, not d's
    for k, D in enumerate(ev.index):
        qa = np.floor(5000 / px['AAA'][k])
        qb = np.floor(5000 / px['BBB'][k])
        long_ = qa * px['AAA'][k] + qb * px['BBB'][k]
        hq = round(long_ / px['QQQ'][k])
        on = lambda t: np.log(op[t][k + 1] / px[t][k]) * 1e4
        valid_b = not (D == '20200131')
        mean_on = (on('AAA') + on('BBB')) / 2 if valid_b else on('AAA')
        gross = long_ / C * mean_on - hq * px['QQQ'][k] / C * on('QQQ')
        sell_b = op['BBB'][k + 1] if valid_b else px['BBB'][k]
        cost = (pb.order_cost('tiered', qa, px['AAA'][k], 'BUY', bs.FEE_DAY)
                + pb.order_cost('tiered', qa, op['AAA'][k + 1], 'SELL',
                                bs.FEE_DAY)
                + pb.order_cost('tiered', qb, px['BBB'][k], 'BUY', bs.FEE_DAY)
                + pb.order_cost('tiered', qb, sell_b, 'SELL', bs.FEE_DAY)
                + pb.order_cost('tiered', hq, px['QQQ'][k], 'SELL', bs.FEE_DAY)
                + pb.order_cost('tiered', hq, op['QQQ'][k + 1], 'BUY',
                                bs.FEE_DAY))
        r = ev.loc[D]
        assert r.gross == pytest.approx(gross, abs=1e-9)
        assert r.cost == pytest.approx(cost / C * 1e4, abs=1e-9)
        assert r.cens == pytest.approx(1.5 * long_ / C)
        assert r.net == pytest.approx(r.gross - r.cost - r.cens)
        assert r.names == (2 if valid_b else 1)
    # a name priced above its slice is not held, and is not in the hedge base
    ev2 = bs.evaluate(N, bk, tk, capital=150.0)
    assert (ev2.held == 1).all()


def test_on_keep_is_load_panel_keep_backstop(tmp_path):
    def crash(d):
        # N003 gaps x0.4 (no split ratio: a backstop night); N007 x0.5 (2:1)
        d = d.copy()
        a = d.index[(d.ticker == 'N003')][100]
        d.loc[a:, ['open', 'close', 'high', 'low', 'p60', 'p15', 'p30']] *= \
            np.where(d.loc[a:, 'ticker'] == 'N003', 0.4, 1.0)[:, None]
        b = d.index[(d.ticker == 'N007')][120]        # a 2:1 split
        d.loc[b:, ['open', 'close', 'high', 'low', 'p60', 'p15', 'p30']] *= \
            np.where(d.loc[b:, 'ticker'] == 'N007', 0.5, 1.0)[:, None]
        return d
    p = synth(str(tmp_path / 'k'), n_names=10, end='2019-07-31', edit=crash)
    base = load_panel(p)
    keep = load_panel(p, keep_backstop=True)
    got = bs.on_keep(base)
    assert np.array_equal(np.isnan(got), np.isnan(keep.on_bps.values))
    m = ~np.isnan(got)
    assert np.allclose(got[m], keep.on_bps.values[m])
    assert np.isnan(base.on_bps.values).sum() > np.isnan(got).sum()


# ------------------------------------------------------------ metrics, rule

def test_max_dd_at_fixed_notional():
    assert bs.max_dd([10, -30, 5, -10]) == pytest.approx(-0.35)
    assert bs.max_dd([-20, 50]) == pytest.approx(-0.20)
    assert bs.max_dd([5, 5]) == 0.0


def mk(rows):
    t = pd.DataFrame(rows, columns=['cfg', 'net', 'sd', 'sharpe', 'maxdd',
                                    'p1'])
    return t.set_index('cfg')


CUR = 'cur'


def test_selection_picks_the_best_qualifying_and_confirms():
    sel = mk([(CUR, 5.0, 100, 0.79, -20, -300),
              ('a', 4.1, 80, 0.81, -15, -250),     # passes S1
              ('b', 3.9, 50, 1.24, -10, -150),     # mean 78% < 80%: out
              ('c', 6.0, 86, 1.11, -12, -200),     # sd 86% > 85%: out
              ('d', 4.0, 70, 0.91, -14, -220),     # passes S1, best Sharpe
              ('e', 4.5, 78, 0.915, -11, -210)])   # 0.92 at 2 dp: best
    conf = mk([(CUR, 3.0, 120, 0.40, -30, -350),
               ('a', 2.0, 90, 0.35, -20, -300),
               ('d', 2.5, 80, 0.50, -25, -340),
               ('e', 2.4, 85, 0.45, -25, -340)])
    s = bs.select(sel, conf, CUR)
    assert sorted(s['candidates']) == ['a', 'd', 'e']
    assert s['pick'] == 'e'
    assert s['proposed'] and s['verdict'].startswith('PROPOSED')


def test_selection_tie_breaks_on_drawdown():
    sel = mk([(CUR, 5.0, 100, 0.79, -20, -300),
              ('d', 4.0, 70, 0.911, -14, -220),
              ('e', 4.5, 78, 0.909, -11, -210)])   # both 0.91 at 2 dp
    conf = mk([(CUR, 3, 120, 0.4, -30, -350), ('d', 3, 1, 0.5, -1, -1),
               ('e', 3, 1, 0.5, -1, -1)])
    assert bs.select(sel, conf, CUR)['pick'] == 'e'


def test_selection_s3_failures_and_no_candidates():
    sel = mk([(CUR, 5.0, 100, 0.79, -20, -300),
              ('a', 4.5, 80, 0.89, -15, -250)])
    for bad, which in (((CUR, 3, 120, 0.40, -30, -350),
                        ('a', 3, 90, 0.39, -20, -300)), 'sharpe'), \
                      (((CUR, 3, 120, 0.40, -30, -350),
                        ('a', 3, 90, 0.50, -31, -300)), 'maxdd'), \
                      (((CUR, 3, 120, 0.40, -30, -350),
                        ('a', 3, 90, 0.50, -20, -351)), 'p1'):
        s = bs.select(sel, mk(list(bad)), CUR)
        assert s['pick'] == 'a' and not s['proposed']
        assert not s['checks'][which] and 'stands' in s['verdict']
        assert sum(s['checks'].values()) == 2
    sel2 = mk([(CUR, 5.0, 100, 0.79, -20, -300),
               ('a', 3.0, 50, 1.5, -5, -100)])
    s = bs.select(sel2, sel2, CUR)
    assert s['pick'] is None and not s['proposed']
    assert 'no SELECTION candidate' in s['verdict']


def test_selection_with_a_non_positive_current_mean():
    sel = mk([(CUR, -1.0, 100, -0.16, -20, -300),
              ('a', -1.1, 80, -0.2, -15, -250),
              ('b', -0.9, 80, -0.18, -15, -250)])
    s = bs.select(sel, sel, CUR)
    assert s['candidates'] == ['b'] and s['mean_floor'] == -1.0


# ------------------------------------------------------------ POST-HOC fixes
# Regression tests for the review's findings (2026-10-14): the dollar P&L
# column, the IBKR volume tiers, the ETP screen, the FORWARD seam splits and
# the paired comparisons. None of them may move a declared number.

def test_dollar_pnl_is_the_shares_held_and_declared_columns_unchanged():
    df, cal, months, tk, px, op = tiny()
    N = bs.Nights(df, cal, months, tk)
    bk = pd.DataFrame(dict(tmonth='2020-01', ticker=['AAA', 'BBB'], nb=2))
    C = 10_000.0
    ev = bs.evaluate(N, bk, tk, capital=C)
    for k, D in enumerate(ev.index):
        qa = np.floor(5000 / px['AAA'][k])
        qb = np.floor(5000 / px['BBB'][k])
        long_ = qa * px['AAA'][k] + qb * px['BBB'][k]
        hq = round(long_ / px['QQQ'][k])
        pa = qa * (op['AAA'][k + 1] - px['AAA'][k])
        pb_ = qb * (op['BBB'][k + 1] - px['BBB'][k])
        if D == '20200131':           # BBB's outcome is masked: the held
            pl = pa * long_ / (qa * px['AAA'][k])   # names' mean, as declared
        else:
            pl = pa + pb_
        short = hq * (op['QQQ'][k + 1] - px['QQQ'][k])
        assert ev.loc[D, 'gross_d'] == pytest.approx((pl - short) / C * 1e4,
                                                     abs=1e-9)
    # the dollar P&L of a long is above its log proxy (convexity)
    assert bs.view(ev, 'dollar').net.ge(ev.net - 1e-9).sum() >= 1
    # the declared view is the declared frame
    pd.testing.assert_frame_equal(bs.view(ev, 'declared')[ev.columns], ev)


def test_tier_total_and_marginal_rates():
    assert bs.tier_total(0) == 0.0
    assert bs.tier_total(300_000) == pytest.approx(1050.0)
    assert bs.tier_total(3_000_000) == pytest.approx(1050 + 2_700_000 * .002)
    assert bs.tier_total(25_000_000) == pytest.approx(
        1050 + 5400 + 17_000_000 * .0015 + 5_000_000 * .001)
    vol = np.array([100_000, 250_000, 0, 1_000_000, 400_000.])
    mon = np.array(['202001', '202001', '202001', '202001', '202002'])
    r = bs.tier_rates(vol, mon)
    assert r[0] == pytest.approx(.0035)
    assert r[1] == pytest.approx((200_000 * .0035 + 50_000 * .002) / 250_000)
    assert r[2] == pytest.approx(.002)            # next share's rate
    assert r[3] == pytest.approx(.002)
    # a new month starts again in the first band
    assert r[4] == pytest.approx((300_000 * .0035 + 100_000 * .002) / 400_000)


def test_order_cost_vec_per_share_override():
    q = np.array([0, 1, 50, 100, 5000, 200000.])
    px = np.array([10, 900, 30, 0.2, 55, 12.])
    for side in ('BUY', 'SELL'):
        a = bs.order_cost_vec('tiered', q, px, side)
        b = bs.order_cost_vec('tiered', q, px, side, per_share=0.0035)
        assert np.array_equal(a, b)
        rates = np.array([.0035, .002, .0015, .001, .0005, .002])
        v = bs.order_cost_vec('tiered', q, px, side, per_share=rates)
        for i in range(len(q)):
            assert v[i] == pytest.approx(bs.order_cost_vec(
                'tiered', q[i:i + 1], px[i:i + 1], side,
                per_share=rates[i])[0], abs=1e-12)
        # the commission falls with the rate, the $0.35 minimum still binds
        assert (v <= a + 1e-12).all()
        assert v[1] == pytest.approx(a[1])        # 1 share: minimum


def test_volume_tiers_by_trade_date_month():
    df, cal, months, tk, px, op = tiny()
    N = bs.Nights(df, cal, months, tk)
    bk = pd.DataFrame(dict(tmonth='2020-01', ticker=['AAA', 'BBB'], nb=2))
    small = bs.evaluate(N, bk, tk, capital=10_000.0)
    # far below 300k shares a month the tiers ARE the flat rate
    assert np.allclose(small.cost_t, small.cost, rtol=0, atol=1e-12)
    import paper_bench as pb
    C = 2e8                                     # ~1e6 shares an order
    ev = bs.evaluate(N, bk, tk, capital=C)
    p, pt = pb.FEES['tiered'], 1 + pb.FEES['tiered']['passthru']
    orders = {}                                 # trade day -> [(q, px, side)]
    for k, D in enumerate(ev.index):
        qa = np.floor(C / 2 / px['AAA'][k])
        qb = np.floor(C / 2 / px['BBB'][k])
        hq = round((qa * px['AAA'][k] + qb * px['BBB'][k]) / px['QQQ'][k])
        sb = op['BBB'][k + 1] if D != '20200131' else px['BBB'][k]
        orders.setdefault(k, []).extend(
            [(qa, px['AAA'][k], 'BUY', D), (qb, px['BBB'][k], 'BUY', D),
             (hq, px['QQQ'][k], 'SELL', D)])
        orders.setdefault(k + 1, []).extend(
            [(qa, op['AAA'][k + 1], 'SELL', D), (qb, sb, 'SELL', D),
             (hq, op['QQQ'][k + 1], 'BUY', D)])
    vol = np.array([sum(o[0] for o in orders.get(i, [])) for i in range(4)])
    rate = bs.tier_rates(vol, [c[:6] for c in cal])
    assert rate[2] < rate[1] < rate[0] <= .0035    # January accumulates
    assert rate[3] > rate[2]                        # February resets
    want = {D: 0.0 for D in ev.index}
    for i, os_ in orders.items():
        for q, x, side, D in os_:
            flat = pb.order_cost('tiered', int(q), float(x), side, bs.FEE_DAY)
            c0 = min(max(p['min'], .0035 * q), p['max_pct'] * q * x) * pt
            c1 = min(max(p['min'], rate[i] * q), p['max_pct'] * q * x) * pt
            want[D] += flat - c0 + c1
    for D in ev.index:
        assert ev.loc[D, 'cost_t'] == pytest.approx(want[D] / C * 1e4,
                                                    rel=1e-9)
        assert ev.loc[D, 'cost_t'] < ev.loc[D, 'cost']


def test_etp_screen_is_time_aware_and_spares_stocks():
    from xsec_backtest import ETF
    assert not (bs.ETP_SCREEN & ETF)
    late = bs.etp_excluded('2026-05')
    for t in ('FNGU', 'TVIX', 'NVDX', 'QQQM', 'SKF', 'SRS', 'SGOV', 'VWO',
              'IEMG', 'UGAZ', 'DGAZ', 'GUSH', 'TECL', 'IWF', 'VCSH', 'ETH'):
        assert t in late, t
    for t in ('GOOG', 'GOOGL', 'PBR', 'COIN', 'BRK.B', 'FOXA', 'VALE',
              'DRIV', 'HERO', 'AETH', 'AABA', 'MSTR', 'NVDA'):
        assert t not in late, t
    early = bs.etp_excluded('2000-05')               # Tandy, Seagram, ...
    for t in ('TAN', 'VO', 'ETH', 'OIL', 'SIL', 'LIT', 'SPYG'):
        assert t not in early and t in late
    prev = {'AAA': 1, 'FNGU': 2, 'SPY': 3, 'BBB': 4}
    assert bs.universe(prev, 4) == {'AAA', 'FNGU', 'BBB'}       # declared
    assert bs.universe(prev, 4, late) == {'AAA', 'BBB'}
    B = pd.DataFrame(dict(cfg=['x'] * 4 + ['y'] * 2,
                          tmonth=['2000-05', '2010-05', '2010-05', '2010-05',
                                  '2010-05', '2010-05'],
                          ticker=['TAN', 'TAN', 'AAA', 'BBB', 'AAA', 'BBB']))
    sh = bs.etp_share(B)
    assert sh['x'] == pytest.approx(0.25) and sh['y'] == 0.0


def test_feature_table_hooks_leave_the_declared_path_alone(tmp_path):
    df = bs.attach_ranks(load_panel(synth(str(tmp_path / 'h'))))
    t0 = bs.feature_table(df, 60)
    t1 = bs.feature_table(df, 60, exclude=lambda T: frozenset())
    pd.testing.assert_frame_equal(t0, t1)
    T = sorted(t0.tmonth.unique())[-2]
    gone = t0[t0.tmonth == T].ticker.iloc[0]
    t2 = bs.feature_table(df, 60, exclude=lambda m: {gone} if m == T else ())
    assert gone not in set(t2[t2.tmonth == T].ticker)
    assert gone in set(t2[t2.tmonth != T].ticker)

    def hook(live, m):
        live = live.copy()
        live.loc[live.ticker == gone, 'lprice'] += 3.0
        return live
    t3 = bs.feature_table(df, 60, raw_hook=bs.lprice_bound_hook({gone: 1e3},
                                                                T), t_from=T)
    assert set(t3.tmonth) == {m for m in t0.tmonth.unique() if m >= T}
    a = t0[t0.tmonth == T].set_index('ticker').lprice
    b = t3[t3.tmonth == T].set_index('ticker').lprice
    assert b[gone] >= a[gone] and b[gone] == b.max()   # ranked to the top
    t4 = bs.feature_table(df, 60, raw_hook=hook, t_from=T)
    assert (t4[t4.tmonth == T].set_index('ticker').lprice[gone] ==
            b[gone])


def test_seam_splits_finds_split_ratio_gaps_only():
    days = ['20260330', '20260331', '20260401', '20260402']
    rows = []
    for tk, c31, o01 in (('SPL', 1000.0, 100.5), ('REV', 2.0, 40.0),
                         ('MOV', 50.0, 53.5), ('NEW', None, 10.0)):
        for i, d in enumerate(days):
            if tk == 'NEW' and i < 2:
                continue
            o = o01 if d == '20260401' else (c31 if i < 2 else o01)
            c = c31 if i < 2 else o01
            rows.append(dict(ticker=tk, day=d, month=f'{d[:4]}-{d[4:6]}',
                             open=o, close=c, dpos=i))
    fac = bs.seam_splits(pd.DataFrame(rows))
    assert set(fac) == {'SPL', 'REV'}
    assert fac['SPL'] == pytest.approx(10.0, rel=0.01)    # 10:1 split
    assert fac['REV'] == pytest.approx(1 / 20, rel=0.01)  # 1:20 reverse


def test_paired_differences():
    idx = pd.Index(['20200102', '20200103', '20200106', '20200107',
                    '20200108'])
    a = pd.DataFrame(dict(net=[3.0, 1, 4, 1, 5]), index=idx)
    b = pd.DataFrame(dict(net=[1.0, 1, 1, np.nan, 1]), index=idx)
    m, t, n = bs.paired(a, b, '20200101', '20200131')
    d = np.array([2.0, 0, 3, 4])
    assert n == 4 and m == pytest.approx(d.mean())
    assert t == pytest.approx(d.mean() / d.std(ddof=1) * 2)
    assert np.isnan(bs.paired(a, b, '20200107', '20200107')[0])
