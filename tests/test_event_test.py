# Part of qqq-microstructure.
#
# event_test.py on synthetic events and panels, no network: the strict
# decision-time rule end to end (15:19 flags night D, 15:21 and 15:29 do not
# under the declared 10-minute margin; 15:29 does and 15:31 does not at the
# bare 15:30 decision time; DST), H1's 364-day rule and its "already
# reported" veto, Holm, the fixed-effects / two-way-clustered estimator, the
# placebo / shuffle / leak-canary machinery (a planted effect is found, a
# shuffled one is not, a post-close reaction shows up only under the leaky
# filingDate placement), and the book counterfactual's arithmetic.

import os, sys, socket
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import edgar_events as ee
import event_test as et

ET = ee.ET


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('a test tried to use the network')
    monkeypatch.setattr(socket.socket, 'connect', boom)


def days(a, b, drop=()):
    return [d.strftime('%Y%m%d') for d in pd.bdate_range(a, b)
            if d.strftime('%Y%m%d') not in drop]


def events(rows):
    """rows: (cik, event, accepted 'YYYY-MM-DD HH:MM' ET, ts_kind[, extra])"""
    out = []
    for i, r in enumerate(rows):
        cik, cls, acc, kind = r[:4]
        a = pd.Timestamp(acc, tz=ET)
        row = dict(cik=cik, event=cls, accession=f'acc{i}',
                   accepted_et=a, accepted_lo_et=a - pd.Timedelta(hours=10),
                   ts_kind=kind, filing_date=a.strftime('%Y-%m-%d'),
                   json_raw=a.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                   role='subject', equity=1.0)
        row.update(r[4] if len(r) > 4 else {})
        out.append(row)
    return pd.DataFrame(out)


def universe(cal, ciks):
    """Every CIK on every night of the calendar, y = 0."""
    rows = [(c, p) for c in ciks for p in range(len(cal.days))]
    nn = pd.DataFrame(rows, columns=['cik', 'pos'])
    nn['y'] = 0.0
    nn['name_id'] = nn.cik
    return nn


def flagged_days(hid, ev, cal, nn, **kw):
    f, _ = et.flags_for(hid, ev, cal, nn, **kw)
    return sorted(set(cal.days[nn.pos.values[f.astype(bool)]]))


# ------------------------------------------------- the strict time rule

def test_strict_rule_end_to_end_declared_margin():
    cal = ee.Calendar(days('2024-01-01', '2024-01-31'))
    nn = universe(cal, [1])
    at = lambda t: flagged_days('H2', events([(1, 'bad', t, 'exact')]),
                                cal, nn)
    assert at('2024-01-09 15:19')[0] == '20240109'     # before 15:20: night D
    assert at('2024-01-09 15:21')[0] == '20240110'     # after: next night
    assert at('2024-01-09 15:29')[0] == '20240110'     # inside the margin
    got = at('2024-01-09 15:19')
    assert got == ['20240109', '20240110', '20240111', '20240112',
                   '20240115']                          # k = 5 trading nights
    assert at('2024-01-13 11:00')[0] == '20240115'     # Saturday -> Monday


def test_strict_rule_at_the_bare_decision_time(monkeypatch):
    monkeypatch.setattr(ee, 'LAG_MIN', 0)
    cal = ee.Calendar(days('2024-01-01', '2024-01-31'))
    nn = universe(cal, [1])
    at = lambda t: flagged_days('H2', events([(1, 'bad', t, 'exact')]),
                                cal, nn)[0]
    assert at('2024-01-09 15:29') == '20240109'        # 15:29 flags night D
    assert at('2024-01-09 15:31') == '20240110'        # 15:31 does not
    assert at('2024-01-09 15:30') == '20240110'        # strictly before


def test_strict_rule_under_dst_and_half_days():
    cal = ee.Calendar(days('2024-03-01', '2024-12-31'),
                      ee.rule_half_days([2024]))
    nn = universe(cal, [1])
    at = lambda t: flagged_days('H3', events([(1, 'nt', t, 'exact')]),
                                cal, nn)[0]
    assert at('2024-03-08 15:19') == '20240308'        # EST
    assert at('2024-03-11 15:19') == '20240311'        # first EDT Monday
    assert at('2024-03-11 15:21') == '20240312'
    assert at('2024-07-09 15:19') == '20240709'
    assert at('2024-11-29 12:19') == '20241129'        # half day: 12:20 cut
    assert at('2024-11-29 12:21') == '20241202'


def test_placements_for_the_canaries():
    cal = ee.Calendar(days('2024-01-01', '2024-01-31'))
    ev = events([(1, 'bad', '2024-01-09 16:05', 'exact')])
    # the JSON of a class-0 filing: wall clock = true ET, but 'Z'
    ev['json_raw'] = '2024-01-09T16:05:00.000Z'
    p = cal.pos
    assert et.event_frame(ev, 'bad', cal, 'true').n[0] == p['20240110']
    assert et.event_frame(ev, 'bad', cal, 'filingdate').n[0] == p['20240109']
    # 'Z' as UTC: 16:05Z = 11:05 ET -> the leak puts it on night D
    assert et.event_frame(ev, 'bad', cal, 'z_as_utc').n[0] == p['20240109']


def test_event_filters_role_and_equity():
    cal = ee.Calendar(days('2024-01-01', '2024-01-31'))
    ev = events([(1, 'sc13d', '2024-01-09 10:00', 'exact', {'role': 'filer'}),
                 (1, 'sc13d', '2024-01-10 10:00', 'exact'),
                 (2, 'supply', '2024-01-09 10:00', 'exact',
                  {'equity': np.nan}),
                 (2, 'supply', '2024-01-10 10:00', 'exact', {'equity': 0.0}),
                 (2, 'supply', '2024-01-11 10:00', 'exact')])
    assert len(et.event_frame(ev, 'sc13d', cal)) == 1     # subject only
    assert len(et.event_frame(ev, 'supply', cal)) == 1    # equity only


# ------------------------------------------------------------------- H1

def test_h1_364_day_rule_and_the_already_reported_veto():
    cal = ee.Calendar(days('2023-04-03', '2024-06-28'))
    src = '2023-05-02'                         # a Tuesday; +364 = 2024-04-30
    ev = events([
        (1, 'earn', f'{src} 16:05', 'exact'),   # after the close: night t
        (2, 'earn', '2023-05-03 07:00', 'exact'),   # pre-open: night before
        (3, 'earn', f'{src} 11:00', 'exact'),   # in session: not a source
        (4, 'earn', f'{src} 16:05', 'exact'),
        (4, 'earn', '2024-04-15 10:00', 'exact'),   # reported already
        (5, 'earn', f'{src} 16:05', 'exact'),
        (5, 'earn', '2024-04-30 15:25', 'exact'),   # after the cut: unknown
        (6, 'earn', f'{src} 16:05', 'exact'),
        (6, 'earn', '2024-04-30 15:00', 'exact'),   # before the cut: known
        (7, 'earn', '2024-03-01 10:00', 'exact'),   # 2023-03-?? none
    ])
    e = et.event_frame(ev, 'earn', cal)
    pairs = et.h1_flags(e, cal)
    got = {c: cal.days[d] for c, d in zip(pairs.cik, pairs.D)}
    assert got == {1: '20240430', 2: '20240430', 5: '20240430'}
    # the 2024 releases of 4/5/6 are themselves sources a year on, beyond
    # this calendar -> no flag; an unresolved source (bounds straddle the
    # open) is not used
    ev2 = events([(9, 'earn', f'{src} 16:05', 'upper')])
    ev2['accepted_lo_et'] = ev2.accepted_et - pd.Timedelta(hours=10)
    assert et.h1_flags(et.event_frame(ev2, 'earn', cal), cal).empty
    ev3 = events([(9, 'earn', f'{src} 20:00', 'upper')])   # 10:00..20:00
    assert et.h1_flags(et.event_frame(ev3, 'earn', cal), cal).empty
    ev4 = events([(9, 'earn', f'{src} 23:00', 'upper')])   # 13:00..23:00?
    ev4['accepted_lo_et'] = ev4.accepted_et - pd.Timedelta(hours=6)
    p4 = et.h1_flags(et.event_frame(ev4, 'earn', cal), cal)
    assert [cal.days[d] for d in p4.D] == ['20240430']     # 17:00..23:00 ok


def test_h1_target_on_a_holiday_moves_to_the_next_day():
    cal = ee.Calendar(days('2023-04-03', '2024-06-28', drop={'20240430'}))
    ev = events([(1, 'earn', '2023-05-02 16:05', 'exact')])
    p = et.h1_flags(et.event_frame(ev, 'earn', cal), cal)
    assert [cal.days[d] for d in p.D] == ['20240501']


# ------------------------------------------------------------ statistics

def test_holm_step_down():
    p = {'a': 0.01, 'b': 0.04, 'c': 0.03, 'd': 0.005, 'e': 0.5, 'f': 0.2}
    r = et.holm(p)
    # sorted d .005 <= .05/6, a .01 <= .05/5, c .03 > .05/4 -> stop
    assert r == {'a': True, 'b': False, 'c': False, 'd': True, 'e': False,
                 'f': False}
    assert not any(et.holm({'x': 0.2, 'y': 0.3}).values())
    assert all(et.holm({'x': 0.001, 'y': 0.02}).values())


def test_one_sided_p():
    assert et.one_sided_p(2.0, +1) == pytest.approx(0.02275, abs=1e-4)
    assert et.one_sided_p(2.0, -1) == pytest.approx(0.97725, abs=1e-4)
    assert et.one_sided_p(-2.0, -1) == pytest.approx(0.02275, abs=1e-4)
    assert et.one_sided_p(2.0, 0) == pytest.approx(0.0455, abs=1e-4)
    assert et.one_sided_p(np.nan, 1) == 1.0


def _panel(seed, n_names=200, n_nights=500, frac=0.05, delta=0.0):
    rng = np.random.default_rng(seed)
    night = np.repeat(np.arange(n_nights), n_names)
    name = np.tile(np.arange(n_names), n_nights)
    shock = rng.normal(0, 80, n_nights)[night]       # the market night
    f = (rng.random(len(night)) < frac).astype(float)
    y = shock + rng.normal(0, 100, len(night)) + delta * f
    return y, f, night, name


def test_fe_twoway_recovers_a_planted_effect_and_absorbs_nights():
    y, f, night, name = _panel(3, delta=-30.0)
    r = et.fe_twoway(y, f, night, name)
    assert r['beta'] == pytest.approx(-30.0, abs=5.0)
    assert r['t'] < -8 and r['n_flag'] == int(f.sum())
    # night fixed effects: adding any per-night constant changes nothing
    r2 = et.fe_twoway(y + 1e3 * np.sin(night), f, night, name)
    assert r2['beta'] == pytest.approx(r['beta'], abs=1e-6)
    assert r2['t'] == pytest.approx(r['t'], rel=1e-6)
    y0, f0, _, _ = _panel(4, delta=0.0)
    assert abs(et.fe_twoway(y0, f0, night, name)['t']) < 2.5
    none = et.fe_twoway(y0, np.zeros_like(f0), night, name)
    assert none['n_flag'] == 0 and np.isnan(none['t'])


def test_active_keys_window_and_clipping():
    k = et.active_keys([1, 2], [5, 9], 3, 10)
    assert set(k) == {(1 << 20) + 5, (1 << 20) + 6, (1 << 20) + 7,
                      2 * (1 << 20) + 9}
    assert len(et.active_keys([], [], 3, 10)) == 0


def test_shuffle_moves_each_event_to_another_night_of_its_name():
    pools = {1: np.array([3, 4, 5, 6]), 2: np.array([10]), 3: np.array([7, 8])}
    c, n = et.shuffle_events([1, 1, 2, 3], [3, 4, 10, 7], pools,
                             np.random.default_rng(1))
    assert list(c) == [1, 1, 3]                 # CIK 2 has one night: dropped
    assert n[0] != 3 and n[1] != 4 and n[2] == 8
    assert all(x in pools[cc] for cc, x in zip(c, n))
    c2, n2 = et.shuffle_events([1, 1, 2, 3], [3, 4, 10, 7], pools,
                               np.random.default_rng(1))
    assert (n2 == n).all()                      # a seed is a fixed draw


# ------------------------------------- placebo / shuffle / canary machinery

def _synthetic(seed=11, n_names=150, start='2015-01-01', end='2017-06-30'):
    rng = np.random.default_rng(seed)
    cal = ee.Calendar(days(start, end))
    nn = universe(cal, list(range(1, n_names + 1)))
    nd = len(cal.days)
    nn['y'] = rng.normal(0, 80, nd)[nn.pos.values] + \
        rng.normal(0, 100, len(nn))
    return rng, cal, nn


def _rand_events(rng, cal, cls, n, ciks, hour):
    rows = []
    for _ in range(n):
        d = cal.days[rng.integers(5, len(cal.days) - 10)]
        t = f'{d[:4]}-{d[4:6]}-{d[6:]} {hour}'
        rows.append((int(rng.choice(ciks)), cls, t, 'exact'))
    return rows


def test_planted_effect_found_placebo_and_shuffle_silent():
    rng, cal, nn = _synthetic()
    ciks = nn.cik.unique()
    ev = events(_rand_events(rng, cal, 'bad', 300, ciks, '10:00') +
                _rand_events(rng, cal, 'vote', 300, ciks, '10:00'))
    f, _ = et.flags_for('H2', ev, cal, nn)
    nn.loc[f.astype(bool), 'y'] -= 40.0            # planted: -40 bps/night
    args = (nn.pos.values, nn.name_id.values)
    r = et.fe_twoway(nn.y.values, et.flags_for('H2', ev, cal, nn)[0], *args)
    assert r['t'] < -6 and r['beta'] == pytest.approx(-40, abs=10)
    p0 = et.fe_twoway(nn.y.values, et.flags_for('P0', ev, cal, nn)[0], *args)
    assert abs(p0['t']) < 2                          # the placebo is silent
    pools = {int(c): np.unique(g.values) for c, g in nn.groupby('cik').pos}
    ts = [et.fe_twoway(nn.y.values, et.flags_for(
        'H2', ev, cal, nn, shuffle_seed=s, cik_nights=pools)[0], *args)['t']
        for s in et.SEEDS]
    assert abs(ts[0]) < 2                            # the binding seed
    assert np.median(np.abs(ts)) < 2


def test_canary_catches_a_post_close_reaction_only_leaky_placements_see():
    rng, cal, nn = _synthetic(seed=5)
    ciks = nn.cik.unique()
    ev = events(_rand_events(rng, cal, 'nt', 400, ciks, '16:30'))
    # the market reacts on the filing night itself (news after the close):
    # only a placement that admits post-cutoff filings can "predict" it
    e = et.event_frame(ev, 'nt', cal, 'filingdate')
    react = set(et.nn_keys(e.cik.values, e.n.values))
    hit = np.isin(et.nn_keys(nn.cik.values, nn.pos.values), list(react))
    nn.loc[hit, 'y'] -= 150.0
    args = (nn.pos.values, nn.name_id.values)
    true = et.fe_twoway(nn.y.values, et.flags_for('H3', ev, cal, nn)[0],
                        *args)
    leaky = et.fe_twoway(nn.y.values, et.flags_for(
        'H3', ev, cal, nn, placement='filingdate')[0], *args)
    assert abs(true['t']) < 2                        # leak-free: nothing
    assert leaky['t'] < -4                           # the leak signature


# ---------------------------------------------------------------- the book

def test_book_eval_skips_flagged_names_and_resizes_the_hedge():
    b = pd.DataFrame({'ticker': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
                      'day': ['d1'] * 3 + ['d2'] * 3 + ['d3'] * 2,
                      'on_bps': [30., 60., 90., 10., 20., np.nan, 5., 7.]})
    qqq = pd.Series({'d1': 10., 'd2': 0., 'd3': 1.})
    flag = np.array([1, 0, 0, 0, 0, 0, 1, 1])
    r = et.book_eval(b, flag, qqq)
    assert r.tilt['d1'] == pytest.approx(50.0)       # (30+60+90)/3 - 10
    assert r.cf['d1'] == pytest.approx(65.0)         # A skipped: 75 - 10
    assert r.cash['d1'] == pytest.approx(65.0 * 2 / 3)
    assert r.cf['d2'] == pytest.approx(r.tilt['d2']) == pytest.approx(15.0)
    assert r.n['d2'] == 2                            # NaN night not held
    assert r.cf['d3'] == 0.0 and r.skipped['d3'] == 2    # all out: flat
    none = et.book_eval(b, np.zeros(len(b)), qqq)
    assert (none.cf == none.tilt).all()


def test_name_nights_universe_and_outcome():
    from xsec_backtest import ETF
    cal = ee.Calendar(['20200102', '20200103', '20200106'])
    df = pd.DataFrame({
        'ticker': ['X', 'X', 'X', 'QQQ', 'QQQ', 'QQQ', 'SPY', 'SPY', 'Y'],
        'day': ['20200102', '20200103', '20200106'] * 2 +
               ['20200103', '20200106', '20200106'],
        'month': ['2020-01'] * 9,
        'on_bps': [np.nan, 30., 50., np.nan, 10., 20., 5., 5., 7.]})
    amap = pd.DataFrame({'ticker': ['X'], 'yr': [2020], 'cik': [42]})
    tiers = pd.Series({'X': 'CORE'})
    d = et.name_nights(df, cal, amap, tiers)
    assert 'SPY' in ETF and 'QQQ' in ETF and not d.ticker.isin(ETF).any()
    x = d[d.ticker == 'X'].sort_values('pos')
    assert list(x.pos) == [0, 1]                 # night D = the day before
    assert list(x.y) == [20.0, 30.0]             # minus QQQ
    assert x.cik.tolist() == [42, 42] and x.tier.tolist() == ['CORE'] * 2
    y = d[d.ticker == 'Y']
    assert y.cik.isna().all() and not y.mapped.any()   # unmapped: kept out


# ------------------------------------------- review fixes (post-hoc, 2026-10-14)

def test_prereg_region_is_pinned_finding_10(tmp_path):
    """Finding 10: the declarations' hash is pinned and checked each run;
    editing the declared region changes it, appending code below the
    machinery rule does not; every run leaves a ledger line."""
    assert et.prereg_sha256() == et.PREREG_SHA256
    src = open(et.__file__).read()
    p = tmp_path / 'e.py'
    p.write_text(src.replace('filed >= 2004-08-23; k = 5',
                             'filed >= 2004-08-23; k = 3', 1))
    assert et.prereg_sha256(str(p)) != et.PREREG_SHA256
    p.write_text(src + '\n# a later post-hoc note\n')
    assert et.prereg_sha256(str(p)) == et.PREREG_SHA256
    log = tmp_path / 'runs.log'
    et.log_run('dry-run', et.PREREG_SHA256, path=str(log))
    et.log_run('real', 'x' * 64, path=str(log))
    lines = log.read_text().splitlines()
    assert 'dry-run' in lines[0] and '\tMATCH\t' in lines[0]
    assert '\treal\t' in lines[1] and '\tEDITED\t' in lines[1]


def test_twoway_is_the_exact_name_and_night_projection():
    rng = np.random.default_rng(0)
    night = np.repeat(np.arange(40), 25)
    name = np.tile(np.arange(25), 40)
    keep = rng.random(len(night)) > 0.3                 # unbalanced
    night, name = night[keep], name[keep]
    v = rng.normal(0, 1, len(night)) + np.sin(night) + 0.3 * name
    tw = et.TwoWay(night, name)
    got = tw(v)
    x = v.copy()                                        # alternating projections
    for _ in range(3000):
        x -= (np.bincount(night, x) / np.bincount(night))[night]
        x -= (np.bincount(name, x) / np.bincount(name))[name]
    assert np.abs(got - x).max() < 1e-8
    assert np.abs(np.bincount(night, got)).max() < 1e-8   # night sums zero
    assert np.abs(np.bincount(name, got)).max() < 1e-8    # name sums zero


def test_name_fe_removes_name_composition_bias_finding_6():
    """Finding 6: names that file events have a higher baseline tilt. With
    night FE only, a NULL event shows a positive 'effect'; name + night FE
    removes it and still recovers a planted one."""
    rng = np.random.default_rng(3)
    n_names, n_nights = 120, 400
    night = np.repeat(np.arange(n_nights), n_names)
    name = np.tile(np.arange(n_names), n_nights)
    a = rng.normal(0, 30, n_names)                       # name baselines
    p = np.where(a > 0, 0.10, 0.02)[name]                # high-a names file
    f = (rng.random(len(name)) < p).astype(float)
    y = rng.normal(0, 60, n_nights)[night] + a[name] + \
        rng.normal(0, 100, len(name))
    biased = et.fe_twoway(y, f, night, name)
    tw = et.TwoWay(night, name)
    clean = et.fe_name_night(y, f, tw)
    assert biased['beta'] > 10 and biased['t'] > 3      # composition, not news
    assert abs(clean['t']) < 2 and abs(clean['beta']) < 5
    planted = et.fe_name_night(y - 30 * f, f, tw)
    assert planted['beta'] == pytest.approx(-30, abs=6) and planted['t'] < -5


def test_lagged_universe_uses_the_month_before_the_decision_finding_1():
    cal = ee.Calendar(['20200130', '20200131', '20200203', '20200204'])
    df = pd.DataFrame({'ticker': ['A', 'A', 'A', 'B', 'B'],
                       'month': ['2020-01', '2020-01', '2020-02', '2020-02',
                                 '2020-02'],
                       'day': ['20200130', '20200131', '20200203',
                               '20200203', '20200204']})
    nn = pd.DataFrame({'ticker': ['A', 'B', 'A', 'A'],
                       'pos': [2, 2, 1, 0]})
    got = et.lagged_member(nn, df, cal)
    # night of Feb 3 (month 2020-02): needs the 2020-01 file -> A yes, B no;
    # nights of Jan 30/31: the month before January is not in the panel
    assert list(got) == [True, False, False, False]


def test_clip_outcome_restores_backstop_nights_finding_7():
    x = np.array([-88251., -5000., -100., 3000., 3500., np.nan])
    got = et.clip_outcome(x, 3365.0)
    assert np.isnan(got[0]) and np.isnan(got[5])        # split error, missing
    assert list(got[1:5]) == [-3365.0, -100.0, 3000.0, 3365.0]


def test_leak_inflation_rule_finding_9():
    assert et.leak_inflation(12.30, 2.07, +1)            # H6 filingDate
    assert et.leak_inflation(5.20, 0.86, +1)             # H5 filingDate
    assert not et.leak_inflation(-4.71, -3.27, -1)       # H2: +1.4 only
    assert not et.leak_inflation(3.0, 2.5, +1)
    assert not et.leak_inflation(9.0, 1.0, 0)            # placebo: no sign
    assert et.leak_inflation(-4.74, 4.84, -1)            # H4 filingDate


def test_h4a_adds_supply_x_to_the_declared_h4_finding_3():
    cal = ee.Calendar(days('2020-01-01', '2020-03-31'))
    nn = universe(cal, [1, 2, 3])
    ev = events([(1, 'supply', '2020-02-03 10:00', 'exact'),
                 (2, 'supply_x', '2020-02-03 10:00', 'exact'),
                 (3, 'supply_x', '2020-02-03 10:00', 'exact',
                  {'equity': 0.0})])
    f4 = et.flags_for('H4', ev, cal, nn)[0].astype(bool)
    f4a = et.flags_for('H4a', ev, cal, nn)[0].astype(bool)
    assert set(nn.cik[f4]) == {1} and f4.sum() == 3      # k = 3
    assert set(nn.cik[f4a]) == {1, 2} and f4a.sum() == 6  # debt one is out


def _basket():
    nights = [f'd{i:02d}' for i in range(40)]
    rows = []
    for i, d in enumerate(nights):
        for nm_ in 'ABCDE':
            y = 10.0
            if nm_ == 'A' and i < 10:
                y = -200.0                                 # A's bad episode
            if nm_ == 'B' and i == 20:
                y = 50.0
            rows.append((nm_, d, y + i % 3))
    b = pd.DataFrame(rows, columns=['ticker', 'day', 'on_bps'])
    fl = ((b.ticker == 'A') & (b.day < 'd10')) | \
        ((b.ticker == 'B') & (b.day == 'd20')) | (b.day == 'd30')
    return b, fl.values, pd.Series(0.0, index=nights)


def test_book_contributions_add_up_to_the_change_finding_5():
    b, fl, qqq = _basket()
    r, chg, c = et.book_contrib(b, fl, qqq)
    assert c.c.sum() == pytest.approx(chg.sum(), abs=1e-9)
    by_night = c.groupby('day').c.sum()
    assert np.allclose(by_night.values, chg[by_night.index].values)
    assert chg['d30'] == pytest.approx(-r.tilt['d30'])      # all out: flat


def test_book_diag_names_the_concentration_finding_5():
    """Finding 5: one name's episode carried the H2 book gain; the
    diagnostics name it, give its share, the t without it and a t that
    treats its nights as one cluster."""
    b, fl, qqq = _basket()
    fl = fl & (b.day != 'd30').values
    x = et.book_diag(b, fl, qqq)
    assert x['top'] == 'A' and x['share'] > 0.9
    assert x['chg_ex'] < 0                         # without A: B's skip hurt
    assert np.isfinite(x['t_name']) and abs(x['t_name']) < abs(x['t'])
    assert x['nights'] == 11
    # the cash reading = re-spread change - level term (finding 8)
    r = et.book_eval(b, fl, qqq)
    assert x['cash'] == pytest.approx(x['chg'] - x['level'], abs=1e-9)


def _wide_panel():
    """161 names x 15 months: N000..N159 ranked by fixed dollar volume, and
    OUT -- the highest-priced name, top-ranked every month but the last,
    when its volume collapses (it leaves the top 150, not the panel)."""
    rng = np.random.default_rng(5)
    dd = pd.bdate_range('2019-01-01', '2020-03-31')
    rows = []
    names = [f'N{i:03d}' for i in range(160)] + ['OUT']
    for j, tk in enumerate(names):
        p0 = 1000.0 if tk == 'OUT' else 10.0 + j % 50
        cl = p0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dd))))
        op = cl * np.exp(rng.normal(0, 0.003, len(dd)))
        for k, d in enumerate(dd):
            dv = 1e9 if tk == 'OUT' else 1e6 * (200 - j)
            if tk == 'OUT' and d >= pd.Timestamp('2020-03-01'):
                dv = 1.0
            rows.append((tk, d.strftime('%Y%m%d'), d.strftime('%Y-%m'),
                         op[k], cl[k], max(op[k], cl[k]) * 1.01,
                         min(op[k], cl[k]) * 0.99, dv, op[k]))
    df = pd.DataFrame(rows, columns=['ticker', 'day', 'month', 'open',
                                     'close', 'high', 'low', 'dollar_vol',
                                     'p60'])
    df['prev'] = df.groupby('ticker').close.shift()
    df['on_bps'] = np.log(df.open / df.prev) * 1e4
    df['cc_bps'] = df.on_bps + np.log(df.close / df.open) * 1e4
    df['on15_bps'] = df.on_bps
    return df, ee.Calendar(df.day.unique())


class _ByPrice:
    def predict(self, X):
        return X['lprice'].values


def test_lagged_book_trades_like_xsec_live_finding_4():
    pytest.importorskip('lightgbm')                  # xsec_live imports it
    df, cal = _wide_panel()
    decl, t = et.book_table(df, _ByPrice())
    lag = et.book_table_lagged(df, _ByPrice(), cal)
    q = lambda b, T: set(b[b.tmonth == T].ticker)
    # 2020-02: every candidate is in both months -> the same basket
    assert q(lag, '2020-02') == q(decl, '2020-02') and 'OUT' in q(decl,
                                                                   '2020-02')
    # 2020-03: OUT left the top 150 in March. The declared replica drops it
    # (it needs month-T membership, unknowable live); the live rule keeps it
    assert 'OUT' not in q(decl, '2020-03') and 'OUT' in q(lag, '2020-03')
    # night D is traded with the basket of D's month: the night into the
    # first March session belongs to February's basket
    dmon = pd.Series(cal.days[lag.pos.values]).str[:6].values
    assert (dmon == lag.tmonth.str.replace('-', '').values).all()
    first_march = lag[lag.day == '20200302']
    assert set(first_march.tmonth) == {'2020-02'}
    assert set(decl[decl.day == '20200302'].tmonth) == {'2020-03'}
