# Part of qqq-microstructure.
#
# The grader's randomization inference (RESULTS 27), printed under the gate
# line and never instead of it. Pinned here: grade() keeps every declared
# column and adds each night's detail; the gate's own lines are byte for byte
# the declared ones, whatever the p-values say; the activity-matched null is
# calibrated (5% of random-decision records at p <= 0.05), has power against
# a record that really picks, is exact where it enumerates, and is
# reproducible from GRADE_SEED; a record that is only exposed (hedged into a
# falling QQQ on random nights) is told apart from one that selects; only the
# three declared records are gated, and a handling suffix is its own record.

import os, sys, re, math, types, itertools
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import guard as g

NAMES = [f'N{i:02d}' for i in range(11)]
DECIDER = 'perplexity/pplx-decider-v1.1-27b-20261010@Perplexity'
JEV = 'typesafe/jev-1.13-20260917@TypeSafe'


def nights(rng, n=63, drift=0.0):
    """Simulated overnight bps: a common factor plus heavy-tailed name moves,
    QQQ tracking the factor (plus a drift, for the exposure case)."""
    f = rng.standard_t(4, n) * 40
    rr = f[:, None] + rng.standard_t(4, (n, len(NAMES))) * 60
    return rr, 0.9 * f + drift + rng.normal(0, 10, n)


def record(rr, q, skip, hedge, backend='test', model='m'):
    """The frame grade() returns, from skip [n, names] and hedge [n] flags,
    with grade()'s own arithmetic."""
    rows = []
    for d in range(len(rr)):
        r = dict(zip(NAMES, rr[d].tolist()))
        kept = [x for x, s in zip(NAMES, skip[d]) if not s]
        frozen = float(np.mean(list(r.values())))
        shadow = float(np.mean([r[x] for x in kept])) if kept else 0.0
        if hedge[d]:
            shadow -= q[d]
        rows.append(dict(date=f'2026{d:04d}', frozen=frozen, shadow=shadow,
                         diff=shadow - frozen, skipped=int(skip[d].sum()),
                         hedged=bool(hedge[d]), backend=backend, model=model,
                         rr=r, kept=kept, q=float(q[d])))
    return pd.DataFrame(rows)


def random_decisions(rng, n=63, m=len(NAMES), hedge_rate=0.25):
    """Decisions that know nothing: 0-2 skips a night on random names,
    hedges on random nights."""
    k = rng.choice(3, n, p=[.5, .3, .2])
    rank = rng.random((n, m)).argsort(1).argsort(1)
    return rank < k[:, None], rng.random(n) < hedge_rate


# ------------------------------------------------------------- grade() --

def _prints():
    D, Dn = '20261013', '20261014'
    px = {}
    for s, c, o in (('MU', 100, 101), ('STX', 100, 90), ('LRCX', 100, 99),
                    ('AMAT', 100, 100.5), ('WDC', 100, 98), ('QQQ', 100, 99)):
        px[(s, D)] = (c, c)
        px[(s, Dn)] = (o, o)
    return px


def test_grade_keeps_its_columns_and_adds_each_nights_detail():
    basket = ['MU', 'STX', 'LRCX', 'AMAT', 'WDC']
    px = _prints()
    r = {s: math.log(px[(s, '20261014')][0] / px[(s, '20261013')][1]) * 1e4
         for s in basket + ['QQQ']}
    paper = pd.DataFrame([dict(month='2026-10', q5=';'.join(basket))])
    daily = pd.DataFrame([dict(date='20261013')])
    log = pd.DataFrame([
        dict(date='20261013', decided_et='2026-10-13 15:20:00', backend='http',
             model='m', question='hold', ticker='STX', p=0.1,
             decision='skip', latency_ms=1, state_sha='x'),
        dict(date='20261013', decided_et='2026-10-13 15:20:00', backend='http',
             model='m', question='hedge_book', ticker=g.BOOK, p=0.9,
             decision='hedge', latency_ms=1, state_sha='x')])
    t = g.grade(log, paper, daily, px)
    assert list(t.columns) == ['date', 'frozen', 'shadow', 'diff', 'skipped',
                               'hedged', 'backend', 'model', 'rr', 'kept', 'q']
    assert t.rr[0].keys() == set(basket)
    assert all(math.isclose(t.rr[0][n], r[n]) for n in basket)
    assert t.kept[0] == [n for n in basket if n != 'STX']
    assert math.isclose(t.q[0], r['QQQ'])
    # the declared values are untouched
    assert math.isclose(t.frozen[0], np.mean([r[n] for n in basket]))
    assert math.isclose(t.shadow[0], np.mean([r[n] for n in basket
                                              if n != 'STX']) - r['QQQ'])
    assert t.skipped[0] == 1 and bool(t.hedged[0])
    late = log.assign(decided_et='2026-10-13 15:55:00')
    assert g.grade(late, paper, daily, px).empty


def test_the_diff_splits_into_its_skip_and_hedge_parts():
    rng = np.random.default_rng(3)
    rr, q = nights(rng)
    t = record(rr, q, *random_decisions(rng))
    line = [x for x in g.verdict(t).split('\n') if 'diff split' in x][0]
    skip, hedge = (float(w.rstrip(',')) for w in line.split()
                   if w[0] in '+-')
    assert abs(skip + hedge - t['diff'].mean()) < 0.01


# ------------------------------------------------------------- the gate --

def _declared_gate(t):
    """guard.verdict as declared (commit e9a68ca), copied so its bytes are
    pinned: the new lines go under it and it never moves."""
    if t.empty:
        return 'no decided-and-graded nights yet'
    n, m = len(t), t['diff'].mean()
    sd = t['diff'].std(ddof=1) if n > 1 else float('nan')
    tt = m / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    head = (f'{n} night(s): shadow - frozen {m:+.2f} bps/night, paired t '
            f'{tt:+.2f}; shadow sd {t.shadow.std():.0f} vs frozen '
            f'{t.frozen.std():.0f}; {int(t.skipped.sum())} name-nights '
            f'skipped, {int(t.hedged.sum())} nights hedged')
    if n < g.GATE_NIGHTS:
        return head + f'\n  gate: not before {g.GATE_NIGHTS} nights -- no verdict'
    if m > 0 and tt > g.GATE_T:
        return head + (f'\n  gate: PASSED (t > {g.GATE_T}) -- the shadow book '
                       f'may be proposed for a dated re-freeze')
    return head + f'\n  gate: not passed (t > {g.GATE_T} needed) -- the frozen ' \
        f'book stands'


def _records():
    rng = np.random.default_rng(11)
    rr, q = nights(rng)
    worst = rr == rr.min(1, keepdims=True)
    passed = record(rr, q, worst, np.zeros(63, bool))
    flat = record(rr, q, *random_decisions(rng))
    old = pd.DataFrame(dict(frozen=rng.normal(0, 100, 70), shadow=0.0,
                            diff=rng.normal(5, 10, 70), skipped=0,
                            hedged=False))
    return [passed, flat, flat.iloc[:20].reset_index(drop=True), old,
            flat.iloc[:1]]


def test_the_gate_lines_are_the_declared_ones_whatever_the_p_values(
        monkeypatch):
    recs = _records()
    gates = [_declared_gate(t) for t in recs]
    assert [x.split('\n')[1].split(' -- ')[0] for x in gates[:3]] == \
        ['  gate: PASSED (t > 2.46)', '  gate: not passed (t > 2.46 needed)',
         '  gate: not before 63 nights']
    for t, gate in zip(recs, gates):
        assert g.verdict(t).startswith(gate + '\n')
    seen = {}
    for pv in (0.0, 1.0):
        monkeypatch.setattr(g, 'randomization', lambda *a, pv=pv, **k: dict(
            p_joint=pv, p_skip=pv, p_hedge=pv))
        monkeypatch.setattr(g, 'signflip_p', lambda *a, pv=pv, **k: pv)
        seen[pv] = [g.verdict(t) for t in recs]
        for v, gate in zip(seen[pv], gates):
            assert v.split('\n')[:2] == gate.split('\n')
    for a, b in zip(seen[0.0], seen[1.0]):
        assert a.split('\n')[-1] == '  selection beats activity-matched random'
        assert b.split('\n')[-1] == '  any edge is exposure, not selection'
    assert g.verdict(pd.DataFrame()) == 'no decided-and-graded nights yet'


def test_the_new_lines_follow_the_gate_in_order():
    v = g.verdict(_records()[1]).split('\n')
    assert [x.split(':')[0] for x in v[2:6]] == [
        '  activity', '  diff split', '  sign-flip p (mean > 0)',
        '  activity-matched null']
    assert v[5].startswith('  activity-matched null: p_joint ')
    assert ', p_skip ' in v[5] and ', p_hedge ' in v[5]
    assert v[6] in ('  selection beats activity-matched random',
                    '  any edge is exposure, not selection')


# ---------------------------------------------------- the activity null --

def test_the_null_is_calibrated_on_records_that_know_nothing():
    """1,000 63-night records whose decisions ignore the outcome, B = 999:
    p_joint <= 0.05 about 5% of the time (each component, too)."""
    rng = np.random.default_rng(20261013)
    ps = []
    for i in range(1000):
        rr, q = nights(rng)
        t = record(rr, q, *random_decisions(rng), model=f'null-{i}')
        r = g.randomization(t, B=999)
        ps.append([r['p_joint'], r['p_skip'], r['p_hedge']])
    share = (np.array(ps, float) <= 0.05).mean(0)
    assert 0.03 <= share[0] <= 0.07, share
    assert ((0.03 <= share) & (share <= 0.07)).all(), share


def test_a_record_that_picks_the_worst_name_beats_random_picks():
    rng = np.random.default_rng(7)
    rr, q = nights(rng)
    t = record(rr, q, rr == rr.min(1, keepdims=True), np.zeros(63, bool))
    r = g.randomization(t)
    assert r['p_joint'] < 0.01 and r['p_skip'] < 0.01
    assert r['p_hedge'] == 'n/a'
    assert g.verdict(t).endswith('selection beats activity-matched random')
    # and one that hedges the nights QQQ fell most
    hedge = q <= np.sort(q)[14]
    r = g.randomization(record(rr, q, np.zeros_like(rr, bool), hedge))
    assert r['p_hedge'] < 0.01 and r['p_skip'] == 'n/a'


def test_exposure_is_not_selection():
    """QQQ falls 60 bps a night on average: hedging ANY nights wins, so the
    paired t passes and the sign-flip p is tiny, but hedging random nights
    wins as much, and the null says so."""
    rng = np.random.default_rng(5)
    rr, q = nights(rng, drift=-60)
    hedge = rng.random(63) < 0.5
    t = record(rr, q, np.zeros_like(rr, bool), hedge)
    v = g.verdict(t)
    assert 'gate: PASSED' in v
    assert g.signflip_p(t['diff'], 'test', 'm') < 0.01
    r = g.randomization(t)
    assert r['p_joint'] > 0.1 and r['p_hedge'] > 0.1
    assert v.endswith('any edge is exposure, not selection')


def test_degenerate_records():
    rng = np.random.default_rng(9)
    rr, q = nights(rng)
    skip, _ = random_decisions(rng)
    # every night hedged: the hedge has one assignment, the observed one
    r = g.randomization(record(rr, q, skip, np.ones(63, bool)))
    assert r['p_hedge'] == 1.0
    assert 0 < r['p_joint'] <= 1 and 0 < r['p_skip'] <= 1
    # no activity: the shadow book is the frozen book, nothing to test
    idle = record(rr, q, np.zeros_like(rr, bool), np.zeros(63, bool))
    assert g.randomization(idle) == dict(p_joint='n/a', p_skip='n/a',
                                         p_hedge='n/a')
    v = g.verdict(idle)
    assert 'activity-matched null: p_joint n/a, p_skip n/a, p_hedge n/a' in v
    assert v.endswith('any edge is exposure, not selection')
    # one night, or a frame without the nights' detail: n/a, never a crash
    assert set(g.randomization(record(rr[:1], q[:1], skip[:1], [True]))
               .values()) == {'n/a'}
    assert set(g.randomization(_records()[3]).values()) == {'n/a'}
    assert 'diff split: n/a' in g.verdict(_records()[3])


def _brute(t):
    """p_joint, p_skip, p_hedge by listing every assignment, plainly."""
    rr = [list(r.values()) for r in t.rr]
    k = list(t.skipped)
    q, h = list(t.q), list(t.hedged)
    n = len(t)

    def T(skips, hedges):
        d = []
        for i in range(n):
            kept = [x for j, x in enumerate(rr[i]) if j not in skips[i]]
            s = (np.mean(kept) if kept else 0.0) - (q[i] if hedges[i] else 0)
            d.append(s - np.mean(rr[i]))
        return np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))

    seen_s = [{j for j, x in enumerate(r) if x not in kept}
              for r, kept in zip(t.rr, t.kept)]
    T0 = T(seen_s, h)
    all_s = list(itertools.product(*[list(map(set, itertools.combinations(
        range(len(rr[i])), k[i]))) for i in range(n)]))
    all_h = [[i in c for i in range(n)]
             for c in itertools.combinations(range(n), sum(h))]
    p = lambda Ts: np.mean(np.array(Ts) >= T0 - 1e-9 * (1 + abs(T0)))
    return (p([T(s, hh) for s in all_s for hh in all_h]),
            p([T(s, h) for s in all_s]), p([T(seen_s, hh) for hh in all_h]))


def test_small_records_are_enumerated_exactly():
    rng = np.random.default_rng(2)
    rr, q = nights(rng, n=5)
    skip = np.zeros_like(rr, bool)
    skip[0, 3] = skip[2, 4] = skip[3, 0] = True
    t = record(rr, q, skip, np.array([False, False, False, True, False]))
    exact = g.randomization(t)                  # 11^3 skip sets x 5 hedges
    assert 11 ** 3 * 5 <= g.RAND_B
    want = _brute(t)
    for key, w in zip(('p_joint', 'p_skip', 'p_hedge'), want):
        assert math.isclose(exact[key], w), (key, exact[key], w)
    # the joint drawn instead of enumerated (B below its count): within
    # Monte Carlo error of the exact p; the components still enumerate
    mc = g.randomization(t, B=6000)
    assert abs(mc['p_joint'] - want[0]) < 0.02
    assert mc['p_skip'] == exact['p_skip']
    assert mc['p_hedge'] == exact['p_hedge']


def test_drawing_by_sort_agrees_with_the_table(monkeypatch):
    rng = np.random.default_rng(4)
    rr, q = nights(rng)
    t = record(rr, q, *random_decisions(rng))
    table = g.randomization(t)
    monkeypatch.setattr(g, 'SKIP_TABLE', 0)     # as if the basket were huge
    sort = g.randomization(t)
    for key in ('p_joint', 'p_skip'):
        assert abs(table[key] - sort[key]) < 0.02, (key, table, sort)


def test_reproducible_from_the_seed(monkeypatch):
    rng = np.random.default_rng(6)
    rr, q = nights(rng)
    t = record(rr, q, *random_decisions(rng), backend='http', model=DECIDER)
    a = g.randomization(t, B=999)
    assert a == g.randomization(t, B=999)
    assert a == g.randomization(t, B=999, backend='http', model=DECIDER)
    assert g.verdict(t) == g.verdict(t, 'http', DECIDER)
    x = t['diff'].to_numpy()
    assert g.signflip_p(x, 'http', DECIDER) == g.signflip_p(x, 'http',
                                                            DECIDER)
    # the stream is GRADE_SEED's and the record's own
    assert a != g.randomization(t, B=999, model=DECIDER + '#perm3.q1')
    monkeypatch.setattr(g, 'GRADE_SEED', 1)
    assert a != g.randomization(t, B=999)


def test_the_sign_flip_p_is_exact_to_20_nights_and_close_beyond():
    rng = np.random.default_rng(8)
    x = rng.normal(3, 10, 12)
    T = x.sum()
    want = np.mean([np.dot(s, x) >= T - 1e-9 for s in
                    itertools.product([1, -1], repeat=12)])
    assert g.signflip_p(x) == want
    # 21 nights: drawn; the exact p by halves (2^10 x 2^11 sums)
    x = rng.normal(2, 10, 21)
    half = lambda v: np.array([np.dot(s, v) for s in
                               itertools.product([1, -1], repeat=len(v))])
    allsums = (half(x[:10])[:, None] + half(x[10:])[None, :]).ravel()
    exact = np.mean(allsums >= x.sum() - 1e-9)
    assert abs(g.signflip_p(x, 'http', DECIDER) - exact) < 0.005
    assert g.signflip_p(-np.abs(x) - 1) == 1.0       # one-sided: mean > 0
    assert math.isnan(g.signflip_p([]))


# ---------------------------------------------------------- the records --

def test_gated_labels():
    assert [b for b, _ in g.GATED] == ['rule', 'http', 'jev']
    assert g.gated('rule', 'rule-earnings-v1')
    assert g.gated('http', DECIDER)
    assert g.gated('jev', JEV)
    assert g.gated('jev', 'typesafe/jev-1.13-20261101@Another Provider')
    for b, m in [('http', 'local:Qwen/Qwen3.5-0.8B'),       # an old decider
                 ('http', 'perplexity/pplx-decider-v1.1-27b@Perplexity'),
                 ('http', 'perplexity/pplx-decider-v1.1-27b-20261010'),
                 ('http', DECIDER + '#perm3.q1'),           # undeclared yet
                 ('jev', JEV + '#perm3.q1'),
                 ('http', JEV), ('jev', DECIDER),           # wrong slot
                 ('rule', 'rule-earnings-v2'),
                 ('jev', 'typesafe/jev-1.13x-20260917@TypeSafe'),
                 ('jev', 'typesafe/jev-1.13-20260917@A@B'),
                 ('jev', 'typesafe/jev-1.13-2026091@TypeSafe')]:
        assert not g.gated(b, m), (b, m)
    # a declared suffix gates exactly that suffix
    rx = g.decider_slot(g.DECIDER[0], '#perm3.q1')
    assert re.fullmatch(rx, DECIDER + '#perm3.q1')
    assert not re.fullmatch(rx, DECIDER + '#perm3.q2')
    assert not re.fullmatch(rx, DECIDER)
    # a model the API serves undated (cloudflare/clef answers as its own
    # slug) is declared dated=False, and its record is the gated one
    for suffix in ('', '#perm3.q1'):
        rx = g.decider_slot('cloudflare/clef', suffix, dated=False)
        for m in ('cloudflare/clef@Cloudflare',
                  'cloudflare/clef-20261009@Cloudflare'):
            assert re.fullmatch(rx, m + suffix), (m, suffix)
        for m in ('cloudflare/clef-flash@Cloudflare', 'cloudflare/clef',
                  'cloudflare/clef@A@B', 'cloudflare/clef-2026100@C'):
            assert not re.fullmatch(rx, m + suffix), (m, suffix)
        assert not re.fullmatch(g.decider_slot('cloudflare/clef', suffix),
                                'cloudflare/clef@Cloudflare' + suffix)


def test_the_gated_decider_is_the_runtimes_pin():
    import decide_local as dl
    assert g.DECIDER[0] == dl.HOSTED['http']
    assert g.DECIDER[1] == ''                       # no suffix declared today


def test_handling_suffixes_grade_as_separate_records(tmp_path, monkeypatch,
                                                     capsys):
    """guard.py --grade end to end, with a stand-in for official prints:
    '#perm3.q1' and '#perm3.q2' are two records, neither pooled with the
    other nor with the plain decider, and neither gated today."""
    basket = ['MU', 'STX', 'LRCX', 'AMAT', 'WDC']
    days = ['20261013', '20261014', '20261015', '20261016']
    rng = np.random.default_rng(12)
    px = {(s, d): tuple(100 * np.exp(rng.normal(0, .01, 2)))
          for s in basket + ['QQQ'] for d in days}
    paper = tmp_path / 'paper.csv'
    pd.DataFrame([dict(month='2026-10', q5=';'.join(basket))]).to_csv(
        paper, index=False)
    daily = tmp_path / 'daily.csv'
    pd.DataFrame(dict(date=days[:3])).to_csv(daily, index=False)
    rows = []

    def decide(backend, model, D, skip=(), hedge=False):
        for n in basket:
            rows.append(dict(date=D, decided_et=f'{D[:4]}-{D[4:6]}-{D[6:]} '
                             f'09:17:00', backend=backend, model=model,
                             question='hold', ticker=n, p=0.5,
                             decision='skip' if n in skip else 'hold',
                             latency_ms=1, state_sha='x'))
        rows.append(dict(date=D, decided_et=f'{D[:4]}-{D[4:6]}-{D[6:]} '
                         f'09:17:00', backend=backend, model=model,
                         question='hedge_book', ticker=g.BOOK, p=0.5,
                         decision='hedge' if hedge else 'none',
                         latency_ms=1, state_sha='x'))
    for D in days[:3]:
        decide('rule', 'rule-earnings-v1', D)
        decide('jev', JEV, D)
        decide('http', DECIDER, D)
        decide('http', DECIDER + '#perm3.q1', D, skip=('STX',))
    decide('http', DECIDER + '#perm3.q2', days[0], hedge=True)
    log = pd.DataFrame(rows)
    monkeypatch.setattr(g, 'read_log', lambda: log)
    monkeypatch.setattr(g, 'PAPER', str(paper))
    monkeypatch.setattr(g, 'DAILY', str(daily))
    monkeypatch.setitem(sys.modules, 'paper_bench', types.SimpleNamespace(
        official_prints=lambda names, lo, hi: px))
    monkeypatch.setattr(sys, 'argv', ['guard.py', '--grade'])
    g.main()
    heads = [x for x in capsys.readouterr().out.split('\n')
             if x.startswith('[')]
    assert len(heads) == 5
    def record_of(x):
        x = x.removeprefix('[not a gated record] ')
        return x[:x.index('] ') + 1]
    by = {record_of(x): x for x in heads}
    assert by[f'[http {DECIDER}]'].startswith(f'[http {DECIDER}] 3 night(s)')
    assert by[f'[jev {JEV}]'].startswith(f'[jev {JEV}] 3 night(s)')
    assert by['[rule rule-earnings-v1]'].startswith('[rule rule-earnings-v1] ')
    q1, q2 = by[f'[http {DECIDER}#perm3.q1]'], by[f'[http {DECIDER}#perm3.q2]']
    assert q1.startswith(f'[not a gated record] [http {DECIDER}#perm3.q1] '
                         f'3 night(s)')
    assert '3 name-nights skipped, 0 nights hedged' in q1
    assert q2.startswith(f'[not a gated record] [http {DECIDER}#perm3.q2] '
                         f'1 night(s)')
    assert '0 name-nights skipped, 1 nights hedged' in q2
    assert sum(x.startswith('[not a gated record]') for x in heads) == 2
