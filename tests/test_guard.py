# Part of qqq-microstructure.
#
# guard.py asks a Jev-style decision model about tonight and grades the shadow
# book it implies. Pinned here: the request is the documented /v1/systemone
# shape (checked by a real local HTTP server that rejects anything else, and
# that answers 429 once to exercise the back-off); a night is never
# re-decided; a decision at/after the MOC cutoff is refused and, if one is in
# the log anyway, the grader drops it; the shadow arithmetic is exact; the
# pre-declared gate says nothing before 63 nights.

import os, sys, json, math, threading, datetime as dt
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import guard as g

NAMES = ['MU', 'STX', 'LRCX', 'AMAT', 'WDC']


class Fake(BaseHTTPRequestHandler):
    calls = 0

    def log_message(self, *a):
        pass

    def do_POST(self):
        Fake.calls += 1
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if Fake.calls == 1:                    # exercise the 429 back-off
            self.send_response(429); self.end_headers(); return
        ok = (self.path == '/v1/systemone' and set(body) == {'state', 'model',
              'questions'} and all(q['type'] == 'noul' and q['instructions']
                                   and set(q['criteria']) == {'true', 'false'}
                                   for q in body['questions'].values()))
        if not ok:
            self.send_response(422); self.end_headers(); return
        ans = {k: {'type': 'noul', 'noul': 0.2 if k == 'hold_STX' else
                   (0.9 if k == 'hedge_book' else 0.8)}
               for k in body['questions']}
        out = json.dumps({'model': 'jev-1.13.0', 'answers': ans,
                          'usage': {'input_tokens': 10, 'output_tokens': 0}})
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(out.encode())


def test_the_client_speaks_the_documented_api(monkeypatch):
    monkeypatch.setattr(g.time, 'sleep', lambda s: None)
    srv = HTTPServer(('127.0.0.1', 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        Fake.calls = 0
        url = f'http://127.0.0.1:{srv.server_port}/v1/systemone'
        p, model, ms = g.ask_http({'x': 1}, NAMES, url, key='k')
        assert Fake.calls == 2                         # 429, then success
        assert model == 'jev-1.13.0'
        assert p['STX'] == 0.2 and p['MU'] == 0.8 and p[g.BOOK] == 0.9
    finally:
        srv.shutdown()


def test_a_shared_bias_skips_nothing_but_an_outlier_is_skipped(tmp_path):
    """The first real run: every name at 0.35-0.41. A bare p < 0.5 would skip
    all of them every night; the pre-declared rule skips only a name the
    model singles out."""
    early = dt.datetime(2026, 10, 13, 15, 20, tzinfo=g.ET)
    names = ['A', 'B', 'C', 'D', 'E']
    flat = lambda s, n: ({'A': .35, 'B': .38, 'C': .40, 'D': .41, 'E': .37,
                          g.BOOK: .29}, 'm', 1)
    path = str(tmp_path / 'a.csv')
    g.decide('http', names, '20261013', early, {}, path, flat)
    assert set(g.read_log(path).decision) == {'hold', 'none'}
    odd = lambda s, n: ({'A': .12, 'B': .38, 'C': .40, 'D': .41, 'E': .37,
                         g.BOOK: .7}, 'm', 1)
    path = str(tmp_path / 'b.csv')
    g.decide('http', names, '20261013', early, {}, path, odd)
    d = g.read_log(path).set_index('ticker').decision
    assert d['A'] == 'skip' and (d.drop(['A', g.BOOK]) == 'hold').all()
    assert d[g.BOOK] == 'hedge'


def test_decide_appends_once_and_refuses_late_or_repeated(tmp_path):
    path = str(tmp_path / 'guard_log.csv')
    ask = lambda s, n: ({'MU': 0.9, 'STX': 0.1, g.BOOK: 0.7}, 'm1', 12)
    early = dt.datetime(2026, 10, 13, 15, 20, tzinfo=g.ET)
    g.decide('http', ['MU', 'STX'], '20261013', early, {'a': 1}, path, ask)
    log = g.read_log(path)
    assert list(log.decision) == ['hold', 'skip', 'hedge']
    for bad in (early, dt.datetime(2026, 10, 14, 15, 51, tzinfo=g.ET)):
        try:
            g.decide('http', ['MU'], bad.strftime('%Y%m%d'), bad, {}, path, ask)
            assert False, 'must refuse'
        except SystemExit as ex:
            assert 'already decided' in str(ex) or 'cutoff' in str(ex)
    assert len(g.read_log(path)) == 3                 # nothing appended
    # the earnings-rule baseline decides the same night once, on its own
    # record; a second rule decision is refused like any other
    g.decide('rule', ['MU', 'STX'], '20261013', early, {'names': {}}, path,
             g.ask_rule)
    try:
        g.decide('rule', ['MU'], '20261013', early, {'names': {}}, path,
                 g.ask_rule)
        assert False, 'must refuse'
    except SystemExit as ex:
        assert 'already decided by the rule backend' in str(ex)
    assert list(g.read_log(path).backend) == ['http'] * 3 + ['rule'] * 3


def _prints():
    D, Dn = '20261013', '20261014'
    px = {}
    for s, c, o in (('MU', 100, 101), ('STX', 100, 90), ('LRCX', 100, 99),
                    ('AMAT', 100, 100.5), ('WDC', 100, 98), ('QQQ', 100, 99)):
        px[(s, D)] = (c, c)
        px[(s, Dn)] = (o, o)
    return px


def test_the_shadow_arithmetic_is_exact():
    px = _prints()
    r = {s: math.log(px[(s, '20261014')][0] / px[(s, '20261013')][1]) * 1e4
         for s in NAMES + ['QQQ']}
    paper = pd.DataFrame([dict(month='2026-10', q5=';'.join(NAMES))])
    daily = pd.DataFrame([dict(date='20261013')])
    log = pd.DataFrame([
        dict(date='20261013', decided_et='2026-10-13 15:20:00', backend='http',
             model='m', question='hold', ticker='STX', p=0.1,
             decision='skip', latency_ms=1, state_sha='x'),
        dict(date='20261013', decided_et='2026-10-13 15:20:00', backend='http',
             model='m', question='hedge_book', ticker=g.BOOK, p=0.9,
             decision='hedge', latency_ms=1, state_sha='x')])
    t = g.grade(log, paper, daily, px)
    frozen = np.mean([r[n] for n in NAMES])
    shadow = np.mean([r[n] for n in NAMES if n != 'STX']) - r['QQQ']
    assert math.isclose(t.frozen[0], frozen) and math.isclose(t.shadow[0], shadow)
    assert t.skipped[0] == 1 and bool(t.hedged[0])
    # a decision stamped after the cutoff is dropped by the grader
    late = log.assign(decided_et='2026-10-13 15:55:00')
    assert g.grade(late, paper, daily, px).empty


def test_the_gate_says_nothing_before_63_nights_and_then_judges():
    rng = np.random.default_rng(1)
    few = pd.DataFrame(dict(frozen=rng.normal(0, 100, 20), shadow=0.0,
                            diff=rng.normal(5, 1, 20), skipped=0, hedged=False))
    assert 'no verdict' in g.verdict(few)
    many = pd.DataFrame(dict(frozen=rng.normal(0, 100, 70), shadow=0.0,
                             diff=rng.normal(5, 10, 70), skipped=0,
                             hedged=False))
    assert 'PASSED' in g.verdict(many)
    flat = many.assign(diff=rng.normal(0, 10, 70) - 2)
    assert 'frozen book stands' in g.verdict(flat)


def test_the_rule_baseline_skips_earnings_and_never_hedges():
    st = {'names': {'MU': {'earnings_tonight': True}, 'STX': {}}}
    p, model, _ = g.ask_rule(st, ['MU', 'STX'])
    assert p == {'MU': 0.0, 'STX': 1.0, g.BOOK: 0.0}


def test_rule_reads_tev1s_opposite_leans_the_same_way():
    """Tev1 0.8B and 4B on 2026-10-12 (dry run, RESULTS 27): one leans to
    hold, the other to skip, on every name. The rule judges each against its
    own night's median, so neither skips the basket wholesale; only the 4B's
    one outlier (SNDK, 0.0584 under a 0.0597 bar) goes."""
    names = ['MU', 'BE', 'STX', 'SNDK', 'NBIS', 'IREN', 'LRCX', 'WDC', 'AMAT',
             'INTC', 'CAT']
    small = dict(zip(names, [.8097, .9977, .9950, .9992, .9970, .7933, .9098,
                             .9403, .6352, .8270, .9662]), **{g.BOOK: .8811})
    big = dict(zip(names, [.3462, .2213, .1196, .0584, .0635, .1597, .1515,
                           .3179, .2095, .0883, .2849]), **{g.BOOK: .6131})
    assert set(g.rule(small, names).values()) == {'hold', 'hedge'}
    d = g.rule(big, names)
    assert [n for n in names if d[n] == 'skip'] == ['SNDK']
    assert d[g.BOOK] == 'hedge'


def test_a_name_exactly_the_margin_under_the_median_is_skipped():
    """Jev answers in hundredths, so ties with the bar happen (its live
    answers on 2026-10-13, RESULTS 27). 'At least 0.10 below' skips every
    exact tie, whatever the float rounding of median - 0.10."""
    for med, low in [(.56, .46), (.59, .49), (.54, .44), (.57, .47),
                     (.58, .48), (.55, .45)]:
        p = {'A': low, 'B': med, 'C': med + .1, g.BOOK: .3}
        assert g.rule(p, ['A', 'B', 'C'])['A'] == 'skip', (med, low)
        p['A'] = round(low + .01, 2)                  # 0.09 under: held
        assert g.rule(p, ['A', 'B', 'C'])['A'] == 'hold', (med, low)


def test_the_state_fits_a_small_deciders_prompt():
    """Ollama's Tev1 refuses a prompt over 2,050 tokens and never truncates
    (it refused schema 1 at 2,762). Schema 2 sends summaries: an 11-name
    state stays far inside, at well under 4 characters a token."""
    names = ['MU', 'BE', 'STX', 'SNDK', 'NBIS', 'IREN', 'LRCX', 'WDC', 'AMAT',
             'INTC', 'CAT']
    idx = pd.bdate_range('2026-07-01', '2026-10-12')
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product([names + ['QQQ'], ['Open', 'Close']])
    px = pd.DataFrame(100 * np.exp(rng.normal(0, .02, (len(idx), len(cols)))
                                   .cumsum(0)), index=idx, columns=cols)
    st = g.build_state(names, '20261012', download=lambda *a, **k: px,
                       earnings={'MU': True})
    assert st['schema'] == g.STATE_SCHEMA == 2
    assert set(st['names']) == set(names) and 'QQQ' in st
    assert st['names']['MU']['earnings_tonight'] is True
    # nothing from the decision day itself
    early = g.build_state(names, '20261012', download=lambda *a, **k:
                          px[px.index < '2026-10-12'])
    assert early['names'] == {**st['names'], 'MU': {**st['names']['MU'],
                                                     'earnings_tonight': False}}
    body = g.request_body(st, names, 'm')
    for q in body['questions'].values():
        assert len(json.dumps(st)) + len(json.dumps(q)) < 2050 * 2
