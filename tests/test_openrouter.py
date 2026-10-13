# Part of qqq-microstructure.
#
# The hosted engine (decide_local --engine hosted): Perplexity Decider and
# Jev through OpenRouter's Decisions API, against a stand-in for it. The
# request is Jev's shape with the pinned model and the key from the
# environment; the record logs the dated build and the provider; an answer
# from any other model is refused; one decider failing never costs the
# others' rows; there is no hosted night without a key.

import os, sys, io, json, glob, types, hashlib, itertools, datetime as dt
import urllib.error
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import guard as g
import decide_local as dl

NAMES = ['MU', 'STX', 'INTC']
NOW = dt.datetime(2026, 10, 13, 9, 17, tzinfo=g.ET)
ST = {'names': {'MU': {'earnings_tonight': True}, 'STX': {}, 'INTC': {}}}
BUILD = {'perplexity/pplx-decider-v1.1-27b':
         ('perplexity/pplx-decider-v1.1-27b-20261010', 'Perplexity'),
         'typesafe/jev-1.13': ('typesafe/jev-1.13-20260917', 'TypeSafe')}


class Decisions:
    """Stands in for POST https://openrouter.ai/api/alpha/decisions. p: a
    probability, or p(n, question id, questions as sent) for the model's
    n-th call; answer_as: model -> (build, provider), or a function of n
    giving one. Keeps every request's bytes."""

    def __init__(self, p=0.7, answer_as=None, fail=None):
        self.p, self.answer_as, self.fail = p, answer_as or {}, fail
        self.calls, self.raw = [], []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data)
        self.calls.append((req.full_url, dict(req.header_items()), body))
        self.raw.append(req.data)
        if body['model'] == self.fail:
            raise urllib.error.HTTPError(req.full_url, 502, 'Bad Gateway', {},
                                         io.BytesIO(b'{"error":"upstream"}'))
        n = sum(c[2]['model'] == body['model'] for c in self.calls)
        who = self.answer_as.get(body['model'], BUILD[body['model']])
        model, prov = who(n) if callable(who) else who
        qs = list(body['questions'])
        p = (lambda q: self.p(n, q, qs)) if callable(self.p) else \
            (lambda q: self.p)
        return io.BytesIO(json.dumps({
            'id': 'gen-dec-1', 'model': model, 'provider': prov,
            'answers': {k: {'type': 'noul', 'noul': p(k)} for k in qs},
            'usage': {'input_tokens': 700, 'output_tokens': 20,
                      'cost': 1e-5}}).encode())


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('QQQ_DECIDER_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-test')
    monkeypatch.setattr(g.time, 'sleep', lambda s: None)   # no real back-off
    return tmp_path


def run(env, opener, name='g.csv'):
    path = str(env / name)
    done = dl.run('kev-4b-q8_0', tonight=lambda t: ('20261013', NAMES, ST),
                  path=path, now=NOW, opener=opener)
    return done, g.read_log(path)


def test_the_decider_and_jev_are_asked_in_jevs_shape_and_logged_by_build(env):
    o = Decisions()
    done, log = run(env, o)
    assert done == ['rule', 'http', 'jev']
    assert {c[2]['model'] for c in o.calls} == set(dl.HOSTED.values())
    for url, headers, body in o.calls:
        assert url == dl.DECISIONS
        assert headers['Authorization'] == 'Bearer sk-test'
        assert set(body) == {'state', 'model', 'questions'}
        assert set(body['questions']) == {f'hold_{n}' for n in NAMES} | \
            {'hedge_book'}
    models = log.groupby('backend').model.first()
    assert models['http'] == \
        'perplexity/pplx-decider-v1.1-27b-20261010@Perplexity'
    assert models['jev'] == 'typesafe/jev-1.13-20260917@TypeSafe'
    assert models['rule'] == 'rule-earnings-v1'
    assert log.state_sha.nunique() == 1                 # one state, all three
    # a second run the same night asks nothing
    o2 = Decisions()
    assert run(env, o2)[0] == [] and o2.calls == []


@pytest.mark.parametrize('impostor', ['typesafe/jev-2.0-20261101',
                                      'typesafe/jev-1.13-mini-20261101',
                                      'typesafe/jev-1.13x'])
def test_an_answer_from_another_model_is_not_logged(env, impostor):
    o = Decisions(answer_as={'typesafe/jev-1.13': (impostor, 'TypeSafe')})
    done, log = run(env, o)
    assert done == ['rule', 'http']                     # Jev refused
    assert 'jev' not in set(log.backend)


def test_one_hosted_failure_never_costs_the_other_rows(env):
    o = Decisions(fail='perplexity/pplx-decider-v1.1-27b')
    done, log = run(env, o)
    assert done == ['rule', 'jev']
    # retried before giving up: a 502 is a provider hiccup, not a verdict
    assert sum(c[2]['model'] == 'perplexity/pplx-decider-v1.1-27b'
               for c in o.calls) == 4


def test_no_key_no_hosted_night(env, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY')
    with pytest.raises(SystemExit, match='OPENROUTER_API_KEY'):
        run(env, Decisions())


def test_proxy_auth_sends_no_key_at_all(env, monkeypatch):
    """Where a network proxy adds the key (measured in a cloud environment,
    RESULTS 27), --proxy-auth sends requests with no Authorization header."""
    monkeypatch.delenv('OPENROUTER_API_KEY')
    o = Decisions()
    done = dl.run('kev-4b-q8_0', tonight=lambda t: ('20261013', NAMES, ST),
                  path=str(env / 'p.csv'), now=NOW, opener=o,
                  proxy_auth=True)
    assert done == ['rule', 'http', 'jev']
    assert all('Authorization' not in h for _, h, _ in o.calls)


def test_the_default_request_is_byte_identical_to_every_logged_night():
    """request_body grew options (order, version, ids, provider, view) for
    the battery (RESULTS 27); its defaults must send exactly the bytes the
    record was declared with, pinned on the real state of 2026-10-13."""
    import hashlib
    s0 = json.load(open(os.path.join(os.path.dirname(HERE), 'bench',
                                     'battery', 's0_20261013.json')))
    b = json.dumps(g.request_body(s0['state'], s0['names'],
                                  'perplexity/pplx-decider-v1.1-27b'))
    assert hashlib.sha256(b.encode()).hexdigest() == \
        '5489fee7f67aa8d81d9649e6df590956b4a592db49158634e0e20abf2dbfb220'


def test_a_seeded_order_moves_questions_and_names_together():
    s0 = json.load(open(os.path.join(os.path.dirname(HERE), 'bench',
                                     'battery', 's0_20261013.json')))
    qids = [f'hold_{n}' for n in s0['names']] + ['hedge_book']
    o = g.question_order('20261013|abc|0|order-v1|http', qids)
    assert o == g.question_order('20261013|abc|0|order-v1|http', qids)
    assert o != qids and sorted(o) == sorted(qids)
    b = g.request_body(s0['state'], s0['names'], 'm', order=o)
    assert list(b['questions']) == o
    assert [f'hold_{n}' for n in b['state']['names']] == \
        [q for q in o if q != 'hedge_book']
    assert list(s0['state']['names']) == s0['names']    # shared state intact


def test_hosted_never_names_a_moving_alias():
    assert not any('~' in v or ':free' in v for v in dl.HOSTED.values())


def test_the_record_gate_is_split_three_ways():
    # three records can pass (decider, reference, control): Bonferroni on the
    # one-sided 2.5% of t > 2 at 62 df
    assert g.GATE_T == 2.46


# ------------------------------------------- the declared handling (R2-R7) --
#
# decide_local.DECLARED says how the decider is asked beyond its pin (RESULTS
# 27, battery spec 5). Today's declaration must send the record's bytes and
# log its rows; a declared handling changes the decider's bytes and suffix
# only, never the reference's, the control's or the shared state's sha.

ROOT = os.path.dirname(HERE)
S0 = json.load(open(os.path.join(ROOT, 'bench', 'battery',
                                 's0_20261013.json')))
PIN, JEV = dl.HOSTED['http'], dl.HOSTED['jev']
SHA = {PIN: '5489fee7f67aa8d81d9649e6df590956b4a592db49158634e0e20abf2dbfb220',
       JEV: '090ea909572942c449dc3717b8bcafe3f0eee6d1f0234a3c6644a4791f3da8e6'}
LOGGED = {PIN: 'perplexity/pplx-decider-v1.1-27b-20261010@Perplexity',
          JEV: 'typesafe/jev-1.13-20260917@TypeSafe'}
QIDS = [f'hold_{n}' for n in S0['names']] + ['hedge_book']


def night(env, opener, name='g.csv', now=NOW, s=S0):
    """One run of the hosted engine on the real state of 2026-10-13."""
    path = str(env / name)
    done = dl.run('kev-4b-q8_0', tonight=lambda t: (s['date'], s['names'],
                                                    s['state']),
                  path=path, now=now, opener=opener)
    return done, g.read_log(path)


def sent(o, model):
    return [r for r, c in zip(o.raw, o.calls) if c[2]['model'] == model]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def declare(monkeypatch, **how):
    """DECLARED['http'] with `how`, and guard's gated slot with it: the two
    are changed together (a half-made change is not asked; see below)."""
    new = dict(dl.DECLARED['http'], **how)
    monkeypatch.setitem(dl.DECLARED, 'http', new)
    monkeypatch.setattr(g, 'DECIDER', (dl.HOSTED['http'], dl.suffix(new))
                        + (() if new['dated'] else (False,)))


def orders(k):
    s = dl.state_sha(S0['state'])
    return [g.question_order(f'20261013|{s}|{j}|order-v1|http', QIDS)
            for j in range(k)]


def decider_log(env):
    return (env / 'home' / 'decider.log').read_text()


def test_todays_declaration_sends_the_records_bytes_and_logs_its_rows(env):
    o = Decisions()
    done, log = night(env, o)
    assert done == ['rule', 'http', 'jev']
    # the bytes every night was declared with: one call each, sha pinned
    assert [sha(b) for b in sent(o, PIN)] == [SHA[PIN]]
    assert [sha(b) for b in sent(o, JEV)] == [SHA[JEV]]
    assert all('provider' not in c[2] for c in o.calls)
    # the logged model strings, unchanged: build@provider, no suffix
    assert log.groupby('backend').model.unique().map(list).to_dict() == {
        'http': [LOGGED[PIN]], 'jev': [LOGGED[JEV]],
        'rule': ['rule-earnings-v1']}
    assert list(log.columns) == g.COLS
    assert set(log.state_sha) == {dl.state_sha(S0['state'])}
    # and the rows are the ones the runner logged before DECLARED existed:
    # a plain guard.ask_http, held to the pin, through guard.decide
    ref = str(env / 'ref.csv')
    for b, pin in dl.HOSTED.items():
        g.decide(b, S0['names'], S0['date'], NOW, S0['state'], ref,
                 lambda st, nm, pin=pin: g.ask_http(
                     st, nm, dl.DECISIONS, 'sk-test', pin,
                     opener=Decisions()))
    cols = [c for c in g.COLS if c not in ('decided_et', 'latency_ms')]
    assert log[log.backend != 'rule'][cols].reset_index(drop=True) \
        .equals(g.read_log(ref)[cols])


def test_k3_asks_three_seeded_orders_in_turn_and_logs_their_mean(
        env, monkeypatch):
    declare(monkeypatch, k=3)
    real, n = g.ask_http, itertools.count(1)

    def timed(*a, **kw):                # call i took 100 * i ms
        p, who, _ = real(*a, **kw)
        return p, who, 100 * next(n)
    monkeypatch.setattr(g, 'ask_http', timed)
    # each answer depends on where the question sat in the request
    o = Decisions(p=lambda n, q, qs: (qs.index(q) + 1) / (len(qs) + 1))
    done, log = night(env, o)
    assert done == ['rule', 'http', 'jev']
    bodies = sent(o, PIN)
    # three calls, three different orders: the night's seeded orders j=0..2,
    # names moved with their questions -- byte for byte the battery's E9
    # S0O{j} items, so the bytes sent are the bytes it tested
    assert len(bodies) == 3
    sent_orders = [list(json.loads(b)['questions']) for b in bodies]
    assert sent_orders == orders(3) and len(set(map(tuple, sent_orders))) == 3
    items = {}
    for line in open(os.path.join(ROOT, 'bench', 'battery',
                                  'items_v1.jsonl')):
        it = json.loads(line)
        if it['experiment'] == 'E9' and it['arm'] in ('S0O0a', 'S0O1a',
                                                      'S0O2a'):
            items[it['meta']['j']] = it['body']
    for j, b in enumerate(bodies):
        want = {'state': items[j]['state'], 'model': PIN,
                'questions': items[j]['questions']}
        assert b == json.dumps(want).encode()
        st = json.loads(b)['state']
        assert [f'hold_{x}' for x in st['names']] == \
            [q for q in sent_orders[j] if q != 'hedge_book']
        assert dl.state_sha(st) == dl.state_sha(S0['state'])
    # the reference: still one raw canonical call, no suffix
    assert [sha(b) for b in sent(o, JEV)] == [SHA[JEV]]
    h = log[log.backend == 'http'].set_index('ticker')
    j = log[log.backend == 'jev']
    assert set(h.model) == {LOGGED[PIN] + '#perm3.q1'}
    assert set(j.model) == {LOGGED[JEV]}
    # the mean of the three draws, per question
    for q in QIDS:
        k = 'BOOK' if q == 'hedge_book' else q[5:]
        mean = sum((o_.index(q) + 1) / (len(QIDS) + 1)
                   for o_ in sent_orders) / 3
        assert h.p[k] == round(mean, 4)
    assert g.rule(h.p.to_dict(), S0['names'])['BOOK'] == h.decision['BOOK']
    # latency: the three calls' sum; the reference its one call
    assert set(h.latency_ms) == {100 + 200 + 300} and set(j.latency_ms) == {400}
    assert log.state_sha.nunique() == 1


def test_a_catch_up_sends_the_same_three_bodies_again(env, monkeypatch):
    declare(monkeypatch, k=3)
    # 09:17: the third call comes back from another build -- nothing logged
    third = lambda n: BUILD[PIN] if n < 3 else (PIN + '-20261011',
                                                 'Perplexity')
    o1 = Decisions(answer_as={PIN: third})
    done, log = night(env, o1)
    assert done == ['rule', 'jev'] and 'http' not in set(log.backend)
    assert len(sent(o1, PIN)) == 3
    # 13:47: the catch-up asks the decider alone, about the cached state
    # (a state built now would differ), with exactly the 09:17 bytes
    later = dict(S0, state=dict(S0['state'], book='something else'))
    o2 = Decisions()
    done, log = night(env, o2, now=NOW.replace(hour=13, minute=47), s=later)
    assert done == ['http']
    assert sent(o2, PIN) == sent(o1, PIN) and sent(o2, JEV) == []
    assert set(log[log.backend == 'http'].model) == {LOGGED[PIN]
                                                     + '#perm3.q1'}
    assert log.state_sha.nunique() == 1


def test_k3_hundredths_that_average_to_one_half_hedge(env, monkeypatch):
    """0.47, 0.58, 0.45: a mean of exactly 0.5, which a plain float sum
    (Python 3.11's sum, numpy's mean) makes 0.49999999999999994 -- logged
    as 0.5000 beside 'none'. The rule hedges iff p >= 0.5, so the night
    hedges, on any Python (guard.mean_of_draws)."""
    declare(monkeypatch, k=3)
    book = {1: 0.47, 2: 0.58, 3: 0.45}
    o = Decisions(p=lambda n, q, qs: book[n] if q == 'hedge_book' else 0.7)
    done, log = night(env, o)
    assert done == ['rule', 'http', 'jev']
    h = log[log.backend == 'http'].set_index('ticker')
    assert h.p['BOOK'] == 0.5 and h.decision['BOOK'] == 'hedge'


@pytest.mark.parametrize('second', [
    (PIN + '-20261011', 'Perplexity'),                  # another build
    (PIN + '-20261010', 'Parasail'),                    # another provider
    ('perplexity/pplx-decider-v1.1-27bx-20261010', 'Perplexity')])  # not it
def test_calls_from_two_builds_or_providers_log_nothing(env, monkeypatch,
                                                        second):
    declare(monkeypatch, k=3)
    o = Decisions(answer_as={PIN: lambda n: second if n == 2
                             else BUILD[PIN]})
    done, log = night(env, o)
    assert done == ['rule', 'jev'] and 'http' not in set(log.backend)
    assert len(sent(o, PIN)) == 2                       # stopped there
    text = decider_log(env)
    assert 'http: ' in text and 'not logged' in text


@pytest.mark.parametrize('second, note', [
    ((PIN + '-20261011', 'Perplexity'), 'build drift'),
    ((PIN + '-20261010', 'Parasail'), 'provider drift')])
def test_a_refused_night_has_no_drift_note(env, monkeypatch, second, note):
    """Two builds or providers in one night: nothing is logged, and the log
    says so -- with what the calls cost, but no drift note, which would say
    the night was logged as answered."""
    declare(monkeypatch, k=3, measured_build=PIN + '-20261010',
            provider='Perplexity')
    o = Decisions(answer_as={PIN: lambda n: second if n == 2
                             else BUILD[PIN]})
    done, log = night(env, o)
    assert done == ['rule', 'jev'] and 'http' not in set(log.backend)
    text = decider_log(env)
    assert 'http: 2 call(s) answered by' in text and 'usage.cost' in text
    assert note not in text and 'drift' not in text
    assert 'a night is one build@provider; not logged' in text


@pytest.mark.parametrize('k,version,views,tag', [
    (1, 'q2', 's3', '#k1.q2s3'), (3, 'q2', 's3', '#perm3.q2s3'),
    (1, 'q1', 'b2', '#k1.q1b2'), (1, 'q2', '', '#k1.q2'),
    (3, 'q1', 'b2s3', '#perm3.q1b2s3')])
def test_a_version_and_a_view_change_only_the_deciders_body(
        env, monkeypatch, k, version, views, tag):
    declare(monkeypatch, k=k, version=version, views=views)
    o = Decisions()
    done, log = night(env, o)
    assert done == ['rule', 'http', 'jev']
    want = [json.dumps(g.request_body(
        S0['state'], S0['names'], PIN, order=None if k == 1 else od,
        version=version, view=g.view_of(views))).encode()
        for od in orders(k)]
    assert sent(o, PIN) == want
    assert [sha(b) for b in sent(o, JEV)] == [SHA[JEV]]  # untouched
    body = json.loads(sent(o, PIN)[0])
    assert ('last_night_z' in body['state']['QQQ']) == ('s3' in views)
    assert (body['state']['book'] == g.BOOK_B2) == ('b2' in views)
    assert body['questions']['hedge_book']['instructions'] == \
        g.QUESTIONS[version]['hedge'][0]
    assert set(log[log.backend == 'http'].model) == {LOGGED[PIN] + tag}
    assert set(log[log.backend == 'jev'].model) == {LOGGED[JEV]}
    # the log keeps the shared schema-2 state's sha, not the view's
    assert set(log.state_sha) == {dl.state_sha(S0['state'])}


def test_a_provider_pin_is_sent_only_when_declared(env, monkeypatch):
    o = Decisions()
    night(env, o, 'a.csv')
    assert all('provider' not in c[2] for c in o.calls)
    declare(monkeypatch, provider='Perplexity')
    o = Decisions()
    done, log = night(env, o, 'b.csv')
    assert done == ['rule', 'http', 'jev']
    pin = {'order': ['Perplexity'], 'allow_fallbacks': False}
    assert sent(o, PIN) == [json.dumps(g.request_body(
        S0['state'], S0['names'], PIN, provider=pin)).encode()]
    assert [sha(b) for b in sent(o, JEV)] == [SHA[JEV]]
    # the provider is already in the logged name: no suffix for a pin
    assert set(log[log.backend == 'http'].model) == {LOGGED[PIN]}


def test_what_the_calls_cost_goes_to_decider_log_never_the_record(
        env, monkeypatch):
    declare(monkeypatch, k=3, measured_build=PIN + '-20261006',
            provider='Parasail')
    monkeypatch.setitem(dl.DECLARED, 'jev',
                        {'measured_build': 'typesafe/jev-1.13-20260917'})
    done, log = night(env, Decisions())
    # drift is noted, never refused: the night is logged as answered
    assert done == ['rule', 'http', 'jev']
    text = decider_log(env)
    assert (f'http: 3 call(s) answered by {LOGGED[PIN]}; usage.cost 3e-05'
            in text)
    assert f'jev: 1 call(s) answered by {LOGGED[JEV]}; usage.cost 1e-05' \
        in text
    drift = [l for l in text.splitlines() if 'drift' in l]
    assert len(drift) == 2                      # the decider's build, provider
    assert 'http: build drift' in drift[0] and PIN + '-20261006' in drift[0]
    assert 'http: provider drift' in drift[1] and 'Parasail' in drift[1]
    assert all('not a refusal' in l for l in drift)
    rec = open(env / 'g.csv').read()
    assert list(log.columns) == g.COLS
    assert not any(w in rec for w in ('cost', 'usage', 'drift', 'e-05'))


def test_the_runner_and_guards_gated_slot_are_one_declaration():
    http = dl.DECLARED['http']
    assert dl.declaration_problems() == []
    assert g.DECIDER == (dl.HOSTED['http'], dl.suffix(http)) == (PIN, '')
    assert g.gated('http', LOGGED[PIN])
    for how, tag in (({'k': 3, 'version': 'q1', 'views': ''}, '#perm3.q1'),
                     ({'k': 3, 'version': 'q2', 'views': 's3'},
                      '#perm3.q2s3'),
                     ({'k': 5, 'version': 'q1', 'views': ''}, '#perm5.q1'),
                     ({'k': 1, 'version': 'q2', 'views': 's3'}, '#k1.q2s3'),
                     ({'k': 1, 'version': 'q1', 'views': ''}, '')):
        assert dl.suffix(how) == tag
        ok = dict(http, **how)
        assert dl.declaration_problems(declared={'http': ok},
                                       decider=(PIN, tag)) == []
        if tag:                         # changed on one side only: refused
            assert any('guard.DECIDER' in p for p in
                       dl.declaration_problems(declared={'http': ok}))
    # an undated build: (pin, suffix, False) in guard, dated False here
    assert dl.declaration_problems(declared={'http': dict(http, dated=False)},
                                   decider=(PIN, '', False)) == []
    assert dl.declaration_problems(decider=(PIN, '', True)) == []
    assert dl.declaration_problems(decider=(PIN, '', False))

    def bad(**kw):
        return dl.declaration_problems(declared={'http': dict(http, **kw)},
                                       decider=(PIN, dl.suffix(
                                           dict(http, **kw))))
    for kw in ({'views': 's3b2'}, {'views': 's3s3'}, {'views': 'z9'},
               {'views': None}, {'version': 'q9'}, {'k': 0}, {'k': True},
               {'k': 2.0}, {'provider': ''}, {'dated': 1},
               {'measured_build': 'typesafe/jev-1.13-20260917'},
               {'canary': True}):
        assert bad(**kw), kw
    assert dl.declaration_problems(
        declared={'http': {k: v for k, v in http.items() if k != 'k'}})
    # the reference takes no handling, only its measured build
    jev = lambda how: dl.declaration_problems(
        declared={'http': http, 'jev': how})
    assert jev({'measured_build': 'typesafe/jev-1.13-20260917'}) == []
    for how in ({'k': 3}, {'version': 'q2'}, {'provider': 'TypeSafe'}):
        assert jev(how), how
    assert dl.declaration_problems(declared={'http': http, 'kev': {}})


def test_a_half_made_declaration_costs_the_decider_alone(env, monkeypatch):
    """DECLARED changed without guard.DECIDER (or the other way round): no
    night is logged under a suffix the grader does not gate -- the decider
    is not asked, decider.log says why, and the control and the reference
    decide. The module still loads, so the nightly run still pulls (where a
    fix would arrive) and logs."""
    src = open(dl.__file__).read()
    old = "DECLARED = {'http': {'k': 1,"
    assert src.count(old) == 1
    mod = types.ModuleType('decide_local_half_made')
    mod.__file__ = dl.__file__
    exec(compile(src.replace(old, "DECLARED = {'http': {'k': 3,"),
                 dl.__file__, 'exec'), mod.__dict__)
    assert any('guard.DECIDER' in p for p in mod.declaration_problems())
    for i, side in enumerate(('DECLARED', 'DECIDER')):
        with monkeypatch.context() as m:
            if side == 'DECLARED':
                m.setitem(dl.DECLARED, 'http',
                          dict(dl.DECLARED['http'], k=3))
            else:
                m.setattr(g, 'DECIDER', (PIN, '#perm3.q1'))
            o = Decisions()
            done, log = night(env, o, f'{side}.csv')
        assert done == ['rule', 'jev'] and sent(o, PIN) == []
        assert 'http' not in set(log.backend)
        why = [l for l in decider_log(env).splitlines()
               if 'http: not asked, the declaration is half-made' in l]
        assert len(why) == i + 1 and 'guard.DECIDER' in why[-1]


def test_a_key_echoed_in_an_error_never_reaches_the_log(env, monkeypatch,
                                                        capsys):
    """A server (or a proxy) that echoes the Authorization header in its
    error body: guard quotes 300 characters of it, and say() writes it to
    decider.log and stdout -- with no key-shaped string left in it."""
    key = 'sk-or-v1-SENTINELSECRET0123456789'
    monkeypatch.setenv('OPENROUTER_API_KEY', key)

    def echo(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 401, 'Unauthorized', {}, io.BytesIO(json.dumps(
                {'error': {'message': 'Invalid key: '
                           + req.get_header('Authorization')}}).encode()))
    done, _ = night(env, echo)
    assert done == ['rule']
    text, out = decider_log(env), capsys.readouterr().out
    assert 'decision server refused (401)' in text
    for where in (text, out):
        assert 'SENTINELSECRET' not in where and 'Bearer sk' not in where
    assert dl.redact('{"echo":"Bearer abc.def"} sk-or-v1-x task-scheduler') \
        == '{"echo":"*** *** task-scheduler'


def battery_declaration(base):
    """What the battery's outputs under `base` (bench/battery) say
    decide_local must declare, or None before a selection lands: the
    decider's pin, its handling in DECLARED['http']'s fields, its suffix and
    the reference. Once iteration has run, the declaration is the 'decider'
    of its one outcome whose next is 'stop' -- the holdout's, or the round
    that stopped -- which carries those fields itself (R6)."""
    find = lambda name: sorted(glob.glob(os.path.join(base, '**', name),
                                         recursive=True))
    found = find('selection_v1.json')
    assert len(found) <= 1, found
    if not found:
        return None
    s = json.load(open(found[0]))
    d = s['decider']
    outs = [json.load(open(f)) for f in find('iter_*_outcome.json')]
    if outs:
        stop = [o for o in outs if o.get('next') == 'stop']
        assert len(stop) == 1, ('iteration lands whole: one outcome says '
                                'next: stop', [o.get('next') for o in outs])
        assert stop[0]['pin'] == d['pin']
        d = stop[0]['decider']
        how = dict(d['declared'])
    else:
        how = {'k': {'k1': 1, 'k3': 3}[d['handling']],
               'version': d['version'], 'views': '',
               'provider': (d['provider_pin'] or {'order': [None]}
                            )['order'][0],
               'dated': d['dated'], 'measured_build': d['build']}
    return {'pin': d['pin'], 'how': how, 'suffix': d['suffix'],
            'reference': s['reference']}


def test_the_declaration_is_the_batterys_outcome():
    """R2, R6: the decider, its handling, its provider pin and its measured
    build are what selection_v1.json chose -- and, once iteration rounds
    have run, what the outcome that stopped them declares. Before the
    selection lands, the declaration of 2026-10-13. The selection, the
    iteration and this declaration land in one deployment PR: any of them
    merged without the others fails here, by design."""
    how = dl.DECLARED['http']
    want = battery_declaration(os.path.join(ROOT, 'bench', 'battery'))
    if want is None:
        assert dl.HOSTED == {'http': 'perplexity/pplx-decider-v1.1-27b',
                             'jev': 'typesafe/jev-1.13'}
        assert {x: how[x] for x in dl.HANDLING} == {
            'k': 1, 'version': 'q1', 'views': '', 'provider': None,
            'dated': True}
        return
    ref = want['reference']
    assert dl.HOSTED['http'] == want['pin']
    assert (ref['status'] == 'retired') == ('jev' not in dl.HOSTED)
    if 'jev' in dl.HOSTED:
        assert dl.HOSTED['jev'] == ref['id']
    assert {x: how.get(x) for x in want['how']} == want['how']
    assert dl.suffix(how) == want['suffix']
    assert dl.declaration_problems() == []      # and guard.DECIDER with it
