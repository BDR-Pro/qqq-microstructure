# Part of qqq-microstructure.
#
# The battery's runner (src/battery_run.py) against a stand-in for
# OpenRouter's Decisions API (tests/fake_decisions.py), spec section 8 items
# 3-7: the bytes sent are the committed items plus the pin; the key never
# leaves the request header and proxy mode never reads it; spend stays under
# the cap and the scale-down is decided once; a crash resumes to exactly the
# planned keys; 429/5xx/401/other 4xx are handled as declared; the smoke stop
# spares I and R; validity is what G1 reads; the dispatch order is the
# seed's. No network anywhere.

import os, sys, json, shutil, hashlib
from collections import Counter
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import guard as g
import battery_run as br
import fake_decisions as fd
from fake_decisions import FakeDecisions, Clock, Crash, SENTINEL

I, R = 'fake/oracle', 'fake/jevlike'


def plan_of(tmp_path, cands=('fake/const',), inc=I, ref=R, name='plan',
            **kw):
    return fd.make_test_plan(tmp_path / name, list(cands), inc, ref, **kw)


def go(plan, out, fake, sleeps=None, **kw):
    # a scratch plan is outside the repository: the tree checks are
    # tested on their own, in a scratch git repository (TestTheTree)
    kw.setdefault('check_tree', False)
    return br.run(plan, out, opener=fake,
                  sleep=(sleeps.append if sleeps is not None
                         else lambda s: None), **kw)


def meta_of(run):
    return json.load(open(os.path.join(run, 'run_meta.json')))


def pinned(body, prov='FakeProv'):
    if body.get('provider') == br.FROM_W:
        return dict(body, provider={'order': [prov],
                                    'allow_fallbacks': False})
    return body


def by_model(run, m):
    return [c for c in br.read_calls(run).values() if c['model'] == m]


# ------------------------------------------------------------- the bytes --

def test_the_bytes_sent_are_the_items_plus_the_pin(tmp_path):
    plan = plan_of(tmp_path)
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.DONE
    p, items, _ = br.read_plan(plan)
    want = Counter(br.send_bytes(pinned(it['body']), m)
                   for m in br.models_of(p) for it in items)
    assert Counter(c['data'] for c in f.posts()) == want   # each once, exact
    s0 = json.load(open(fd.S0_FILE))
    for m in br.models_of(p):
        # the canonical S0 item is the record's own request, and the pilot
        # (E0 W) is every model's first call
        w = json.dumps(g.request_body(s0['state'], s0['names'], m)).encode()
        assert f.posts(m)[0]['data'] == w
    for c in f.posts():
        assert c['url'] == br.DECISIONS
        assert c['headers']['Content-type'] == 'application/json'
        assert list(json.loads(c['data']))[:3] == ['state', 'model',
                                                   'questions']
    calls = br.read_calls(tmp_path / 'run')
    assert set(calls) == br.planned(p, items)
    assert all(c['valid'] and c['final'] for c in calls.values())
    pp = calls[f'{I}|E0|S0|PP']
    assert pp['provider'] == 'FakeProv' and pp['valid']


# ------------------------------------------------------------------ auth --

def test_proxy_mode_sends_no_key_and_never_reads_it(tmp_path, monkeypatch):
    class Trap(dict):
        def __getitem__(self, k):
            assert k != 'OPENROUTER_API_KEY', 'proxy mode read the key'
            return super().__getitem__(k)

        def get(self, k, d=None):
            assert k != 'OPENROUTER_API_KEY', 'proxy mode read the key'
            return super().get(k, d)
    monkeypatch.setenv('OPENROUTER_API_KEY', SENTINEL)
    monkeypatch.setattr(br.os, 'environ', Trap(os.environ))
    f = FakeDecisions()
    assert go(plan_of(tmp_path), tmp_path / 'run', f, auth='proxy') == \
        br.DONE
    assert {c['method'] for c in f.calls} == {'GET', 'POST'}
    assert not any('Authorization' in c['headers'] for c in f.calls)


def test_env_mode_the_key_reaches_only_the_request_header(tmp_path,
                                                          monkeypatch,
                                                          capsys):
    monkeypatch.setenv('OPENROUTER_API_KEY', SENTINEL)
    # the /key label is the sentinel too, and fake/leaky echoes it in a 400
    plan = plan_of(tmp_path, cands=['fake/const', 'fake/leaky'])
    f = FakeDecisions(label=SENTINEL)
    assert go(plan, tmp_path / 'run', f, auth='env') == br.DONE
    assert all(c['headers'].get('Authorization') == f'Bearer {SENTINEL}'
               for c in f.calls)
    out = capsys.readouterr()
    assert 'SENTINEL' not in out.out + out.err
    for d, _, fs in os.walk(tmp_path):
        for x in fs:
            assert b'SENTINEL' not in open(os.path.join(d, x), 'rb').read(), x
    assert br.scrub_check(tmp_path / 'run') == []
    meta = meta_of(tmp_path / 'run')
    assert set(meta['key_before']) == {'limit_remaining', 'usage'}
    assert set(meta['key_after']) == {'limit_remaining', 'usage'}
    leaky = by_model(tmp_path / 'run', 'fake/leaky')
    assert any(c['error'].startswith('HTTP 400') and '***' in c['error']
               for c in leaky)


def test_env_mode_without_a_key_is_an_auth_failure(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    f = FakeDecisions()
    assert go(plan_of(tmp_path), tmp_path / 'run', f, auth='env') == br.AUTH
    assert f.calls == [] and not os.path.exists(tmp_path / 'run')


def test_scrub_and_scrub_check(tmp_path):
    s = br.scrub(f'x {SENTINEL} y Authorization: Bearer abc.def z sk-or-ab')
    assert 'sk-' not in s and 'Bearer' not in s and 'SENTINEL' not in s
    d = tmp_path / 'run'
    d.mkdir()
    (d / 'calls.jsonl').write_text('{"error": "HTTP 400: ***"}\n')
    assert br.scrub_check(d) == []
    (d / 'notes.txt').write_text('oops sk-or-v1-planted')
    hits = br.scrub_check(d)
    assert hits == ['notes.txt: sk-or-'] and 'planted' not in ''.join(hits)


# ----------------------------------------------------------------- spend --

def test_the_cap_bills_at_most_five_calls_then_resumes(tmp_path):
    # $0.01 a call, priced so the catalogue estimate before the pilot agrees
    cands = [{'id': m, 'price_in': 2.6e-7, 'X': 0.5}
             for m in ('fake/const', I, R)]
    plan = plan_of(tmp_path, cands)
    f = FakeDecisions(cost=0.01)
    assert go(plan, tmp_path / 'run', f, max_usd=0.05) == br.CAPPED
    assert sum(f.answered.values()) <= 5 and f.billed <= 0.05 + 1e-12
    meta = meta_of(tmp_path / 'run')
    assert meta['status'] == 'capped' and meta['spent_usd'] <= 0.05 + 1e-12
    sd = meta['scale_down']
    assert sd == {'dropped_blocks': [4], 'dropped_models': []}
    # a resume may raise the cap, up to the ceiling; the scale-down stays
    assert go(plan, tmp_path / 'run', FakeDecisions(cost=0.01),
              max_usd=12.01) == br.REFUSED
    assert go(plan, tmp_path / 'run', FakeDecisions(cost=0.01),
              max_usd=5) == br.DONE
    meta2 = meta_of(tmp_path / 'run')
    assert meta2['scale_down'] == sd and \
        meta2['projection'] == meta['projection']
    p, items, _ = br.read_plan(plan)
    assert set(br.read_calls(tmp_path / 'run')) == br.planned(p, items, sd)
    assert len(meta2['sessions']) == 2         # the refusal wrote nothing


def test_the_projection_drops_block_4_then_the_dearest_unscored_model(
        tmp_path):
    costs = {I: 0.001, R: 0.01, 'fake/const': 0.006, 'fake/uniform': 0.004,
             'fake/rule': 0.002}
    cands = [{'id': I, 'price_in': 2e-8, 'X': 1.0},
             {'id': R, 'price_in': 4.2e-8, 'X': None},   # dearest, but R
             {'id': 'fake/const', 'price_in': 1.5e-7, 'X': None},
             {'id': 'fake/uniform', 'price_in': 5e-8, 'X': None},
             {'id': 'fake/rule', 'price_in': 2e-8, 'X': 0.6}]
    plan = plan_of(tmp_path, cands)
    p, items, _ = br.read_plan(plan)
    n4 = sum(not (it['block'] == 4 and it['state_id'] != 'S0')
             for it in items)
    tot = sum(costs.values())
    # dropping block 4 is not enough; dropping it and fake/const is
    lim = (n4 * tot + n4 * (tot - costs['fake/const']) +
           costs['fake/const']) / 2
    f = FakeDecisions(costs=costs)
    go(plan, tmp_path / 'run', f, max_usd=lim / 0.8)
    meta = meta_of(tmp_path / 'run')
    assert meta['scale_down'] == {'dropped_blocks': [4],
                                  'dropped_models': ['fake/const']}
    assert meta['pilot_costs'] == pytest.approx(costs)
    assert meta['projection']['usd'] > lim >= \
        meta['projection']['after_scale_down_usd']
    assert len(f.posts('fake/const')) == 1           # its pilot call only
    assert len(f.posts(R)) > 1 and len(f.posts(I)) > 1


def test_refused_below_the_credit_floor(tmp_path):
    plan = plan_of(tmp_path)
    f = FakeDecisions(limit_remaining=17.99)         # max_usd 8 + 10 needed
    assert go(plan, tmp_path / 'run', f) == br.REFUSED
    assert f.posts() == [] and not os.path.exists(tmp_path / 'run')
    assert go(plan, tmp_path / 'run', FakeDecisions(limit_remaining=18.0)) \
        == br.DONE


def test_a_first_run_cannot_raise_the_planned_cap(tmp_path):
    f = FakeDecisions()
    assert go(plan_of(tmp_path), tmp_path / 'run', f, max_usd=8.01) == \
        br.REFUSED
    assert f.calls == []


def test_the_scale_down_cannot_be_rewritten(tmp_path):
    br.write_meta(tmp_path, {'scale_down': {'dropped_blocks': [],
                                            'dropped_models': []}})
    with pytest.raises(ValueError, match='immutable'):
        br.write_meta(tmp_path, {'scale_down': {'dropped_blocks': [4],
                                                'dropped_models': []}})


# ---------------------------------------------------------------- resume --

def test_a_crash_at_call_n_resumes_to_exactly_the_planned_keys(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/const', 'fake/rule'])
    run = tmp_path / 'run'
    with pytest.raises(Crash):
        go(plan, run, FakeDecisions(crash_at=60))
    settled = set(br.read_calls(run))
    p, items, _ = br.read_plan(plan)
    keys = br.planned(p, items)
    assert 0 < len(settled) < len(keys)
    with open(run / 'calls.jsonl', 'a') as fh:
        fh.write('{"key": "fake/const|E1|S0')      # a line cut by the crash
    f = FakeDecisions()
    assert go(plan, run, f) == br.DONE
    finals = Counter(ln['key'] for ln in br.read_lines(run) if ln['final'])
    assert set(finals) == keys and max(finals.values()) == 1
    assert len(f.posts()) == len(keys) - len(settled)  # nothing settled resent
    assert meta_of(run)['status'] == 'done'


def test_one_run_directory_per_plan(tmp_path):
    plan = plan_of(tmp_path)
    runs = tmp_path / 'runs'
    assert go(plan, runs / 'v1', FakeDecisions()) == br.DONE
    f = FakeDecisions()
    assert go(plan, runs / 'v1b', f) == br.REFUSED and f.calls == []
    other = plan_of(tmp_path, name='plan2', dispatch_seed=7)
    assert go(other, runs / 'v1', f) == br.REFUSED and f.calls == []
    # the same command again resumes the finished run: nothing to send
    f = FakeDecisions()
    assert go(plan, runs / 'v1', f) == br.DONE and f.posts() == []


def test_items_that_are_not_the_locked_bytes_are_refused(tmp_path):
    plan = plan_of(tmp_path)
    path = tmp_path / 'plan' / 'items_v1.jsonl'
    path.write_text(path.read_text().replace('"block": 4', '"block": 3', 1))
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.REFUSED and f.calls == []


def test_a_pilot_that_is_not_the_records_request_is_refused(tmp_path):
    plan = plan_of(tmp_path)
    d = tmp_path / 'plan'
    items = [json.loads(ln) for ln in open(d / 'items_v1.jsonl')]
    for it in items:
        if it['item_id'] == 'E0|S0|W':           # committed, locked, wrong
            it['body']['state'] = dict(it['body']['state'], date='20261014')
            it['body_sha256'] = hashlib.sha256(
                json.dumps(it['body']).encode()).hexdigest()
    (d / 'items_v1.jsonl').write_text(''.join(json.dumps(it) + '\n'
                                              for it in items))
    lock = json.load(open(d / 'lock_v1.json'))
    lock['items'] = hashlib.sha256((d / 'items_v1.jsonl').read_bytes()) \
        .hexdigest()
    (d / 'lock_v1.json').write_text(json.dumps(lock))
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.REFUSED and f.calls == []


# ------------------------------------------------------------- transport --

@pytest.mark.parametrize('after,wait', [('7', 7.0), ('600', 60.0)])
def test_a_429_waits_for_retry_after(tmp_path, after, wait):
    plan = plan_of(tmp_path, cands=['fake/ratelimit'])
    sleeps = []
    assert go(plan, tmp_path / 'run', FakeDecisions(retry_after=after),
              sleeps=sleeps) == br.DONE
    rl = by_model(tmp_path / 'run', 'fake/ratelimit')
    assert rl and all(c['valid'] and c['attempts'] == 2 for c in rl)
    assert sleeps == [wait] * len(rl)


def test_a_502_storm_is_tried_six_times_then_recorded_invalid(tmp_path):
    plan = plan_of(tmp_path, cands=[], inc='fake/storm')    # I: no smoke stop
    sleeps, f = [], FakeDecisions()
    assert go(plan, tmp_path / 'run', f, sleeps=sleeps) == br.DONE
    run = tmp_path / 'run'
    w = [ln for ln in br.read_lines(run) if ln['key'] == 'fake/storm|E0|S0|W']
    # six tries in the pilot, six more in the end-of-run pass, then final
    assert [(ln['attempts'], ln['final']) for ln in w] == [(6, False),
                                                            (6, True)]
    lines = [ln for ln in br.read_lines(run) if ln['model'] == 'fake/storm'
             and ln['attempts']]
    assert all(ln['http_status'] == 502 and not ln['valid'] for ln in lines)
    assert len(f.posts('fake/storm')) == 6 * len(lines)
    # the breaker: three calls in a row failed every try -- parked in the
    # main pass while R answered, again in the retry pass: the rest of its
    # keys are settled 'unavailable', never sent (a resume sends nothing)
    calls = br.read_calls(run)
    storm = [c for c in calls.values() if c['model'] == 'fake/storm']
    assert all(c['final'] and not c['valid'] for c in storm)
    sent = [c for c in storm if c['attempts']]
    assert len(sent) == 1 + br.BREAKER                      # W + the trip
    assert Counter(c['error'].split(' after ')[0] for c in storm
                   if not c['attempts']) == \
        {'unavailable': len(storm) - len(sent) - 1, 'no W provider': 1}
    assert calls['fake/storm|E0|S0|PP']['error'] == 'no W provider'
    assert meta_of(run)['breaker'] == {
        'fake/storm': {'trips': 2, 'others_answered': True}}
    # jittered back-off: 1, 2, 4, 8, 16 s (+-25%) between the six tries
    band = Counter(next(b for b in br.BACKOFF if 0.75 * b <= s <= 1.25 * b)
                   for s in sleeps)
    assert band == {b: len(lines) for b in (1, 2, 4, 8, 16)}


def test_a_401_aborts_the_run_with_exit_2(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/unauth'])
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f, concurrency=1) == br.AUTH
    assert len(f.posts('fake/unauth')) == 1           # not retried
    assert len(f.posts()) <= 3                        # nothing after it
    assert meta_of(tmp_path / 'run')['status'] == 'auth_failed'
    w = br.read_lines(tmp_path / 'run')
    assert [ln['final'] for ln in w if ln['model'] == 'fake/unauth'] == \
        [False]                                       # a resume asks again
    f = FakeDecisions(key_status=401)
    assert go(plan_of(tmp_path, name='p2'), tmp_path / 'run2', f) == br.AUTH
    assert f.posts() == []


def test_other_4xx_are_recorded_invalid_and_not_retried(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/nopin'], ref='fake/leaky')
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.DONE
    leaky = by_model(tmp_path / 'run', 'fake/leaky')
    sent = [c for c in leaky if c['attempts']]
    assert len(f.posts('fake/leaky')) == len(sent) == len(leaky) - 1  # no PP
    assert all(c['attempts'] == 1 and c['http_status'] == 400 and
               c['final'] and not c['valid'] for c in sent)
    assert not any('smoke' in c['error'] for c in leaky)   # R is never stopped
    calls = br.read_calls(tmp_path / 'run')
    pp = calls['fake/nopin|E0|S0|PP']
    assert pp['error'].startswith('pin unsupported') and pp['attempts'] == 1
    assert all(c['valid'] for k, c in calls.items()
               if c['model'] == 'fake/nopin' and k != 'fake/nopin|E0|S0|PP')


def test_the_smoke_stop_ends_a_model_after_three_invalid_but_never_I_or_R(
        tmp_path):
    plan = plan_of(tmp_path, cands=['fake/impostor', 'fake/const'],
                   inc='test/inc')
    f = FakeDecisions(behaviours={'test/inc': 'impostor'})
    assert go(plan, tmp_path / 'run', f) == br.DONE
    imp = by_model(tmp_path / 'run', 'fake/impostor')
    assert len(f.posts('fake/impostor')) == 3
    assert all(c['final'] and not c['valid'] for c in imp)
    err = Counter(c['error'] for c in imp)
    assert err['smoke stop'] == len(imp) - 4 and err['no W provider'] == 1
    inc = by_model(tmp_path / 'run', 'test/inc')
    assert len(f.posts('test/inc')) == len(inc) - 1   # all but PP sent
    assert not any('smoke' in c['error'] for c in inc)
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.DONE and f.posts() == []


# ------------------------------------------------------- what G1 reads --

def test_what_counts_as_a_valid_call(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/split50', 'fake/split95',
                                    'fake/nocost', 'fake/partial'],
                   ref='fake/impostor')
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f) == br.DONE
    run = tmp_path / 'run'
    imp = [c for c in by_model(run, 'fake/impostor') if c['attempts']]
    assert imp and all(not c['valid'] and c['answered_as'] ==
                       f'fake/impostor-mini-{fd.BUILD}' and
                       c['error'].startswith('answered as') for c in imp)
    s50 = by_model(run, 'fake/split50')
    assert all(c['valid'] for c in s50)
    n = Counter(c['provider'] for c in s50)
    assert abs(n['ProvA'] - n['ProvB']) <= 1
    s95 = by_model(run, 'fake/split95')
    assert all(c['valid'] for c in s95)
    assert Counter(c['provider'] for c in s95)['ProvB'] == len(s95) // 20
    nc = by_model(run, 'fake/nocost')
    assert all(c['valid'] and c['cost'] is None and
               c['usage']['input_tokens'] > 0 for c in nc)
    meta = meta_of(run)
    w = br.read_calls(run)['fake/nocost|E0|S0|W']
    assert meta['pilot_costs']['fake/nocost'] == \
        pytest.approx(w['usage']['input_tokens'] * 2e-8 * 2)
    assert meta['spent_usd'] > f.billed - 1e-4 * len(nc)   # estimated, not 0
    part = [c for c in by_model(run, 'fake/partial') if c['attempts']]
    assert len(part) == 3 and all('missing' in c['error'] and
                                  None in c['answers'].values()
                                  for c in part)


@pytest.mark.parametrize('who,ok', [
    ('m/x', True), ('m/x-20261001', True), ('m/x-mini-20261001', False),
    ('m/xx', False), ('m/x-2026100', False), ('m/x-20261001@P', False),
    (None, False)])
def test_answered_as_must_be_the_pin_or_a_dated_build(who, ok):
    assert br.valid_call('m/x', 200, {'a': 0.5}, who, 'P')[0] is ok


def test_validity_needs_every_id_a_p_in_range_and_a_provider():
    assert br.valid_call('m', 200, {'a': 0.0, 'b': 1.0}, 'm', 'P')[0]
    assert not br.valid_call('m', 200, {'a': 0.5, 'b': None}, 'm', 'P')[0]
    assert not br.valid_call('m', 200, {'a': 1.2}, 'm', 'P')[0]
    assert not br.valid_call('m', 200, {'a': 0.5}, 'm', '')[0]
    assert not br.valid_call('m', 503, {'a': 0.5}, 'm', 'P')[0]
    assert not br.valid_call('m', 200, {}, 'm', 'P')[0]


def test_a_slow_model_completes_under_the_fake_clock(tmp_path):
    clock = Clock()
    plan = plan_of(tmp_path, cands=['fake/slow'])
    f = FakeDecisions(clock=clock, slow_s=30)
    assert br.run(plan, tmp_path / 'run', opener=f, sleep=clock.sleep,
                  now=clock.now, check_tree=False) == br.DONE
    slow = by_model(tmp_path / 'run', 'fake/slow')
    assert slow and all(c['valid'] and c['latency_ms'] >= 30000
                        for c in slow)


def test_the_deadline_stops_sending_and_a_resume_finishes(tmp_path):
    clock = Clock()
    plan = plan_of(tmp_path, cands=['fake/slow'])
    kw = dict(sleep=clock.sleep, now=clock.now, check_tree=False)
    f = FakeDecisions(clock=clock, slow_s=30)
    assert br.run(plan, tmp_path / 'run', opener=f, deadline_min=5,
                  **kw) == br.DEADLINE
    assert meta_of(tmp_path / 'run')['status'] == 'deadline'
    f2 = FakeDecisions(clock=clock, slow_s=30)
    assert br.run(plan, tmp_path / 'run', opener=f2, **kw) == br.DONE
    p, items, _ = br.read_plan(plan)
    assert set(br.read_calls(tmp_path / 'run')) == br.planned(p, items)


# -------------------------------------------------------------- dispatch --

def test_the_dispatch_order_is_reproducible_from_the_seed(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/const', 'fake/rule'])
    p, items, _ = br.read_plan(plan)
    by = {it['item_id']: it for it in items}
    o = br.dispatch_order(p, items)
    assert o == br.dispatch_order(p, items)
    assert sorted(o) == sorted((m, it['item_id']) for m in br.models_of(p)
                               for it in items)
    blocks = [by[i]['block'] for _, i in o]
    assert blocks == sorted(blocks)                       # blocks in order
    o2 = br.dispatch_order(dict(p, seeds=dict(p['seeds'], dispatch=1)),
                           items)
    assert o2 != o and sorted(o2) == sorted(o)
    for m in br.models_of(p):                 # a B2B triple: one slot
        at = [k for k, (x, i) in enumerate(o)
              if x == m and by[i]['arm'].startswith('B2B')]
        assert at == list(range(at[0], at[0] + 3))
    # one call at a time, they go out in exactly this order, pilot first
    f = FakeDecisions()
    assert go(plan, tmp_path / 'run', f, concurrency=1) == br.DONE
    w = 'E0|S0|W'
    want = [(m, i) for m, i in o if i == w] + [(m, i) for m, i in o if i != w]
    assert [c['data'] for c in f.posts()] == \
        [br.send_bytes(pinned(by[i]['body']), m) for m, i in want]


@pytest.mark.parametrize('concurrency,cap', [(None, 16), (5, 5)])
def test_calls_in_flight_stay_under_the_limits(tmp_path, concurrency, cap):
    cands = ['fake/const', 'fake/rule', 'fake/uniform', 'fake/lethargic',
             'fake/posbias', 'fake/joint', 'fake/tickerprior']
    f = FakeDecisions(delay=0.002)
    assert go(plan_of(tmp_path, cands), tmp_path / 'run', f,
              concurrency=concurrency) == br.DONE
    assert 3 < f.max_inflight <= cap
    assert max(f.max_inflight_model.values()) <= 3


# ---------------------------------------------------------- the stand-in --

def test_the_stand_ins_answer_as_declared(tmp_path):
    cands = ['fake/lethargic', 'fake/tickerprior', 'fake/caching',
             'fake/posbias', 'fake/joint']
    plan = plan_of(tmp_path, cands)
    assert go(plan, tmp_path / 'run', FakeDecisions()) == br.DONE
    p, items, d = br.read_plan(plan)
    states = json.load(open(os.path.join(d, 'states_v1.json')))
    c = br.read_calls(tmp_path / 'run')

    def ans(m, iid):                    # canonical id -> p
        qmap = next(it['qmap'] for it in items if it['item_id'] == iid)
        return {qmap[q]: v for q, v in c[f'{m}|{iid}']['answers'].items()}
    for sid, s in states.items():
        a = ans(I, f'E1|{sid}|Ca')       # the oracle follows the criteria
        dec = g.rule({k[5:] if k != 'hedge_book' else g.BOOK: v
                      for k, v in a.items()}, s['labels'])
        for t in s['labels']:
            assert dec[t] == ('skip' if s['conditions'][t] in (
                'Earnings', 'ExtremeDown', 'ExtremeUp') else 'hold')
        assert dec[g.BOOK] == ('hedge' if s['regime'] == 'stressed'
                               else 'none')
        assert ans(R, f'E1|{sid}|Ca') != ans(R, f'E1|{sid}|Cb')  # noisy
        assert ans('fake/posbias', f'E1|{sid}|O1') != \
            ans('fake/posbias', f'E1|{sid}|Ca')
        if s['e7_kind'] == 'LD':         # the ticker prior ignores the rows
            assert ans('fake/tickerprior', f'E7|{sid}|LD') == \
                ans('fake/tickerprior', f'E1|{sid}|Ca')
        if s['e8']:                      # the oracle reads a negation
            ng = ans(I, f'E8|{sid}|NG')
            assert all(abs(ng[k] + v - 1) < 1e-9 for k, v in a.items())
        if s['cx_type'] != 1:            # alone: the joint answerer shifts
            cx = ans('fake/joint', f'E3|{sid}|CX')
            full = ans('fake/joint', f'E1|{sid}|Ca')
            assert all(abs(cx[k] - full[k]) > 0.09 for k in cx)
    leth = [v for k, v in ans('fake/lethargic', 'E0|S0|W').items()]
    assert all(0.46 <= v <= 0.61 for v in leth[:-1]) and leth[-1] == 0.558
    b2b = [ans('fake/caching', f'E9|S0|B2B{i}') for i in (1, 2, 3)]
    assert b2b[0] == b2b[1] == b2b[2]    # identical bytes, back to back


# ------------------------------------------------- statuses the API sends --

def test_a_408_is_a_timeout_and_is_retried(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/flaky408'])
    assert go(plan, tmp_path / 'run', FakeDecisions()) == br.DONE
    sent = [c for c in by_model(tmp_path / 'run', 'fake/flaky408')
            if c['attempts']]
    assert sent and all(c['valid'] and c['attempts'] == 2 for c in sent)


def test_a_402_stops_sending_and_leaves_its_key_open(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/const'])
    run = tmp_path / 'run'
    f = FakeDecisions(credit_after=40)
    assert go(plan, run, f, concurrency=1) == br.CAPPED
    assert meta_of(run)['status'] == 'no_credit'
    assert len(f.posts()) == 41                       # nothing after it
    lines = br.read_lines(run)
    hit = [ln for ln in lines if ln['http_status'] == 402]
    assert len(hit) == 1 and hit[0]['attempts'] == 1 and not hit[0]['final']
    assert not any(ln['final'] and not ln['valid'] for ln in lines)
    # topped up, the same command finishes exactly the planned keys
    assert go(plan, run, FakeDecisions()) == br.DONE
    p, items, _ = br.read_plan(plan)
    calls = br.read_calls(run)
    assert set(calls) == br.planned(p, items)
    assert all(c['valid'] for c in calls.values())


# ----------------------------------------------------- the pilot's cost --

def test_a_failed_pilot_is_projected_at_the_measured_cost(tmp_path):
    # two unscored models whose W fails -- an unbilled 400, a 502 storm not
    # yet retried -- are not $0.01 a call: that projected $1.14 here and
    # dropped block 4 for everyone at a $1 cap
    cands = [{'id': m, 'price_in': 4e-8, 'X': None}
             for m in ('fake/leaky', 'fake/storm')]
    plan = plan_of(tmp_path, cands)
    go(plan, tmp_path / 'run', FakeDecisions(cost=1e-4), max_usd=1.0)
    meta = meta_of(tmp_path / 'run')
    assert meta['scale_down'] == {'dropped_blocks': [], 'dropped_models': []}
    assert meta['pilot_unmeasured'] == ['fake/leaky', 'fake/storm']
    assert meta['pilot_costs'] == pytest.approx(
        {m: 1e-4 for m in ('fake/leaky', 'fake/storm', I, R)})
    assert meta['projection']['usd'] < 0.8


# ----------------------------------------------------------- the breaker --

def test_a_model_that_goes_down_is_parked_and_the_rest_finish(tmp_path):
    # fake/hang answers W, then every call times out after 60 s: six tries
    # each, ~6.5 minutes a call. Unparked, its 56 keys held every block
    # boundary until the 90-minute deadline, and every resume again.
    clock = Clock()
    plan = plan_of(tmp_path, cands=['fake/const', 'fake/rule', 'fake/hang'])
    run = tmp_path / 'run'
    f = FakeDecisions(clock=clock, slow_s=60)
    assert br.run(plan, run, opener=f, sleep=clock.sleep, now=clock.now,
                  check_tree=False) == br.DONE
    calls = br.read_calls(run)
    p, items, _ = br.read_plan(plan)
    assert set(calls) == br.planned(p, items)
    assert all(c['valid'] for c in calls.values() if c['model'] != 'fake/hang')
    hang = by_model(run, 'fake/hang')
    assert calls['fake/hang|E0|S0|W']['valid']
    gone = [c for c in hang if (c['error'] or '').startswith('unavailable')]
    assert len(gone) > 40 and all(c['final'] and not c['attempts']
                                  for c in gone)
    # parked after three calls in the main pass and three in the retry pass
    # (plus what was already in flight), never more
    assert len(f.posts('fake/hang')) <= 1 + 2 * (br.BREAKER + 2) * br.TRIES
    assert meta_of(run)['breaker'] == {
        'fake/hang': {'trips': 2, 'others_answered': True}}
    assert go(plan, run, FakeDecisions()) == br.DONE   # nothing left to send


def test_an_outage_parks_everyone_settles_nothing_and_resumes(tmp_path):
    clock = Clock()
    plan = plan_of(tmp_path, cands=['fake/const'])
    run = tmp_path / 'run'
    down = {m: 'hang' for m in ('fake/const', I, R)}
    f = FakeDecisions(clock=clock, slow_s=60, behaviours=down)
    assert br.run(plan, run, opener=f, sleep=clock.sleep, now=clock.now,
                  check_tree=False) == br.CAPPED
    assert meta_of(run)['status'] == 'outage'
    finals = [ln for ln in br.read_lines(run) if ln['final']]
    assert {ln['item_id'] for ln in finals} == {'E0|S0|W'}  # nothing settled
    assert go(plan, run, FakeDecisions()) == br.DONE
    p, items, _ = br.read_plan(plan)
    calls = br.read_calls(run)
    assert set(calls) == br.planned(p, items)
    assert all(c['valid'] for c in calls.values())


# ------------------------------------------------- the back-to-back triple --

def test_the_back_to_back_triple_goes_out_one_member_at_a_time(
        tmp_path, monkeypatch):
    seen, send = [], br._Run._send

    def spy(self, ex, flight, per, m, iid, final):
        if 'B2B' in iid:
            seen.append((m, iid, [v[1] for v in flight.values()
                                  if v[0] == m and 'B2B' in v[1]]))
        return send(self, ex, flight, per, m, iid, final)
    monkeypatch.setattr(br._Run, '_send', spy)
    cands = ['fake/const', 'fake/rule', 'fake/caching']
    assert go(plan_of(tmp_path, cands), tmp_path / 'run',
              FakeDecisions(delay=0.005)) == br.DONE
    assert len(seen) == 3 * 5
    assert all(busy == [] for _, _, busy in seen)      # never two in flight
    for m in {m for m, _, _ in seen}:
        assert [i for x, i, _ in seen if x == m] == \
            [f'E9|S0|B2B{k}' for k in (1, 2, 3)]


# -------------------------------------------------------------- resuming --

def _redeclare(d, edit):
    """Edit the items under the plan and renew their shas and the lock: a
    consistent declaration, but not the one a run started with."""
    items = [json.loads(ln) for ln in open(d / 'items_v1.jsonl')]
    for it in items:
        if edit(it):
            it['body_sha256'] = hashlib.sha256(
                json.dumps(it['body']).encode()).hexdigest()
    (d / 'items_v1.jsonl').write_text(''.join(json.dumps(it) + '\n'
                                              for it in items))
    lock = json.load(open(d / 'lock_v1.json'))
    lock['items'] = hashlib.sha256((d / 'items_v1.jsonl').read_bytes()) \
        .hexdigest()
    (d / 'lock_v1.json').write_text(json.dumps(lock))


def test_a_resume_goes_on_only_with_the_bytes_it_started_with(tmp_path):
    plan = plan_of(tmp_path, cands=['fake/const'])
    run = tmp_path / 'run'
    with pytest.raises(Crash):
        go(plan, run, FakeDecisions(crash_at=40))
    meta = meta_of(run)
    assert set(meta['shas']) == {'plan', 'items', 'states', 's0', 'anchors',
                                 'lock'}

    def edit(it):
        if it['item_id'] != 'E1|S02|O1':
            return False
        it['body']['state'] = dict(it['body']['state'], date='20261014')
        return True
    _redeclare(tmp_path / 'plan', edit)
    f = FakeDecisions()
    assert go(plan, run, f) == br.REFUSED and f.calls == []


def test_a_scored_run_is_frozen(tmp_path):
    plan = plan_of(tmp_path)
    run = tmp_path / 'run'
    with pytest.raises(Crash):
        go(plan, run, FakeDecisions(crash_at=30))
    for name in ('summary.json', 'selection_v1.json'):
        (run / name).write_text('{}')
        f = FakeDecisions()
        assert go(plan, run, f) == br.REFUSED and f.calls == []
        os.remove(run / name)
    assert go(plan, run, FakeDecisions()) == br.DONE


# --------------------------------------------------------------- the tree --

def _git(repo, *args):
    import subprocess
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c',
                    'user.email=t@example.com', '-c', 'commit.gpgsign=false',
                    *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A scratch repository with the plan committed in bench/battery, and
    the runner's ROOT pointed at it."""
    root = tmp_path / 'repo'
    plan = fd.make_test_plan(root / 'bench' / 'battery', ['fake/const'], I, R)
    (root / 'src').mkdir()
    (root / 'src' / 'a.py').write_text('x = 1\n')
    (root / 'tests').mkdir()
    (root / 'tests' / 't.py').write_text('y = 1\n')
    _git(root, 'init', '-q')
    _git(root, 'add', '-A')
    _git(root, 'commit', '-qm', 'the declaration')
    monkeypatch.setattr(br, 'ROOT', str(root))
    return root, plan


def tree_go(plan, out, f=None):
    f = f or FakeDecisions()
    return br.run(plan, out, opener=f, sleep=lambda s: None), f


def test_a_clean_tree_runs_and_records_it(repo):
    root, plan = repo
    rc, f = tree_go(plan, root / 'bench' / 'battery' / 'runs' / 'v1')
    assert rc == br.DONE and f.posts()
    meta = meta_of(root / 'bench' / 'battery' / 'runs' / 'v1')
    assert meta['tree'] == 'clean' and meta['sessions'][0]['tree'] == 'clean'
    assert meta['commit'] and meta['sessions'][0]['commit'] == meta['commit']
    assert meta['plan'] == 'bench/battery/plan_v1.json'


@pytest.mark.parametrize('out', ['elsewhere/v1', 'src', 'bench/battery',
                                 'bench/battery/runs',
                                 'bench/battery/runs/a/b'])
def test_the_run_directory_is_one_child_of_runs(repo, out):
    root, plan = repo
    (root / 'src' / 'b.py').write_text('dirty\n')      # an out of src/ hid it
    rc, f = tree_go(plan, root / out)
    assert rc == br.REFUSED and f.calls == []


@pytest.mark.parametrize('dirty', ['src/a.py', 'src/new.py', 'tests/new.py',
                                   'bench/battery/notes.txt'])
def test_an_uncommitted_change_is_refused(repo, dirty):
    root, plan = repo
    (root / dirty).write_text('changed\n')
    rc, f = tree_go(plan, root / 'bench' / 'battery' / 'runs' / 'v1')
    assert rc == br.REFUSED and f.calls == []


def test_a_plan_outside_the_repository_or_uncommitted_is_refused(repo,
                                                                tmp_path):
    root, plan = repo
    for where in (tmp_path / 'copy', root / 'reports' / 'battery'):
        shutil.copytree(os.path.dirname(plan), where)
        rc, f = tree_go(where / 'plan_v1.json', where / 'runs' / 'v1')
        assert rc == br.REFUSED and f.calls == []
    # verify's view of the same: the committed plan is clean
    p, items, d = br.read_plan(plan)
    assert br.tree_problems(plan, p, d) == []
    q, _, e = br.read_plan(root / 'reports' / 'battery' / 'plan_v1.json')
    probs = br.tree_problems(str(root / 'reports' / 'battery' /
                                 'plan_v1.json'), q, e)
    assert any(x.startswith('not committed') for x in probs)


def test_a_second_run_of_a_plan_is_refused_on_disk_and_in_history(repo):
    root, plan = repo
    runs = root / 'bench' / 'battery' / 'runs'
    assert tree_go(plan, runs / 'v1')[0] == br.DONE
    _git(root, 'add', '-A')
    _git(root, 'commit', '-qm', 'the run')
    rc, f = tree_go(plan, runs / 'v2')                 # v1 is on disk
    assert rc == br.REFUSED and f.calls == []
    shutil.rmtree(runs / 'v1')                         # ... and then lost
    _git(root, 'add', '-A')
    _git(root, 'commit', '-qm', 'lose the run')
    for out in ('v1', 'v2'):
        rc, f = tree_go(plan, runs / out)
        assert rc == br.REFUSED and f.calls == []
