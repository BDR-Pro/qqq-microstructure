# Part of qqq-microstructure.
#
# The decider battery end to end (RESULTS 27; spec v1, section 8): the real
# generator's 24 states and 377 items per model, made by the CLI's
# make-items in a scratch directory, re-declared as a test plan whose
# candidates are the stand-ins of tests/fake_decisions.py (the lethargic one
# as the incumbent, the Jev-like one as the reference); then verify, run,
# score, select and scrub-check through src/decider_battery.py, exactly as
# the cloud session calls them, against the fake Decisions API. The stand-
# ins' scores are held to section 8's table and the selection to section 3.
#
# Twice: at $0.0001 a call (all 4 blocks), and at $0.002 a call, where the
# pilot's projection ($7.54 > 0.8 x $8) drops block 4's synthetic states
# and the analysis set is the floor's 3 blocks. One call in flight
# (concurrency 1): the noisy stand-ins draw from a per-model call counter,
# so this pins every answer and the run is reproducible (concurrent
# dispatch is tested in test_battery_run.py). No network: every request
# goes to the fake, and the clock only moves when something sleeps on it.

import os, sys, json, math, shutil, hashlib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import guard
import battery_items as bi
import battery_run as br
import battery_score as bs
import decider_battery as db
import fake_decisions as fd

BENCH = os.path.join(ROOT, 'bench', 'battery')
STAND_INS = ['oracle', 'const', 'uniform', 'lethargic', 'rule', 'jevlike',
             'posbias', 'joint', 'tickerprior', 'caching']
I, R = 'fake/lethargic', 'fake/jevlike'
SKIP_CRITERIA = ('Earnings', 'ExtremeDown', 'ExtremeUp')
# $ per call -> the scale-down the pilot must decide
RUNS = {'nominal': (1e-4, {'dropped_blocks': [], 'dropped_models': []}),
        'floor': (2e-3, {'dropped_blocks': [4], 'dropped_models': []})}


def _dump(obj):
    return (json.dumps(obj, indent=1) + '\n').encode()


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def redeclare(plan_path, real):
    """The generated plan with the stand-ins as its candidates; I and R
    carry the real incumbent's and reference's X, the rest are unscored, as
    in the real plan. The lock is renewed for the new plan bytes only."""
    plan = json.load(open(plan_path))
    X = {c['id']: c['X'] for c in real['candidates']}
    xs = {I: X[real['incumbent']], R: X[real['reference']]}
    plan.update(candidates=[{'id': f'fake/{b}', 'price_in': 2e-8,
                             'X': xs.get(f'fake/{b}')} for b in STAND_INS],
                incumbent=I, reference=R)
    with open(plan_path, 'wb') as f:
        f.write(_dump(plan))
    lock_path = bi.lock_path(plan_path)
    lock = json.load(open(lock_path))
    lock['plan'] = _sha(plan_path)
    with open(lock_path, 'wb') as f:
        f.write(_dump(lock))
    return plan


@pytest.fixture(scope='module', params=list(RUNS))
def battery(request, tmp_path_factory):
    """The whole pipeline, once per cost: make-items, re-declare, verify,
    run, score, select, scrub-check. Every exit code is kept."""
    cost, scale_down = RUNS[request.param]
    d = tmp_path_factory.mktemp(request.param)
    for f in (bi.S0_FILE, bi.ANCHORS_FILE):
        shutil.copyfile(os.path.join(BENCH, f), d / f)
    plan_path = str(d / 'plan_v1.json')
    run = str(d / 'runs' / 'v1')
    codes = {'make-items': db.main(['make-items', '--plan', plan_path])}
    generated = {f: (d / f).read_bytes() for f in os.listdir(d)}
    real = json.load(open(plan_path))
    plan = redeclare(plan_path, real)
    codes['verify'] = db.main(['verify', '--plan', plan_path])
    clock = fd.Clock()
    fake = fd.FakeDecisions(clock=clock, cost=cost)
    codes['run'] = db.main(['run', '--plan', plan_path, '--out', run,
                            '--auth', 'proxy', '--concurrency', '1',
                            '--per-model', '1'],
                           opener=fake, sleep=clock.sleep, now=clock.now)
    codes['score'] = db.main(['score', '--run', run])
    codes['select'] = db.main(['select', '--run', run])
    codes['scrub-check'] = db.main(['scrub-check', '--run', run])
    L = bs.load(plan_path)
    summary = json.load(open(os.path.join(run, 'summary.json')))
    A = summary['analysis_set']['states']
    return {'name': request.param, 'scale_down': scale_down,
            'dir': str(d), 'plan_path': plan_path, 'plan': plan,
            'real': real, 'generated': generated, 'run': run, 'codes': codes,
            'fake': fake, 'clock': clock,
            'states': {s: L['states'][s] for s in A},
            'items': {it['item_id']: it for it in L['items']},
            's0': L['s0'], 'summary': summary,
            'selection': json.load(open(os.path.join(run,
                                                     'selection_v1.json')))}


def model(battery, b):
    return battery['summary']['models'][f'fake/{b}']


# ------------------------------------------------------------ the items --

def test_every_step_exits_0(battery):
    assert battery['codes'] == {'make-items': 0, 'verify': 0, 'run': 0,
                                'score': 0, 'select': 0, 'scrub-check': 0}


def test_the_items_are_the_committed_declaration(battery):
    gen, real = battery['generated'], battery['real']
    if real['generator']['numpy'] != '.'.join(np.__version__.split('.')[:2]):
        pytest.skip('the committed bytes were made with another numpy')
    for f in ('plan_v1.json', 'states_v1.json', 'items_v1.jsonl',
              'lock_v1.json'):
        assert gen[f] == open(os.path.join(BENCH, f), 'rb').read(), f
    # the test plan changes who is asked, never what they are asked
    plan = battery['plan']
    assert {k: v for k, v in plan.items() if k not in (
        'candidates', 'incumbent', 'reference')} == \
        {k: v for k, v in real.items() if k not in (
            'candidates', 'incumbent', 'reference')}
    assert bi.verify(battery['plan_path']) == []


# -------------------------------------------------------------- the run --

def test_the_run_settles_every_planned_key_on_the_first_try(battery):
    plan_path, run, fake = battery['plan_path'], battery['run'], \
        battery['fake']
    plan, items, _ = br.read_plan(plan_path)
    calls = br.read_calls(run)
    meta = json.load(open(os.path.join(run, 'run_meta.json')))
    assert meta['status'] == 'done'
    sd = meta['scale_down']
    assert sd == battery['scale_down']
    assert set(calls) == br.planned(plan, items, sd)
    assert all(c['valid'] and c['attempts'] == 1 for c in calls.values())
    assert battery['clock'].slept == []             # no back-off at all
    # a dropped block takes only its synthetic states: S0's items sent in
    # block 4 (a repeat, singles, an O-draw) are reported apart, and stay
    sent = {(c['state_id'] == 'S0', c['block']) for c in calls.values()}
    assert (True, 4) in sent
    assert ((False, 4) in sent) == (sd['dropped_blocks'] == [])
    n = 377 if not sd['dropped_blocks'] else 290      # spec section 2
    assert battery['summary']['analysis_set']['blocks'] == \
        [b for b in (1, 2, 3, 4) if b not in sd['dropped_blocks']]
    s0 = battery['s0']
    for b in STAND_INS:
        m = f'fake/{b}'
        posts = fake.posts(m)
        assert len(posts) == n
        assert model(battery, b)['ops']['completion'] == 1.0
        # the pilot is the record's own request, byte for byte
        assert posts[0]['data'] == json.dumps(guard.request_body(
            s0['state'], s0['names'], m)).encode()
        pp = [json.loads(p['data']) for p in posts if 'provider' in
              json.loads(p['data'])]
        assert [p['provider'] for p in pp] == [
            {'order': ['FakeProv'], 'allow_fallbacks': False}]
        # the back-to-back triple went out on one slot
        sent = [p['data'] for p in posts]
        canon = posts[0]['data']
        at = [i for i, x in enumerate(sent) if x == canon]
        assert any(at[k + 2] - at[k] == 2 for k in range(len(at) - 2))


# ---------------------------------------- spec section 8's stand-in table --

def test_the_offline_baselines(battery):
    base = battery['summary']['baselines']
    # battery_items' offline Q_rule (the plan's note, on all 24 states) and
    # battery_score's (on the analysis set) are one computation
    q_rule, *_ = bi.rule_q(battery['states'])
    assert base['Q_rule'] == pytest.approx(q_rule, abs=1e-12)
    q24, *_ = bi.rule_q(bs.load(battery['plan_path'])['states'])
    assert f'Q_rule = {q24:.4f}' in battery['plan']['baselines_note']
    assert base['constant_0.5'] == 0.5
    assert base['skip_nothing_always_hedge'] == 0.5
    assert base['criteria_oracle'] == 1.0


def q_ceiling(states):
    """The highest Q any decider can reach through guard.rule on these
    states: a name is skipped only 0.10 under the night's median p, so at
    most (n - 1) // 2 of n names can be skipped. The generator keeps at
    least 6 Ordinary names in 11, so no state has more skip-criteria names
    than that (a 6-of-11 state, S17 of the first draw, once capped every
    model below 1.0)."""
    crit = [(sum(c in SKIP_CRITERIA for c in st['conditions'].values()),
             (len(st['labels']) - 1) // 2) for st in states.values()]
    return 0.25 * (sum(min(k, cap) for k, cap in crit)
                   / sum(k for k, _ in crit) + 3)


def test_every_label_is_reachable_through_the_rule(battery):
    top = q_ceiling(battery['states'])
    assert top == 1.0                   # the criteria oracle's 1.00 is real
    for r in battery['summary']['models'].values():
        assert r['headline']['Q_h'] <= top + 1e-12


def test_the_oracle_passes_every_gate(battery):
    o = model(battery, 'oracle')
    assert o['failed'] == []
    assert o['headline']['Q_h'] >= 0.95


def test_the_constant_is_a_coin_and_fails_G2a_G3_G5(battery):
    c = model(battery, 'const')
    assert c['headline']['Q_h'] == 0.5
    assert c['noise']['D'] < 0.01                   # no dispersion at all
    assert {'G2a', 'G3a', 'G3b', 'G5'} <= set(c['failed'])


def test_uniform_noise_is_a_coin_and_fails_G3_G4(battery):
    u = model(battery, 'uniform')
    assert abs(u['headline']['Q_h'] - 0.5) <= 0.05
    assert {'G3a', 'G3b', 'G4'} <= set(u['failed'])


def test_the_lethargic_incumbent_fails_G3a_G3b_G5(battery):
    assert {'G3a', 'G3b', 'G5'} <= set(model(battery, 'lethargic')['failed'])


def test_the_rule_clone_scores_Q_rule_exactly_and_fails_G3b_G5(battery):
    r = model(battery, 'rule')
    assert r['h'] == 'k1'
    assert r['headline']['Q_h'] == battery['summary']['baselines']['Q_rule']
    assert r['failed'] == ['G3b', 'G5']


def repeat_noise(battery, m):
    """Mean squared gap, per state and question, between the two k1
    decision sets' p (C^a, C^b) and between the two k3 sets' p (the means
    of O1-O3 and of O4-O6)."""
    calls, items = br.read_calls(battery['run']), battery['items']
    P = {c['item_id']: bs.pdict(c, items[c['item_id']])
         for c in calls.values() if c['model'] == m}
    k1, k3 = [], []
    for s in battery['states']:
        a, b = P[f'E1|{s}|Ca'], P[f'E1|{s}|Cb']
        o = [P[f'E1|{s}|O{i}'] for i in range(1, 7)]
        x, y = bs.mean_p(o[:3]), bs.mean_p(o[3:])
        k1 += [(a[q] - b[q]) ** 2 for q in a]
        k3 += [(x[q] - y[q]) ** 2 for q in x]
    return float(np.mean(k1)), float(np.mean(k3))


def test_the_jev_like_model_is_handled_at_k3_and_passes(battery):
    j = model(battery, 'jevlike')
    assert j['h'] == 'k3' and j['failed'] == []
    # Section 8 expects flip_k1 > flip_k3. Its stand-in is the oracle plus
    # N(0, 0.03), and the oracle's p sits far from guard.rule's bars on
    # every name but one (S17's ASML, 0.089 under the median), so each
    # handling flips that one decision once (1/288 each, 1/216 at the
    # floor). What k3 buys is still measurable a step earlier, in p --
    # three draws averaged, so the two k3 sets sit ~1/3 as far apart as
    # C^a and C^b.
    assert j['E1']['flip_k1'] >= j['E1']['flip_k3']
    k1, k3 = repeat_noise(battery, R)
    assert k3 < 0.6 * k1


def test_position_bias_is_handled_at_k3(battery):
    assert model(battery, 'posbias')['h'] == 'k3'


def test_the_joint_answerer_fails_G2b(battery):
    assert 'G2b' in model(battery, 'joint')['failed']


def test_the_ticker_prior_fails_G6(battery):
    assert 'G6' in model(battery, 'tickerprior')['failed']


def test_the_caching_model_is_flagged_by_rho_b2b(battery):
    e9 = model(battery, 'caching')['E9']
    assert e9['caching_flag'] and e9['rho_b2b'] < bs.CACHE_RHO_FLAG
    for b in STAND_INS:                 # and nobody else is
        if b != 'caching':
            assert not model(battery, b)['E9']['caching_flag'], b


# ------------------------------------------------------- the selection --

def ranked(summary, X, ids):
    """Section 3.4's order, from summary.json alone: Q_h, then AUC_comb,
    mean(lift_E, lift_M6, delta_h), flip_h (lower), X (unscored last),
    median cost (lower), id."""
    def v(x, worst):
        return worst if x is None else round(x, 12)

    def key(m):
        h = summary['models'][m]['headline']
        return (-v(h['Q_h'], -math.inf), -v(h['AUC_comb'], -math.inf),
                -v(h['tiebreak_mean'], -math.inf), v(h['flip_h'], math.inf),
                -v(X.get(m), -math.inf), v(h['median_cost_usd'], math.inf),
                m)
    return sorted(ids, key=key)


def test_the_ineligible_incumbent_gives_way_to_the_best_eligible(battery):
    summary, sel = battery['summary'], battery['selection']
    models = summary['models']
    E = [m for m in models if not models[m]['failed']]
    assert I not in E and 'fake/oracle' in E
    assert sel['eligible'] == E and sel['branch'] == 'b'
    assert sel['switch_test'] is None
    X = {c['id']: c['X'] for c in battery['plan']['candidates']}
    order = ranked(summary, X, E)
    assert [r['id'] for r in sel['ranking']] == order
    W = order[0]
    dec = sel['decider']
    assert dec['pin'] == sel['top'] == W
    assert models[W]['headline']['Q_h'] == max(models[m]['headline']['Q_h']
                                               for m in E)
    # Section 8 expects the oracle here. On these items it is eligible and
    # within one decision of the top, but S17 (6 skip-criteria names of 11)
    # sits under guard.rule's median bar, and a noisy stand-in's draws can
    # clear it where the noiseless oracle's p (ASML 0.089 under the median)
    # cannot: the declared winner is whoever ranks first, by the rule.
    gap = models[W]['headline']['Q_h'] - \
        models['fake/oracle']['headline']['Q_h']
    assert 0 <= gap <= q_ceiling(battery['states']) - \
        models['fake/oracle']['headline']['Q_h'] + 1e-12
    assert dec['handling'] == models[W]['h']
    assert dec['suffix'] == ('#perm3.q1' if dec['handling'] == 'k3' else '')
    assert dec['version'] == 'q1' and dec['provider_pin'] is None
    ref = sel['reference']
    assert ref['id'] == R and ref['suffix'] == ''
    assert ref['status'] == ('retired' if W == R and dec['handling'] == 'k1'
                             else 'raw')
    assert sel['summary_sha256'] == _sha(os.path.join(battery['run'],
                                                      'summary.json'))
    assert sum(sel['sensitivity']['P_rank1'].values()) == pytest.approx(1)
    assert sel['gate_unchanged'] == {'GATE_NIGHTS': guard.GATE_NIGHTS,
                                     'GATE_T': guard.GATE_T}


# ------------------------------------------------- the CLI's exit codes --

def test_select_exits_6_when_fewer_than_3_blocks_are_complete(battery,
                                                              tmp_path):
    run = tmp_path / 'truncated'
    shutil.copytree(battery['run'], run, ignore=shutil.ignore_patterns(
        'summary.json', 'selection_v1.json'))
    # blocks 3 and 4 never landed (S0's items, sent in every block, did)
    keep = [ln for ln in open(run / 'calls.jsonl')
            if json.loads(ln)['state_id'] == 'S0'
            or json.loads(ln)['block'] <= 2]
    (run / 'calls.jsonl').write_text(''.join(keep))
    assert db.main(['select', '--run', str(run)]) == 6
    s = json.load(open(run / 'summary.json'))
    assert s['analysis_set']['blocks'] == [1, 2]
    assert not s['analysis_set']['sufficient']
    assert not (run / 'selection_v1.json').exists()


def test_scrub_check_exits_7_on_a_planted_key(battery, tmp_path):
    run = tmp_path / 'leaked'
    shutil.copytree(battery['run'], run)
    with open(run / 'calls.jsonl', 'a') as f:
        f.write('{"key": "x", "error": "sk-or-v1-planted"}\n')
    assert db.main(['scrub-check', '--run', str(run)]) == 7
    assert db.main(['scrub-check', '--run', battery['run']]) == 0


def test_verify_and_make_items_refuse_bytes_that_are_not_the_lock(
        battery, tmp_path):
    d = tmp_path / 'tampered'
    shutil.copytree(battery['dir'], d, ignore=shutil.ignore_patterns('runs'))
    plan = str(d / 'plan_v1.json')
    assert db.main(['verify', '--plan', plan]) == 0
    items = (d / 'items_v1.jsonl').read_text().splitlines(keepends=True)
    it = json.loads(items[5])
    it['body']['state']['date'] = '20261014'      # one byte of one body
    items[5] = json.dumps(it) + '\n'
    (d / 'items_v1.jsonl').write_text(''.join(items))
    assert db.main(['verify', '--plan', plan]) == 5
    # make-items will not overwrite a declaration it cannot reproduce: the
    # test plan's lock names stand-ins the generator never writes
    before = {f: (d / f).read_bytes() for f in os.listdir(d)}
    assert db.main(['make-items', '--plan', plan]) == 5
    assert {f: (d / f).read_bytes() for f in os.listdir(d)} == before
