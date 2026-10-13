# Part of qqq-microstructure.
#
# The battery's iteration mode end to end (RESULTS 27; spec v1, section 4):
# from a selection as the battery leaves one, make-iter declares a round
# whose arms are request variants of the ONE selected model, and verify,
# run, score and select go through src/decider_battery.py exactly as the
# cloud session calls them, against the fake Decisions API. The stand-in
# decider (fake/zreader) judges a move against the name's own volatility
# only when the request states it -- the s3 view's *_z fields -- so s3 must
# be adopted in round 1 and nothing that changes nothing (q2, b2) ever is;
# round 2 then runs on top of s3, adopts nothing, and the holdout keeps
# q1s3 against the selection's own q1. Also: an arm's items go to that arm
# alone, with the pin in the bytes and the arm in the log; a q1 arm's bytes
# are the record's request; make-iter's bytes are deterministic; k5 and sep
# only where the selection allowed them; and the stopping rule decides
# which round may come. And the outcome is what decide_local declares: its
# suffix, its handling, the selection's provider pin in every arm's bytes.
# No network: every request goes to the fake.

import os, sys, json, shutil, hashlib, itertools
from collections import Counter
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
import decide_local as dl
import fake_decisions as fd
import test_openrouter as tor

PIN = 'fake/zreader'


def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def outcome_name(rnd):
    return 'iter_holdout_outcome.json' if rnd == 'holdout' else \
        f'iter_r{rnd}_outcome.json'


def go(src, rnd, variants, out, cost=1e-4):
    """make-iter, verify, run, score and select one round, as the cloud
    session would. The plan is a scratch copy outside the repository, so
    verify and run skip the tree checks (test_battery_run.py tests them).
    cost: what the fake bills a call."""
    args = ['make-iter', '--selection', src, '--round', rnd, '--out', out]
    if variants:
        args += ['--variants', variants]
    codes = {'make-iter': db.main(args)}
    name = 'iter_holdout' if rnd == 'holdout' else f'iter_r{rnd}'
    plan = os.path.join(out, f'plan_{name}.json')
    run = os.path.join(out, 'runs', 'r')
    codes['verify'] = db.main(['verify', '--plan', plan], check_tree=False)
    clock = fd.Clock()
    fake = fd.FakeDecisions(clock=clock, cost=cost)
    codes['run'] = db.main(['run', '--plan', plan, '--out', run,
                            '--auth', 'proxy'], opener=fake,
                           sleep=clock.sleep, now=clock.now,
                           check_tree=False)
    codes['score'] = db.main(['score', '--run', run])
    codes['select'] = db.main(['select', '--run', run])
    outcome = os.path.join(run, outcome_name(rnd))
    return {'codes': codes, 'plan_path': plan, 'run': run, 'fake': fake,
            'plan': json.load(open(plan)), 'outcome_path': outcome,
            'outcome': json.load(open(outcome)),
            'summary': json.load(open(os.path.join(run, 'summary.json')))}


@pytest.fixture(scope='module')
def protocol(tmp_path_factory):
    """Section 4 in full: round 1 (q2, s3, b2 on the selection's q1),
    round 2 (q2, b2 on round 1's adopted version), the holdout."""
    d = tmp_path_factory.mktemp('iter')     # the battery directory
    sel = fd.make_selection(d, PIN)
    r1 = go(sel, '1', 'q2,s3,b2', str(d / 'iter_r1'))
    r2 = go(r1['outcome_path'], '2', 'q2,b2', str(d / 'iter_r2'))
    ho = go(r2['outcome_path'], 'holdout', '', str(d / 'iter_holdout'))
    return {'dir': d, 'selection': sel, 'r1': r1, 'r2': r2, 'holdout': ho}


def test_every_step_exits_0(protocol):
    for k in ('r1', 'r2', 'holdout'):
        assert protocol[k]['codes'] == {'make-iter': 0, 'verify': 0,
                                        'run': 0, 'score': 0,
                                        'select': 0}, k


# -------------------------------------------------------- the arms --

def test_the_arms_are_request_variants_of_one_pin(protocol):
    p = protocol['r1']['plan']
    assert p['mode'] == 'iterate' and p['round'] == 1 and p['pin'] == PIN
    assert (p['base'], p['handling'], p['variants']) == \
        ('q1', 'k1', ['q2', 's3', 'b2'])
    assert [(a['role'], a['tag']) for a in p['arms']] == [
        ('control', 'k1.q1'), ('variant', 'k1.q2'), ('variant', 'k1.q1s3'),
        ('variant', 'k1.q1b2')]
    assert p['candidates'] == [{'id': f'{PIN}#{a["tag"]}', 'pin': PIN,
                                'price_in': 2e-8,
                                'X': 1.0 if a['role'] == 'control' else None}
                               for a in p['arms']]
    assert p['incumbent'] == p['reference'] == f'{PIN}#k1.q1'
    assert p['seeds']['items'] == 20261021
    assert p['blocks'] == 3 and p['spend']['max_usd'] == 3
    assert p['gates']['analysis_set']['blocks_min'] == 2
    # the spec's 261 calls per arm, plus the one pilot each
    for a in p['arms']:
        n = p['calls_per_model']['per_arm'][a['id']]['nominal']
        assert n == {'E0': 1, 'E1': 144, 'E2': 18, 'E3': 18, 'E4': 18,
                     'E5': 18, 'E6': 18, 'E7': 18, 'E8': 9}
    assert protocol['holdout']['plan']['seeds']['items'] == 20261029
    assert protocol['holdout']['plan']['spend']['max_usd'] == 2


def test_an_arms_items_are_sent_for_that_arm_only(protocol):
    r1 = protocol['r1']
    plan, items, _ = br.read_plan(r1['plan_path'])
    by = {it['item_id']: it for it in items}
    calls = br.read_calls(r1['run'])
    assert set(calls) == br.planned(plan, items, {'dropped_blocks': [],
                                                  'dropped_models': []})
    for c in calls.values():
        it = by[c['item_id']]
        # the record names the arm; the answer came from the pin's build
        assert it.get('for_arm') in (None, c['model'])
        assert c['valid'] and c['answered_as'] == f'{PIN}-{fd.BUILD}'
    # every request named the pin, and nothing else was sent
    posts = r1['fake'].posts()
    assert {json.loads(x['data'])['model'] for x in posts} == {PIN}
    assert Counter(x['data'] for x in posts) == Counter(
        br.send_bytes(by[c['item_id']]['body'], PIN) for c in calls.values())
    # only the s3 arm's requests carry the view's fields
    z = [x for x in posts if b'"last_night_z"' in x['data']]
    assert len(z) == sum(1 for it in items
                         if it.get('for_arm') == f'{PIN}#k1.q1s3')
    assert sum(b'equal weight: buy at the close' in x['data']
               for x in posts) == 261


def test_a_q1_arms_bytes_are_the_records_request(protocol):
    plan, items, _ = br.read_plan(protocol['r1']['plan_path'])
    by = {it['item_id']: it for it in items}
    states = json.load(open(os.path.join(os.path.dirname(
        protocol['r1']['plan_path']), plan['files']['states'])))
    for sid, st in states.items():
        rec = json.dumps(guard.request_body(st['state'], st['labels'],
                                            PIN)).encode()
        assert br.send_bytes(by[f'E1|{sid}|Ca#k1.q1']['body'], PIN) == rec
        # the other arms ask about the same state in their own way
        s3 = by[f'E1|{sid}|Ca#k1.q1s3']['body']
        assert s3 == bi._body(st['state'], st['labels'],
                              view=guard.view_s3)
    # the pilot, sent to every arm, is the control's first request
    w = by['E0|S01|W']
    assert 'for_arm' not in w
    assert w['body'] == by['E1|S01|Ca#k1.q1']['body']


# ------------------------------------------------------ the outcomes --

def test_s3_is_adopted_and_variants_that_change_nothing_are_not(protocol):
    o = protocol['r1']['outcome']
    assert o['adopted'] == 's3' and o['next'] == 'round 2'
    ch = o['checks']
    s3 = ch[f'{PIN}#k1.q1s3']
    assert s3['ok'] and s3['gates'] and s3['delta_Q'] >= 0.02
    assert s3['lower_bound'] > 0 and s3['alpha'] == pytest.approx(.05 / 3)
    for t in ('k1.q2', 'k1.q1b2'):
        assert ch[f'{PIN}#{t}']['delta_Q'] == 0
        assert not ch[f'{PIN}#{t}']['ok']
    assert o['final'] == {'version': 'q1s3', 'handling': 'k1',
                          'tag': 'k1.q1s3', 'suffix': '#k1.q1s3',
                          'declared': {'k': 1, 'version': 'q1',
                                       'views': 's3'}}
    # without the view, the stand-in misses the moves that went on ret5d
    m = protocol['r1']['summary']['models']
    assert m[f'{PIN}#k1.q1']['E1']['skip_recall'] < \
        m[f'{PIN}#k1.q1s3']['E1']['skip_recall'] == 1.0
    assert m[f'{PIN}#k1.q1s3']['h'] == 'k1'


def test_the_round_2_control_is_round_1s_adopted_version(protocol):
    p = protocol['r2']['plan']
    assert (p['base'], p['handling']) == ('q1s3', 'k1')
    assert [(a['role'], a['tag']) for a in p['arms']] == [
        ('control', 'k1.q1s3'), ('variant', 'k1.q2s3'),
        ('variant', 'k1.q1b2s3')]
    assert p['seeds']['items'] == 20261022
    assert p['prior']['sha256'] == _sha(protocol['r1']['outcome_path'])
    assert p['prior']['adopted'] == 's3'
    assert p['origin'] == protocol['r1']['plan']['origin']
    _, items, _ = br.read_plan(protocol['r2']['plan_path'])
    ctrl = [it for it in items if it.get('for_arm') == f'{PIN}#k1.q1s3']
    assert ctrl and all('last_night_z' in json.dumps(it['body'])
                        for it in ctrl)
    o = protocol['r2']['outcome']
    assert o['adopted'] is None and o['next'] == 'holdout'
    assert o['final']['tag'] == 'k1.q1s3' and o['decider'] is None


def test_the_holdout_keeps_the_final_version_against_q1(protocol):
    p = protocol['holdout']['plan']
    assert [(a['role'], a['tag']) for a in p['arms']] == [
        ('control', 'k1.q1'), ('final', 'k1.q1s3')]
    assert p['variants'] == [] and p['drop_order'] == []
    o = protocol['holdout']['outcome']
    assert o['holdout'] and o['kept'] and o['next'] == 'stop'
    f = o['checks'][f'{PIN}#k1.q1s3']
    assert f['gates'] and f['lower_bound'] > 0
    assert f['alpha'] == pytest.approx(.05)
    sel = json.load(open(protocol['selection']))
    assert o['decider'] == dict(
        sel['decider'], version='q1s3', handling='k1', suffix='#k1.q1s3',
        declared={'k': 1, 'version': 'q1', 'views': 's3', 'provider': None,
                  'dated': True, 'measured_build': f'{PIN}-{fd.BUILD}'})


def test_after_the_holdout_nothing_more_may_run(protocol, tmp_path, capsys):
    src = protocol['holdout']['outcome_path']
    for rnd in ('1', '2', 'holdout'):
        assert db.main(['make-iter', '--selection', src, '--round', rnd,
                        '--out', str(tmp_path / rnd)]) == 5
        assert "the stopping rule says next: 'stop'" in \
            capsys.readouterr().out
        assert not (tmp_path / rnd).exists()


# ---------------------------------------- what decide_local declares --

def test_every_arm_is_named_as_decide_local_would_log_it():
    """R3, R4: one suffix scheme (guard.handling_suffix). Every arm a round
    can make is a well-formed DECLARED handling, and guard.DECIDER set to
    its suffix is the runner's own -- '#k1.q2s3', never another name."""
    for w, views, k in itertools.product(('q1', 'q2'),
                                         ('', 'b2', 's3', 'b2s3'), (1, 3, 5)):
        h, version = f'k{k}', w + views
        how = bi.declared_of(h, version)
        assert how == {'k': k, 'version': w, 'views': views}
        suffix = bi.suffix_of(h, version)
        assert suffix == dl.suffix(how) == guard.handling_suffix(k, w, views)
        assert dl.declaration_problems(
            declared={'http': dict(dl.DECLARED['http'], **how)},
            decider=(dl.HOSTED['http'], suffix)) == []
    # a version names its views as DECLARED does: sorted
    assert bi.apply_tag('q1s3', 'k1', 'b2') == ('q1b2s3', 'k1')
    assert bi.apply_tag('q2b2', 'k3', 's3') == ('q2b2s3', 'k3')
    with pytest.raises(ValueError):
        bi.parse_version('q1s3b2')
    # the selection's own: the record's request, and k3's '#perm3.q1'
    assert (bi.suffix_of('k1', 'q1'), bi.suffix_of('k3', 'q1')) == \
        ('', '#perm3.q1')
    assert bi.declared_of('k1', 'q1sep') is None


def test_the_outcome_is_what_decide_local_declares(protocol, tmp_path,
                                                   monkeypatch):
    """R6: the one outcome whose next is 'stop' (here the holdout's) is the
    declaration. Copied into DECLARED and guard.DECIDER it is accepted, and
    the runtime then sends, byte for byte, the request the battery tested
    for the arm it adopted."""
    o = protocol['holdout']['outcome']
    want = tor.battery_declaration(str(protocol['dir']))
    assert want['pin'] == PIN and want['suffix'] == '#k1.q1s3'
    assert want['how'] == o['decider']['declared']
    assert dl.declaration_problems(
        hosted=dict(dl.HOSTED, http=PIN), declared={'http': want['how']},
        decider=(PIN, want['suffix'])) == []
    monkeypatch.setenv('QQQ_DECIDER_HOME', str(tmp_path))
    plan, items, d = br.read_plan(protocol['holdout']['plan_path'])
    by = {it['item_id']: it for it in items}
    states = json.load(open(os.path.join(d, plan['files']['states'])))
    for sid in ('S01', 'S12'):
        st, fake = states[sid], fd.FakeDecisions()
        _, who, _ = dl.hosted_ask(PIN, None, fake, backend='http',
                                  today='20261013', how=want['how'])(
            st['state'], st['labels'])
        assert who == f'{PIN}-{fd.BUILD}@FakeProv#k1.q1s3'
        assert [x['data'] for x in fake.posts()] == [
            br.send_bytes(by[f'E1|{sid}|Ca#k1.q1s3']['body'], PIN)]


def test_a_pinned_selection_pins_every_arm_as_the_runtime_would(
        tmp_path, monkeypatch):
    """R5: a selection that pins its decider's provider has every arm --
    the pilot W, the control, the holdout's final -- sent with that pin, as
    decide_local sends every call once it is declared; verify holds every
    body to it."""
    pv = {'order': ['FakeProv'], 'allow_fallbacks': False}
    sel = fd.make_selection(tmp_path, PIN, provider_pin=True)
    assert json.load(open(sel))['decider']['provider_pin'] == pv
    r = go(sel, '1', 'q2,s3', str(tmp_path / 'iter_r1'))
    assert set(r['codes'].values()) == {0}
    assert r['plan']['provider_pin'] == pv
    plan, items, d = br.read_plan(r['plan_path'])
    assert all(it['body']['provider'] == pv for it in items)
    assert all(json.loads(x['data'])['provider'] == pv
               for x in r['fake'].posts())
    monkeypatch.setenv('QQQ_DECIDER_HOME', str(tmp_path / 'home'))
    by = {it['item_id']: it for it in items}
    st = json.load(open(os.path.join(d, plan['files']['states'])))['S01']
    for a in plan['arms']:
        how = dict(a['declared'], provider='FakeProv', dated=True,
                   measured_build=None)
        fake = fd.FakeDecisions()
        dl.hosted_ask(PIN, None, fake, backend='http', today='20261013',
                      how=how)(st['state'], st['labels'])
        assert [x['data'] for x in fake.posts()] == [
            br.send_bytes(by[f'E1|S01|Ca#{a["tag"]}']['body'], PIN)]
    # an item without the pin is caught
    t = tmp_path / 'tampered'
    shutil.copytree(d, t, ignore=shutil.ignore_patterns('runs'))
    lines = (t / plan['files']['items']).read_text().splitlines(True)
    it = json.loads(lines[5])
    del it['body']['provider']
    it['body_sha256'] = bi.body_sha(it['body'])
    lines[5] = json.dumps(it) + '\n'
    (t / plan['files']['items']).write_text(''.join(lines))
    probs = bi.verify(str(t / 'plan_iter_r1.json'))
    assert any("is not the selection's pin" in p for p in probs)


# --------------------------------------------------------- make-iter --

def tree(d):
    return {f: open(os.path.join(d, f), 'rb').read()
            for f in sorted(os.listdir(d)) if f != 'runs'}


def test_make_iter_writes_the_same_bytes_every_time(protocol, tmp_path,
                                                    capsys):
    """Built twice from one source, in either variant order, the bytes are
    the declaration round 1 ran. make-iter into its own directory again
    changes nothing; another declaration there is refused -- without
    --force, and with it, since that declaration has a run."""
    sel = protocol['selection']
    src = db.iter_source(sel, 1)
    for k, v in (('a', ['b2', 's3', 'q2']), ('b', ['q2', 's3', 'b2'])):
        os.makedirs(tmp_path / k)
        bi.make_iter(str(tmp_path / k), dict(src, variants=v))
    a = tree(tmp_path / 'a')
    assert a == tree(tmp_path / 'b')
    r1 = os.path.dirname(protocol['r1']['plan_path'])
    assert a == tree(r1)
    assert sorted(a) == ['anchors_20261013.json', 'items_iter_r1.jsonl',
                         'lock_iter_r1.json', 'plan_iter_r1.json',
                         'states_iter_r1.json']
    lock = json.loads(a['lock_iter_r1.json'])
    assert lock == {k: hashlib.sha256(a[f]).hexdigest() for k, f in (
        ('plan', 'plan_iter_r1.json'), ('states', 'states_iter_r1.json'),
        ('items', 'items_iter_r1.jsonl'),
        ('anchors', 'anchors_20261013.json'))}
    assert a['anchors_20261013.json'] == open(fd.ANCHORS_FILE, 'rb').read()
    assert bi.verify(str(tmp_path / 'a' / 'plan_iter_r1.json')) == []
    args = ['make-iter', '--selection', sel, '--round', '1', '--out', r1]
    assert db.main(args + ['--variants', 'q2,s3,b2']) == 0
    assert db.main(args + ['--variants', 's3']) == 5
    assert '--force replaces it' in capsys.readouterr().out
    assert db.main(args + ['--variants', 's3', '--force']) == 5
    assert 'holds a run of its declaration' in capsys.readouterr().out
    assert tree(r1) == a


def test_a_round_is_declared_once_inside_its_battery(protocol, tmp_path,
                                                     capsys):
    """Round 1 again from the same selection is the same 18 states (the
    seed is fixed), re-rolled: refused anywhere in the battery directory,
    and outside it make-iter could not see the first, so that is refused
    too. Likewise round 2 after round 2."""
    d, sel = protocol['dir'], protocol['selection']
    for out, why in ((d / 'iter_r1_again', 'iter_r1 is already declared'),
                     (d / 'more' / 'iter_r1', 'iter_r1 is already declared'),
                     (tmp_path / 'iter_r1', 'is not inside')):
        assert db.main(['make-iter', '--selection', sel, '--round', '1',
                        '--variants', 's3', '--out', str(out)]) == 5
        assert why in capsys.readouterr().out
        assert not out.exists()
    assert db.main(['make-iter', '--selection',
                    protocol['r1']['outcome_path'], '--round', '2',
                    '--variants', 'q2,b2', '--out',
                    str(d / 'iter_r2_again')]) == 5
    assert 'iter_r2 is already declared' in capsys.readouterr().out


def test_round_2_is_the_remaining_menu():
    """Round 2 is chosen after round 1's per-variant results: it tests the
    remaining menu on top, not a hand-picked part of it (which would also
    loosen alpha = 0.05 / #variants)."""
    origin = {'k5_allowed': False, 'sep_allowed': False}
    arms = bi.iter_arms(PIN, 2, ['q1s3', 'k1'], ['b2', 'q2'], origin)
    assert [a['tag'] for a in arms] == ['k1.q1s3', 'k1.q2s3', 'k1.q1b2s3']
    for v in (['b2'], ['q2'], ['q2', 'b2', 's3']):
        with pytest.raises(ValueError, match='--variants q2,b2$'):
            bi.iter_arms(PIN, 2, ['q1s3', 'k1'], v, origin)
    # what the selection opened belongs to it; sep, never yet
    assert bi.menu_left(['q1s3', 'k3'], dict(origin, k5_allowed=True)) == \
        ['q2', 'b2', 'k5']
    assert bi.menu_left(['q1', 'k1'], dict(origin, sep_allowed=True)) == \
        ['q2', 's3', 'b2']
    # round 1 may name any of the menu
    assert len(bi.iter_arms(PIN, 1, ['q1', 'k1'], ['b2'], origin)) == 2


def test_verify_catches_an_item_sent_to_the_wrong_arm(protocol, tmp_path):
    d = tmp_path / 'tampered'
    shutil.copytree(os.path.dirname(protocol['r1']['plan_path']), d,
                    ignore=shutil.ignore_patterns('runs'))
    lines = (d / 'items_iter_r1.jsonl').read_text().splitlines(True)
    it = json.loads(lines[1])
    it['for_arm'] = f'{PIN}#k1.q1s3'
    lines[1] = json.dumps(it) + '\n'
    (d / 'items_iter_r1.jsonl').write_text(''.join(lines))
    probs = bi.verify(str(d / 'plan_iter_r1.json'))
    assert any('items sha256 mismatch' in p for p in probs)
    assert any('id is not base_id + arm tag' in p for p in probs)


@pytest.mark.parametrize('variants, why', [
    ('k5', 'k5 is not allowed'), ('sep', 'sep is not allowed'),
    ('q2,s3,k5', 'k5 is not allowed'), ('x9', 'not on the menu'),
    ('q2,q2', 'named twice'), ('', 'at least one variant')])
def test_k5_and_sep_need_the_selections_reason(tmp_path, capsys, variants,
                                               why):
    sel = fd.make_selection(tmp_path, PIN, handling='k1', sd_rep=0.005,
                            ex_ctx=0.02)
    out = tmp_path / 'iter_r1'
    assert db.main(['make-iter', '--selection', sel, '--round', '1',
                    '--variants', variants, '--out', str(out)]) == 5
    assert why in capsys.readouterr().out
    assert not out.exists()


def test_sep_is_refused_until_the_runtime_can_send_it(tmp_path, capsys):
    """decide_local sends every question in one request: an adopted sep
    could never be declared, so it is not tested even where ex_ctx opens
    it."""
    sel = fd.make_selection(tmp_path, PIN, ex_ctx=0.021)
    out = tmp_path / 'iter_r1'
    assert db.main(['make-iter', '--selection', sel, '--round', '1',
                    '--variants', 'q2,sep', '--out', str(out)]) == 5
    assert 'sep cannot be declared yet' in capsys.readouterr().out
    assert not out.exists()


def test_k5_and_sep_where_the_selection_allows_them(tmp_path, monkeypatch):
    """sd_rep > 0.005 opens k5 (five seeded orders a set, not three) and
    ex_ctx > 0.02 opens sep (one request per question); both are scored,
    k5 at its own handling and sep's answers merged back per request. sep
    is refused by make-iter until decide_local can send it; its items and
    scoring are kept, and tested here, for that day."""
    monkeypatch.setattr(bi, 'RUNTIME_MENU', bi.MENU)
    sel = fd.make_selection(tmp_path, PIN, handling='k3', sd_rep=0.0051,
                            ex_ctx=0.021)
    r = go(sel, '1', 'sep,k5,s3', str(tmp_path / 'iter_r1'))
    assert r['codes'] == {'make-iter': 0, 'verify': 0, 'run': 0, 'score': 0,
                          'select': 0}
    p = r['plan']
    assert [a['tag'] for a in p['arms']] == ['k3.q1', 'k3.q1s3', 'k5.q1',
                                             'k3.q1sep']
    per = p['calls_per_model']['per_arm']
    assert per[f'{PIN}#k5.q1']['nominal']['E1'] == 18 * 12
    _, items, _ = br.read_plan(r['plan_path'])
    k5 = {it['arm'] for it in items if it.get('for_arm') == f'{PIN}#k5.q1'
          and it['experiment'] == 'E1'}
    assert k5 == {'Ca', 'Cb'} | {f'O{i}' for i in range(1, 11)}
    sep = [it for it in items if it.get('for_arm') == f'{PIN}#k3.q1sep']
    assert all(len(it['body']['questions']) == 1 for it in sep)
    ca = [it for it in sep if it['base_id'] == 'E1|S01|Ca']
    whole = next(it for it in items if it['item_id'] == 'E1|S01|Ca#k3.q1')
    assert len(ca) == 12 and [it['body']['questions'] for it in ca] == [
        {q: x} for q, x in whole['body']['questions'].items()]
    assert all(it['body']['state'] == whole['body']['state'] for it in ca)
    m = r['summary']['models']
    assert m[f'{PIN}#k5.q1']['h'] == 'k5' and 'k5' in \
        m[f'{PIN}#k5.q1']['q_counts']
    assert m[f'{PIN}#k3.q1']['h'] == 'k3' and 'k5' not in \
        m[f'{PIN}#k3.q1']['q_counts']
    # the stand-in reads each question alone and ignores order: the k5
    # and sep arms answer exactly as the control, so neither is adopted
    for t in ('k5.q1', 'k3.q1sep'):
        assert m[f'{PIN}#{t}']['q_counts'][m[f'{PIN}#{t}']['h']] == \
            m[f'{PIN}#k3.q1']['q_counts']['k3']
        assert m[f'{PIN}#{t}']['noise']['n_rep_pairs'] == \
            m[f'{PIN}#k3.q1']['noise']['n_rep_pairs']
    o = r['outcome']
    assert o['adopted'] == 's3' and o['final']['suffix'] == '#perm3.q1s3'


def test_a_sep_controls_pilot_is_a_whole_request():
    """A round on a sep control: W, on which the cost per call (a whole
    request's) is measured, is the control's first request whole, so each
    one-question call is priced at its share of the bytes -- not as a whole
    request, which projected about 3x and dropped what the cap had room
    for."""
    arms = [{'id': f'{PIN}#k1.q1sep', 'tag': 'k1.q1sep', 'role': 'control',
             'menu': None, 'version': 'q1sep', 'handling': 'k1'}]
    states, items = bi.iter_build(json.load(open(fd.ANCHORS_FILE)),
                                  20261022, 1, arms, 'iter_r2')
    w, rest = items[0], items[1:]
    st = states['S01']
    assert w['item_id'] == 'E0|S01|W' and 'for_arm' not in w
    assert w['body'] == bi._body(st['state'], st['labels'])
    assert w['meta']['body_of'] == 'E1|S01|Ca#k1.q1sep'
    assert all(len(it['body']['questions']) == 1 for it in rest)
    share = (bi.iter_cost(items, PIN, 1.0) - 1) / len(rest)
    assert 0.2 < share < 0.6


def test_the_stopping_rule_names_the_only_next_step(protocol, tmp_path,
                                                    capsys):
    sel, r1 = protocol['selection'], protocol['r1']['outcome_path']
    for src, rnd, variants, why in (
            (sel, '2', 'q2', 'a selection starts round 1'),
            (sel, 'holdout', '', 'a selection starts round 1'),
            (r1, '1', 'q2', "says next: 'round 2'"),
            (r1, 'holdout', '', "says next: 'round 2'")):
        out = tmp_path / f'{rnd}-{len(variants)}'
        assert db.main(['make-iter', '--selection', src, '--round', rnd,
                        '--variants', variants, '--out', str(out)]) == 5
        assert why in capsys.readouterr().out
        assert not out.exists()


def test_a_selection_whose_summary_changed_is_refused(tmp_path, capsys):
    sel = fd.make_selection(tmp_path, PIN)
    s = tmp_path / 'runs' / 'v1' / 'summary.json'
    s.write_text(s.read_text().replace('0.002', '0.009'))   # sd_rep
    assert db.main(['make-iter', '--selection', sel, '--round', '1',
                    '--variants', 'k5', '--out',
                    str(tmp_path / 'iter_r1')]) == 5
    assert 'not the one it was made from' in capsys.readouterr().out


def test_a_hand_edited_selection_or_outcome_unlocks_nothing(tmp_path,
                                                            capsys):
    """make-iter re-derives the decision from the summary and plan beside
    the file: an edited decider, stopping rule, adopted version or origin
    is refused, though every sha beside it still matches."""
    sel = fd.make_selection(tmp_path, PIN)
    honest = json.load(open(sel))
    for edit in ({'decider': dict(honest['decider'], provider_pin={
                     'order': ['FakeProv'], 'allow_fallbacks': False})},
                 {'decider': dict(honest['decider'], handling='k3',
                                  suffix='#perm3.q1')},
                 {'branch': 'b'}):
        with open(sel, 'w') as f:
            json.dump(dict(honest, **edit), f)
        assert db.main(['make-iter', '--selection', sel, '--round', '1',
                        '--variants', 'q2', '--out',
                        str(tmp_path / 'iter_r1')]) == 5
        assert 'is not what its summary and plan decide' in \
            capsys.readouterr().out
    with open(sel, 'w') as f:
        json.dump(honest, f, indent=1)
    # an honest round 1 that adopts nothing: the rule says stop
    r1 = go(sel, '1', 'q2,b2', str(tmp_path / 'iter_r1'))
    o = r1['outcome']
    assert set(r1['codes'].values()) == {0}
    assert o['adopted'] is None and o['next'] == 'stop'
    s3 = {'version': 'q1s3', 'handling': 'k1', 'tag': 'k1.q1s3',
          'suffix': '#k1.q1s3',
          'declared': {'k': 1, 'version': 'q1', 'views': 's3'}}
    for edit, variants in (
            ({'next': 'round 2'}, 'q2,s3,b2'),
            ({'next': 'round 2', 'adopted': 's3', 'final': s3}, 'q2,b2'),
            ({'next': 'round 2', 'origin': dict(o['origin'],
                                                k5_allowed=True)}, 'k5')):
        with open(r1['outcome_path'], 'w') as f:
            json.dump(dict(o, **edit), f)
        assert db.main(['make-iter', '--selection', r1['outcome_path'],
                        '--round', '2', '--variants', variants, '--out',
                        str(tmp_path / 'iter_r2')]) == 5
        assert 'is not what its summary and plan decide' in \
            capsys.readouterr().out
        assert not (tmp_path / 'iter_r2').exists()


def test_a_projection_over_the_cap_drops_a_block_then_the_last_variant(
        tmp_path):
    """$0.004 a call: 4 arms of 262 project $4.20 > 0.8 x $3; dropping
    block 3 gives $2.80, still over; dropping b2 (the last on the menu)
    gives $2.09."""
    sel = fd.make_selection(tmp_path / 'v1', PIN, cost=0.004)
    assert db.main(['make-iter', '--selection', sel, '--round', '1',
                    '--variants', 'q2,s3,b2',
                    '--out', str(tmp_path / 'v1' / 'r1')]) == 0
    p = json.load(open(tmp_path / 'v1' / 'r1' / 'plan_iter_r1.json'))
    pr = p['projection']
    assert pr['dropped_blocks'] == [3] and pr['dropped_variants'] == ['b2']
    assert p['blocks'] == 2 and p['variants'] == ['q2', 's3']
    assert pr['usd'] <= pr['limit_usd'] == pytest.approx(2.4)
    assert p['drop_order'] == [f'{PIN}#k1.q1s3', f'{PIN}#k1.q2']
    # at the floor, the run's scale-down may drop no further block
    c = p['calls_per_model']
    assert c['floor'] == c['nominal']
    # at $0.02 a call even the floor is over: refused
    sel = fd.make_selection(tmp_path / 'v2', PIN, cost=0.02)
    assert db.main(['make-iter', '--selection', sel, '--round', '1',
                    '--variants', 'q2', '--out',
                    str(tmp_path / 'v2' / 'r2')]) == 5


def test_a_plan_at_its_floor_drops_a_variant_never_a_block(tmp_path):
    """make-iter cut this plan to its 2-block floor ($0.004 a call: block 3
    and b2 dropped). The pilot then measures 17% more: the run drops the
    last variant on the menu (s3) -- not block 2, which would have left one
    block, no outcome, and the money spent."""
    sel = fd.make_selection(tmp_path, PIN, cost=0.004)
    r = go(sel, '1', 'q2,s3,b2', str(tmp_path / 'iter_r1'), cost=0.0047)
    p = r['plan']
    assert p['blocks'] == 2 and p['variants'] == ['q2', 's3']
    assert r['codes'] == {'make-iter': 0, 'verify': 0, 'run': 0, 'score': 0,
                          'select': 0}
    meta = json.load(open(os.path.join(r['run'], 'run_meta.json')))
    assert meta['projection']['usd'] > meta['projection']['limit_usd']
    assert meta['scale_down'] == {'dropped_blocks': [],
                                  'dropped_models': [f'{PIN}#k1.q1s3']}
    assert r['summary']['analysis_set']['blocks'] == [1, 2]
    assert r['outcome']['checks'][f'{PIN}#k1.q1s3']['dropped']
    # verify's bound says the same; a selection plan still drops block 4
    plan, items, _ = br.read_plan(r['plan_path'])
    pr = db.projection(plan, items)
    assert br.last_block(plan) == [] and \
        pr['scale_down']['dropped_blocks'] == []
    assert br.last_block(dict(plan, blocks=3)) == [3]
    v1 = json.load(open(os.path.join(ROOT, 'bench', 'battery',
                                     'plan_v1.json')))
    assert br.last_block(v1) == [4]


def test_k_draws_average_as_the_runtime_logs_them():
    """R3: an iteration arm's k3 and k5 decisions are scored on the p the
    runtime logs (guard.mean_of_draws): hundredths averaging to 0.5 hedge.
    A selection keeps v1's arithmetic, with which its run was scored."""
    ps = [{'BOOK': 0.47}, {'BOOK': 0.58}, {'BOOK': 0.45}]
    assert bs.mean_draws(ps) == guard.mean_of_draws(ps) == {'BOOK': 0.5}
    assert guard.rule(bs.mean_draws(ps), [])['BOOK'] == 'hedge'
    assert bs.mean_p(ps)['BOOK'] < 0.5                  # v1's, unchanged
    for a, b in itertools.product(range(101), repeat=2):
        c = 150 - a - b
        if 0 <= c <= 100:
            assert guard.mean_of_draws(
                [{'x': a / 100}, {'x': b / 100}, {'x': c / 100}]) == \
                {'x': 0.5}
