# Part of qqq-microstructure.
#
# The battery's committed stimuli (src/battery_items.py, RESULTS 27): the
# lock and every body's sha hold; make_items regenerates the same bytes; no
# body names a model, and S0's canonical item plus a pin is the record's
# request byte for byte; each experiment's manipulation is exactly what the
# spec declares (coherent moves, derangements, swaps, negations, runtime k3
# orders), every sent id maps back, the counts are the spec's 377 per model,
# and the generator never touches an outcome.

import os, sys, ast, csv, json, math, re, shutil, hashlib, datetime as dt
from collections import Counter
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import guard
import battery_items as bi

BENCH = os.path.join(ROOT, 'bench', 'battery')
PLAN = os.path.join(BENCH, 'plan_v1.json')
D = bi.load(PLAN)
ITEMS = {it['item_id']: it for it in D['items']}
STATES, S0 = D['states'], D['s0']
NOMINAL = {'E0': 2, 'E1': 192, 'E2': 24, 'E3': 36, 'E4': 24, 'E5': 24,
           'E6': 24, 'E7': 24, 'E8': 13, 'E9': 14}
FLOOR = {'E0': 2, 'E1': 144, 'E2': 18, 'E3': 30, 'E4': 18, 'E5': 18,
         'E6': 18, 'E7': 18, 'E8': 10, 'E9': 14}


def of(exp, arm=None):
    return [it for it in D['items'] if it['experiment'] == exp
            and (arm is None or it['arm'] == arm)]


def send(it, pin):
    """The bytes the run sends for an item and a model (the contract)."""
    s = {'state': it['body']['state'], 'model': pin,
         'questions': it['body']['questions']}
    if 'provider' in it['body']:
        s['provider'] = it['body']['provider']
    return json.dumps(s).encode()


def by_block(sids):
    out = {}
    for sid in sids:
        out.setdefault(STATES[sid]['block'], []).append(sid)
    return out


# ------------------------------------------------------ lock and bytes --

def test_the_lock_and_every_body_sha_hold():
    assert bi.verify(PLAN) == []
    lock = json.load(open(os.path.join(BENCH, 'lock_v1.json')))
    for k, f in [('plan', 'plan_v1.json'), ('states', 'states_v1.json'),
                 ('items', 'items_v1.jsonl'), ('s0', 's0_20261013.json'),
                 ('anchors', 'anchors_20261013.json')]:
        assert lock[k] == hashlib.sha256(
            open(os.path.join(BENCH, f), 'rb').read()).hexdigest(), k
    for it in D['items']:
        assert it['body_sha256'] == hashlib.sha256(
            json.dumps(it['body']).encode()).hexdigest()


def test_verify_catches_a_tampered_body(tmp_path):
    for f in os.listdir(BENCH):
        if os.path.isfile(os.path.join(BENCH, f)):
            shutil.copy(os.path.join(BENCH, f), tmp_path / f)
    lines = open(tmp_path / 'items_v1.jsonl').read().splitlines()
    it = json.loads(lines[5])
    it['body']['state']['date'] = '20261014'
    lines[5] = json.dumps(it)
    (tmp_path / 'items_v1.jsonl').write_text('\n'.join(lines) + '\n')
    probs = bi.verify(str(tmp_path / 'plan_v1.json'))
    assert any('lock: items' in p for p in probs)
    assert any(it['item_id'] in p and 'body_sha256' in p for p in probs)


def test_make_items_regenerates_identical_bytes(tmp_path):
    rec = D['plan']['generator']['numpy']
    if '.'.join(np.__version__.split('.')[:2]) != rec:
        pytest.skip(f'items were generated with numpy {rec}')
    for f in ('s0_20261013.json', 'anchors_20261013.json'):
        shutil.copy(os.path.join(BENCH, f), tmp_path / f)
    before = {f: (tmp_path / f).read_bytes() for f in os.listdir(tmp_path)}
    bi.make_items(str(tmp_path))
    for f in ('plan_v1.json', 'states_v1.json', 'items_v1.jsonl',
              'lock_v1.json'):
        assert (tmp_path / f).read_bytes() == \
            open(os.path.join(BENCH, f), 'rb').read(), f
    for f, b in before.items():                     # inputs untouched
        assert (tmp_path / f).read_bytes() == b


def test_no_body_names_a_model():
    for it in D['items']:
        assert 'model' not in it['body']
        assert list(it['body']) in (['state', 'questions'],
                                    ['state', 'questions', 'provider'])
    assert [it['item_id'] for it in D['items'] if 'provider' in it['body']] \
        == ['E0|S0|PP']
    assert ITEMS['E0|S0|PP']['body']['provider'] == '__FROM_W__'


@pytest.mark.parametrize('pin', [bi.INCUMBENT, bi.REFERENCE,
                                 'respan/span-01'])
def test_s0s_canonical_item_with_a_pin_is_the_records_request(pin):
    want = json.dumps(guard.request_body(S0['state'], S0['names'],
                                         pin)).encode()
    canon = ['E0|S0|W'] + [f'E9|S0|S0rep{r}' for r in range(1, 5)] + \
        [f'E9|S0|B2B{i}' for i in range(1, 4)]
    for iid in canon:
        assert send(ITEMS[iid], pin) == want, iid
    if pin == bi.INCUMBENT:             # the sha test_openrouter pins
        assert hashlib.sha256(want).hexdigest() == \
            '5489fee7f67aa8d81d9649e6df590956b4a592db49158634e0e20abf2dbfb220'
    pp = dict(ITEMS['E0|S0|PP'], body=dict(ITEMS['E0|S0|PP']['body']))
    prov = {'order': ['Perplexity'], 'allow_fallbacks': False}
    pp['body']['provider'] = prov
    assert send(pp, pin) == json.dumps(guard.request_body(
        S0['state'], S0['names'], pin, provider=prov)).encode()


# ------------------------------------------------------- the states --

def test_every_synthetic_state_is_well_formed():
    assert list(STATES) == [f'S{s:02d}' for s in range(1, 25)]
    for sid, st in STATES.items():
        c = Counter(st['conditions'].values())
        assert c['Ordinary'] >= 5
        assert c['Earnings'] + c['ExtremeDown'] + c['ExtremeUp'] >= 2
        assert len(st['labels']) == 11 == len(set(st['labels']))
        assert set(st['labels']) <= set(bi.POOL)
        s = st['state']
        assert list(s) == ['schema', 'date', 'book', 'names', 'QQQ']
        assert s['schema'] == 2 and s['date'] == '20261013'
        assert s['book'] == S0['state']['book']
        assert list(s['names']) == st['labels']
        for t, row in s['names'].items():
            assert list(row) == ['last_night', 'nights5_avg', 'ret5d',
                                 'vol20d', 'earnings_tonight']
            assert row == st['rows'][t]
            assert row['earnings_tonight'] == \
                (st['conditions'][t] == 'Earnings')
        assert s['QQQ'] == st['qqq'][st['regime']]
        assert list(st['qqq']['calm']) == list(st['qqq']['stressed']) == \
            ['last_night', 'nights5_avg', 'ret5d', 'vol20d']
        assert 2 <= st['stressed_m'] <= 3 and st['stressed_sign'] in (1, -1)
    for b, sids in by_block(STATES).items():
        assert len(sids) == 6
        assert Counter(STATES[s]['regime'] for s in sids) == \
            {'stressed': 3, 'calm': 3}
        assert [STATES[s]['cx_type'] for s in sids] == [1, 1, 2, 2, 3, 3]
        assert Counter(STATES[s]['e7_kind'] for s in sids) == \
            {'LD': 4, 'AN': 2}
        assert {STATES[s]['regime'] for s in sids
                if STATES[s]['e7_kind'] == 'AN'} == {'stressed', 'calm'}
        assert sum(STATES[s]['e8'] for s in sids) == 3


def test_states_regenerate_and_moves_are_coherent():
    """Extreme and Moderate names: the move lands on last_night or ret5d
    with the declared coherent update, and the z recomputed from the final
    row still says what the condition says."""
    vols = D['anchors']['vol20d']
    moved = Counter()
    for sid, st in STATES.items():
        rec, trace, _ = bi.gen_state(int(sid[1:]), st['regime'], vols)
        assert rec['rows'] == st['rows'] and rec['qqq'] == st['qqq']
        assert rec['labels'] == st['labels']
        for t in st['labels']:
            c, row = st['conditions'][t], st['rows'][t]
            if c in ('Ordinary', 'Earnings'):
                assert t not in trace
                son = 0.5 * row['vol20d']
                assert abs(row['last_night']) <= son + 0.5
                assert abs(row['nights5_avg']) <= son / math.sqrt(5) + 0.5
                assert abs(row['ret5d']) <= row['vol20d'] * math.sqrt(5) + .5
                continue
            tr, b = trace[t], trace[t]['before']
            d = tr['D']
            if tr['field'] == 'last_night':
                assert tr['new'] == tr['sign'] * round(
                    tr['z'] * 0.5 * b['vol20d'])
                assert d == tr['new'] - b['last_night']
                assert row['last_night'] == tr['new']
                assert row['nights5_avg'] == b['nights5_avg'] + round(d / 5)
                assert row['ret5d'] == b['ret5d'] + d
                assert row['vol20d'] == round(math.sqrt(
                    b['vol20d'] ** 2 + d ** 2 / 19))
                z = row['last_night'] / (0.5 * row['vol20d'])
            else:
                assert tr['new'] == tr['sign'] * round(
                    tr['z'] * b['vol20d'] * math.sqrt(5))
                assert d == tr['new'] - b['ret5d']
                assert row['ret5d'] == tr['new']
                assert (row['last_night'], row['nights5_avg']) == \
                    (b['last_night'], b['nights5_avg'])
                assert row['vol20d'] == round(math.sqrt(
                    b['vol20d'] ** 2 + d ** 2 / 95))
                z = row['ret5d'] / (row['vol20d'] * math.sqrt(5))
            lo, hi = (1.5, 2.5) if c == 'Moderate' else (3.5, 5.0)
            assert lo <= tr['z'] <= hi
            # the vol update shrinks z; worst cases worked out by hand: an
            # extreme move stays >= ~2.44 sd, a moderate one <= ~2.46
            if c != 'Moderate':
                assert tr['sign'] == (-1 if c == 'ExtremeDown' else 1)
                assert z * tr['sign'] >= 2.4
            else:
                assert 1.25 <= z * tr['sign'] <= 2.5
            moved[tr['field']] += 1
    assert moved['last_night'] > 10 and moved['ret5d'] > 10


def test_stress_scales_names_and_qqq_rows_follow_the_spec():
    vols = D['anchors']['vol20d']
    a = vols['QQQ']
    for sid, st in STATES.items():
        calm_rec, _, _ = bi.gen_state(int(sid[1:]), 'calm', vols)
        stress_rec, _, _ = bi.gen_state(int(sid[1:]), 'stressed', vols)
        f = stress_rec['name_vol_factor']
        assert 1.3 <= f <= 1.8 and calm_rec['qqq'] == stress_rec['qqq']
        for t in st['labels']:
            if st['conditions'][t] == 'Ordinary':
                assert abs(stress_rec['rows'][t]['vol20d'] - f * calm_rec[
                    'rows'][t]['vol20d']) <= 0.5 + f / 2
        s = st['qqq']['stressed']
        sg = st['stressed_sign']
        assert s['vol20d'] == round(a * st['stressed_m'])
        assert 2.5 * 0.5 * a - 1 <= sg * s['last_night'] <= 4 * 0.5 * a + 1
        assert 2.5 * a * math.sqrt(5) - 1 <= sg * s['ret5d'] <= \
            4 * a * math.sqrt(5) + 1
        assert 1 * 0.5 * a / math.sqrt(5) - 1 <= sg * s['nights5_avg'] <= \
            2 * 0.5 * a / math.sqrt(5) + 1
        c = st['qqq']['calm']
        assert abs(c['last_night']) <= 0.5 * c['vol20d'] + 0.5


# ------------------------------------------------------- the experiments --

def test_counts_per_model_are_the_spec_table():
    c = Counter(it['experiment'] for it in D['items'])
    assert c == NOMINAL and sum(c.values()) == 377
    floor = Counter(it['experiment'] for it in D['items']
                    if it['state_id'] == 'S0' or it['block'] != 4)
    assert floor == FLOOR and sum(floor.values()) == 290
    assert D['plan']['calls_per_model'] == {'nominal': NOMINAL,
                                            'floor': FLOOR}
    assert len(ITEMS) == len(D['items'])
    for it in D['items']:
        assert it['item_id'] == \
            f'{it["experiment"]}|{it["state_id"]}|{it["arm"]}'
        want = 0 if it['experiment'] == 'E0' else (
            STATES[it['state_id']]['block'] if it['state_id'] != 'S0'
            else None)
        assert it['block'] == want if want is not None else \
            1 <= it['block'] <= 4
    assert [ITEMS[f'E9|S0|S0rep{r}']['block'] for r in range(1, 5)] == \
        [1, 2, 3, 4]
    assert len({ITEMS[f'E9|S0|B2B{i}']['block'] for i in (1, 2, 3)}) == 1


def test_qmap_maps_every_sent_id_back():
    for it in D['items']:
        q, qm = it['body']['questions'], it['qmap']
        labels = S0['names'] if it['state_id'] == 'S0' else \
            STATES[it['state_id']]['labels']
        canon = set(bi.qids_of(labels))
        assert set(qm) == set(q)
        assert set(qm.values()) <= canon and len(set(qm.values())) == len(qm)
        shown = it['meta'].get('anon', {})
        for sent, cid in qm.items():
            text = q[sent]['instructions']
            if cid == 'hedge_book':
                assert 'basket' in text
            else:
                t = cid[len('hold_'):]
                assert f'`{shown.get(t, t)}`' in text, (it['item_id'], sent)
        if it['meta'].get('factors', {}).get('opaque'):
            assert all(re.fullmatch(r'q[0-9a-f]{8}', k) for k in q)
        elif it['arm'] == 'Cb':
            assert all(qm[k] + '_b' == k for k in q)
        elif it['arm'] != 'AN':
            assert all(qm[k] == k for k in q)


def test_e1_retest_and_runtime_orders():
    for sid, st in STATES.items():
        core = bi._body(st['state'], st['labels'])
        ca, cb = ITEMS[f'E1|{sid}|Ca'], ITEMS[f'E1|{sid}|Cb']
        assert json.dumps(ca['body']) == json.dumps(core)
        # C^b: the same state and questions, in the same order, under other
        # ids -- other bytes, so a cache cannot answer the retest with C^a
        assert send(cb, 'm') != send(ca, 'm')
        assert cb['body']['state'] == ca['body']['state']
        assert [(cb['qmap'][k], v) for k, v in
                cb['body']['questions'].items()] == \
            list(ca['body']['questions'].items())
        assert list(cb['body']['questions']) == \
            [k + D['plan']['generator']['cb_id_suffix']
             for k in ca['body']['questions']]
        qids = bi.qids_of(st['labels'])
        orders = []
        for i in range(1, 7):
            it = ITEMS[f'E1|{sid}|O{i}']
            o = guard.question_order(f'battery-v1|{sid}|O{i}', qids)
            assert list(it['body']['questions']) == o
            assert ['hold_' + t for t in it['body']['state']['names']] == \
                [k for k in o if k != 'hedge_book']
            assert it['body']['state']['names'] == st['state']['names']
            orders.append(tuple(o))
        assert len(set(orders)) == 6


def test_e2_nuisance_draws():
    seen = Counter()
    for sid, st in STATES.items():
        it = ITEMS[f'E2|{sid}|V']
        f, s = it['meta']['factors'], it['body']['state']
        assert s['date'] == bi._bday('20261013', f['date_shift'])
        assert 0 <= f['date_shift'] <= 4
        assert np.is_busday(np.datetime64(
            f'{s["date"][:4]}-{s["date"][4:6]}-{s["date"][6:]}'))
        assert s['book'] == D['plan']['texts']['book_paraphrases'][
            f['paraphrase']]
        core = st['state']
        for t in st['labels']:
            for k in ('last_night', 'nights5_avg', 'ret5d'):
                assert abs(s['names'][t][k] - core['names'][t][k]) <= \
                    (3 if f['jitter'] else 0)
            assert abs(s['names'][t]['vol20d'] - core['names'][t]['vol20d'])\
                <= (0.01 * core['names'][t]['vol20d'] + 0.5 if f['jitter']
                    else 0)
            assert s['names'][t]['earnings_tonight'] == \
                core['names'][t]['earnings_tonight']
        if not f['keyshuffle']:
            assert list(s) == list(core)
        seen.update(k for k in ('opaque', 'keyshuffle', 'jitter') if f[k])
        assert list(it['qmap'].values()) != bi.qids_of(st['labels']) or \
            f['opaque']
    assert all(4 <= seen[k] <= 20 for k in ('opaque', 'keyshuffle',
                                            'jitter'))


def test_e3_context_items():
    for sid, st in STATES.items():
        it = ITEMS[f'E3|{sid}|CX']
        q = list(it['body']['questions'])
        assert it['body']['state'] == st['state']
        assert q == it['meta']['cx_subset']
        if st['cx_type'] == 1:
            assert len(q) == 6 and q[-1] == 'hedge_book'
            canon = bi.qids_of(st['labels'])
            assert q == [k for k in canon if k in q]
        elif st['cx_type'] == 2:
            assert len(q) == 1 and q[0].startswith('hold_')
        else:
            assert q == ['hedge_book']
    alone = of('E3')[24:]
    assert [it['arm'] for it in alone] == \
        ['alone_' + k for k in bi.qids_of(S0['names'])]
    for it in alone:
        assert list(it['body']['questions']) == [it['arm'][len('alone_'):]]
        assert json.dumps(it['body']['state']) == json.dumps(S0['state'])
    # S0's 21 single-call slots (12 singles, 6 O-draws, V, NG, the B2B
    # triple) are dealt round-robin over blocks 1-4 after a seeded shuffle
    slots = [it['block'] for it in D['items'] if it['state_id'] == 'S0'
             and it['experiment'] != 'E0' and 'rep' not in it['meta']
             and it['meta'].get('seq', 1) == 1]
    assert len(slots) == 21
    assert sorted(Counter(slots).values()) == [5, 5, 5, 6]


def test_e4_flips_only_the_treated_names_earnings_flag():
    for sid, st in STATES.items():
        it = ITEMS[f'E4|{sid}|EF']
        j = it['meta']['treated']
        assert st['conditions'][j] == 'Ordinary'
        names = it['body']['state']['names']
        assert names[j] == dict(st['rows'][j], earnings_tonight=True)
        assert {t: r for t, r in names.items() if t != j} == \
            {t: r for t, r in st['rows'].items() if t != j}
        assert list(it['body']['questions']) == bi.qids_of(st['labels'])
        assert it['body']['state']['QQQ'] == st['state']['QQQ']


def test_e5_dose_moves_are_coherent_and_balanced():
    for sid, st in STATES.items():
        it = ITEMS[f'E5|{sid}|XM']
        m = it['meta']
        j, k, sg = m['treated'], m['dose'], m['sign']
        assert st['conditions'][j] == 'Ordinary' and sg in (1, -1)
        old, new = st['rows'][j], it['body']['state']['names'][j]
        ln = sg * round(k * old['vol20d'])
        d = ln - old['last_night']
        assert m['D'] == d
        assert new == {'last_night': ln,
                       'nights5_avg': old['nights5_avg'] + round(d / 5),
                       'ret5d': old['ret5d'] + d,
                       'vol20d': round(math.sqrt(old['vol20d'] ** 2
                                                 + d ** 2 / 19)),
                       'earnings_tonight': False}
        assert {t: r for t, r in it['body']['state']['names'].items()
                if t != j} == {t: r for t, r in st['rows'].items() if t != j}
    for b, sids in by_block(STATES).items():
        for reg in ('stressed', 'calm'):
            assert sorted(ITEMS[f'E5|{s}|XM']['meta']['dose'] for s in sids
                          if STATES[s]['regime'] == reg) == [1.5, 3.0, 6.0]


def test_e6_swaps_only_the_qqq_row():
    for sid, st in STATES.items():
        it = ITEMS[f'E6|{sid}|QS']
        other = 'calm' if st['regime'] == 'stressed' else 'stressed'
        s = it['body']['state']
        assert s['QQQ'] == st['qqq'][other] != st['state']['QQQ']
        assert json.dumps(s['names']) == json.dumps(st['state']['names'])
        assert list(it['body']['questions']) == bi.qids_of(st['labels'])


def test_e7_derangements_have_no_fixed_points_and_rows_travel_whole():
    for sid, st in STATES.items():
        it = ITEMS[f'E7|{sid}|{st["e7_kind"]}']
        names = it['body']['state']['names']
        assert it['body']['state']['QQQ'] == st['state']['QQQ']
        if st['e7_kind'] == 'LD':
            p = it['meta']['perm']
            assert sorted(p) == sorted(p.values()) == sorted(st['labels'])
            assert all(p[t] != t for t in p)
            assert list(names) == st['labels']
            for t in st['labels']:
                assert names[t] == st['rows'][p[t]]   # incl. earnings flag
        else:
            an = it['meta']['anon']
            assert sorted(an.values()) == [f'S{i:02d}' for i in range(1, 12)]
            assert list(names) == [an[t] for t in st['labels']]
            for t in st['labels']:
                assert names[an[t]] == st['rows'][t]
            assert it['qmap'] == dict(
                {f'hold_{an[t]}': f'hold_{t}' for t in st['labels']},
                hedge_book='hedge_book')


def test_e8_negates_every_question_with_the_declared_texts():
    neg = D['plan']['texts']['negation']
    q1 = guard.QUESTIONS['q1']
    assert neg['hold'] == ('Should the long position in `{n}` be left out '
                           'tonight -- not bought at today\'s closing auction '
                           'and not sold at tomorrow\'s opening auction?')
    assert neg['hold_criteria'] == {'true': q1['hold'][1]['false'],
                                    'false': q1['hold'][1]['true']}
    assert neg['hedge_criteria'] == {'true': q1['hedge'][1]['false'],
                                     'false': q1['hedge'][1]['true']}
    assert 'left unhedged' in neg['hedge']
    ng = of('E8')
    assert [it['state_id'] for it in ng] == \
        [s for s in STATES if STATES[s]['e8']] + ['S0']
    for it in ng:
        labels = S0['names'] if it['state_id'] == 'S0' else \
            STATES[it['state_id']]['labels']
        state = S0['state'] if it['state_id'] == 'S0' else \
            STATES[it['state_id']]['state']
        assert json.dumps(it['body']['state']) == json.dumps(state)
        q = it['body']['questions']
        assert list(q) == bi.qids_of(labels)
        for k, v in q.items():
            if k == 'hedge_book':
                assert v == {'type': 'noul', 'instructions': neg['hedge'],
                             'criteria': neg['hedge_criteria']}
            else:
                assert v == {'type': 'noul',
                             'instructions': neg['hold'].format(n=k[5:]),
                             'criteria': neg['hold_criteria']}


def test_state_sha_is_the_one_the_record_logs(tmp_path):
    """E9's orders are keyed on the logged state_sha: ask guard.decide to
    log S0 (to a scratch file) and read the sha back."""
    path = str(tmp_path / 'log.csv')
    p = {n: 0.6 for n in S0['names']}
    p[guard.BOOK] = 0.4
    guard.decide('http', S0['names'], '20261013',
                 dt.datetime(2026, 10, 13, 9, 17, tzinfo=guard.ET),
                 S0['state'], path=path, ask=lambda s, n: (p, 'm', 0))
    logged = {r['state_sha'] for r in csv.DictReader(open(path))}
    assert logged == {bi.state_sha(S0['state'])} == {'c889c3bda805'}


def test_e9_bodies_are_the_runtime_k3_bodies():
    qids = bi.qids_of(S0['names'])
    sha = bi.state_sha(S0['state'])
    seen = set()
    for j in range(3):
        o = guard.question_order(f'20261013|{sha}|{j}|order-v1|http', qids)
        want = json.dumps(guard.request_body(S0['state'], S0['names'],
                                             bi.INCUMBENT, order=o)).encode()
        for c in 'ab':
            assert send(ITEMS[f'E9|S0|S0O{j}{c}'], bi.INCUMBENT) == want
        seen.add(want)
    assert len(seen) == 3
    v = ITEMS['E9|S0|S0V']
    assert v['meta']['factors']['date_shift'] in range(5)
    assert [it['arm'] for it in of('E9')] == \
        [f'S0rep{r}' for r in range(1, 5)] + [f'B2B{i}' for i in (1, 2, 3)] \
        + [f'S0O{j}{c}' for j in range(3) for c in 'ab'] + ['S0V']


def test_s0_is_untouched():
    lock = json.load(open(os.path.join(BENCH, 'lock_v1.json')))
    raw = open(os.path.join(BENCH, 's0_20261013.json'), 'rb').read()
    assert hashlib.sha256(raw).hexdigest() == lock['s0']
    for it in D['items']:
        if it['state_id'] == 'S0' and it['arm'] != 'S0V':
            assert it['body']['state'] == S0['state'], it['item_id']
    assert ITEMS['E9|S0|S0V']['body']['state'] != S0['state']


# ------------------------------------------------------- the plan --

def test_plan_candidates_x_and_spend():
    plan = D['plan']
    c = {x['id']: x for x in plan['candidates']}
    assert len(c) == 18
    assert not any(k.startswith('~') or ':free' in k for k in c)
    assert plan['incumbent'] == 'perplexity/pplx-decider-v1.1-27b' in c
    assert plan['reference'] == 'typesafe/jev-1.13' in c
    x = {'perplexity/pplx-decider-v1.1-27b': 1.000,
         'typesafe/jev-1.13': 0.799, 'inception/mercury-decide': 0.766,
         'cloudflare/clef': 0.594, 'jaredpalmer/kev-4b': 0.000}
    norm = lambda di=None, jb=None: np.mean(
        [v for v in ((di - 39.5) / (62.75 - 39.5) if di else None,
                     (jb - 33.9) / (75.7 - 33.9) if jb else None)
         if v is not None])
    exact = {'perplexity/pplx-decider-v1.1-27b': norm(62.75, 75.7),
             'typesafe/jev-1.13': norm(60.11, 63.6),
             'inception/mercury-decide': norm(jb=65.9),
             'cloudflare/clef': norm(53.08, 59.1),
             'jaredpalmer/kev-4b': 0.0}
    for k, v in c.items():
        if k in x:                  # the formula; the spec's table rounds
            assert abs(v['X'] - exact[k]) < 1e-6, k
            assert abs(v['X'] - x[k]) <= 0.001, k
        else:
            assert v['X'] is None, k
    assert c['respan/span-01-lite']['price_in'] == 0
    assert c['cloudflare/clef-omni']['price_in'] == 1.5e-7
    assert plan['spend'] == {'max_usd': 8, 'max_usd_ceiling': 12,
                             'credit_margin': 10,
                             'pessimistic_per_call': 0.0012,
                             'price_ref': 3.1e-8}
    assert plan['seeds'] == {'items': 20261013, 'analysis': 20261014,
                             'dispatch': 20261015}
    assert (plan['bootstrap_B'], plan['signflip_draws'], plan['deadline_min'],
            plan['timeout_s'], plan['concurrency'], plan['per_model']) == \
        (10000, 200000, 90, 60, 16, 3)
    assert plan['mode'] == 'select' and plan['version'] == 'v1'
    assert re.search(r'Q_rule = 0\.[56]\d{3}', plan['baselines_note'])


def test_plan_gates_carry_every_threshold():
    g = D['plan']['gates']
    assert g['G1'] == {'valid_rate_min': 0.98, 'completion_min': 0.95,
                       'modal_share_min': 0.90, 'p95_latency_s_max': 60.0,
                       'median_cost_usd_max': 0.01}
    assert g['G2a'] == {'ex_nuis_max': 0.05, 'rel_max': 0.5, 'D_min': 0.01}
    assert g['G2b'] == {'ex_ctx_max': 0.05, 'lb_alpha': 0.05}
    assert g['G3a'] == {'E_bar_max': -0.05, 'frac_neg_min': 0.75,
                        'holm_p_lt': 0.05}
    assert g['G3b'] == {'G_bar_min': 0.05, 'delta_h_gt': 0.0,
                        'holm_p_lt': 0.05}
    assert g['G4'] == {'flip_h_max': 0.10}
    assert g['G5'] == {'Q_minus_Q_rule_min': 0.05}
    assert g['G6'] == {'Phi_min': 0.5, 'Phi_ab_floor_min': 0.2}
    assert g['handling']['k3_if_sd_rep_gt'] == 0.005
    assert g['handling']['k3_if_ex_O_gt'] == 0.01
    assert g['handling']['phi_factor'] == 1.5
    assert g['E8_flag'] == {'CE_gt': 0.15, 'rho_neg_max': 0.0,
                            'permutations': 10000}
    assert g['selection']['switch_margin_min'] == 0.03
    assert g['analysis_set'] == {'block_final_share_min': 0.95,
                                 'blocks_min': 3}


def test_book_paraphrase_zero_is_guards_book_sentence():
    src = open(os.path.join(ROOT, 'src', 'guard.py')).read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == 'build_state')
    book = next(v.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
                for k, v in zip(n.keys, n.values)
                if isinstance(k, ast.Constant) and k.value == 'book')
    para = D['plan']['texts']['book_paraphrases']
    assert para[0] == book == S0['state']['book']
    assert len(para) == 3 == len(set(para))
    assert all('basis points' in p and 'overnight' in p for p in para)


# ------------------------------------------------------- outcome-free --

BANNED = ('guard_log', 'xsec_paper_daily', 'official_prints', 'paper_bench',
          'yfinance', 'GLOG', 'DAILY', 'PAPER')
GUARD_OK = {'request_body', 'question_order', 'rule', 'ask_rule', 'BOOK',
            'QUESTIONS', 'STATE_SCHEMA'}


def test_battery_items_is_outcome_free():
    """Nothing in the generator can reach an outcome: no banned name in any
    identifier, attribute, import or string, and only guard's request and
    rule functions are touched."""
    tree = ast.parse(open(os.path.join(ROOT, 'src',
                                       'battery_items.py')).read())
    words = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            words.append(n.id)
        elif isinstance(n, ast.Attribute):
            words.append(n.attr)
        elif isinstance(n, ast.alias):
            words += [n.name, n.asname or '']
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            words.append(n.value)
        elif isinstance(n, (ast.FunctionDef, ast.arg)):
            words.append(getattr(n, 'name', None) or n.arg)
    for w in words:
        for b in BANNED:
            assert b not in w, (b, w)
    imported = {a.name for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names}
    assert imported <= {'os', 'sys', 'json', 'math', 'copy', 'hashlib',
                        'numpy', 'guard'}
    touched = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
               and isinstance(n.value, ast.Name) and n.value.id == 'guard'}
    assert touched <= GUARD_OK, touched - GUARD_OK
