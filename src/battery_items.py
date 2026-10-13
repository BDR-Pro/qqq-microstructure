# Part of qqq-microstructure.
#
# The decider battery's stimuli (RESULTS 27; spec v1, sections 1-3 and 7):
# every request body the battery will send, generated once by make_items()
# and committed as bytes, with a lock of their sha256, before any live call.
# The committed bytes are the declaration: every model is scored on stimuli
# fixed before a single answer was seen, and anyone can regenerate them from
# the seeds and check the lock.
#
# What goes in: only the committed real state S0 (built from bars strictly
# before 2026-10-13) and the anchor vols from the same bars. Nothing here
# reads a log, an outcome, a price after a state's date or the network; a
# test scans this file's syntax tree and the guard functions it touches.
#
# An item is a body WITHOUT 'model' (the run inserts each pin), built by
# guard.request_body, so the bytes tested are the bytes the record sends:
# for S0's canonical item, body + pin is byte-identical to the request every
# logged night was asked with.
#
#   python src/battery_items.py make bench/battery
#   python src/battery_items.py verify bench/battery/plan_v1.json

import os, sys, json, math, copy, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard

VERSION = 'v1'
DATE = '20261013'
SEEDS = {'items': 20261013, 'analysis': 20261014, 'dispatch': 20261015}
BLOCKS, PER_BLOCK = 4, 6
INCUMBENT = 'perplexity/pplx-decider-v1.1-27b'
REFERENCE = 'typesafe/jev-1.13'
S0_FILE, ANCHORS_FILE = 's0_20261013.json', 'anchors_20261013.json'
EXPERIMENTS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']

# (id, $ per input token): the paid, pinned decision models on the
# catalogue (spec section 1). Moving aliases (~...) and :free routes are
# never candidates: a record must name a dated build.
CANDIDATES = [
    ('perplexity/pplx-decider-v1.1-27b', 2e-8),
    ('typesafe/jev-1.13', 4.2e-8),
    ('inception/mercury-decide', 2e-8),
    ('cloudflare/clef', 4.2e-8),
    ('jaredpalmer/kev-4b', 4.2e-8),
    ('openai/gpt-6-luna-decisions', 1e-7),
    ('cloudflare/clef-omni', 1.5e-7),
    ('cloudflare/clef-flash', 2.1e-8),
    ('microsoft/microsoft-decision-1', 4.2e-8),
    ('upstage/solar-decide', 5e-8),
    ('upstage/solar-decide-flash', 5e-8),
    ('nace-ai/drex-v2.0', 6e-8),
    ('nace-ai/drex-v1.5', 4e-8),
    ('liquid/d1', 4e-8),
    ('scaledown/classify-1', 4e-8),
    ('togethercomputer/tev1-4b-experimental', 4.2e-8),
    ('respan/span-01', 2e-8),
    ('respan/span-01-lite', 0.0),
]
# The two independent boards, where a model is scored: Decision Index (DI)
# and JevBench intelligence (JB). X, the late tie-break, is the mean of the
# available scores normalised so the best board model is 1 and the worst 0;
# an unscored model has X = null and ranks below every scored one.
BOARDS = {
    'perplexity/pplx-decider-v1.1-27b': {'DI': 62.75, 'JB': 75.7},
    'typesafe/jev-1.13': {'DI': 60.11, 'JB': 63.6},
    'inception/mercury-decide': {'JB': 65.9},
    'cloudflare/clef': {'DI': 53.08, 'JB': 59.1},
    'jaredpalmer/kev-4b': {'DI': 39.5},
}
BOARD_RANGE = {'DI': (39.5, 62.75), 'JB': (33.9, 75.7)}

# The 17 names ever in a q5 basket (q1-only names are never used).
POOL = ['LRCX', 'NBIS', 'AMAT', 'MU', 'CAT', 'ASML', 'AMD', 'TSM', 'SOFI',
        'GEV', 'WDC', 'BE', 'CRWV', 'INTC', 'STX', 'SNDK', 'IREN']
CONDITIONS = ['Ordinary', 'Earnings', 'ExtremeDown', 'ExtremeUp', 'Moderate']
SKIP_CRITERIA = ('Earnings', 'ExtremeDown', 'ExtremeUp')

GENERATOR = {
    'pool': POOL,
    'n_labels': 11,
    'condition_p': {'Ordinary': 0.50, 'Earnings': 0.15, 'ExtremeDown': 0.10,
                    'ExtremeUp': 0.10, 'Moderate': 0.15},
    'min_ordinary': 6, 'min_skip_criteria': 2,
    'vol_log_sd': 0.25,                 # vol = round(anchor exp(N(0, sd^2)))
    'on_frac': 0.5,                     # sigma_on = on_frac * vol
    'z_base': [-1.0, 1.0],
    'z_extreme': [3.5, 5.0],
    'z_moderate': [1.5, 2.5],
    'p_move_on_night': 0.5,             # else the move goes on ret5d
    'night_vol_div': 19, 'ret5d_vol_div': 95, 'nights5_div': 5,
    'qqq_calm_log_sd': 0.15,
    'qqq_stressed_m': [2.0, 3.0],
    'qqq_stressed_last_night': [2.5, 4.0],
    'qqq_stressed_ret5d': [2.5, 4.0],
    'qqq_stressed_nights5': [1.0, 2.0],
    'stressed_name_vol': [1.3, 1.8],
    'stressed_per_block': 3,
    'cx_type_by_position': [1, 1, 2, 2, 3, 3],
    'cx1_holds': 5,
    'e5_doses': [1.5, 3.0, 6.0],
    'e7_ld_per_block': 4, 'e7_an_per_block': 2,
    'e8_per_block': 3,
    'v_p_opaque': 0.5, 'v_p_keyshuffle': 0.5, 'v_p_jitter': 0.5,
    'v_jitter_bps': 3, 'v_jitter_vol': 0.01,
    'v_date_shift_bdays': [0, 4],
    'rounding': 'Python round(), half to even',
    'rng_state': 'default_rng(SeedSequence(seeds.items, spawn_key=(s,))), '
                 's = 1..24 for S01..S24, s = 0 for S0',
    'rng_block': 'default_rng(SeedSequence(seeds.items, spawn_key=(100 + b,)))'
                 ', b = 1..4',
    'e1_order_seed': 'battery-v1|{state_id}|O{i}',
    # C^b is C^a with every question id suffixed: the same question to the
    # model (ids only key the answers), other bytes for any cache, so the
    # retest is a fresh draw and phi is identified even behind a cache
    'cb_id_suffix': '_b',
    'e9_order_seed': '{date}|{state_sha}|{j}|order-v1|http',
    'state_sha': 'sha256(json.dumps(state, sort_keys=True))[:12], as logged',
    'numpy': '.'.join(np.__version__.split('.')[:2]),
}

# The book sentence guard.build_state writes (index 0, byte for byte), and
# two faithful paraphrases for E2's nuisance draws.
BOOK_PARAPHRASES = [
    'long these names overnight: buy at the close auction, sell at the next '
    'open auction; moves in basis points',
    'an overnight long position in each of these names: bought at the '
    'closing auction, sold at the next opening auction; all moves are in '
    'basis points',
    'these names are held long overnight -- bought in the close auction and '
    'sold in the next open auction; moves are given in basis points',
]
# E8: every question asked the other way round, criteria swapped, so a
# coherent model answers p_NG = 1 - p.
NEGATION = {
    'hold': ('Should the long position in `{n}` be left out tonight -- not '
             'bought at today\'s closing auction and not sold at '
             'tomorrow\'s opening auction?'),
    'hold_criteria': {'true': 'skip: earnings tonight or an extreme recent '
                              'move',
                      'false': 'hold: an ordinary night for this name'},
    'hedge': ('Should tonight\'s whole basket be left unhedged -- with no '
              'equal-dollar short position in QQQ held from today\'s close '
              'to tomorrow\'s open?'),
    'hedge_criteria': {'true': 'do not hedge: an ordinary night',
                       'false': 'hedge: elevated risk of a market-wide '
                                'overnight gap'},
}

# Every threshold the scorer and select() apply (spec sections 2-4). Key
# suffixes: _min means >=, _max means <=, _gt means >, _lt means <.
GATES = {
    'G1': {'valid_rate_min': 0.98, 'completion_min': 0.95,
           'modal_share_min': 0.90, 'p95_latency_s_max': 60.0,
           'median_cost_usd_max': 0.01},
    'G2a': {'ex_nuis_max': 0.05, 'rel_max': 0.5, 'D_min': 0.01},
    # gated on the one-sided (1 - lb_alpha) bootstrap lower bound of ex_ctx
    # (states resampled within CX type, phi's error included), so a model
    # with no context effect is not failed by its noise alone
    'G2b': {'ex_ctx_max': 0.05, 'lb_alpha': 0.05},
    'G3a': {'E_bar_max': -0.05, 'frac_neg_min': 0.75, 'holm_p_lt': 0.05},
    'G3b': {'G_bar_min': 0.05, 'delta_h_gt': 0.0, 'holm_p_lt': 0.05},
    'G4': {'flip_h_max': 0.10},
    'G5': {'Q_minus_Q_rule_min': 0.05},
    'G6': {'Phi_min': 0.5, 'Phi_ab_floor_min': 0.2},
    'smoke_stop': {'invalid_first_calls': 3, 'never_for': ['incumbent',
                                                           'reference']},
    'handling': {'k3_if_sd_rep_gt': 0.005, 'k3_if_ex_O_gt': 0.01,
                 'phi_factor': 1.5},
    'holm': {'fwer': 0.05, 'family': ['G3a', 'G3b'],
             'over': 'models that pass G1',
             'signflip_exact_max_states': 20},
    'E8_flag': {'CE_gt': 0.15, 'rho_neg_max': 0.0, 'permutations': 10000},
    # report-only, never a gate: back-to-back identical bytes varying far less
    # than the same bytes spread over the run means a cache
    'caching': {'rho_b2b_lt': 0.25, 'report_only': True},
    'analysis_set': {'block_final_share_min': 0.95, 'blocks_min': 3},
    'selection': {'switch_margin_min': 0.03, 'lower_bound_gt': 0.0,
                  'alpha': 0.05, 'alpha_divisor': 'eligible - 1',
                  'bootstrap': 'paired, states resampled within regime',
                  'tie_break': ['AUC_comb desc',
                                'mean(lift_E, lift_M6, delta_h) desc',
                                'flip_h asc', 'X desc', 'median cost asc',
                                'id asc']},
    'iteration': {'rounds_max': 2, 'variants_max': 4,
                  'delta_Q_min': 0.02, 'lower_bound_gt': 0.0,
                  'alpha': 0.05, 'alpha_divisor': 'variants',
                  'AUC_comb_drop_max': 0.02, 'Phi_drop_max': 0.05,
                  'k5_if_sd_rep_gt': 0.005, 'sep_if_ex_ctx_gt': 0.02,
                  'items_seeds': [20261021, 20261022],
                  'holdout_seed': 20261029,
                  'round_cap_usd': 3, 'holdout_cap_usd': 2},
}

# Calls per model, per experiment (spec section 2's table). The floor drops
# the synthetic states of block 4 only; S0's items are always sent.
CALLS = {
    'nominal': {'E0': 2, 'E1': 192, 'E2': 24, 'E3': 36, 'E4': 24, 'E5': 24,
                'E6': 24, 'E7': 24, 'E8': 13, 'E9': 14},
    'floor': {'E0': 2, 'E1': 144, 'E2': 18, 'E3': 30, 'E4': 18, 'E5': 18,
              'E6': 18, 'E7': 18, 'E8': 10, 'E9': 14},
}
SPEND = {'max_usd': 8, 'max_usd_ceiling': 12, 'credit_margin': 10,
         'pessimistic_per_call': 0.0012, 'price_ref': 3.1e-8}


# ---------------------------------------------------------------- helpers --

def rnd(x):
    return int(round(float(x)))


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def body_sha(body):
    return sha256_bytes(json.dumps(body).encode())


def state_sha(state):
    """The state_sha the record logs for a night (guard.decide), which the
    runtime's k3 orders are keyed on."""
    return hashlib.sha256(json.dumps(state, sort_keys=True)
                          .encode()).hexdigest()[:12]


def x_score(cid):
    b = BOARDS.get(cid)
    if not b:
        return None
    v = [(b[k] - lo) / (hi - lo) for k, (lo, hi) in BOARD_RANGE.items()
         if k in b]
    return round(float(np.mean(v)), 6)


def _rng(key):
    return np.random.default_rng(np.random.SeedSequence(SEEDS['items'],
                                                        spawn_key=(key,)))


def _body(state, names, **kw):
    """guard.request_body without 'model': the run inserts each pin in the
    same place, so state, questions[, provider] keep the record's order."""
    b = guard.request_body(state, names, None, **kw)
    del b['model']
    return b


def _item(exp, sid, arm, block, body, qmap=None, meta=None):
    return {'item_id': f'{exp}|{sid}|{arm}', 'experiment': exp, 'arm': arm,
            'state_id': sid, 'block': block, 'body': body,
            'body_sha256': body_sha(body),
            'qmap': qmap if qmap is not None else
            {k: k for k in body['questions']},
            'meta': meta or {}}


def qids_of(labels):
    return [f'hold_{t}' for t in labels] + ['hedge_book']


def canonical(rows, labels, qqq, date=DATE, book=BOOK_PARAPHRASES[0]):
    """The schema-2 state, in guard.build_state's key order."""
    return {'schema': guard.STATE_SCHEMA, 'date': date, 'book': book,
            'names': {t: rows[t] for t in labels}, 'QQQ': qqq}


# ----------------------------------------------------- coherent updates --

def shock_night(row, new):
    """last_night becomes `new`; the 5-night mean, the 5-day return and the
    20-day vol move with it, so the row stays a possible history (a single
    night of D adds D/5 to the mean, D to the 5-day sum, ~D^2/19 to the
    20-day variance)."""
    r = dict(row)
    d = new - r['last_night']
    r['last_night'] = new
    r['nights5_avg'] += rnd(d / GENERATOR['nights5_div'])
    r['ret5d'] += d
    r['vol20d'] = rnd(math.sqrt(row['vol20d'] ** 2
                                + d ** 2 / GENERATOR['night_vol_div']))
    return r, d


def shock_ret5d(row, new):
    """ret5d becomes `new`, spread over 5 days (~D^2/95 of 20-day
    variance)."""
    r = dict(row)
    d = new - r['ret5d']
    r['ret5d'] = new
    r['vol20d'] = rnd(math.sqrt(row['vol20d'] ** 2
                                + d ** 2 / GENERATOR['ret5d_vol_div']))
    return r, d


# -------------------------------------------------------------- generator --

def block_plan(b):
    """Block b's assignments, from the block's own RNG: the 3/3 regime
    permutation (E6), E5's doses (each once Stressed, once Calm), E7's two
    anonymised states (one per regime, so the AN arm sees both hedge
    classes) and E8's three negated states."""
    g = GENERATOR
    rng = _rng(100 + b)
    k = g['stressed_per_block']
    regime = [str(x) for x in rng.permutation(['stressed'] * k
                                              + ['calm'] * (PER_BLOCK - k))]
    S = [i for i in range(PER_BLOCK) if regime[i] == 'stressed']
    C = [i for i in range(PER_BLOCK) if regime[i] == 'calm']
    dose = [None] * PER_BLOCK
    for grp in (S, C):
        for i, j in zip(grp, rng.permutation(len(g['e5_doses']))):
            dose[i] = float(g['e5_doses'][int(j)])
    an = {int(rng.choice(S)), int(rng.choice(C))}
    e8 = {int(i) for i in rng.choice(PER_BLOCK, g['e8_per_block'],
                                     replace=False)}
    return {'regime': regime, 'dose': dose,
            'e7_kind': ['AN' if i in an else 'LD' for i in range(PER_BLOCK)],
            'e8': [i in e8 for i in range(PER_BLOCK)],
            'cx_type': list(g['cx_type_by_position'])}


def gen_state(s, regime, anchors):
    """State s (1..24): labels, conditions, name rows and both QQQ rows, from
    the state's own RNG. Every name draws the same numbers whatever its
    condition, and the stress vol factor is drawn whatever the regime, so a
    name's z stays the same across regimes. Returns (record, trace, rng):
    trace keeps each moved name's pre-move row (tests check the update);
    rng goes on to draw the state's items."""
    g = GENERATOR
    rng = _rng(s)
    labels = [str(t) for t in rng.choice(POOL, g['n_labels'], replace=False)]
    p = [g['condition_p'][c] for c in CONDITIONS]
    while True:
        cond = [CONDITIONS[int(i)] for i in rng.choice(len(CONDITIONS),
                                                       len(labels), p=p)]
        if cond.count('Ordinary') >= g['min_ordinary'] and \
                sum(c in SKIP_CRITERIA for c in cond) >= \
                g['min_skip_criteria']:
            break
    f = float(rng.uniform(*g['stressed_name_vol']))
    scale = f if regime == 'stressed' else 1.0
    r5 = math.sqrt(5)
    rows, trace = {}, {}
    for t, c in zip(labels, cond):
        eps = rng.normal(0, g['vol_log_sd'])
        zl, za, z5 = rng.uniform(*g['z_base'], 3)
        u_on, u_z, u_sign = rng.random(3)
        vol = rnd(anchors[t] * math.exp(eps) * scale)
        son = g['on_frac'] * vol
        row = {'last_night': rnd(zl * son), 'nights5_avg': rnd(za * son / r5),
               'ret5d': rnd(z5 * vol * r5), 'vol20d': vol,
               'earnings_tonight': c == 'Earnings'}
        if c in ('ExtremeDown', 'ExtremeUp', 'Moderate'):
            lo, hi = g['z_moderate'] if c == 'Moderate' else g['z_extreme']
            z = lo + (hi - lo) * float(u_z)
            sign = {'ExtremeDown': -1, 'ExtremeUp': 1}.get(
                c, 1 if u_sign < 0.5 else -1)
            on = bool(u_on < g['p_move_on_night'])
            new = sign * rnd(z * son) if on else sign * rnd(z * vol * r5)
            moved, d = (shock_night if on else shock_ret5d)(row, new)
            trace[t] = {'before': row, 'field': 'last_night' if on
                        else 'ret5d', 'z': z, 'sign': sign, 'new': new,
                        'D': d}
            row = moved
        rows[t] = row
    a = anchors['QQQ']
    eps = rng.normal(0, g['qqq_calm_log_sd'])
    zl, za, z5 = rng.uniform(*g['z_base'], 3)
    vol = rnd(a * math.exp(eps))
    son = g['on_frac'] * vol
    calm = {'last_night': rnd(zl * son), 'nights5_avg': rnd(za * son / r5),
            'ret5d': rnd(z5 * vol * r5), 'vol20d': vol}
    m = float(rng.uniform(*g['qqq_stressed_m']))
    sg = 1 if rng.random() < 0.5 else -1
    uL = rng.uniform(*g['qqq_stressed_last_night'])
    uR = rng.uniform(*g['qqq_stressed_ret5d'])
    uN = rng.uniform(*g['qqq_stressed_nights5'])
    stressed = {'last_night': sg * rnd(uL * g['on_frac'] * a),
                'nights5_avg': sg * rnd(uN * g['on_frac'] * a / r5),
                'ret5d': sg * rnd(uR * a * r5), 'vol20d': rnd(a * m)}
    qqq = {'calm': calm, 'stressed': stressed}
    rec = {'regime': regime, 'labels': labels,
           'conditions': dict(zip(labels, cond)), 'rows': rows, 'qqq': qqq,
           'stressed_m': m, 'stressed_sign': sg, 'name_vol_factor': f}
    return rec, trace, rng


# ---------------------------------------------------------------- the items --

def _bday(date, k):
    d = np.busday_offset(np.datetime64(f'{date[:4]}-{date[4:6]}-{date[6:]}'),
                         k, roll='forward')
    return str(d).replace('-', '')


def _shuffle_keys(d, rng):
    ks = list(d)
    return {ks[int(i)]: d[ks[int(i)]] for i in rng.permutation(len(ks))}


def v_draw(rng, state, labels):
    """One nuisance draw (E2, and S0's V): always a shuffled question order,
    a date shift of 0-4 business days and a book paraphrase; each of opaque
    ids, shuffled keys and numeric jitter on with probability 1/2."""
    g = GENERATOR
    u = rng.random(3)
    f = {'opaque': bool(u[0] < g['v_p_opaque']),
         'keyshuffle': bool(u[1] < g['v_p_keyshuffle']),
         'jitter': bool(u[2] < g['v_p_jitter']),
         'date_shift': int(rng.integers(g['v_date_shift_bdays'][0],
                                        g['v_date_shift_bdays'][1] + 1)),
         'paraphrase': int(rng.integers(0, len(BOOK_PARAPHRASES)))}
    qids = qids_of(labels)
    order = [qids[int(i)] for i in rng.permutation(len(qids))]
    st = copy.deepcopy(state)
    st['date'] = _bday(state['date'], f['date_shift'])
    st['book'] = BOOK_PARAPHRASES[f['paraphrase']]
    if f['jitter']:
        j = g['v_jitter_bps']
        for row in list(st['names'].values()) + [st['QQQ']]:
            for k in ('last_night', 'nights5_avg', 'ret5d'):
                row[k] += int(rng.integers(-j, j + 1))
            row['vol20d'] = rnd(row['vol20d'] * (1 + rng.uniform(
                -g['v_jitter_vol'], g['v_jitter_vol'])))
    if f['keyshuffle']:
        st = _shuffle_keys(st, rng)
        st['names'] = {t: _shuffle_keys(r, rng)
                       for t, r in st['names'].items()}
        st['QQQ'] = _shuffle_keys(st['QQQ'], rng)
    ids = None
    if f['opaque']:
        while True:
            sent = [f'q{int(x):08x}' for x in rng.integers(0, 2 ** 32,
                                                          len(qids))]
            if len(set(sent)) == len(sent) and not set(sent) & set(qids):
                break
        ids = dict(zip(qids, sent))
    body = _body(st, labels, order=order, ids=ids)
    qmap = {(ids or {}).get(k, k): k for k in order}
    return body, qmap, {'factors': f}


def negate(body):
    """E8: every question of a canonical body asked the other way round."""
    for k, q in body['questions'].items():
        if k == 'hedge_book':
            q['instructions'] = NEGATION['hedge']
            q['criteria'] = dict(NEGATION['hedge_criteria'])
        else:
            q['instructions'] = NEGATION['hold'].format(n=k[len('hold_'):])
            q['criteria'] = dict(NEGATION['hold_criteria'])
    return body


def _cx(state, labels, ask):
    """A body that asks only the canonical ids in `ask`, about the full
    state: request_body always appends hedge_book, so drop it if unasked."""
    names = [k[len('hold_'):] for k in ask if k != 'hedge_book']
    b = _body(state, names)
    if 'hedge_book' not in ask:
        del b['questions']['hedge_book']
    return b


def state_items(sid, rec, rng, b):
    """E1-E8 items of one synthetic state, drawn from its RNG in a fixed
    order: E2's V, E3's subset, E4's and E5's treated names, E5's sign,
    E7's derangement or anonymisation."""
    labels, rows, core = rec['labels'], rec['rows'], rec['state']
    reg = rec['regime']
    qids = qids_of(labels)
    ordinary = [t for t in labels if rec['conditions'][t] == 'Ordinary']
    out = {e: [] for e in EXPERIMENTS}
    out['E1'].append(_item('E1', sid, 'Ca', b, _body(core, labels)))
    ids = {q: q + GENERATOR['cb_id_suffix'] for q in qids}
    out['E1'].append(_item('E1', sid, 'Cb', b, _body(core, labels, ids=ids),
                           {x: q for q, x in ids.items()},
                           {'ids': 'canonical + '
                            + GENERATOR['cb_id_suffix']}))
    for i in range(1, 7):
        seed = f'battery-v1|{sid}|O{i}'
        order = guard.question_order(seed, qids)
        out['E1'].append(_item('E1', sid, f'O{i}', b,
                               _body(core, labels, order=order),
                               meta={'order_seed': seed}))
    body, qmap, meta = v_draw(rng, core, labels)
    out['E2'].append(_item('E2', sid, 'V', b, body, qmap, meta))
    cx = rec['cx_type']
    if cx == 1:
        pick = sorted(int(i) for i in rng.choice(
            len(labels), GENERATOR['cx1_holds'], replace=False))
        ask = [qids[i] for i in pick] + ['hedge_book']
    elif cx == 2:
        ask = [qids[int(rng.integers(len(labels)))]]
    else:
        ask = ['hedge_book']
    out['E3'].append(_item('E3', sid, 'CX', b, _cx(core, labels, ask),
                           meta={'cx_type': cx, 'cx_subset': ask}))
    j = ordinary[int(rng.integers(len(ordinary)))]
    st = copy.deepcopy(core)
    st['names'][j]['earnings_tonight'] = True
    out['E4'].append(_item('E4', sid, 'EF', b, _body(st, labels),
                           meta={'treated': j}))
    j = ordinary[int(rng.integers(len(ordinary)))]
    sign = 1 if rng.random() < 0.5 else -1
    k = rec['dose']
    st = copy.deepcopy(core)
    st['names'][j], d = shock_night(rows[j], sign * rnd(k * rows[j]['vol20d']))
    out['E5'].append(_item('E5', sid, 'XM', b, _body(st, labels),
                           meta={'treated': j, 'dose': k, 'sign': sign,
                                 'D': d}))
    other = 'calm' if reg == 'stressed' else 'stressed'
    st = canonical(rows, labels, rec['qqq'][other])
    out['E6'].append(_item('E6', sid, 'QS', b, _body(st, labels),
                           meta={'core_regime': reg, 'swapped_in': other,
                                 'stressed_m': rec['stressed_m'],
                                 'stressed_sign': rec['stressed_sign']}))
    n = len(labels)
    if rec['e7_kind'] == 'LD':
        while True:                         # a uniform derangement
            p = [int(i) for i in rng.permutation(n)]
            if all(p[i] != i for i in range(n)):
                break
        perm = {labels[i]: labels[p[i]] for i in range(n)}
        st = canonical({t: rows[perm[t]] for t in labels}, labels,
                       rec['qqq'][reg])
        out['E7'].append(_item('E7', sid, 'LD', b, _body(st, labels),
                               meta={'perm': perm}))
    else:
        p = [int(i) for i in rng.permutation(n)]
        anon = {labels[i]: f'S{p[i] + 1:02d}' for i in range(n)}
        names = [anon[t] for t in labels]
        st = canonical({anon[t]: rows[t] for t in labels}, names,
                       rec['qqq'][reg])
        qmap = {f'hold_{anon[t]}': f'hold_{t}' for t in labels}
        qmap['hedge_book'] = 'hedge_book'
        out['E7'].append(_item('E7', sid, 'AN', b, _body(st, names), qmap,
                               meta={'anon': anon}))
    if rec['e8']:
        out['E8'].append(_item('E8', sid, 'NG', b,
                               negate(_body(core, labels)),
                               meta={'negated': True}))
    return out


def s0_items(s0):
    """Every item on the real state S0 (E0, E3's singles, E8, E9). S0 is
    diagnostic only and carries no labels. Its V draw and the blocks its
    single items are dispatched in come from S0's RNG (key 0); the four
    canonical repeats go one per block."""
    names, state = s0['names'], s0['state']
    qids = qids_of(names)
    rng = _rng(0)
    canon = _body(state, names)
    out = {e: [] for e in EXPERIMENTS}
    out['E0'].append(_item('E0', 'S0', 'W', 0, canon,
                           meta={'role': 'warm-up, cost pilot'}))
    out['E0'].append(_item('E0', 'S0', 'PP', 0,
                           _body(state, names, provider='__FROM_W__'),
                           meta={'provider': 'filled from W at run time'}))
    v_body, v_qmap, v_meta = v_draw(rng, state, names)
    # one dispatch slot each: 12 singles, 6 O-draws, V, NG and the B2B triple
    slots = [('E3', f'alone_{q}') for q in qids] + \
        [('E9', f'S0O{j}{c}') for j in range(3) for c in 'ab'] + \
        [('E9', 'S0V'), ('E8', 'NG'), ('E9', 'B2B')]
    blk = {}
    for i, k in enumerate(rng.permutation(len(slots))):
        blk[slots[int(k)]] = i % BLOCKS + 1
    for q in qids:
        out['E3'].append(_item('E3', 'S0', f'alone_{q}',
                               blk[('E3', f'alone_{q}')],
                               _cx(state, names, [q]),
                               meta={'cx_type': 'alone', 'cx_subset': [q]}))
    out['E8'].append(_item('E8', 'S0', 'NG', blk[('E8', 'NG')],
                           negate(_body(state, names)),
                           meta={'negated': True}))
    for r in range(1, BLOCKS + 1):
        out['E9'].append(_item('E9', 'S0', f'S0rep{r}', r, canon,
                               meta={'rep': r}))
    for i in range(1, 4):
        out['E9'].append(_item('E9', 'S0', f'B2B{i}', blk[('E9', 'B2B')],
                               canon, meta={'group': 'B2B', 'seq': i}))
    sha = state_sha(state)
    for j in range(3):
        seed = f'{s0["date"]}|{sha}|{j}|order-v1|http'
        for c in 'ab':
            out['E9'].append(_item(
                'E9', 'S0', f'S0O{j}{c}', blk[('E9', f'S0O{j}{c}')],
                _body(state, names, order=guard.question_order(seed, qids)),
                meta={'j': j, 'copy': c, 'order_seed': seed}))
    out['E9'].append(_item('E9', 'S0', 'S0V', blk[('E9', 'S0V')], v_body,
                           v_qmap, v_meta))
    return out


def build(s0, anchors):
    """(states, items) from the two committed inputs; pure and seeded."""
    assert sorted(POOL) == sorted(k for k in anchors['vol20d'] if k != 'QQQ')
    assert s0['state']['book'] == BOOK_PARAPHRASES[0]
    vols = anchors['vol20d']
    states, per = {}, {e: [] for e in EXPERIMENTS}
    syn = []
    for b in range(1, BLOCKS + 1):
        bp = block_plan(b)
        for pos in range(PER_BLOCK):
            s = (b - 1) * PER_BLOCK + pos + 1
            sid = f'S{s:02d}'
            rec, _, rng = gen_state(s, bp['regime'][pos], vols)
            core = canonical(rec['rows'], rec['labels'],
                             rec['qqq'][rec['regime']])
            states[sid] = {
                'block': b, 'regime': rec['regime'], 'labels': rec['labels'],
                'conditions': rec['conditions'], 'rows': rec['rows'],
                'qqq': rec['qqq'], 'stressed_m': rec['stressed_m'],
                'stressed_sign': rec['stressed_sign'],
                'cx_type': bp['cx_type'][pos], 'e7_kind': bp['e7_kind'][pos],
                'e8': bp['e8'][pos], 'state': core}
            rec.update(state=core, cx_type=bp['cx_type'][pos],
                       e7_kind=bp['e7_kind'][pos], e8=bp['e8'][pos],
                       dose=bp['dose'][pos])
            syn.append(state_items(sid, rec, rng, b))
    s0i = s0_items(s0)
    for e in EXPERIMENTS:
        for o in syn:
            per[e] += o[e]
        per[e] += s0i[e]
    items = [it for e in EXPERIMENTS for it in per[e]]
    return states, items


# ------------------------------------------------------------- the plan --

def rule_q(states):
    """Q of the earnings-rule control on the core states: it skips exactly
    the Earnings names and never hedges, so Q = 1/4 [recall + 1 + 0 + 1].
    battery_score recomputes this from the locked items; this note is for
    transparency."""
    hit = crit = fs = nord = 0
    for st in states.values():
        p, _, _ = guard.ask_rule(st['state'], st['labels'])
        d = guard.rule(p, st['labels'])
        for t in st['labels']:
            c = st['conditions'][t]
            if c in SKIP_CRITERIA:
                crit += 1
                hit += d[t] == 'skip'
            elif c == 'Ordinary':
                nord += 1
                fs += d[t] == 'skip'
    q = 0.25 * (hit / crit + (1 - fs / nord) + 0 + 1)
    return q, hit, crit, fs, nord


def make_plan(states):
    q, hit, crit, fs, nord = rule_q(states)
    note = (f'Offline baselines on the {len(states)} core states (C^a = C^b '
            f'for a deterministic baseline): Q_rule = {q:.4f} (the earnings '
            f'rule skips {hit} of {crit} skip-criteria names and {fs} of '
            f'{nord} Ordinary names, never hedges); constant 0.5 = 0.50; '
            f'skip-nothing / always-hedge = 0.50; criteria oracle = 1.00. '
            f'battery_score.score recomputes them from the locked items on '
            f'the analysis set, and that value is the one G5 uses.')
    return {
        'version': VERSION,
        'mode': 'select',
        'incumbent': INCUMBENT,
        'reference': REFERENCE,
        'candidates': [{'id': c, 'price_in': p, 'X': x_score(c)}
                       for c, p in CANDIDATES],
        'x_boards': {'scores': BOARDS,
                     'normalise': {k: list(v) for k, v in BOARD_RANGE.items()}},
        'seeds': dict(SEEDS),
        'date': DATE,
        'blocks': BLOCKS, 'states_per_block': PER_BLOCK,
        'generator': GENERATOR,
        'texts': {'questions_version': 'q1',
                  'book_paraphrases': BOOK_PARAPHRASES,
                  'negation': NEGATION},
        'gates': GATES,
        'gates_note': 'key suffixes: _min >=, _max <=, _gt >, _lt <',
        'calls_per_model': CALLS,
        'calls_note': 'floor = block 4\'s synthetic states dropped; S0 items '
                      '(state_id S0) are always sent. B2B1-3 (meta.group '
                      'B2B) go back to back on one dispatch slot.',
        'bootstrap_B': 10000, 'signflip_draws': 200000,
        'spend': SPEND,
        'deadline_min': 90, 'timeout_s': 60, 'concurrency': 16,
        'per_model': 3,
        'files': {'states': f'states_{VERSION}.json',
                  'items': f'items_{VERSION}.jsonl',
                  's0': S0_FILE, 'anchors': ANCHORS_FILE},
        'baselines_note': note,
    }


def _dump(obj):
    return (json.dumps(obj, indent=1) + '\n').encode()


def lock_path(plan_path):
    d, f = os.path.split(os.path.abspath(plan_path))
    return os.path.join(d, 'lock_' + (f[len('plan_'):]
                                      if f.startswith('plan_') else f))


def make_items(plan_dir):
    """Write plan, states, items and lock into plan_dir from the committed
    S0 and anchors there. Deterministic bytes: run twice, same files."""
    s0 = json.load(open(os.path.join(plan_dir, S0_FILE)))
    anchors = json.load(open(os.path.join(plan_dir, ANCHORS_FILE)))
    states, items = build(s0, anchors)
    plan = make_plan(states)
    blobs = {'plan': _dump(plan), 'states': _dump(states),
             'items': ''.join(json.dumps(it) + '\n'
                              for it in items).encode()}
    paths = {'plan': os.path.join(plan_dir, f'plan_{VERSION}.json'),
             'states': os.path.join(plan_dir, plan['files']['states']),
             'items': os.path.join(plan_dir, plan['files']['items'])}
    for k, p in paths.items():
        with open(p, 'wb') as f:
            f.write(blobs[k])
    lock = {k: sha256_bytes(v) for k, v in blobs.items()}
    for k in ('s0', 'anchors'):
        lock[k] = sha256_bytes(open(os.path.join(
            plan_dir, plan['files'][k]), 'rb').read())
    with open(lock_path(paths['plan']), 'wb') as f:
        f.write(_dump(lock))
    return {'plan': plan, 'states': states, 'items': items, 'lock': lock,
            'paths': paths}


def load(plan_path):
    d = os.path.dirname(os.path.abspath(plan_path))
    plan = json.load(open(plan_path))
    fs = plan['files']
    with open(os.path.join(d, fs['items'])) as f:
        items = [json.loads(line) for line in f if line.strip()]
    return {'plan': plan,
            'states': json.load(open(os.path.join(d, fs['states']))),
            'items': items,
            's0': json.load(open(os.path.join(d, fs['s0']))),
            'anchors': json.load(open(os.path.join(d, fs['anchors']))),
            'dir': d}


def verify(plan_path):
    """Problems with a committed plan (empty = ok): the lock, every body's
    sha, no 'model' in any body, ids and qmaps, the per-experiment counts,
    and S0's canonical body against the record's bytes."""
    probs = []
    try:
        d = load(plan_path)
    except (OSError, ValueError, KeyError) as ex:
        return [f'cannot load {plan_path}: {type(ex).__name__}: {ex}']
    plan, items = d['plan'], d['items']
    try:
        lock = json.load(open(lock_path(plan_path)))
    except (OSError, ValueError) as ex:
        return [f'no lock: {type(ex).__name__}: {ex}']
    files = dict(plan['files'], plan=os.path.basename(plan_path))
    for k in ('plan', 'states', 'items', 's0', 'anchors'):
        p = os.path.join(d['dir'], files[k])
        got = sha256_bytes(open(p, 'rb').read()) if os.path.exists(p) \
            else None
        if lock.get(k) != got:
            probs.append(f'lock: {k} sha256 mismatch ({files[k]})')
    for c in plan['candidates']:
        if c['id'].startswith('~') or ':free' in c['id']:
            probs.append(f'candidate {c["id"]} is a moving alias or :free')
    ids = [c['id'] for c in plan['candidates']]
    for role in ('incumbent', 'reference'):
        if plan[role] not in ids:
            probs.append(f'{role} {plan[role]} is not a candidate')
    labels = {sid: st['labels'] for sid, st in d['states'].items()}
    labels['S0'] = d['s0']['names']
    seen, count = set(), {}
    for it in items:
        iid, body = it['item_id'], it['body']
        if iid in seen:
            probs.append(f'{iid}: duplicate item id')
        seen.add(iid)
        if iid != f'{it["experiment"]}|{it["state_id"]}|{it["arm"]}':
            probs.append(f'{iid}: id does not match experiment|state|arm')
        count[it['experiment']] = count.get(it['experiment'], 0) + 1
        if 'model' in body:
            probs.append(f'{iid}: body contains "model"')
        if list(body)[:2] != ['state', 'questions']:
            probs.append(f'{iid}: body keys {list(body)}')
        if body_sha(body) != it['body_sha256']:
            probs.append(f'{iid}: body_sha256 mismatch')
        canon = set(qids_of(labels.get(it['state_id'], [])))
        if set(it['qmap']) != set(body['questions']):
            probs.append(f'{iid}: qmap does not cover the sent ids')
        if not set(it['qmap'].values()) <= canon or \
                len(set(it['qmap'].values())) != len(it['qmap']):
            probs.append(f'{iid}: qmap maps outside the canonical ids')
    want = plan.get('calls_per_model', {}).get('nominal', {})
    if count != want:
        probs.append(f'counts per experiment {count} != plan {want}')
    w = next((it for it in items if it['item_id'] == 'E0|S0|W'), None)
    s0 = d['s0']
    if w is None or json.dumps(w['body']) != \
            json.dumps(_body(s0['state'], s0['names'])):
        probs.append('E0|S0|W is not the record\'s canonical S0 body')
    return probs


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'make':
        out = make_items(sys.argv[2])
        print(f'{len(out["items"])} items, {len(out["states"])} states -> '
              f'{sys.argv[2]}')
        for k, v in out['lock'].items():
            print(f'  {k:8s} {v}')
    elif len(sys.argv) == 3 and sys.argv[1] == 'verify':
        problems = verify(sys.argv[2])
        print('\n'.join(problems) or 'ok')
        sys.exit(1 if problems else 0)
    else:
        sys.exit('usage: battery_items.py make DIR | verify PLAN')
