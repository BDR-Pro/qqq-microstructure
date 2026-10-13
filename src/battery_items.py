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
# Iteration (spec section 4, below make_iter): after the selection, the
# selected decider is re-asked in the menu's other ways -- request variants
# of the one pinned model, never other model ids -- on fresh states from a
# round's own seed, by the same recipe and the same experiments (no S0).
# decider_battery.py make-iter is the command.
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


def _rng(key, seed=None):
    return np.random.default_rng(np.random.SeedSequence(
        SEEDS['items'] if seed is None else seed, spawn_key=(key,)))


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

def block_plan(b, seed=None):
    """Block b's assignments, from the block's own RNG: the 3/3 regime
    permutation (E6), E5's doses (each once Stressed, once Calm), E7's two
    anonymised states (one per regime, so the AN arm sees both hedge
    classes) and E8's three negated states. seed: the items seed (v1's by
    default; an iteration round's own)."""
    g = GENERATOR
    rng = _rng(100 + b, seed)
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


def gen_state(s, regime, anchors, seed=None):
    """State s (1..24): labels, conditions, name rows and both QQQ rows, from
    the state's own RNG. Every name draws the same numbers whatever its
    condition, and the stress vol factor is drawn whatever the regime, so a
    name's z stays the same across regimes. Returns (record, trace, rng):
    trace keeps each moved name's pre-move row (tests check the update);
    rng goes on to draw the state's items."""
    g = GENERATOR
    rng = _rng(s, seed)
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


def v_draw(rng, state, labels, body=None):
    """One nuisance draw (E2, and S0's V): always a shuffled question order,
    a date shift of 0-4 business days and a book paraphrase; each of opaque
    ids, shuffled keys and numeric jitter on with probability 1/2. body: how
    a request is written (an iteration arm's wording and view); the draws
    never depend on it."""
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
    body = (body or _body)(st, labels, order=order, ids=ids)
    qmap = {(ids or {}).get(k, k): k for k in order}
    return body, qmap, {'factors': f}


def negate(body, hedge_extra=''):
    """E8: every question of a canonical body asked the other way round.
    hedge_extra: what a wording adds to the hedge question (q2's sentence
    about the QQQ row), kept in its negation too."""
    for k, q in body['questions'].items():
        if k == 'hedge_book':
            q['instructions'] = NEGATION['hedge'] + hedge_extra
            q['criteria'] = dict(NEGATION['hedge_criteria'])
        else:
            q['instructions'] = NEGATION['hold'].format(n=k[len('hold_'):])
            q['criteria'] = dict(NEGATION['hold_criteria'])
    return body


def _cx(state, labels, ask, body=None):
    """A body that asks only the canonical ids in `ask`, about the full
    state: request_body always appends hedge_book, so drop it if unasked."""
    names = [k[len('hold_'):] for k in ask if k != 'hedge_book']
    b = (body or _body)(state, names)
    if 'hedge_book' not in ask:
        del b['questions']['hedge_book']
    return b


def state_items(sid, rec, rng, b, body=None, neg=None, orders=6,
                order_seed='battery-v1|{sid}|O{i}'):
    """E1-E8 items of one synthetic state, drawn from its RNG in a fixed
    order: E2's V, E3's subset, E4's and E5's treated names, E5's sign,
    E7's derangement or anonymisation. An iteration arm passes how it writes
    a request (body, neg) and its number of E1 orders (2k: k5 asks ten);
    the draws never depend on them, so every arm gets the same stimuli."""
    mk = body or _body
    neg = neg or negate
    labels, rows, core = rec['labels'], rec['rows'], rec['state']
    reg = rec['regime']
    qids = qids_of(labels)
    ordinary = [t for t in labels if rec['conditions'][t] == 'Ordinary']
    out = {e: [] for e in EXPERIMENTS}
    out['E1'].append(_item('E1', sid, 'Ca', b, mk(core, labels)))
    ids = {q: q + GENERATOR['cb_id_suffix'] for q in qids}
    out['E1'].append(_item('E1', sid, 'Cb', b, mk(core, labels, ids=ids),
                           {x: q for q, x in ids.items()},
                           {'ids': 'canonical + '
                            + GENERATOR['cb_id_suffix']}))
    for i in range(1, orders + 1):
        seed = order_seed.format(sid=sid, i=i)
        order = guard.question_order(seed, qids)
        out['E1'].append(_item('E1', sid, f'O{i}', b,
                               mk(core, labels, order=order),
                               meta={'order_seed': seed}))
    v, qmap, meta = v_draw(rng, core, labels, mk)
    out['E2'].append(_item('E2', sid, 'V', b, v, qmap, meta))
    cx = rec['cx_type']
    if cx == 1:
        pick = sorted(int(i) for i in rng.choice(
            len(labels), GENERATOR['cx1_holds'], replace=False))
        ask = [qids[i] for i in pick] + ['hedge_book']
    elif cx == 2:
        ask = [qids[int(rng.integers(len(labels)))]]
    else:
        ask = ['hedge_book']
    out['E3'].append(_item('E3', sid, 'CX', b, _cx(core, labels, ask, mk),
                           meta={'cx_type': cx, 'cx_subset': ask}))
    j = ordinary[int(rng.integers(len(ordinary)))]
    st = copy.deepcopy(core)
    st['names'][j]['earnings_tonight'] = True
    out['E4'].append(_item('E4', sid, 'EF', b, mk(st, labels),
                           meta={'treated': j}))
    j = ordinary[int(rng.integers(len(ordinary)))]
    sign = 1 if rng.random() < 0.5 else -1
    k = rec['dose']
    st = copy.deepcopy(core)
    st['names'][j], d = shock_night(rows[j], sign * rnd(k * rows[j]['vol20d']))
    out['E5'].append(_item('E5', sid, 'XM', b, mk(st, labels),
                           meta={'treated': j, 'dose': k, 'sign': sign,
                                 'D': d}))
    other = 'calm' if reg == 'stressed' else 'stressed'
    st = canonical(rows, labels, rec['qqq'][other])
    out['E6'].append(_item('E6', sid, 'QS', b, mk(st, labels),
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
        out['E7'].append(_item('E7', sid, 'LD', b, mk(st, labels),
                               meta={'perm': perm}))
    else:
        p = [int(i) for i in rng.permutation(n)]
        anon = {labels[i]: f'S{p[i] + 1:02d}' for i in range(n)}
        names = [anon[t] for t in labels]
        st = canonical({anon[t]: rows[t] for t in labels}, names,
                       rec['qqq'][reg])
        qmap = {f'hold_{anon[t]}': f'hold_{t}' for t in labels}
        qmap['hedge_book'] = 'hedge_book'
        out['E7'].append(_item('E7', sid, 'AN', b, mk(st, names), qmap,
                               meta={'anon': anon}))
    if rec['e8']:
        out['E8'].append(_item('E8', sid, 'NG', b,
                               neg(mk(core, labels)),
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


def synthetic(s, b, bp, pos, vols, seed=None):
    """State s (block b, position pos in it): (its states-file record, the
    generator's record with the block's assignments, the RNG that goes on
    to draw its items)."""
    rec, _, rng = gen_state(s, bp['regime'][pos], vols, seed)
    core = canonical(rec['rows'], rec['labels'], rec['qqq'][rec['regime']])
    entry = {
        'block': b, 'regime': rec['regime'], 'labels': rec['labels'],
        'conditions': rec['conditions'], 'rows': rec['rows'],
        'qqq': rec['qqq'], 'stressed_m': rec['stressed_m'],
        'stressed_sign': rec['stressed_sign'],
        'cx_type': bp['cx_type'][pos], 'e7_kind': bp['e7_kind'][pos],
        'e8': bp['e8'][pos], 'state': core}
    rec.update(state=core, cx_type=bp['cx_type'][pos],
               e7_kind=bp['e7_kind'][pos], e8=bp['e8'][pos],
               dose=bp['dose'][pos])
    return entry, rec, rng


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
            states[sid], rec, rng = synthetic(s, b, bp, pos, vols)
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
    """The plan and its files; s0 is None for an iteration plan (no S0)."""
    d = os.path.dirname(os.path.abspath(plan_path))
    plan = json.load(open(plan_path))
    fs = plan['files']
    with open(os.path.join(d, fs['items'])) as f:
        items = [json.loads(line) for line in f if line.strip()]
    return {'plan': plan,
            'states': json.load(open(os.path.join(d, fs['states']))),
            'items': items,
            's0': json.load(open(os.path.join(d, fs['s0'])))
            if fs.get('s0') else None,
            'anchors': json.load(open(os.path.join(d, fs['anchors']))),
            'dir': d}


def verify(plan_path):
    """Problems with a committed plan (empty = ok): the lock, every body's
    sha, no 'model' in any body, ids and qmaps, the per-experiment counts,
    and S0's canonical body against the record's bytes. An iteration plan
    is checked by verify_iter."""
    try:
        if json.load(open(plan_path)).get('mode') == 'iterate':
            return verify_iter(plan_path)
    except (OSError, ValueError) as ex:
        return [f'cannot load {plan_path}: {type(ex).__name__}: {ex}']
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


# ------------------------------------------- iteration (spec section 4) --
#
# After the selection, the selected decider may be re-asked in other ways
# from a fixed menu -- a pre-written text or a view computed in code, none
# holding a threshold or a number -- against how it is asked now, on fresh
# states. An ARM is one way of asking the ONE pinned model: a version (the
# wording q1|q2, the views s3 and b2, and sep: one request per question)
# and a handling (k1|k3|k5, the draws averaged). Arms are never other model
# ids: the bytes carry the pin; the arm id '<pin>#<handling>.<version>'
# only names the arm's record in calls.jsonl, and each arm's items carry
# 'for_arm' so the run sends them to that arm alone. Every arm gets the
# same stimuli -- one generator draw per state, written each arm's way --
# so arms are compared state by state. An arm is sent exactly as the
# runtime would send it once declared: the selection's provider pin (R5)
# rides in every body, and its suffix is guard.handling_suffix's, the one
# decide_local logs.

MENU = ('q2', 's3', 'b2', 'k5', 'sep')
# sep is on the spec's menu, but decide_local sends every question in one
# request: an adopted sep could never be declared, so make-iter refuses it
# until the runtime can send it (its items and scoring are kept for then)
RUNTIME_MENU = ('q2', 's3', 'b2', 'k5')
VIEW_TAGS = ('b2', 's3')                # sorted, as decide_local declares
ITER_BLOCKS, ITER_FLOOR_BLOCKS = 3, 2


def parse_version(v):
    """{'wording', 'b2', 's3', 'sep'} of a version string: the wording
    first ('q1', 'q2'), then the views in VIEW_TAGS order, then 'sep' --
    'q1', 'q2s3', 'q1b2s3', 'q1b2sep'. Anything else is a ValueError."""
    v = str(v)
    if v[:2] not in guard.QUESTIONS:
        raise ValueError(f'not a version: {v!r}')
    out, rest = {'wording': v[:2]}, v[2:]
    for t in VIEW_TAGS + ('sep',):
        out[t] = rest.startswith(t)
        rest = rest[len(t):] if out[t] else rest
    if rest:
        raise ValueError(f'not a version: {v!r}')
    return out


def version_str(p):
    return p['wording'] + ''.join(t for t in VIEW_TAGS + ('sep',) if p[t])


def arm_tag(version, handling):
    return f'{handling}.{version}'


def applied(version, handling):
    """The menu tags a (version, handling) already carries."""
    p = parse_version(version)
    return [t for t in MENU if (t == 'q2' and p['wording'] == 'q2')
            or (t == 'k5' and handling == 'k5') or p.get(t) is True]


def apply_tag(version, handling, tag):
    """(version, handling) with one menu tag on top."""
    if tag not in MENU:
        raise ValueError(f'{tag!r} is not on the menu {", ".join(MENU)}')
    if tag in applied(version, handling):
        raise ValueError(f'{tag} is already in {arm_tag(version, handling)}')
    p = parse_version(version)
    if tag == 'q2':
        p['wording'] = 'q2'
    elif tag == 'k5':
        handling = 'k5'
    else:
        p[tag] = True
    return version_str(p), handling


def declared_of(handling, version):
    """An arm as decide_local.DECLARED['http'] states it: k (int), version
    (the wording) and views (sorted guard.VIEWS tags). sep has no runtime
    form: None."""
    p = parse_version(version)
    if p['sep']:
        return None
    return {'k': int(handling[1:]), 'version': p['wording'],
            'views': ''.join(t for t in VIEW_TAGS if p[t])}


def suffix_of(handling, version):
    """The suffix the record logs after the model for a declared (handling,
    version) (spec R3/R4): guard.handling_suffix, the runtime's own -- none
    for one draw of q1, the record's request; '#perm3.q1' as the selection
    declares k3; '#k1.q2s3' for one draw of another version. A sep arm,
    which has no runtime form, is named the same way ('#k1.q1sep')."""
    p = parse_version(version)
    return guard.handling_suffix(
        int(handling[1:]), p['wording'],
        ''.join(t for t in VIEW_TAGS if p[t]) + ('sep' if p['sep'] else ''))


def arm_writer(version, provider=None):
    """(body, neg) for a version: guard.request_body with its wording, view
    (computed from the state at request time, as the runtime would) and
    provider pin (the selection's, R5: decide_local sends it on every call
    once declared), and E8's negation, which keeps what the wording adds to
    the hedge question."""
    p = parse_version(version)
    w, view = p['wording'], guard.view_of(version)
    q1, qw = guard.QUESTIONS['q1']['hedge'][0], \
        guard.QUESTIONS[w]['hedge'][0]
    assert qw.startswith(q1), 'a wording only ever adds to the hedge text'

    def body(state, names, **kw):
        return _body(state, names, version=w, view=view, provider=provider,
                     **kw)
    return body, (lambda b: negate(b, qw[len(q1):]))


def split(it):
    """sep: one item as one request per question, each with the item's
    whole state (its names in the item's order) and that question alone.
    The scorer merges the answers back by base_id."""
    out = []
    for q, x in it['body']['questions'].items():
        body = dict(it['body'], questions={q: x})   # its provider pin too
        c = it['qmap'][q]
        out.append(dict(it, item_id=f'{it["item_id"]}/{c}', body=body,
                        body_sha256=body_sha(body), qmap={q: c},
                        meta=dict(it['meta'], question=c)))
    return out


def menu_left(base, origin):
    """The menu tags round 2 tests on top of round 1's version (spec 4: the
    remaining menu items, in menu order, at most variants_max): those not
    already in it that the runtime can send and the selection allowed."""
    it = GATES['iteration']
    have = applied(*base)
    return [t for t in MENU if t not in have and t in RUNTIME_MENU
            and (t not in ('k5', 'sep') or origin.get(f'{t}_allowed'))
            ][:it['variants_max']]


def iter_arms(pin, rnd, base, variants, origin):
    """The arms of a round (the control = base, then one arm per menu tag
    on top of it, in menu order) or of the holdout (the control = the
    selection's own q1 at its handling, against the final version = base).
    Refuses (ValueError) a tag off the menu or already in the base, more
    than variants_max, k5 unless the selection's sd_rep allowed it, sep
    unless its ex_ctx did (and, until the runtime can send it, at all), a
    round 2 that is not the remaining menu (menu_left: chosen after round
    1's results, a subset would be a choice the stopping rule does not
    make, and would loosen its alpha), and a holdout of the selection's own
    version."""
    it = GATES['iteration']

    def arm(role, version, handling, menu=None):
        tag = arm_tag(version, handling)
        return {'id': f'{pin}#{tag}', 'tag': tag, 'role': role,
                'menu': menu, 'version': version, 'handling': handling,
                'suffix': suffix_of(handling, version),
                'declared': declared_of(handling, version)}
    if rnd == 'holdout':
        dec = origin['decider']
        ctrl = arm('control', dec['version'], dec['handling'])
        if variants:
            raise ValueError('the holdout takes no variants: it is the final '
                             'version against the selection\'s own')
        if arm_tag(*base) == ctrl['tag']:
            raise ValueError('the final version is the selection\'s own: '
                             'there is nothing to hold out')
        return [ctrl, arm('final', *base)]
    tags = list(dict.fromkeys(variants))
    if not tags:
        raise ValueError('a round needs at least one variant')
    for t in tags:
        if t not in MENU:
            raise ValueError(f'{t!r} is not on the menu {", ".join(MENU)}')
    if len(tags) != len(variants):
        raise ValueError('a variant is named twice')
    if len(tags) > it['variants_max']:
        raise ValueError(f'at most {it["variants_max"]} variants a round')
    for t, why in (('k5', f'sd_rep {origin.get("sd_rep")} <= '
                          f'{it["k5_if_sd_rep_gt"]}'),
                   ('sep', f'ex_ctx {origin.get("ex_ctx")} <= '
                           f'{it["sep_if_ex_ctx_gt"]}')):
        if t in tags and not origin.get(f'{t}_allowed'):
            raise ValueError(f'{t} is not allowed: the selection measured '
                             f'{why}')
    for t in tags:
        if t not in RUNTIME_MENU:
            raise ValueError(f'{t} cannot be declared yet: decide_local sends '
                             f'every question in one request, so an adopted '
                             f'{t} could never run')
    if rnd == 2:
        left = menu_left(base, origin)
        if sorted(tags, key=MENU.index) != left:
            raise ValueError(f'round 2 tests the remaining menu on top of '
                             f'{arm_tag(*base)}: --variants '
                             f'{",".join(left) or "(none left)"}')
    out = [arm('control', *base)]
    for t in sorted(tags, key=MENU.index):
        out.append(arm('variant', *apply_tag(*base, t), menu=t))
    return out


def iter_build(anchors, seed, blocks, arms, version, provider=None):
    """(states, items) of an iteration plan: `blocks` blocks of 6 fresh
    states from the round's seed by v1's recipe (no S0), and each state's
    E1-E8 items once per arm, written the arm's way (with the selection's
    provider pin, if any); then one warm-up W, the control's first request
    whole -- a full request even for a sep control, since the cost per call
    it measures is a full request's, and each call is weighed against it by
    its bytes -- which every arm sends once (the pilot: the cost per call,
    the smoke stop) and no metric reads."""
    vols = anchors['vol20d']
    states, items = {}, []
    per = {a['id']: {e: [] for e in EXPERIMENTS} for a in arms}
    for b in range(1, blocks + 1):
        bp = block_plan(b, seed)
        for pos in range(PER_BLOCK):
            s = (b - 1) * PER_BLOCK + pos + 1
            sid = f'S{s:02d}'
            for a in arms:
                # each arm draws the state afresh from its own key, so every
                # arm's items come from the very same draws
                states[sid], rec, rng = synthetic(s, b, bp, pos, vols, seed)
                body, neg = arm_writer(a['version'], provider)
                k = int(a['handling'][1:])
                out = state_items(sid, rec, rng, b, body, neg,
                                  orders=max(6, 2 * k),
                                  order_seed=f'battery-{version}|{{sid}}|'
                                             f'O{{i}}')
                for e in EXPERIMENTS:
                    per[a['id']][e] += out[e]
    for a in arms:
        sep = parse_version(a['version'])['sep']
        for e in EXPERIMENTS:
            for it in per[a['id']][e]:
                for x in (split(it) if sep else [it]):
                    items.append(dict(x, item_id=f'{x["item_id"]}#{a["tag"]}',
                                      base_id=it['item_id'],
                                      for_arm=a['id']))
    first = next(it for e in EXPERIMENTS for it in per[arms[0]['id']][e])
    w = _item('E0', first['state_id'], 'W', 0, first['body'], first['qmap'],
              {'role': 'warm-up, cost pilot: sent once to every arm',
               'body_of': f'{first["item_id"]}#{arms[0]["tag"]}'})
    return states, [w] + items


def _sent(body, pin):
    # the bytes the run sends (battery_run.send_bytes, the contract)
    send = {'state': body['state'], 'model': pin,
            'questions': body['questions']}
    if 'provider' in body:
        send['provider'] = body['provider']
    return json.dumps(send).encode()


def iter_cost(items, pin, per_call):
    """What an iteration plan's calls cost at per_call for a request the
    size of W -- each call priced by its bytes, since the price is per input
    token and a sep request is a fraction of a full one: W once per arm,
    then each arm's own items."""
    ref = len(_sent(items[0]['body'], pin))
    arms = {it['for_arm'] for it in items if it.get('for_arm')}
    return per_call * (len(arms) + sum(len(_sent(it['body'], pin))
                                       for it in items
                                       if it.get('for_arm')) / ref)


def origin_of(sel, summary, src_plan, where):
    """What every round starts from, read once from the selection: the
    decider it declared (iteration only ever re-asks that pin), and the
    measurements that open the conditional menu items -- sd_rep for k5,
    ex_ctx for sep, against the selection plan's declared bars -- with the
    cost per call the rounds are projected at (the catalogue's pessimistic
    figure, or the measured median if higher)."""
    dec = sel['decider']
    m = (summary.get('models') or {}).get(dec['pin']) or {}
    sd_rep = (m.get('noise') or {}).get('sd_rep')
    ex_ctx = (m.get('E3') or {}).get('ex_ctx')
    measured = (m.get('ops') or {}).get('median_cost_usd')
    price = next((c.get('price_in') for c in src_plan.get('candidates', [])
                  if c['id'] == dec['pin']), None)
    bars = (src_plan.get('gates') or {}).get('iteration') or \
        GATES['iteration']
    sp = src_plan.get('spend') or SPEND
    cat = sp['pessimistic_per_call'] * (price / sp['price_ref']
                                        if price is not None else 1.0)
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    return {'selection': where, 'branch': sel.get('branch'),
            'decider': dec, 'sd_rep': sd_rep, 'ex_ctx': ex_ctx,
            'k5_allowed': bool(num(sd_rep)
                               and sd_rep > bars['k5_if_sd_rep_gt']),
            'sep_allowed': bool(num(ex_ctx)
                                and ex_ctx > bars['sep_if_ex_ctx_gt']),
            'price_in': price,
            'cost_per_call_usd': max([cat] + ([measured] if num(measured)
                                              else []))}


def iter_plan(src, version, arms, states, items, blocks, projection):
    """An iteration plan in the contract's format, mode 'iterate'. The
    control is both 'incumbent' and 'reference', so the run never drops it
    or smoke-stops it; a scale-down at run time drops the last block only
    while that leaves the analysis floor of 2 blocks (spec 4), then
    variants from the end of the menu (drop_order). provider_pin: the
    selection's, in every body (verify_iter holds them to it)."""
    it = GATES['iteration']
    gates = json.loads(json.dumps(GATES))
    gates['analysis_set']['blocks_min'] = ITER_FLOOR_BLOCKS
    ctrl = arms[0]
    hold = src['round'] == 'holdout'
    # the run's scale-down drops the last block only above the floor
    # (battery_run.last_block); at the floor it drops variants alone
    drop = blocks if blocks > ITER_FLOOR_BLOCKS else None
    count = {}
    for x in items:
        for a in arms:
            if x.get('for_arm') in (None, a['id']):
                c = count.setdefault(a['id'], {'nominal': {}, 'floor': {}})
                for k in ('nominal', 'floor'):
                    if k == 'floor' and x['block'] == drop:
                        continue
                    c[k][x['experiment']] = c[k].get(x['experiment'], 0) + 1
    seed = it['holdout_seed'] if hold else it['items_seeds'][src['round'] - 1]
    words = sorted({parse_version(a['version'])['wording'] for a in arms})
    note = (f'Iteration {version} on {src["pin"]}: {len(states)} fresh '
            f'states, {len(arms)} arms. battery_score.score computes Q_rule '
            f'and the other offline baselines from these items; G5 uses '
            f'that value.')
    return {
        'version': version,
        'mode': 'iterate',
        'round': src['round'],
        'pin': src['pin'],
        'base': src['base'][0], 'handling': src['base'][1],
        'variants': [a['menu'] for a in arms if a['role'] == 'variant'],
        'arms': arms,
        'incumbent': ctrl['id'], 'reference': ctrl['id'],
        'candidates': [{'id': a['id'], 'pin': src['pin'],
                        'price_in': src['origin']['price_in'],
                        'X': None if a['role'] == 'variant' else 1.0}
                       for a in arms],
        'drop_order': [a['id'] for a in reversed(arms)
                       if a['role'] == 'variant'],
        'origin': src['origin'], 'prior': src.get('prior'),
        'provider_pin': src['origin']['decider'].get('provider_pin'),
        'seeds': dict(SEEDS, items=seed),
        'date': DATE,
        'blocks': blocks, 'states_per_block': PER_BLOCK,
        'generator': dict(
            GENERATOR,
            rng_state=f'default_rng(SeedSequence(seeds.items, spawn_key='
                      f'(s,))), s = 1..{len(states)} for S01..'
                      f'S{len(states):02d}; no S0',
            rng_block=f'default_rng(SeedSequence(seeds.items, spawn_key='
                      f'(100 + b,))), b = 1..{blocks}',
            e1_order_seed=f'battery-{version}|{{state_id}}|O{{i}}',
            e1_orders='O1-O6; O1-O10 for a k5 arm',
            sep='one request per question: the item\'s state, one question',
            pilot='E0|S01|W: the control\'s first request, once per arm'),
        'texts': {'questions': {w: guard.QUESTIONS[w] for w in words},
                  'views': {'s3': 'guard.view_s3: last_night_z = round('
                                  'last_night / vol20d, 1), ret5d_z = round('
                                  'ret5d / (vol20d sqrt 5), 1), for the '
                                  'names and QQQ',
                            'b2': guard.view_of('b2')({})['book']},
                  'book_paraphrases': BOOK_PARAPHRASES,
                  'negation': NEGATION},
        'gates': gates,
        'gates_note': 'key suffixes: _min >=, _max <=, _gt >, _lt <',
        'calls_per_model': dict(count[ctrl['id']], per_arm=count),
        'calls_note': (f'floor = block {drop}\'s states dropped' if drop
                       else f'floor = nominal: {blocks} blocks is the '
                            f'floor, so a scale-down drops variants only')
        + '. W (E0|S01|W) goes to every arm; every other item carries '
          'for_arm and goes to that arm alone.',
        'bootstrap_B': 10000, 'signflip_draws': 200000,
        'spend': dict(SPEND, max_usd=projection['cap_usd'],
                      max_usd_ceiling=projection['cap_usd']),
        'projection': projection,
        'deadline_min': 90, 'timeout_s': 60,
        # every arm is the same model: its 3 calls in flight are shared
        'concurrency': 3, 'per_model': 3,
        'files': {'states': f'states_{version}.json',
                  'items': f'items_{version}.jsonl',
                  'anchors': ANCHORS_FILE},
        'baselines_note': note,
    }


def make_iter(out_dir, src):
    """Write an iteration round's (or the holdout's) plan, states, items, a
    copy of the anchors and the lock into out_dir: deterministic bytes.
    src: {'round': 1|2|'holdout', 'pin', 'base': [version, handling],
    'variants': [menu tags], 'origin' (origin_of), 'prior', 'anchors':
    path}. Projected at the origin's cost per call; above 0.8 x the cap
    (the run's own scale-down bar) the last block goes first (down to the
    floor of 2), then variants from the end of the menu. Refusals raise
    ValueError."""
    it = GATES['iteration']
    rnd, pin = src['round'], src['pin']
    hold = rnd == 'holdout'
    if rnd not in (1, 2, 'holdout'):
        raise ValueError(f'round {rnd!r}: 1, 2 or holdout')
    version = 'iter_holdout' if hold else f'iter_r{rnd}'
    seed = it['holdout_seed'] if hold else it['items_seeds'][rnd - 1]
    cap = it['holdout_cap_usd' if hold else 'round_cap_usd']
    arms = iter_arms(pin, rnd, src['base'], src['variants'], src['origin'])
    with open(src['anchors'], 'rb') as f:
        anchors_bytes = f.read()
    anchors = json.loads(anchors_bytes)
    per_call = src['origin']['cost_per_call_usd']
    proj = {'per_call_usd': per_call, 'cap_usd': cap,
            'limit_usd': 0.8 * cap, 'requested': [a['menu'] for a in arms
                                                  if a['menu']],
            'dropped_blocks': [], 'dropped_variants': []}
    blocks = ITER_BLOCKS
    provider = src['origin']['decider'].get('provider_pin')
    while True:
        states, items = iter_build(anchors, seed, blocks, arms, version,
                                   provider)
        usd = iter_cost(items, pin, per_call)
        if usd <= proj['limit_usd']:
            break
        if blocks > ITER_FLOOR_BLOCKS:
            proj['dropped_blocks'].append(blocks)
            blocks -= 1
        elif not hold and len(arms) > 2:
            proj['dropped_variants'].append(arms.pop()['menu'])
        else:
            raise ValueError(f'projected ${usd:.2f} > ${proj["limit_usd"]:.2f}'
                             f' (0.8 x the ${cap} cap) even at the floor')
    proj.update(usd=usd, blocks=blocks, calls=len(items) - 1 + len(arms))
    plan = iter_plan(dict(src), version, arms, states, items, blocks, proj)
    blobs = {'plan': _dump(plan), 'states': _dump(states),
             'items': ''.join(json.dumps(x) + '\n' for x in items).encode(),
             'anchors': anchors_bytes}
    paths = {'plan': os.path.join(out_dir, f'plan_{version}.json'),
             'states': os.path.join(out_dir, plan['files']['states']),
             'items': os.path.join(out_dir, plan['files']['items']),
             'anchors': os.path.join(out_dir, plan['files']['anchors'])}
    for k, p in paths.items():
        with open(p, 'wb') as f:
            f.write(blobs[k])
    lock = {k: sha256_bytes(v) for k, v in blobs.items()}
    with open(lock_path(paths['plan']), 'wb') as f:
        f.write(_dump(lock))
    return {'plan': plan, 'states': states, 'items': items, 'lock': lock,
            'paths': paths}


def verify_iter(plan_path):
    """Problems with an iteration plan (empty = ok): the lock; every body's
    sha, no 'model' in a body; one W, sent to every arm; every other item
    for one of the plan's arms, its id its base id and arm tag, its qmap
    back to the state's canonical ids; the counts per arm; every body
    carries the selection's provider pin, or none if it declared none; and
    every arm's canonical requests are guard.request_body's for its version
    and that pin (for a q1 arm with no view or pin, the record's own
    bytes)."""
    probs = []
    try:
        d = load(plan_path)
        lock = json.load(open(lock_path(plan_path)))
    except (OSError, ValueError, KeyError) as ex:
        return [f'cannot load {plan_path}: {type(ex).__name__}: {ex}']
    plan, items, states = d['plan'], d['items'], d['states']
    files = dict(plan['files'], plan=os.path.basename(plan_path))
    for k in ('plan', 'states', 'items', 'anchors'):
        p = os.path.join(d['dir'], files[k])
        got = sha256_bytes(open(p, 'rb').read()) if os.path.exists(p) \
            else None
        if lock.get(k) != got:
            probs.append(f'lock: {k} sha256 mismatch ({files[k]})')
    pin, arms = plan['pin'], {a['id']: a for a in plan['arms']}
    pv = plan.get('provider_pin')
    if pin.startswith('~') or ':free' in pin:
        probs.append(f'{pin} is a moving alias or :free')
    if pv != ((plan.get('origin') or {}).get('decider') or {}).get(
            'provider_pin'):
        probs.append('the provider pin is not the selection\'s')
    if [c['id'] for c in plan['candidates']] != list(arms) or \
            any(c.get('pin') != pin for c in plan['candidates']):
        probs.append('the candidates are not the arms of the one pin')
    for a in arms.values():
        if a['id'] != f'{pin}#{a["tag"]}' or \
                a['tag'] != arm_tag(a['version'], a['handling']):
            probs.append(f'{a["id"]}: id, tag and version disagree')
        if a.get('suffix') != suffix_of(a['handling'], a['version']) or \
                a.get('declared') != declared_of(a['handling'], a['version']):
            probs.append(f'{a["id"]}: suffix or declaration is not the '
                         f'runtime\'s for {a["tag"]}')
    seen, count = set(), {}
    for it in items:
        iid, body, a = it['item_id'], it['body'], it.get('for_arm')
        if iid in seen:
            probs.append(f'{iid}: duplicate item id')
        seen.add(iid)
        if 'model' in body or list(body)[:2] != ['state', 'questions']:
            probs.append(f'{iid}: body keys {list(body)}')
        if body_sha(body) != it['body_sha256']:
            probs.append(f'{iid}: body_sha256 mismatch')
        if body.get('provider') != pv:
            probs.append(f'{iid}: provider {body.get("provider")!r} is not '
                         f'the selection\'s pin {pv!r}')
        base = f'{it["experiment"]}|{it["state_id"]}|{it["arm"]}'
        if a is None:
            if it['experiment'] != 'E0' or iid != base:
                probs.append(f'{iid}: only the pilot W goes to every arm')
        elif a not in arms:
            probs.append(f'{iid}: for_arm {a} is not an arm')
        else:
            q = it['meta'].get('question')
            want = base + (f'/{q}' if q else '') + f'#{arms[a]["tag"]}'
            if it.get('base_id') != base or iid != want:
                probs.append(f'{iid}: id is not base_id + arm tag')
            c = count.setdefault(a, {})
            c[it['experiment']] = c.get(it['experiment'], 0) + 1
        canon = set(qids_of(states.get(it['state_id'], {}).get('labels',
                                                                 [])))
        if set(it['qmap']) != set(body['questions']) or \
                not set(it['qmap'].values()) <= canon:
            probs.append(f'{iid}: qmap does not map the sent ids back')
    if sum(it['experiment'] == 'E0' for it in items) != 1:
        probs.append('the plan needs exactly one pilot W')
    per = plan.get('calls_per_model', {}).get('per_arm', {})
    for a in arms:
        want = {k: v for k, v in per.get(a, {}).get('nominal', {}).items()
                if k != 'E0'}
        if count.get(a, {}) != want:
            probs.append(f'{a}: counts {count.get(a)} != plan {want}')
    by = {it['item_id']: it for it in items}
    for a in arms.values():
        if parse_version(a['version'])['sep']:
            continue
        mk, _ = arm_writer(a['version'], pv)
        for sid, st in states.items():
            it = by.get(f'E1|{sid}|Ca#{a["tag"]}')
            if it is None or json.dumps(it['body']) != \
                    json.dumps(mk(st['state'], st['labels'])):
                probs.append(f'{a["id"]}: E1|{sid}|Ca is not guard\'s '
                             f'request for {a["version"]}')
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
