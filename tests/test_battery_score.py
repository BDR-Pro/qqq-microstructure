# Part of qqq-microstructure.
#
# The battery's scorer and selection (src/battery_score.py, RESULTS 27):
# each section-2 formula on numbers small enough to check by hand; the
# sign-flip test exact against Monte Carlo and Holm against reference
# values; a whole run scored end to end on a tiny synthetic plan written per
# the contract (states, items and calls built here, read by a stand-in that
# answers from the bytes it is sent); and every branch of section 3's
# select(). Nothing here touches the network or the committed battery.

import os, sys, re, json, math, hashlib
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import guard
import battery_score as bs

BOOK = guard.BOOK
GATES = {
    'G1': {'valid_rate_min': 0.98, 'completion_min': 0.95,
           'modal_share_min': 0.90, 'p95_latency_s_max': 60.0,
           'median_cost_usd_max': 0.01},
    'G2a': {'ex_nuis_max': 0.05, 'rel_max': 0.5, 'D_min': 0.01},
    'G2b': {'ex_ctx_max': 0.05, 'lb_alpha': 0.05},
    'G3a': {'E_bar_max': -0.05, 'frac_neg_min': 0.75, 'holm_p_lt': 0.05},
    'G3b': {'G_bar_min': 0.05, 'delta_h_gt': 0.0, 'holm_p_lt': 0.05},
    'G4': {'flip_h_max': 0.10},
    'G5': {'Q_minus_Q_rule_min': 0.05},
    'G6': {'Phi_min': 0.5, 'Phi_ab_floor_min': 0.2},
    'handling': {'k3_if_sd_rep_gt': 0.005, 'k3_if_ex_O_gt': 0.01,
                 'phi_factor': 1.5},
    'holm': {'fwer': 0.05, 'family': ['G3a', 'G3b'],
             'signflip_exact_max_states': 20},
    'E8_flag': {'CE_gt': 0.15, 'rho_neg_max': 0.0, 'permutations': 2000},
    'analysis_set': {'block_final_share_min': 0.95, 'blocks_min': 3},
    'selection': {'switch_margin_min': 0.03, 'lower_bound_gt': 0.0,
                  'alpha': 0.05},
    'iteration': {'delta_Q_min': 0.02, 'lower_bound_gt': 0.0, 'alpha': 0.05,
                  'AUC_comb_drop_max': 0.02, 'Phi_drop_max': 0.05}}
SEEDS = {'items': 20261013, 'analysis': 20261014, 'dispatch': 20261015}


def plan_dict(**kw):
    p = {'version': 'v1', 'mode': 'select', 'incumbent': 'fake/inc',
         'reference': 'fake/ref', 'candidates': [], 'seeds': SEEDS,
         'date': '20261013', 'blocks': 3, 'states_per_block': 4,
         'gates': GATES, 'bootstrap_B': 2000, 'signflip_draws': 20000,
         'files': {'states': 'states_v1.json', 'items': 'items_v1.jsonl',
                   's0': 's0_20261013.json'}}
    p.update(kw)
    return p


# ------------------------------------------------- formulas, by hand --

LAB = ['A', 'B', 'C']
HAND_STATES = {
    'S1': {'block': 1, 'regime': 'stressed', 'labels': LAB,
           'conditions': {'A': 'Ordinary', 'B': 'Moderate',
                          'C': 'Earnings'},
           'stressed_m': 2.5, 'stressed_sign': 1},
    'S2': {'block': 1, 'regime': 'calm', 'labels': LAB,
           'conditions': {'A': 'Ordinary', 'B': 'ExtremeUp',
                          'C': 'ExtremeDown'},
           'stressed_m': 2.2, 'stressed_sign': -1}}


def hand_ctx():
    items = []
    for s, st in HAND_STATES.items():
        core = {'names': {t: {'earnings_tonight':
                              st['conditions'][t] == 'Earnings'}
                          for t in LAB}}
        items.append({'item_id': f'E1|{s}|Ca', 'experiment': 'E1',
                      'state_id': s, 'arm': 'Ca', 'block': 1,
                      'body': {'state': core}, 'qmap': {}, 'meta': {}})
    items.append({'item_id': 'E4|S1|EF', 'experiment': 'E4', 'state_id':
                  'S1', 'arm': 'EF', 'block': 1, 'body': {}, 'qmap': {},
                  'meta': {'treated': 'A'}})
    s0 = {'date': '20261013', 'names': ['A'],
          'state': {'names': {'A': {'earnings_tonight': False}}}}
    return bs.Ctx(plan_dict(), HAND_STATES, items, s0, ['S1', 'S2'])


def test_noise_primitives_and_handling_by_hand():
    P = {'E1|S1|Ca': {'A': .9, 'B': .8, 'C': .1, BOOK: .7},
         'E1|S1|Cb': {'A': .7, 'B': .8, 'C': .1, BOOK: .7},
         'E1|S1|O1': {'A': 1.0, 'B': .8, 'C': .1, BOOK: .7},
         'E1|S2|Ca': {'A': .5, 'B': .5, 'C': .5, BOOK: .2},
         'E1|S2|Cb': {'A': .5, 'B': .5, 'C': .5, BOOK: .2}}
    r = bs.score_model(P, hand_ctx())
    # sd_rep^2 = mean over the 8 (s, x) of (p_Ca - p_Cb)^2 / 2: one 0.2 gap
    assert r['noise']['sd_rep'] == pytest.approx(math.sqrt(0.2 ** 2 / 2 / 8))
    assert r['noise']['phi'] == pytest.approx(1.5 * 0.2 ** 2 / 2 / 8)
    # D: the mean over states of the sd of p_C-bar across names
    assert r['noise']['D'] == pytest.approx(
        (np.std([.8, .8, .1], ddof=1) + 0.0) / 2)
    # ex(O): O1 sits 0.2 off p_C-bar on one of its 4 answers
    assert r['noise']['ex_O'] == pytest.approx(
        math.sqrt(0.2 ** 2 / 4 - 1.5 * 0.2 ** 2 / 2 / 8))
    assert r['h'] == 'k3'                    # sd_rep 0.05 > 0.005


def test_handling_follows_order_noise_alone():
    base = {'A': .9, 'B': .8, 'C': .1, BOOK: .7}
    P = {'E1|S1|Ca': base, 'E1|S1|Cb': base}
    assert bs.score_model(P, hand_ctx())['h'] == 'k1'
    P['E1|S1|O1'] = dict(base, A=.95)        # ex(O) = sqrt(.05^2 / 4) > 0.01
    r = bs.score_model(P, hand_ctx())
    assert r['noise']['sd_rep'] == 0 and r['h'] == 'k3'


def hand_decisions():
    s1 = {'A': .9, 'B': .8, 'C': .1, BOOK: .7}
    s2 = {'A': .5, 'B': .5, 'C': .405, BOOK: .2}
    P = {'E1|S1|Ca': s1, 'E1|S1|Cb': s1, 'E1|S2|Ca': s2, 'E1|S2|Cb': s2}
    for i in range(1, 7):
        P[f'E1|S1|O{i}'] = s1
        # the two triples straddle the skip bar on C (median 0.5 - 0.10)
        P[f'E1|S2|O{i}'] = dict(s2, C=.40 if i <= 3 else .41)
    return P


def test_decision_sets_flips_Q_and_AUC_by_hand():
    P = hand_decisions()
    r = bs.score_model(P, hand_ctx())
    e1 = r['E1']
    assert r['h'] == 'k1'
    assert e1['flip_k1'] == 0 and e1['flip_k3'] == pytest.approx(1 / 8)
    # k1: S1 skips its Earnings name in both sets, hedges (Stressed); S2
    # skips nothing (0.5 - 0.405 < 0.10), does not hedge (Calm)
    assert e1['Q_k1'] == pytest.approx(0.25 * (2 / 6 + 1 + 1 + 1))
    # k3: the first triple's mean (C = 0.40) skips C, the second does not
    assert e1['Q_k3'] == pytest.approx(0.25 * (3 / 6 + 1 + 1 + 1))
    assert e1['Q_h'] == e1['Q_k1']
    # Q_noE: Earnings names leave the labels; Extreme names alone remain
    assert e1['Q_noE'] == pytest.approx(0.25 * (0 / 4 + 1 + 1 + 1))
    # AUC_hold: S1 [.9] vs [.1] -> 1; S2 [.5] vs [.5, .405] -> (1/2 + 1)/2
    assert e1['AUC_hold'] == pytest.approx((1 + 1 + .75 + .75) / 4)
    assert e1['AUC_hedge'] == 1.0
    assert e1['AUC_comb'] == pytest.approx((.875 + 1) / 2)
    # the earnings rule skips C in S1 and never hedges: S1 differs on BOOK
    assert e1['agree_rule'] == pytest.approx((3 + 3 + 4 + 4) / 16)


def test_did_hedge_swap_negation_and_lift_by_hand():
    P = hand_decisions()
    P['E4|S1|EF'] = {'A': .1, 'B': .82, 'C': .25, BOOK: .7}
    P['E6|S1|QS'] = {'A': .9, 'B': .8, 'C': .1, BOOK: .2}   # calm swapped in
    P['E6|S2|QS'] = {'A': .5, 'B': .5, 'C': .405, BOOK: .6}  # stressed in
    P['E8|S1|NG'] = {'A': .1, 'B': .25, 'C': .9, BOOK: .3}
    r = bs.score_model(P, hand_ctx())
    e4 = r['E4']
    # (p_EF(j) - p_C(j)) - mean_{k != j}(p_EF(k) - p_C(k))
    assert e4['DiD'] == [pytest.approx(-.8 - (.02 + .15) / 2)]
    assert e4['frac_neg'] == 1.0
    assert e4['lift_E'] == 1.0                # skipped vs held in C^a, C^b
    assert e4['spillover'] == pytest.approx(
        math.sqrt((.02 ** 2 + .15 ** 2) / 2))
    assert e4['sham']['values'] == [0.0]      # C^b vs C^a, identical
    e6 = r['E6']
    assert e6['dH'] == [pytest.approx(.5), pytest.approx(.4)]
    assert e6['G_bar'] == pytest.approx(.45) and e6['delta_h'] == 1.0
    assert e6['up'] == pytest.approx(.5) and e6['down'] == pytest.approx(.4)
    assert e6['kendall_tau_m'] == 1.0         # m 2.5 vs 2.2, dH .5 vs .4
    # CE = mean |p_C + p_NG - 1|: only B is off, by 0.05
    assert r['E8']['CE'] == pytest.approx(.05 / 4)


def test_did_and_phi_and_small_helpers():
    assert bs.did({'A': .1, 'B': .82, 'C': .3}, {'A': .9, 'B': .8, 'C': .2},
                  'A', ['A', 'B', 'C']) == pytest.approx(-.8 - .06)
    y = np.array([1., -1, 0, .5, -.5])
    xt = np.array([0., 1, -1, .2, -.2])
    assert bs.phi_ols(y, y, xt, .2)[0] == pytest.approx(1)      # features
    assert bs.phi_ols(xt, y, xt, .2)[0] == pytest.approx(0)     # ticker
    assert bs.phi_ols(y + xt, y, xt, .2)[0] == pytest.approx(.5)
    assert bs.phi_ols(.05 * y, y, xt, .2)[0] == 0               # neither
    assert bs.phi_ols(-y, y, xt, .2)[0] == pytest.approx(-1)
    assert bs.kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)
    assert bs.kendall_tau([1, 1, 2], [1, 2, 3]) == \
        pytest.approx(2 / math.sqrt(2 * 3))                      # tau-b
    assert bs.auc([.9, .5], [.5, .1]) == pytest.approx((1 + 1 + .5 + 1) / 4)
    assert bs.ex([.01, .03], .005) == pytest.approx(math.sqrt(.015))
    assert bs.ex([.001], .005) == 0.0
    assert bs.mean_p([{'A': .2, 'B': .4}, None, {'A': .4}]) == \
        {'A': pytest.approx(.3)}


def _ctx_draws(sd, shift, seed):
    """24 states, 8 per CX type (6, 1 and 1 questions asked): C^a, C^b and
    the CX answers as oracle + N(0, sd), CX shifted by `shift`; (point ex,
    the lower bound) as score_model computes them."""
    rng = np.random.default_rng(seed)
    strata = {f'S{i:02d}': str(i % 3 + 1) for i in range(24)}
    nq = {'1': 6, '2': 1, '3': 1}
    dev, rep, sq_x, sq_r = {}, {}, [], []
    for s, t in strata.items():
        ca, cb = rng.normal(0, sd, 12), rng.normal(0, sd, 12)
        x = rng.normal(0, sd, nq[t]) + shift
        d = (x - (ca + cb)[:nq[t]] / 2) ** 2
        r = (ca - cb) ** 2 / 2
        dev[s], rep[s] = (float(d.sum()), nq[t]), (float(r.sum()), 12)
        sq_x, sq_r = sq_x + list(d), sq_r + list(r)
    return (bs.ex(sq_x, 1.5 * np.mean(sq_r)),
            bs.ex_lb(dev, rep, strata, 1.5, 2000, seed, 0.05))


def test_g2b_gates_on_a_lower_bound_so_noise_alone_does_not_fail():
    # no context effect at all, sd 0.10: the point estimate crosses 0.05 in
    # about a fifth of draws (64 answers against 0.05^2); the bound rarely
    noise = [_ctx_draws(0.10, 0.0, k) for k in range(100)]
    assert np.mean([pt > .05 for pt, _ in noise]) > .1
    assert np.mean([lb > .05 for _, lb in noise]) <= .03
    assert all(lb <= pt + 1e-12 for pt, lb in noise)
    # a real context shift of 0.08 on a quiet model still fails, every time
    shift = [_ctx_draws(0.02, 0.08, k) for k in range(30)]
    assert all(lb > .05 for _, lb in shift)


def test_a_context_effect_the_same_in_every_state_fails_at_its_size():
    # the joint answerer: every CX answer of a type off by the same amount;
    # resampling within the type cannot move the bound below the point
    strata = {f'S{i:02d}': str(i % 3 + 1) for i in range(24)}
    off = {'1': .05, '2': .0917, '3': .0917}
    n = {'1': 6, '2': 1, '3': 1}
    dev = {s: (n[t] * off[t] ** 2, n[t]) for s, t in strata.items()}
    rep = {s: (0.0, 12) for s in strata}
    lb = bs.ex_lb(dev, rep, strata, 1.5, 2000, 1, .05)
    point = math.sqrt(sum(v for v, _ in dev.values())
                      / sum(k for _, k in dev.values()))
    assert lb == pytest.approx(point) and lb > .05
    assert math.isnan(bs.ex_lb({}, rep, strata, 1.5, 100, 1, .05))


def test_a_cache_that_never_expires_is_flagged():
    # identical bytes answered identically all run long: np.var of equal
    # floats is not always 0 (0.05 three times: ~7e-35), and rho is 0/0
    assert np.var([.05] * 3, ddof=1) != 0
    cx = hand_ctx()
    arms = ['S0rep1', 'S0rep2', 'S0rep3', 'B2B1', 'B2B2', 'B2B3']
    for a in arms:
        cx.item[f'E9|S0|{a}'] = {'item_id': f'E9|S0|{a}', 'state_id': 'S0',
                                 'arm': a, 'meta': {}}
    P = {f'E9|S0|{a}': {'A': .05, BOOK: .05} for a in arms}
    ctrl = {'A': .05, BOOK: .05}
    # C^b under its own ids varied (phi > 0): a cache, not a constant model
    e9 = bs.score_s0(P, cx, 'k1', 1.5 * .03 ** 2, ctrl)
    assert e9['caching_flag'] and e9['var_b2b'] <= bs.TINY
    assert e9['caching_reason'].startswith('identical bytes')
    # C^b answered as C^a (phi 0): a deterministic model, however much a
    # reordering moves it, is not a cache
    assert not bs.score_s0(P, cx, 'k1', 0.0, ctrl)['caching_flag']


def test_the_decider_records_whether_its_build_is_dated():
    ms = {'inc/m': model(qrows()), 'ref/m': model(qrows(ord_skip=3))}
    p = splan('inc/m', 'ref/m')
    assert bs.choose(summary(ms), p)['decider']['dated'] is True
    ms['inc/m']['ops']['build'] = 'inc/m'           # served undated
    assert bs.choose(summary(ms), p)['decider']['dated'] is False


def test_negation_coherence_and_its_permutation_p():
    pc = [[.9, .8, .1, .3], [.2, .6, .7, .4], [.5, .9, .1, .8]]
    good = [(p, [1 - v for v in p]) for p in pc]
    rho, p = bs.neg_coherence(good, 2000, 20261014)
    assert rho == pytest.approx(1) and p < 0.01
    bad = [(p, p) for p in pc]                # answers the question twice
    rho, p = bs.neg_coherence(bad, 2000, 20261014)
    assert rho == pytest.approx(-1) and p == 1.0
    assert bs.neg_coherence(good, 2000, 1) == bs.neg_coherence(good, 2000, 1)


def test_dose_slope_ols_and_bootstrap():
    x = np.log2([1.5, 3, 6, 1.5, 3, 6])
    y = -0.1 * x + np.array([.01, -.01, 0, 0, .01, -.01])
    b = bs._slopes(x, y, np.ones((1, 6)))[0]
    assert b == pytest.approx(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------ statistics --

def test_signflip_exact_by_hand():
    assert bs.signflip_p([1, 2, 3], 1, 10) == 1 / 8
    assert bs.signflip_p([1, -1], 1, 10) == 3 / 4
    assert bs.signflip_p([], 1, 10) == 1.0


def test_signflip_exact_and_monte_carlo_agree():
    x = np.random.default_rng(7).normal(0.45, 1, 14)
    exact = bs.signflip_p(x, 20261014, 200000, exact_max=20)
    mc = bs.signflip_p(x, 20261014, 200000, exact_max=0)
    assert 0.005 < exact < 0.5
    assert abs(exact - mc) < 0.002
    # reproducible from the seed; Monte Carlo p is never 0
    assert mc == bs.signflip_p(x, 20261014, 200000, exact_max=0)
    assert bs.signflip_p([5.0] * 25, 1, 1000) == 1 / 1001


def test_holm_matches_reference_values():
    h = bs.holm({'a': .01, 'b': .04, 'c': .03, 'd': .005}, .05)
    assert {k: round(v['p_holm'], 10) for k, v in h.items()} == \
        {'d': .02, 'a': .03, 'c': .06, 'b': .06}
    assert {k for k, v in h.items() if v['reject']} == {'a', 'd'}
    # the textbook ladder: 5 x .01 = .05 is not < .05, so nothing rejects
    h = bs.holm({f'h{i}': p for i, p in enumerate([.01, .02, .03, .04, .05])},
                .05)
    assert [round(h[f'h{i}']['p_holm'], 10) for i in range(5)] == \
        [.05, .08, .09, .09, .09]
    assert not any(v['reject'] for v in h.values())


def test_stratified_weights_stay_in_their_regime_and_pair_models():
    strata = ['stressed'] * 3 + ['calm'] * 5
    w = bs.strat_weights(strata, 500, 20261014)
    assert (w[:, :3].sum(1) == 3).all() and (w[:, 3:].sum(1) == 5).all()
    assert (w == bs.strat_weights(strata, 500, 20261014)).all()
    rows = qrows(8, ord_skip=3)
    assert bs.paired_bootstrap_lb(rows, rows, strata, 500, 1, .05) == 0
    lb = bs.paired_bootstrap_lb(qrows(8, ord_skip=1), rows, strata, 500, 1,
                                .05)
    assert lb == pytest.approx(0.05)          # a uniform lead is the bound


# ---------------------------------------------------------- operations --

def call(model, iid, valid=True, build=None, prov='P1', lat=500, cost=1e-4,
         err=None):
    return {'key': f'{model}|{iid}', 'model': model, 'item_id': iid,
            'latency_ms': lat, 'http_status': 200 if valid else 400,
            'answered_as': build or f'{model}-20261010', 'provider': prov,
            'answers': {}, 'usage': {'cost': cost}, 'cost': cost,
            'valid': valid, 'final': True, 'error': err}


def ops_items(n):
    it = {f'E1|S{i}|Ca': {'experiment': 'E1', 'arm': 'Ca'}
          for i in range(n)}
    it['E0|S0|W'] = {'experiment': 'E0', 'arm': 'W'}
    it['E0|S0|PP'] = {'experiment': 'E0', 'arm': 'PP'}
    return it


def test_g1_operations_modal_share_pin_probe_and_latency():
    items = ops_items(100)
    ids = [i for i in items if i.startswith('E1')]
    m = 'x/m'
    f = {i: call(m, i, prov='P1' if k < 95 else 'P2')
         for k, i in enumerate(ids)}
    f['E0|S0|PP'] = call(m, 'E0|S0|PP', valid=False, err='pin unsupported')
    f['E0|S0|W'] = call(m, 'E0|S0|W', lat=90000)     # a cold start
    o = bs.model_ops(f, items, GATES['G1'])
    assert o['modal_share'] == .95 and o['provider_share'] == .95
    assert o['pin_probe'] == 'pin unsupported' and o['valid_rate'] == 1.0
    assert o['p95_latency_s'] == .5 and o['pass_ops']   # W left out
    split = {i: call(m, i, prov='P1' if k % 2 else 'P2')
             for k, i in enumerate(ids)}
    assert not bs.model_ops(split, items, GATES['G1'])['pass_ops']
    bad = dict(f, **{ids[0]: call(m, ids[0], valid=False),
                     ids[1]: call(m, ids[1], valid=False),
                     ids[2]: call(m, ids[2], valid=False)})
    o = bs.model_ops(bad, items, GATES['G1'])
    assert o['valid_rate'] == .97 and not o['pass_ops']
    slow = {i: call(m, i, lat=61000) for i in ids}
    assert not bs.model_ops(slow, items, GATES['G1'])['pass_ops']
    dear = {i: call(m, i, cost=.02) for i in ids}
    assert not bs.model_ops(dear, items, GATES['G1'])['pass_ops']
    smoke = dict(f, **{ids[0]: call(m, ids[0], valid=False,
                                    err='smoke stop')})
    assert bs.model_ops(smoke, items, GATES['G1'])['smoke_stopped']


# ------------------------------------------- a whole run, end to end --

R5 = math.sqrt(5)
POOL = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']
VOL = dict(zip(POOL, [200, 250, 300, 180, 220, 260, 240]))
NEG_HOLD = 'Should the long position in `{n}` be left out tonight?'
NEG_HEDGE = 'Should tonight\'s whole basket be left unhedged?'


def _body(state, names, **kw):
    b = guard.request_body(state, names, None, **kw)
    del b['model']
    return b


def _item(e, s, arm, block, body, qmap=None, meta=None):
    return {'item_id': f'{e}|{s}|{arm}', 'experiment': e, 'arm': arm,
            'state_id': s, 'block': block, 'body': body,
            'body_sha256': hashlib.sha256(json.dumps(body).encode())
            .hexdigest(),
            'qmap': qmap or {k: k for k in body['questions']},
            'meta': meta or {}}


def _canon(rows, labels, qqq):
    return {'schema': 2, 'date': '20261013', 'book': 'long overnight',
            'names': {t: dict(rows[t]) for t in labels}, 'QQQ': dict(qqq)}


def _negate(body):
    for k, q in body['questions'].items():
        q['instructions'] = NEG_HEDGE if k == 'hedge_book' else \
            NEG_HOLD.format(n=k[len('hold_'):])
    return body


def build_plan(d, models, incumbent, reference, X):
    """3 blocks x 4 states (2 Stressed, 2 Calm), 7 names each, every
    experiment's items per the contract, and S0's."""
    states, items = {}, []
    k = 0
    for b in (1, 2, 3):
        for pos in range(4):
            k += 1
            sid = f'S{k:02d}'
            reg = 'stressed' if pos in (0, 2) else 'calm'
            lab = POOL[k % 7:] + POOL[:k % 7]
            cond = ['Ordinary'] * 4 + ['Earnings', 'ExtremeDown' if k % 2
                                       else 'ExtremeUp', 'Moderate']
            cond = dict(zip(lab, cond[k % 3:] + cond[:k % 3]))
            f = 1.5 if reg == 'stressed' else 1.0
            rows = {}
            for i, t in enumerate(lab):
                v = int(round(VOL[t] * f))
                r = {'last_night': (i * 7 + k) % 40 - 20, 'nights5_avg': 3,
                     'ret5d': (i * 13 + k) % 200 - 100, 'vol20d': v,
                     'earnings_tonight': cond[t] == 'Earnings'}
                if cond[t] == 'ExtremeDown':
                    r['last_night'] = -int(round(4 * .5 * v))
                elif cond[t] == 'ExtremeUp':
                    r['ret5d'] = int(round(4 * v * R5))
                elif cond[t] == 'Moderate':
                    r['last_night'] = int(round(2 * .5 * v))
                rows[t] = r
            m, sg = 2.0 + .07 * k, 1 if k % 2 else -1
            qqq = {'calm': {'last_night': 5, 'nights5_avg': 1, 'ret5d': 30,
                            'vol20d': 96},
                   'stressed': {'last_night': sg * 300, 'nights5_avg': sg * 60,
                                'ret5d': sg * 900,
                                'vol20d': int(round(96 * m))}}
            core = _canon(rows, lab, qqq[reg])
            cx = [1, 2, 3, 1][pos]
            e7 = ['LD', 'AN', 'AN', 'LD'][pos]
            states[sid] = {'block': b, 'regime': reg, 'labels': lab,
                           'conditions': cond, 'rows': rows, 'qqq': qqq,
                           'stressed_m': m, 'stressed_sign': sg,
                           'cx_type': cx, 'e7_kind': e7, 'e8': pos in (0, 3),
                           'state': core}
            qids = [f'hold_{t}' for t in lab] + ['hedge_book']
            for arm in ('Ca', 'Cb'):
                items.append(_item('E1', sid, arm, b, _body(core, lab)))
            for i in range(1, 7):
                o = guard.question_order(f'battery-v1|{sid}|O{i}', qids)
                items.append(_item('E1', sid, f'O{i}', b,
                                   _body(core, lab, order=o)))
            order = qids[::-1]
            ids = {q: f'q{(k * 97 + j):08x}' for j, q in enumerate(qids)} \
                if k % 2 else None
            v = _body(dict(core, date='20261014'), lab, order=order, ids=ids)
            items.append(_item('E2', sid, 'V', b, v,
                               {(ids or {}).get(q, q): q for q in order},
                               {'factors': {'opaque': bool(ids),
                                            'keyshuffle': False,
                                            'jitter': False, 'date_shift': 1,
                                            'paraphrase': k % 3}}))
            ask = {1: qids[:3] + ['hedge_book'], 2: [qids[1]],
                   3: ['hedge_book']}[cx]
            cb = _body(core, [q[5:] for q in ask if q != 'hedge_book'])
            if 'hedge_book' not in ask:
                del cb['questions']['hedge_book']
            items.append(_item('E3', sid, 'CX', b, cb,
                               meta={'cx_type': cx, 'cx_subset': ask}))
            ords = [t for t in lab if cond[t] == 'Ordinary']
            st = _canon(rows, lab, qqq[reg])
            st['names'][ords[0]]['earnings_tonight'] = True
            items.append(_item('E4', sid, 'EF', b, _body(st, lab),
                               meta={'treated': ords[0]}))
            j, dose, sign = ords[1], [1.5, 3.0, 6.0][k % 3], sg
            st = _canon(rows, lab, qqq[reg])
            r = st['names'][j]
            new = sign * int(round(dose * r['vol20d']))
            D = new - r['last_night']
            r.update(last_night=new, nights5_avg=r['nights5_avg']
                     + int(round(D / 5)), ret5d=r['ret5d'] + D,
                     vol20d=int(round(math.sqrt(r['vol20d'] ** 2
                                                + D ** 2 / 19))))
            items.append(_item('E5', sid, 'XM', b, _body(st, lab),
                               meta={'treated': j, 'dose': dose,
                                     'sign': sign, 'D': D}))
            other = 'calm' if reg == 'stressed' else 'stressed'
            items.append(_item('E6', sid, 'QS', b,
                               _body(_canon(rows, lab, qqq[other]), lab),
                               meta={'core_regime': reg,
                                     'swapped_in': other}))
            if e7 == 'LD':
                perm = {lab[i]: lab[(i + 1) % 7] for i in range(7)}
                items.append(_item('E7', sid, 'LD', b, _body(_canon(
                    {t: rows[perm[t]] for t in lab}, lab, qqq[reg]), lab),
                    meta={'perm': perm}))
            else:
                anon = {t: f'S{7 - i:02d}' for i, t in enumerate(lab)}
                names = [anon[t] for t in lab]
                qm = {f'hold_{anon[t]}': f'hold_{t}' for t in lab}
                qm['hedge_book'] = 'hedge_book'
                items.append(_item('E7', sid, 'AN', b, _body(_canon(
                    {anon[t]: rows[t] for t in lab}, names, qqq[reg]),
                    names), qm, {'anon': anon}))
            if states[sid]['e8']:
                items.append(_item('E8', sid, 'NG', b,
                                   _negate(_body(core, lab)),
                                   meta={'negated': True}))
    names = POOL[:5]
    s0st = _canon({t: {'last_night': -30 + 9 * i, 'nights5_avg': -4,
                       'ret5d': 60 - 20 * i, 'vol20d': VOL[t],
                       'earnings_tonight': False}
                   for i, t in enumerate(names)}, names,
                  {'last_night': -20, 'nights5_avg': -3, 'ret5d': -80,
                   'vol20d': 96})
    s0 = {'date': '20261013', 'names': names, 'state': s0st}
    qids = [f'hold_{t}' for t in names] + ['hedge_book']
    canon = _body(s0st, names)
    items += [_item('E0', 'S0', 'W', 0, canon),
              _item('E0', 'S0', 'PP', 0,
                    _body(s0st, names, provider='__FROM_W__'))]
    for i, q in enumerate(qids):
        bb = _body(s0st, [q[5:]] if q != 'hedge_book' else [])
        if q != 'hedge_book':
            del bb['questions']['hedge_book']
        items.append(_item('E3', 'S0', f'alone_{q}', i % 3 + 1, bb,
                           meta={'cx_type': 'alone', 'cx_subset': [q]}))
    items.append(_item('E8', 'S0', 'NG', 2, _negate(_body(s0st, names)),
                       meta={'negated': True}))
    for r in (1, 2, 3):
        items.append(_item('E9', 'S0', f'S0rep{r}', r, canon))
    for i in (1, 2, 3):
        items.append(_item('E9', 'S0', f'B2B{i}', 1, canon,
                           meta={'group': 'B2B', 'seq': i}))
    for j in range(3):
        o = guard.question_order(f'20261013|abc|{j}|order-v1|http', qids)
        for c in 'ab':
            items.append(_item('E9', 'S0', f'S0O{j}{c}', j + 1,
                               _body(s0st, names, order=o)))
    items.append(_item('E9', 'S0', 'S0V', 2,
                       _body(dict(s0st, date='20261015'), names,
                             order=qids[::-1]),
                       meta={'factors': {'date_shift': 2}}))
    plan = plan_dict(incumbent=incumbent, reference=reference,
                     candidates=[{'id': m, 'price_in': 2e-8, 'X': X.get(m)}
                                 for m in models])
    os.makedirs(d, exist_ok=True)
    for name, obj in (('plan_v1.json', plan), ('states_v1.json', states),
                      ('s0_20261013.json', s0)):
        with open(os.path.join(d, name), 'w') as f:
            json.dump(obj, f, indent=1)
    with open(os.path.join(d, 'items_v1.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    write_lock(d)
    return os.path.join(d, 'plan_v1.json'), items


def write_lock(d):
    lock = {k: hashlib.sha256(open(os.path.join(d, f), 'rb').read())
            .hexdigest() for k, f in (('plan', 'plan_v1.json'),
                                      ('states', 'states_v1.json'),
                                      ('items', 'items_v1.jsonl'),
                                      ('s0', 's0_20261013.json'))}
    with open(os.path.join(d, 'lock_v1.json'), 'w') as f:
        json.dump(lock, f)


def stand_in(kind, model, item):
    """Answers from the bytes sent, as a hosted model would: each question's
    instructions name its ticker; a negated one is answered the other way
    round by every kind that reads the text."""
    body, out = item['body'], {}
    for sent, q in body['questions'].items():
        st = body['state']
        m = re.search(r'`([^`]+)`', q['instructions'])
        neg = 'left out' in q['instructions'] or 'left unhedged' in \
            q['instructions']
        if m:
            r = st['names'][m.group(1)]
            crit = r['earnings_tonight'] or abs(r['last_night']) >= \
                1.5 * r['vol20d'] or abs(r['ret5d']) >= 3 * r['vol20d'] * R5
            p = .1 + abs(r['ret5d']) / 1e6 if crit else \
                .9 - abs(r['last_night']) / 1e4
        else:
            p = .9 if st['QQQ']['vol20d'] >= 150 else .1
        if kind == 'const':
            p, neg = .5, False
        elif kind == 'rule':
            p = (0. if st['names'][m.group(1)]['earnings_tonight'] else 1.) \
                if m else 0.
            neg = False
        elif kind == 'noisy':
            h = int(hashlib.sha256(f'{model}|{item["item_id"]}|{sent}'
                                   .encode()).hexdigest()[:8], 16)
            p = round(min(1., max(0., p + np.random.default_rng(h)
                                  .normal(0, .03))), 2)
        out[sent] = 1 - p if neg else p
    return out


KINDS = {'fake/oracle': 'oracle', 'fake/noisy': 'noisy',
         'fake/const': 'const', 'fake/rule': 'rule'}


def write_run(run, plan_path, items, skip=lambda m, it: False,
              status='done', **meta):
    os.makedirs(run, exist_ok=True)
    with open(os.path.join(run, 'calls.jsonl'), 'w') as f:
        for m, kind in KINDS.items():
            for it in items:
                if skip(m, it):
                    continue
                c = call(m, it['item_id'])
                c.update(experiment=it['experiment'],
                         state_id=it['state_id'], block=it['block'],
                         answers=stand_in(kind, m, it))
                f.write(json.dumps(c) + '\n')
        f.write('{"key": "torn')              # a crash mid-line is skipped
    with open(os.path.join(run, 'run_meta.json'), 'w') as f:
        json.dump(dict({'plan': plan_path, 'status': status,
                        'scale_down': {'dropped_blocks': [],
                                       'dropped_models': []}}, **meta), f)


@pytest.fixture(scope='module')
def scored(tmp_path_factory):
    d = tmp_path_factory.mktemp('battery')
    plan_path, items = build_plan(str(d / 'plan'), list(KINDS), 'fake/const',
                                  'fake/noisy', {'fake/noisy': .9,
                                                 'fake/oracle': .5})
    run = str(d / 'run')
    write_run(run, plan_path, items)
    return run, plan_path, items, bs.score(run)


def test_score_reads_every_experiment_and_the_baselines(scored):
    run, _, _, s = scored
    assert s['analysis_set']['blocks'] == [1, 2, 3]
    assert s['analysis_set']['sufficient'] and s['lock'] == 'ok'
    b = s['baselines']
    # the rule skips the Earnings name (1 of 2 skip-criteria per state),
    # never skips an Ordinary name, never hedges
    assert b['Q_rule'] == pytest.approx(.25 * (.5 + 1 + 0 + 1))
    assert b['constant_0.5'] == .5 and b['skip_nothing_always_hedge'] == .5
    assert b['criteria_oracle'] == 1.0
    o = s['models']['fake/oracle']
    assert o['h'] == 'k1' and o['headline']['Q_h'] == 1.0
    assert o['failed'] == []
    assert o['E4']['E_bar'] < -.7 and o['E4']['frac_neg'] == 1
    assert o['E4']['p_signflip'] == 1 / 2 ** 12
    assert o['E6']['G_bar'] == pytest.approx(.8, abs=.01)
    assert o['E7']['Phi'] == pytest.approx(1)
    assert o['E7']['Q_AN'] == o['E7']['Q_Ca_same_states'] == 1.0
    assert o['E2']['ex_nuis'] == 0 and o['E3']['ex_ctx'] == 0
    assert o['E8']['CE'] < 1e-9 and o['E8']['rho_neg'] == pytest.approx(1)
    assert set(o['E5']['by_dose']) == {'1.5', '3.0', '6.0'}
    assert o['E5']['lift_M6'] == 1.0
    assert o['E9']['n_reps'] == 3 and o['E9']['n_b2b'] == 3
    assert o['E9']['agree_rule_k1'] == 1.0
    assert not o['E9']['caching_flag']
    c = s['models']['fake/const']
    assert c['headline']['Q_h'] == .5 and c['noise']['D'] == 0
    assert {'G2a', 'G3a', 'G3b', 'G5'} <= set(c['failed'])
    r = s['models']['fake/rule']
    assert r['headline']['Q_h'] == b['Q_rule']        # exactly
    assert {'G3b', 'G5'} <= set(r['failed'])
    n = s['models']['fake/noisy']
    assert n['h'] == 'k3' and n['noise']['sd_rep'] > .02
    assert n['failed'] == []
    # Holm runs over (models passing G1) x {G3a, G3b}
    assert len(s['holm']) == 8
    with open(os.path.join(run, 'summary.json')) as f:
        assert json.load(f) == s


def test_score_is_reproducible_bytes(scored, tmp_path):
    run = scored[0]
    first = open(os.path.join(run, 'summary.json'), 'rb').read()
    bs.score(run)
    assert open(os.path.join(run, 'summary.json'), 'rb').read() == first


def test_select_end_to_end_jev_like_winner_at_k3(scored):
    run, plan_path, _, s = scored
    sel = bs.select(run)
    # the incumbent (constant) is ineligible: the best eligible model is
    # declared with no margin; oracle and noisy tie on Q, AUC, the lift
    # mean and flip_h, so X decides
    assert sel['branch'] == 'b' and sel['top'] == 'fake/noisy'
    assert sorted(sel['eligible']) == ['fake/noisy', 'fake/oracle']
    d = sel['decider']
    assert d['pin'] == 'fake/noisy' and d['handling'] == 'k3'
    assert d['suffix'] == '#perm3.q1' and d['provider_pin'] is None
    assert sel['reference']['id'] == 'fake/noisy'
    assert sel['reference']['status'] == 'raw'
    assert sel['reference']['suffix'] == ''
    assert sum(sel['sensitivity']['P_rank1'].values()) == pytest.approx(1)
    assert sel['summary_sha256'] == hashlib.sha256(
        open(os.path.join(run, 'summary.json'), 'rb').read()).hexdigest()
    assert os.path.exists(os.path.join(run, 'selection_v1.json'))


def test_select_rescores_a_summary_older_than_the_calls(scored, tmp_path):
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'resumed')
    write_run(run, plan_path, items, skip=lambda m, it: m == 'fake/oracle'
              and it['item_id'] == 'E4|S01|EF')
    assert bs.score(run)['models']['fake/oracle']['E4']['n_states'] == 11
    with open(os.path.join(run, 'calls.jsonl'), 'a') as f:    # the resume
        it = next(i for i in items if i['item_id'] == 'E4|S01|EF')
        c = call('fake/oracle', it['item_id'])
        c['answers'] = stand_in('oracle', 'fake/oracle', it)
        f.write('\n' + json.dumps(c) + '\n')
    bs.select(run)
    s = json.load(open(os.path.join(run, 'summary.json')))
    assert s['models']['fake/oracle']['E4']['n_states'] == 12


def test_a_truncated_run_has_too_few_blocks_and_select_exits_6(scored,
                                                               tmp_path):
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'trunc')
    write_run(run, plan_path, items, skip=lambda m, it: m == 'fake/oracle'
              and it['block'] == 3 and it['state_id'] != 'S0')
    s = bs.score(run)
    assert s['analysis_set']['blocks'] == [1, 2]
    assert not s['analysis_set']['sufficient']
    with pytest.raises(SystemExit) as e:
        bs.select(run)
    assert e.value.code == 6
    assert not os.path.exists(os.path.join(run, 'selection_v1.json'))


def test_only_the_modal_build_and_provider_is_analysed(scored, tmp_path):
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'split')
    write_run(run, plan_path, items)
    # one state's C^a from another provider, answering nonsense
    with open(os.path.join(run, 'calls.jsonl'), 'a') as f:
        c = call('fake/oracle', 'E1|S01|Ca', prov='P9')
        c['answers'] = {q: 0.0 for q in stand_in('oracle', 'x', next(
            it for it in items if it['item_id'] == 'E1|S01|Ca'))}
        f.write('\n' + json.dumps(c) + '\n')
    s = bs.score(run)
    o = s['models']['fake/oracle']['ops']
    assert o['provider'] == 'P1' and o['modal_share'] < 1
    assert s['models']['fake/oracle']['headline']['Q_h'] == 1.0


def test_a_lock_mismatch_is_refused(scored, tmp_path):
    run, plan_path, _, _ = scored
    lock = os.path.join(os.path.dirname(plan_path), 'lock_v1.json')
    good = open(lock, 'rb').read()
    try:
        with open(lock, 'w') as f:
            json.dump({'items': '0' * 64}, f)
        with pytest.raises(ValueError, match='lock mismatch'):
            bs.score(run)
        os.remove(lock)               # a run never goes without its lock
        with pytest.raises(ValueError, match='no lock'):
            bs.score(run)
    finally:
        with open(lock, 'wb') as f:
            f.write(good)


def test_a_run_is_scored_only_against_the_bytes_it_recorded(scored,
                                                           tmp_path):
    # items regenerated and re-locked after the run started: the lock and
    # the files agree, but not with what run_meta recorded
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'other')
    write_run(run, plan_path, items, items_sha='0' * 64)
    with pytest.raises(ValueError, match='items do not match the run'):
        bs.score(run)
    write_run(run, plan_path, items, shas={'lock': '0' * 64})
    with pytest.raises(ValueError, match='lock do not match the run'):
        bs.score(run)


@pytest.mark.parametrize('status,last,ok', [
    ('done', 8, True), ('capped', 12, True), ('capped', 8, False),
    ('running', 8, False), ('deadline', 12, False), ('no_credit', 12, False),
    ('outage', 8, False), ('incomplete', 8, False), (None, 8, False)])
def test_only_a_finished_run_is_scored(scored, tmp_path, status, last, ok):
    # a crash, a deadline or a cap below the ceiling is resumed first:
    # nobody reads a partial result and then decides whether to extend
    plan = {'spend': {'max_usd': 8, 'max_usd_ceiling': 12}}
    meta = {'status': status, 'sessions': [{'max_usd': 8},
                                           {'max_usd': last}]}
    assert bs.finished(meta, plan)[0] is ok
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'run')
    write_run(run, plan_path, items, status=status)
    if status == 'done':
        assert bs.select(run)['run']['status'] == 'done'
    elif status != 'capped':
        with pytest.raises(ValueError, match='not finished'):
            bs.score(run)
        with pytest.raises(ValueError, match='not finished'):
            bs.select(run)
        assert not os.path.exists(os.path.join(run, 'summary.json'))


def test_select_never_replaces_an_outcome_of_another_summary(scored,
                                                            tmp_path):
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'run')
    write_run(run, plan_path, items)
    first = bs.select(run)
    assert bs.select(run) == first                   # the same: rewritten
    with open(os.path.join(run, 'calls.jsonl'), 'a') as f:
        it = next(i for i in items if i['item_id'] == 'E4|S01|EF')
        c = call('fake/oracle', it['item_id'], prov='P2')
        c['answers'] = stand_in('oracle', 'fake/oracle', it)
        f.write('\n' + json.dumps(c) + '\n')
    with pytest.raises(ValueError, match='never replaced'):
        bs.select(run)
    sel = json.load(open(os.path.join(run, 'selection_v1.json')))
    assert sel['summary_sha256'] == first['summary_sha256']


def test_completion_counts_s0_only_where_it_was_dispatched(scored, tmp_path):
    # a run that stopped after block 2: S0's items dispatched in block 3
    # never went out, and are not held against anyone's completion
    _, plan_path, items, _ = scored
    run = str(tmp_path / 'stopped')
    write_run(run, plan_path, items, skip=lambda m, it: it['block'] == 3)
    s = bs.score(run)
    assert s['analysis_set']['blocks'] == [1, 2]
    assert any(it['state_id'] == 'S0' and it['block'] == 3
               and it['experiment'] != 'E0' for it in items)
    for m, r in s['models'].items():
        assert r['ops']['completion'] == 1.0, m


# ------------------------------------------------------ select's branches --

STRATA = ['stressed'] * 4 + ['calm'] * 4


def qrows(n=8, ord_skip=0, ord_n=10, crit=(2, 2), first=None):
    """Per-state Q counts: every criteria name skipped (crit = skips, n),
    Stressed states hedged in both sets, Calm never."""
    out = []
    for i in range(n):
        cs, cn = first if first and i == 0 else crit
        out.append([cs, cn, ord_skip, ord_n, 2 if i < n // 2 else 0, 2,
                    cs, cn])
    return out


def model(rows, h='k1', failed=(), auc=.9, tb=.5, flip=0., cost=1e-4,
          provider_share=1., probe='ok'):
    st = [s == 'stressed' for s in STRATA]
    q = float(bs.q_value(rows, st)[0])
    return {'h': h, 'failed': list(failed),
            'gates': {g: g not in failed for g in bs.GATE_NAMES},
            'headline': {'h': h, 'Q_h': q, 'Q_k1': q, 'Q_k3': q,
                         'Q_noE': q, 'AUC_comb': auc, 'flip_h': flip,
                         'tiebreak_mean': tb, 'median_cost_usd': cost},
            'q_counts': {'k1': rows, 'k3': rows},
            'ops': {'build': 'b-20261010', 'provider': 'P1',
                    'modal_share': 1., 'provider_share': provider_share,
                    'pin_probe': probe},
            'E7': {'Phi': 1.}, 'E9': {'decisions_h': [{'A': 'hold'}]}}


def summary(models, sufficient=True):
    return {'analysis_set': {'blocks': [1, 2, 3], 'states': list(range(8)),
                             'strata': STRATA, 'sufficient': sufficient,
                             'blocks_min': 3},
            'holm': {}, 'models': models, 'plan_sha256': 'x'}


def splan(*ids, I='inc/m', R='ref/m', X=None):
    return plan_dict(incumbent=I, reference=R,
                     candidates=[{'id': i, 'X': (X or {}).get(i)}
                                 for i in ids])


def test_the_incumbent_stays_unless_margin_and_bound_both_clear():
    p = splan('inc/m', 'ch/w', 'ref/m')
    ref = model(qrows(ord_skip=5))
    # a uniform 0.05 lead: margin >= 0.03 and the bound is 0.05 > 0
    s = bs.choose(summary({'inc/m': model(qrows(ord_skip=3)),
                           'ch/w': model(qrows(ord_skip=1)),
                           'ref/m': ref}), p)
    assert s['branch'] == 'a' and s['switch_test']['switched']
    assert s['decider']['pin'] == 'ch/w'
    assert s['switch_test']['alpha'] == pytest.approx(.05 / 2)
    assert s['switch_test']['lower_bound'] == pytest.approx(.05)
    # a uniform 0.02 lead: the bound is > 0 but the margin is short
    s = bs.choose(summary({'inc/m': model(qrows(ord_skip=3, ord_n=25)),
                           'ch/w': model(qrows(ord_skip=1, ord_n=25)),
                           'ref/m': ref}), p)
    assert s['switch_test']['margin'] == pytest.approx(.02)
    assert s['switch_test']['lower_bound'] > 0
    assert not s['switch_test']['switched']
    assert s['decider']['pin'] == 'inc/m'
    # a 0.052 lead that lives in one state: the bound is 0, I stays
    s = bs.choose(summary({'inc/m': model(qrows(crit=(1, 2), first=(5, 10))),
                           'ch/w': model(qrows(crit=(1, 2), first=(10, 10))),
                           'ref/m': ref}), p)
    assert s['switch_test']['margin'] >= .03
    assert s['switch_test']['lower_bound'] <= 0
    assert s['decider']['pin'] == 'inc/m'


def test_an_ineligible_incumbent_is_replaced_without_a_margin():
    s = bs.choose(summary({
        'inc/m': model(qrows(ord_skip=0), failed=['G5']),
        'ch/w': model(qrows(ord_skip=1), h='k3'),
        'ch/v': model(qrows(ord_skip=2)),
        'ref/m': model(qrows(ord_skip=3))}),
        splan('inc/m', 'ch/w', 'ch/v', 'ref/m'))
    assert s['branch'] == 'b' and s['switch_test'] is None
    assert s['decider'] == dict(s['decider'], pin='ch/w', handling='k3',
                                suffix='#perm3.q1', version='q1')
    assert [r['id'] for r in s['ranking']] == ['ch/w', 'ch/v', 'ref/m']
    assert s['reference']['status'] == 'raw'


def test_nothing_eligible_keeps_the_incumbent_at_k1_q1():
    s = bs.choose(summary({
        'inc/m': model(qrows(), h='k3', failed=['G3a', 'G5']),
        'ref/m': model(qrows(), failed=['G6'])}), splan('inc/m', 'ref/m'))
    assert s['branch'] == 'none' and s['eligible'] == []
    assert s['decider']['pin'] == 'inc/m'
    assert s['decider']['handling'] == 'k1' and s['decider']['suffix'] == ''
    assert s['decider']['provider_pin'] is None
    assert s['results_note'] == 'no candidate qualified'
    assert s['failed_gates'] == {'inc/m': ['G3a', 'G5'], 'ref/m': ['G6']}
    assert s['reference']['status'] == 'raw'


def test_jev_winning_at_k3_keeps_the_raw_reference_at_k1_retires_it():
    p = splan('inc/m', 'ref/m')
    for h, status, suffix in (('k3', 'raw', '#perm3.q1'),
                              ('k1', 'retired', '')):
        s = bs.choose(summary({'inc/m': model(qrows(), failed=['G2a']),
                               'ref/m': model(qrows(), h=h)}), p)
        assert s['decider']['pin'] == 'ref/m'
        assert s['decider']['suffix'] == suffix
        assert s['reference']['status'] == status
        assert s['reference']['suffix'] == ''


def test_a_reference_failing_g1_is_dropped():
    s = bs.choose(summary({'inc/m': model(qrows()),
                           'ref/m': model(qrows(), failed=['G1'])}),
                  splan('inc/m', 'ref/m'))
    assert s['reference']['status'] == 'dropped'
    assert s['decider']['pin'] == 'inc/m'


def test_provider_pin_only_when_split_and_the_probe_passed():
    for share, probe, pinned in ((.95, 'ok', True), (1., 'ok', False),
                                 (.95, 'pin unsupported', False)):
        s = bs.choose(summary({
            'inc/m': model(qrows(), provider_share=share, probe=probe),
            'ref/m': model(qrows(ord_skip=3))}), splan('inc/m', 'ref/m'))
        assert (s['decider']['provider_pin'] ==
                {'order': ['P1'], 'allow_fallbacks': False}) == pinned


@pytest.mark.parametrize('better, worse, X', [
    (dict(auc=.95, tb=0, flip=.09, cost=1), dict(auc=.9), {}),
    (dict(tb=.6, flip=.09, cost=1), dict(tb=.5), {}),
    (dict(flip=0.01, cost=1), dict(flip=.02), {'m/a': 0, 'm/b': 1}),
    (dict(cost=1), dict(cost=1), {'m/a': .2, 'm/b': .1}),
    (dict(cost=1), dict(cost=1), {'m/a': 0., 'm/b': None}),  # unscored last
    (dict(cost=.001), dict(cost=.002), {}),
    (dict(cost=None), dict(cost=None), {}),                   # then the id
])
def test_the_tie_break_chain_is_followed_in_order(better, worse, X):
    rows = qrows(ord_skip=2)
    ms = {'m/b': model(rows, **worse), 'm/a': model(rows, **better)}
    assert bs.rank(['m/b', 'm/a'], ms, X) == ['m/a', 'm/b']
    # Q_h comes first of all: a hair more Q beats every tie-break
    ms['m/b'] = model(qrows(ord_skip=1), **worse)
    assert bs.rank(['m/b', 'm/a'], ms, X) == ['m/b', 'm/a']


def test_p_rank1_sums_to_one_and_never_changes_the_decision():
    def lead(i):                    # better than the rest in state i only
        r = qrows(ord_skip=3)
        r[i][2] = 0
        return r
    ms = {'m/a': model(lead(0), auc=.9), 'm/b': model(lead(1), auc=.8),
          'm/c': model(lead(5), auc=.7)}
    s = bs.choose(summary(ms), splan(*ms, I='m/c', R='m/b'))
    p1 = s['sensitivity']['P_rank1']
    assert set(p1) == set(ms) and sum(p1.values()) == pytest.approx(1)
    assert all(0 < v < 1 for v in p1.values())
    # equal Q, so AUC_comb ranks m/a first; a tie in a draw goes the same way
    assert s['top'] == 'm/a' and p1['m/a'] == max(p1.values())
    assert s['decider']['pin'] == 'm/c'             # no margin: I stays
    assert s['sensitivity']['winner_AUC_comb'] == 'm/a'


@pytest.mark.parametrize('bad', ['~perplexity/pplx-decider-27b-latest',
                                 'inception/mercury-decide:free'])
def test_moving_aliases_and_free_routes_are_refused(bad):
    ms = {'inc/m': model(qrows()), 'ref/m': model(qrows())}
    with pytest.raises(ValueError, match='refused'):
        bs.choose(summary(ms), splan('inc/m', 'ref/m', bad))
    with pytest.raises(ValueError, match='refused'):
        bs.choose(summary(dict(ms, **{bad: model(qrows())})),
                  splan('inc/m', 'ref/m'))
    with pytest.raises(ValueError, match='refused'):
        bs.choose(summary(ms), splan('ref/m', I=bad))


def test_an_insufficient_analysis_set_exits_6():
    with pytest.raises(SystemExit) as e:
        bs.choose(summary({'inc/m': model(qrows()),
                           'ref/m': model(qrows())}, sufficient=False),
                  splan('inc/m', 'ref/m'))
    assert e.value.code == bs.EXIT_INSUFFICIENT == 6


def test_iteration_adopts_the_best_variant_that_clears_every_bar():
    q = lambda k: qrows(ord_skip=k, ord_n=50)
    ms = {'ctl': model(q(15)), 'v1': model(q(5)),         # +0.05
          'v2': model(q(0), auc=.85),                     # AUC fell .05
          'v3': model(q(13))}                             # +0.01 < 0.02
    p = plan_dict(mode='iterate', control='ctl', variants=['v1', 'v2', 'v3'],
                  round=1)
    o = bs.iterate_outcome(summary(ms), p)
    assert o['adopted'] == 'v1'
    assert not o['checks']['v2']['auc_ok'] and o['checks']['v2']['lb_ok']
    assert not o['checks']['v3']['dq_ok'] and o['checks']['v3']['lb_ok']
    assert o['checks']['v1']['alpha'] == pytest.approx(.05 / 3)
