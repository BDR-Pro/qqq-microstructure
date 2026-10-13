# Part of qqq-microstructure.
#
# The decider battery's scorer and its pre-declared selection (RESULTS 27;
# spec v1, sections 2, 3 and 7). score() turns a run's calls into every
# metric, gate and test the plan declares and writes summary.json; select()
# applies section 3 to those numbers mechanically and writes the decider
# they imply. There is no hand step between the calls and the declaration.
#
# What goes in: the locked plan, states and items, and the run's settled
# calls. Nothing here reads a log, an outcome, a price or the network, so
# the scores are a function of committed bytes and the models' answers.
#
# Why it is built this way:
#   - decisions go through guard.rule, the record's own rule, so Q scores
#     the decisions the runtime would log, not a proxy of them
#   - every random draw (sign flips, bootstraps, permutations) comes from
#     the plan's analysis seed: re-scoring a committed run gives the same
#     bytes
#   - only complete blocks, and only each model's modal (build, provider),
#     are analysed: a model is judged as the one object it would be
#     declared as
#   - the incumbent stays unless a challenger clears a margin AND a
#     Bonferroni bootstrap bound: the top model is the best of up to 17, so
#     its lead is inflated by the selection itself
#
#   python src/battery_score.py score bench/battery/runs/v1
#   python src/battery_score.py select bench/battery/runs/v1

import os, sys, re, json, math, hashlib, datetime as dt
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = guard.BOOK
SKIP_CRITERIA = ('Earnings', 'ExtremeDown', 'ExtremeUp')
EXTREME = ('ExtremeDown', 'ExtremeUp')
GATE_NAMES = ['G1', 'G2a', 'G2b', 'G3a', 'G3b', 'G4', 'G5', 'G6']
EXIT_INSUFFICIENT = 6
# A float computed two ways can miss a threshold it equals by 1e-16; gates
# compare with this slack so a tie with a threshold is decided by the rule
# as written (<=, >=), not by rounding.
SLACK = 1e-12
# rho_b2b well below 1 means back-to-back identical bytes vary less than
# the same bytes spread over the run: a cache. Report-only, never a gate.
CACHE_RHO_FLAG = 0.25
# Q counts per state and decision set: skip-criteria skips and names,
# Ordinary skips and names, hedges and decision sets, Extreme-only skips
# and names (for Q_noE).
QROW = ['crit_skip', 'crit_n', 'ord_skip', 'ord_n', 'hedge', 'sets',
        'ext_skip', 'ext_n']


class InsufficientAnalysisSet(SystemExit):
    """Fewer complete blocks than the plan's floor: select exits 6 and the
    run must be resumed (spec section 7)."""

    def __init__(self, msg):
        super().__init__(EXIT_INSUFFICIENT)
        self.msg = msg


# ---------------------------------------------------------------- loading --

def _json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _sha_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(plan_path):
    """The locked declaration, read from the plan's directory as the
    contract lays it out (battery_items is not imported: the scorer reads
    bytes, whoever wrote them)."""
    d = os.path.dirname(os.path.abspath(plan_path))
    plan = _json(plan_path)
    fs = plan['files']
    with open(os.path.join(d, fs['items']), encoding='utf-8') as f:
        items = [json.loads(ln) for ln in f if ln.strip()]
    return {'plan': plan, 'states': _json(os.path.join(d, fs['states'])),
            'items': items, 's0': _json(os.path.join(d, fs['s0'])), 'dir': d}


def read_calls(run_dir):
    """{key: the LAST final line}. A torn line (a crash mid-write) is
    skipped: that call was never settled."""
    out, path = {}, os.path.join(run_dir, 'calls.jsonl')
    if not os.path.exists(path):
        return out
    with open(path, encoding='utf-8') as f:
        for ln in f:
            try:
                c = json.loads(ln)
            except ValueError:
                continue
            if isinstance(c, dict) and c.get('final'):
                out[c.get('key') or f'{c["model"]}|{c["item_id"]}'] = c
    return out


def plan_of(run_dir, plan_path=None):
    """The plan a run was made from: given, or the one run_meta names."""
    if plan_path:
        return plan_path
    meta = _json(os.path.join(run_dir, 'run_meta.json'))
    p = meta.get('plan') or meta.get('plan_path')
    if isinstance(p, dict):
        p = p.get('path')
    if not p:
        raise ValueError('run_meta.json names no plan; pass plan_path')
    for c in (p, os.path.join(ROOT, p), os.path.join(run_dir, p)):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f'plan {p} named by run_meta.json not found')


def check_lock(plan_path, plan, meta):
    """Refuse to score against bytes other than the ones the run used: the
    plan sha run_meta recorded, and the lock beside the plan."""
    d = os.path.dirname(os.path.abspath(plan_path))
    shas = {'plan': _sha_file(plan_path)}
    for k in ('states', 'items', 's0', 'anchors'):
        p = os.path.join(d, plan['files'].get(k) or '')
        if plan['files'].get(k) and os.path.exists(p):
            shas[k] = _sha_file(p)
    want = (meta or {}).get('plan_sha') or (meta or {}).get('plan_sha256')
    if want and want != shas['plan']:
        raise ValueError('the plan does not match the run (run_meta '
                         'plan_sha) -- refusing to score')
    lock = os.path.join(d, plan['files'].get('lock',
                                             f'lock_{plan["version"]}.json'))
    if not os.path.exists(lock):
        return shas, 'absent'
    lk = _json(lock)
    bad = [k for k in shas if k in lk and lk[k] != shas[k]]
    if bad:
        raise ValueError(f'lock mismatch: {", ".join(bad)} -- refusing to '
                         f'score')
    return shas, 'ok'


def models_of(plan):
    """Every model the plan tests: the candidates, with I and R always in."""
    ids = [c['id'] for c in plan['candidates']]
    return ids + [m for m in (plan['incumbent'], plan['reference'])
                  if m not in ids]


def refuse_aliases(ids):
    """A moving alias (~...) or a :free route never names a record: the
    declaration must be a dated build of a paid route (spec section 3.8)."""
    bad = sorted({i for i in ids if i.startswith('~') or ':free' in i})
    if bad:
        raise ValueError('moving alias or :free id refused: '
                         + ', '.join(bad))


# ------------------------------------------------------------- statistics --

def _ok(x):
    return x is not None and isinstance(x, (int, float, np.floating,
                                            np.integer)) and math.isfinite(x)


def _le(x, t):
    return _ok(x) and x <= t + SLACK


def _ge(x, t):
    return _ok(x) and x >= t - SLACK


def _mean(xs):
    xs = [x for x in xs if _ok(x)]
    return float(np.mean(xs)) if xs else float('nan')


def signflip_p(x, seed, draws, exact_max=20):
    """One-sided sign-flip p for mean(x) > 0 over states: exact enumeration
    of all 2^n sign vectors when n <= exact_max, otherwise `draws` random
    vectors from `seed` with p = (1 + #{T* >= T}) / (1 + draws)."""
    x = np.asarray([v for v in x if _ok(v)], float)
    n = len(x)
    if n == 0:
        return 1.0
    t = float(x.sum())
    tol = 1e-12 * max(1.0, float(np.abs(x).sum()))
    if n <= exact_max:
        s = np.zeros(1)
        for v in x:
            s = np.concatenate([s + v, s - v])
        return float(np.mean(s >= t - tol))
    rng = np.random.default_rng(seed)
    hits, left = 0, int(draws)
    while left:
        k = min(left, 20000)
        sg = rng.integers(0, 2, size=(k, n), dtype=np.int8) * 2 - 1
        hits += int(np.sum(sg @ x >= t - tol))
        left -= k
    return (1 + hits) / (1 + int(draws))


def holm(pdict, alpha):
    """Holm's step-down over a family {key: p}: adjusted p (monotone) and
    reject iff adjusted p < alpha. Ties in p keep key order."""
    keys = sorted(pdict, key=lambda k: (pdict[k], k))
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * float(pdict[k])))
        out[k] = {'p': float(pdict[k]), 'p_holm': run, 'reject': run < alpha}
    return out


def strat_weights(strata, B, seed):
    """(B, n) multiplicities: each stratum's states resampled with
    replacement within the stratum. Drawn once and shared by every model,
    so model differences are paired draw by draw."""
    strata = list(strata)
    rng = np.random.default_rng(seed)
    w = np.zeros((int(B), len(strata)))
    for g in sorted(set(strata)):
        idx = [i for i, s in enumerate(strata) if s == g]
        draw = rng.integers(0, len(idx), size=(int(B), len(idx)))
        for k, i in enumerate(idx):
            w[:, i] = (draw == k).sum(1)
    return w


def q_value(rows, stressed, w=None, noE=False):
    """Q = 1/4 [skip_recall + (1 - false_skip) + hedge_tpr + (1 - hedge_fpr)]
    from per-state counts (QROW), pooled over states and decision sets. w:
    (B, n) bootstrap multiplicities (None: each state once). An array of
    one Q per row of w; a class with no members gives nan."""
    rows = np.asarray(rows, float).reshape(-1, len(QROW))
    s = np.asarray(stressed, bool)
    w = np.ones((1, len(rows))) if w is None else np.asarray(w, float)
    a, S, C = w @ rows, (w * s) @ rows, (w * ~s) @ rows
    with np.errstate(divide='ignore', invalid='ignore'):
        rec = a[:, 6] / a[:, 7] if noE else a[:, 0] / a[:, 1]
        return 0.25 * (rec + 1 - a[:, 2] / a[:, 3] + S[:, 4] / S[:, 5]
                       + 1 - C[:, 4] / C[:, 5])


def paired_bootstrap_lb(rows_a, rows_b, strata, B, seed, alpha, w=None):
    """One-sided lower bound (the alpha quantile) of Q_a - Q_b over a paired
    bootstrap that resamples states within each regime. Pass w to share one
    set of index draws across many comparisons."""
    if w is None:
        w = strat_weights(strata, B, seed)
    s = [x == 'stressed' for x in strata]
    d = q_value(rows_a, s, w) - q_value(rows_b, s, w)
    d = d[np.isfinite(d)]
    return float(np.quantile(d, alpha)) if len(d) else float('nan')


def kendall_tau(x, y):
    """Kendall's tau-b (ties in either variable allowed)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return float('nan')
    i, j = np.triu_indices(len(x), 1)
    dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
    den = math.sqrt(float(np.sum(dx != 0)) * float(np.sum(dy != 0)))
    return float(np.sum(dx * dy) / den) if den else float('nan')


def auc(hi, lo):
    """P(a > b) over all pairs a in hi, b in lo, ties counted 1/2."""
    hi, lo = np.asarray(hi, float), np.asarray(lo, float)
    if not len(hi) or not len(lo):
        return float('nan')
    d = hi[:, None] - lo[None, :]
    return float(np.mean((d > 0) + 0.5 * (d == 0)))


def ex(sq, phi):
    """Excess deviation: sqrt(max(0, mean squared deviation - phi)), phi the
    deviation pure repeat noise alone would give."""
    sq = [v for v in sq if _ok(v)]
    if not sq:
        return float('nan')
    return math.sqrt(max(0.0, float(np.mean(sq)) - (phi if _ok(phi)
                                                     else 0.0)))


def neg_coherence(pairs, B, seed):
    """E8: Pearson rho of the state-demeaned p_C and the state-demeaned
    1 - p_NG, pooled, and its one-sided p from B permutations within each
    state (a state's own values are only ever exchanged among its own
    questions)."""
    xs, ys = [], []
    for pc, ng in pairs:
        x, y = np.asarray(pc, float), 1 - np.asarray(ng, float)
        if len(x) > 1:
            xs.append(x - x.mean())
            ys.append(y - y.mean())
    if not xs:
        return float('nan'), float('nan')
    den = math.sqrt(sum(float(x @ x) for x in xs)
                    * sum(float(y @ y) for y in ys))
    if den == 0:
        return float('nan'), float('nan')
    rho = sum(float(x @ y) for x, y in zip(xs, ys)) / den
    rng = np.random.default_rng(seed)
    num = np.zeros(int(B))
    for x, y in zip(xs, ys):
        perm = np.argsort(rng.random((int(B), len(y))), axis=1)
        num += y[perm] @ x
    hits = int(np.sum(num / den >= rho - 1e-12))
    return rho, (1 + hits) / (1 + int(B))


def phi_ols(y, xf, xt, floor):
    """E7: the pooled no-intercept OLS y = a x_f + b x_t on state-demeaned
    answers, and Phi = a / (|a| + |b|) -- the share of the answer that
    follows the feature row rather than the ticker. Below the floor the
    model follows neither, and Phi is 0."""
    if len(y) < 2:
        return float('nan'), float('nan'), float('nan')
    a, b = map(float, np.linalg.lstsq(np.column_stack([xf, xt]),
                                      np.asarray(y, float), rcond=None)[0])
    return (a / (abs(a) + abs(b)) if abs(a) + abs(b) >= floor else 0.0), a, b


def _slopes(x, y, w):
    """OLS slope of y on x (with intercept) per row of weights w."""
    sw = w.sum(1)
    mx, my = (w @ x) / sw, (w @ y) / sw
    dx = x[None, :] - mx[:, None]
    num = (w * dx * (y[None, :] - my[:, None])).sum(1)
    den = (w * dx * dx).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 1e-15, num / den, np.nan)


# -------------------------------------------------------------- operations --

def _cost(c):
    v = c.get('cost')
    if not _ok(v):
        v = (c.get('usage') or {}).get('cost')
    return float(v) if _ok(v) else None


def model_ops(finals, items, g1):
    """G1's operational numbers over a model's settled calls. E0 is left
    out: W is the warm-up and cost pilot (excluded from every metric), and a
    refused provider pin (PP) is recorded, never held against the model."""
    calls = [c for i, c in finals.items() if items[i]['experiment'] != 'E0']
    valid = [c for c in calls if c.get('valid')]
    pairs = Counter((str(c.get('answered_as')), str(c.get('provider')))
                    for c in valid)
    # the modal (build, provider); an exact tie goes to the smaller pair
    modal = max(sorted(pairs), key=pairs.get) if pairs else None
    lat = [c['latency_ms'] / 1000 for c in valid
           if _ok(c.get('latency_ms'))]
    cost = [x for x in (_cost(c) for c in valid) if x is not None]
    pp = [c for i, c in finals.items() if items[i]['experiment'] == 'E0'
          and items[i]['arm'] == 'PP']
    w = [c for i, c in finals.items() if items[i]['experiment'] == 'E0'
         and items[i]['arm'] == 'W']
    r = {'attempted': len(calls), 'valid': len(valid),
         'valid_rate': len(valid) / len(calls) if calls else float('nan'),
         'build': modal[0] if modal else None,
         'provider': modal[1] if modal else None,
         'modal_share': pairs[modal] / len(valid) if modal else float('nan'),
         'provider_share': (sum(c.get('provider') == modal[1] for c in valid)
                            / len(valid)) if modal else float('nan'),
         'pairs': {f'{b}@{p}': n for (b, p), n in sorted(pairs.items())},
         'p95_latency_s': float(np.percentile(lat, 95)) if lat
         else float('nan'),
         'median_cost_usd': float(np.median(cost)) if cost else None,
         'pin_probe': ('ok' if pp[0].get('valid') else 'pin unsupported')
         if pp else 'not run',
         'pin_probe_status': pp[0].get('http_status') if pp else None,
         'W_valid': bool(w[0].get('valid')) if w else None,
         'W_cost_usd': _cost(w[0]) if w else None,
         'smoke_stopped': any('smoke stop' in str(c.get('error') or '')
                              for c in finals.values())}
    # an unreported cost is not evidence of a high one (the median is over
    # the calls that report it); it is flagged instead
    r['cost_unmeasured'] = r['median_cost_usd'] is None
    r['pass_ops'] = bool(
        _ge(r['valid_rate'], g1['valid_rate_min'])
        and _ge(r['modal_share'], g1['modal_share_min'])
        and _le(r['p95_latency_s'], g1['p95_latency_s_max'])
        and (r['cost_unmeasured']
             or _le(r['median_cost_usd'], g1['median_cost_usd_max']))
        and not r['smoke_stopped'])
    return r


def pdict(call, item):
    """{ticker or BOOK: p} for one call: sent ids mapped back to canonical
    ids through the item's qmap (opaque ids, anonymised labels)."""
    out = {}
    for sent, p in (call.get('answers') or {}).items():
        cid = item['qmap'].get(sent)
        if cid is None or not _ok(p):
            continue
        out[BOOK if cid == 'hedge_book' else cid[len('hold_'):]] = float(p)
    return out


def mean_p(ps):
    """The per-question mean of the given answer sets (missing ones
    skipped), over the questions every one of them answered."""
    ps = [p for p in ps if p]
    if not ps:
        return None
    return {k: float(np.mean([p[k] for p in ps])) for k in ps[0]
            if all(k in p for p in ps)}


def qrow(decisions, st):
    """One state's Q counts (QROW) over its decision sets."""
    r = np.zeros(len(QROW))
    for d in decisions:
        for t, c in st['conditions'].items():
            if t not in d:
                continue
            sk = d[t] == 'skip'
            if c in SKIP_CRITERIA:
                r[0] += sk
                r[1] += 1
            if c in EXTREME:
                r[6] += sk
                r[7] += 1
            if c == 'Ordinary':
                r[2] += sk
                r[3] += 1
        if BOOK in d:
            r[4] += d[BOOK] == 'hedge'
            r[5] += 1
    return r


def did(p, base, j, labels):
    """Within-request DiD: the treated name's move minus the mean move of
    the untreated names, both against the same control."""
    rest = [k for k in labels if k != j and k in p and k in base]
    if j not in p or j not in base or not rest:
        return None
    return (p[j] - base[j]) - float(np.mean([p[k] - base[k] for k in rest]))


# ------------------------------------------------------------ per model --

class Ctx:
    """What every model is scored against: the analysis set's states, the
    locked items, S0 and the plan's numbers."""

    def __init__(self, plan, states, items, s0, A):
        self.plan, self.states, self.s0, self.A = plan, states, s0, A
        self.item = {it['item_id']: it for it in items}
        self.G = plan['gates']
        self.seed = plan['seeds']['analysis']
        self.B = int(plan['bootstrap_B'])
        self.draws = int(plan['signflip_draws'])
        self.exact_max = int(self.G['holm'].get('signflip_exact_max_states',
                                                20))
        self.perms = int(self.G['E8_flag']['permutations'])
        self.stressed = [states[s]['regime'] == 'stressed' for s in A]
        self.date = str(plan.get('date') or s0.get('date'))

    def meta(self, e, s, arm):
        it = self.item.get(f'{e}|{s}|{arm}')
        return it['meta'] if it else {}


def score_model(P, cx):
    """Every section-2 metric for one model. P: {item_id: {ticker or BOOK:
    p}} from its valid calls of the modal (build, provider)."""
    g = lambda e, s, a: P.get(f'{e}|{s}|{a}')
    G = cx.G
    out = {}

    # noise primitives, on the core states
    pc, sq_rep, disp, sq_O = {}, [], [], []
    for s in cx.A:
        st = cx.states[s]
        ca, cb = g('E1', s, 'Ca'), g('E1', s, 'Cb')
        if ca and cb:
            sq_rep += [(ca[x] - cb[x]) ** 2 / 2 for x in ca if x in cb]
        m = mean_p([ca, cb])
        if m:
            pc[s] = m
            v = [m[t] for t in st['labels'] if t in m]
            if len(v) > 1:
                disp.append(float(np.std(v, ddof=1)))
            for i in range(1, 7):
                o = g('E1', s, f'O{i}')
                if o:
                    sq_O += [(o[x] - m[x]) ** 2 for x in o if x in m]
    sd_rep2 = _mean(sq_rep)
    phi = G['handling']['phi_factor'] * sd_rep2 if _ok(sd_rep2) else 0.0
    D = _mean(disp)
    ex_O = ex(sq_O, phi)
    sd_rep = math.sqrt(sd_rep2) if _ok(sd_rep2) else float('nan')
    hg = G['handling']
    # fixed from E1 noise and order alone, before Q is looked at
    h = 'k3' if (_ok(sd_rep) and sd_rep > hg['k3_if_sd_rep_gt']) or \
        (_ok(ex_O) and ex_O > hg['k3_if_ex_O_gt']) else 'k1'
    out['noise'] = {'sd_rep': sd_rep, 'phi': phi, 'D': D, 'ex_O': ex_O,
                    'n_rep_pairs': len(sq_rep), 'n_states': len(pc)}
    out['h'] = h

    # E1: decision sets, flips, Q, AUC, Goodhart diagnostics
    reps, sets = {'k1': {}, 'k3': {}}, {'k1': {}, 'k3': {}}
    for s in cx.A:
        lab = cx.states[s]['labels']
        o = [g('E1', s, f'O{i}') for i in range(1, 7)]
        reps['k1'][s] = [p for p in (g('E1', s, 'Ca'), g('E1', s, 'Cb'))
                         if p]
        reps['k3'][s] = [p for p in (mean_p(o[:3]), mean_p(o[3:])) if p]
        for k in ('k1', 'k3'):
            sets[k][s] = [guard.rule(p, lab) for p in reps[k][s]]
    e1, rows = {}, {}
    for k in ('k1', 'k3'):
        fl = []
        for s in cx.A:
            if len(sets[k][s]) == 2:
                a, b = sets[k][s]
                fl += [a[x] != b[x] for x in a if x in b]
        e1[f'flip_{k}'] = _mean(fl)
        rows[k] = [qrow(sets[k][s], cx.states[s]) for s in cx.A]
        e1[f'Q_{k}'] = float(q_value(rows[k], cx.stressed)[0]) \
            if rows[k] else float('nan')
    e1['flip_h'], e1['Q_h'] = e1[f'flip_{h}'], e1[f'Q_{h}']
    e1['Q_noE'] = float(q_value(rows[h], cx.stressed, noE=True)[0]) \
        if rows[h] else float('nan')
    tot = np.sum(rows[h], 0) if rows[h] else np.zeros(len(QROW))
    S = np.sum([r for r, x in zip(rows[h], cx.stressed) if x], 0) \
        if any(cx.stressed) else np.zeros(len(QROW))
    C = tot - S
    with np.errstate(divide='ignore', invalid='ignore'):
        e1.update(skip_recall=tot[0] / tot[1], false_skip=tot[2] / tot[3],
                  hedge_tpr=S[4] / S[5], hedge_fpr=C[4] / C[5])
    ah, bs, bc = [], [], []
    for s in cx.A:
        st = cx.states[s]
        for p in reps[h][s]:
            ordi = [p[t] for t in st['labels'] if t in p
                    and st['conditions'][t] == 'Ordinary']
            crit = [p[t] for t in st['labels'] if t in p
                    and st['conditions'][t] in SKIP_CRITERIA]
            if ordi and crit:
                ah.append(auc(ordi, crit))
            if BOOK in p:
                (bs if st['regime'] == 'stressed' else bc).append(p[BOOK])
    e1['AUC_hold'], e1['AUC_hedge'] = _mean(ah), auc(bs, bc)
    e1['AUC_comb'] = _mean([e1['AUC_hold'], e1['AUC_hedge']]) \
        if _ok(e1['AUC_hold']) and _ok(e1['AUC_hedge']) else float('nan')
    agree = []
    for s in cx.A:
        lab = cx.states[s]['labels']
        core = cx.item[f'E1|{s}|Ca']['body']['state']
        rd = guard.rule(guard.ask_rule(core, lab)[0], lab)
        for d in sets[h][s]:
            agree += [d[x] == rd[x] for x in d if x in rd]
    e1['agree_rule'] = _mean(agree)
    out['E1'] = e1

    # E2: nuisance invariance, and what each factor adds (reported)
    sq_V, X, Y = [], [], []
    for s in cx.A:
        v = g('E2', s, 'V')
        if not v or s not in pc:
            continue
        d = [(v[x] - pc[s][x]) ** 2 for x in v if x in pc[s]]
        sq_V += d
        f = cx.meta('E2', s, 'V').get('factors', {})
        X.append([1.0, float(bool(f.get('opaque'))),
                  float(bool(f.get('keyshuffle'))),
                  float(bool(f.get('jitter'))),
                  float((f.get('date_shift') or 0) > 0),
                  float((f.get('paraphrase') or 0) != 0)])
        Y.append(_mean(d))
    ex_nuis = ex(sq_O + sq_V, phi)
    fac = None
    if X:
        coef = np.linalg.lstsq(np.asarray(X), np.asarray(Y), rcond=None)[0]
        fac = dict(zip(['intercept', 'opaque', 'keyshuffle', 'jitter',
                        'date_shift', 'paraphrase'], map(float, coef)))
    out['E2'] = {'ex_nuis': ex_nuis,
                 'rel': ex_nuis / D if _ok(D) and D > 0 else float('nan'),
                 'D': D, 'ex_V': ex(sq_V, phi), 'n_states': len(Y),
                 'factor_ols': fac}

    # E3: context independence on the synthetic states; S0 apart
    sq_X, by_t = [], {}
    for s in cx.A:
        c = g('E3', s, 'CX')
        if not c or s not in pc:
            continue
        d = [(c[x] - pc[s][x]) ** 2 for x in c if x in pc[s]]
        sq_X += d
        by_t.setdefault(str(cx.meta('E3', s, 'CX').get('cx_type')),
                        []).extend(d)
    s0c = mean_p([P.get(i) for i, it in cx.item.items()
                  if it['state_id'] == 'S0' and it['arm'].startswith('S0rep')])
    sq_S0 = []
    if s0c:
        for i, it in cx.item.items():
            if it['state_id'] == 'S0' and it['arm'].startswith('alone_') \
                    and P.get(i):
                sq_S0 += [(P[i][x] - s0c[x]) ** 2 for x in P[i] if x in s0c]
    out['E3'] = {'ex_ctx': ex(sq_X, phi), 'n_answers': len(sq_X),
                 'ex_by_cx_type': {t: ex(v, phi) for t, v in
                                   sorted(by_t.items())},
                 'S0': {'rms_alone': math.sqrt(_mean(sq_S0)) if sq_S0
                        else float('nan'), 'ex_alone': ex(sq_S0, phi),
                        'n_answers': len(sq_S0)}}

    # E4: earnings flip, within-request DiD (gate G3a)
    dE, tE, bE, spill, sham = [], [], [], [], []
    for s in cx.A:
        ef = g('E4', s, 'EF')
        if not ef or s not in pc:
            continue
        lab, j = cx.states[s]['labels'], cx.meta('E4', s, 'EF')['treated']
        v = did(ef, pc[s], j, lab)
        if v is None:
            continue
        dE.append(v)
        tE.append(guard.rule(ef, lab).get(j) == 'skip')
        bE.append(_mean([guard.rule(p, lab).get(j) == 'skip'
                         for p in reps['k1'][s]]))
        spill += [(ef[k] - pc[s][k]) ** 2 for k in lab
                  if k != j and k in ef and k in pc[s]]
        ca, cb = g('E1', s, 'Ca'), g('E1', s, 'Cb')
        if ca and cb:
            sh = did(cb, ca, j, lab)
            if sh is not None:
                sham.append(sh)
    out['E4'] = {'E_bar': _mean(dE),
                 'frac_neg': _mean([v < 0 for v in dE]),
                 'lift_E': _mean(tE) - _mean(bE) if tE else float('nan'),
                 'spillover': ex(spill, phi), 'n_states': len(dE),
                 'DiD': dE,
                 'p_signflip': signflip_p([-v for v in dE], cx.seed,
                                          cx.draws, cx.exact_max),
                 'sham': {'mean': _mean(sham),
                          'sd': float(np.std(sham, ddof=1))
                          if len(sham) > 1 else float('nan'),
                          'q05': float(np.quantile(sham, 0.05))
                          if sham else float('nan'),
                          'q95': float(np.quantile(sham, 0.95))
                          if sham else float('nan'),
                          'n': len(sham), 'values': sham}}

    # E5: extreme-move dose-response (reported; lift_M6 breaks ties)
    dM, dose, sign, tM, bM = [], [], [], [], []
    for s in cx.A:
        xm = g('E5', s, 'XM')
        if not xm or s not in pc:
            continue
        lab, mt = cx.states[s]['labels'], cx.meta('E5', s, 'XM')
        v = did(xm, pc[s], mt['treated'], lab)
        if v is None:
            continue
        dM.append(v)
        dose.append(float(mt['dose']))
        sign.append(int(mt['sign']))
        tM.append(guard.rule(xm, lab).get(mt['treated']) == 'skip')
        bM.append(_mean([guard.rule(p, lab).get(mt['treated']) == 'skip'
                         for p in reps['k1'][s]]))
    e5 = {'n_states': len(dM), 'DiD': dM,
          'by_dose': {str(k): _mean([v for v, z in zip(dM, dose) if z == k])
                      for k in sorted(set(dose))},
          'up': _mean([v for v, z in zip(dM, sign) if z > 0]),
          'down': _mean([v for v, z in zip(dM, sign) if z < 0])}
    top = max(dose) if dose else None
    six = [i for i, z in enumerate(dose) if z == top]
    e5['lift_M6'] = (_mean([tM[i] for i in six]) - _mean([bM[i] for i in six])
                     if six else float('nan'))
    if len(dM) > 1:
        x, y = np.log2(dose), np.asarray(dM)
        e5['slope'] = float(_slopes(x, y, np.ones((1, len(y))))[0])
        w = strat_weights([0] * len(y), cx.B, cx.seed)
        b = _slopes(x, y, w)
        b = b[np.isfinite(b)]
        e5['slope_ci95'] = [float(np.quantile(b, 0.025)),
                            float(np.quantile(b, 0.975))] if len(b) \
            else [float('nan')] * 2
    else:
        e5['slope'], e5['slope_ci95'] = float('nan'), [float('nan')] * 2
    out['E5'] = e5

    # E6: hedge, within-state QQQ swap (gate G3b)
    dH, dh, mm, sg, sp = [], [], [], [], []
    for s in cx.A:
        qs = g('E6', s, 'QS')
        st = cx.states[s]
        if not qs or s not in pc or BOOK not in qs or BOOK not in pc[s]:
            continue
        hi = st['regime'] == 'stressed'
        dH.append((pc[s][BOOK] - qs[BOOK]) if hi else (qs[BOOK]
                                                       - pc[s][BOOK]))
        raw = g('E1', s, 'Ca') or g('E1', s, 'Cb')     # C^a for the core side
        if raw and BOOK in raw:
            hc = guard.rule(raw, st['labels'])[BOOK] == 'hedge'
            hq = guard.rule(qs, st['labels'])[BOOK] == 'hedge'
            dh.append(float(hc) - float(hq) if hi else float(hq) - float(hc))
        mm.append(float(st['stressed_m']))
        sg.append(int(st['stressed_sign']))
        sp += [(qs[t] - pc[s][t]) ** 2 for t in st['labels']
               if t in qs and t in pc[s]]
    out['E6'] = {'G_bar': _mean(dH), 'delta_h': _mean(dh),
                 'n_states': len(dH), 'dH': dH,
                 'kendall_tau_m': kendall_tau(dH, mm),
                 'up': _mean([v for v, z in zip(dH, sg) if z > 0]),
                 'down': _mean([v for v, z in zip(dH, sg) if z < 0]),
                 'hold_spillover': ex(sp, phi),
                 'p_signflip': signflip_p(dH, cx.seed, cx.draws,
                                          cx.exact_max)}

    # E7: ticker versus features (gate G6); anonymised Q (reported)
    ys, xf, xt, an_rows, ca_rows, an_st = [], [], [], [], [], []
    for s in cx.A:
        st = cx.states[s]
        lab = st['labels']
        ld = g('E7', s, 'LD')
        if ld and s in pc:
            perm = cx.meta('E7', s, 'LD')['perm']
            ts = [t for t in lab if t in ld and t in pc[s]
                  and perm.get(t) in pc[s]]
            if len(ts) > 1:
                for vec, dst in ((ld, ys), ({t: pc[s][perm[t]] for t in ts},
                                            xf), (pc[s], xt)):
                    v = np.asarray([vec[t] for t in ts])
                    dst.extend(v - v.mean())
        an, ca = g('E7', s, 'AN'), g('E1', s, 'Ca')
        if an and ca:
            an_rows.append(qrow([guard.rule(an, lab)], st))
            ca_rows.append(qrow([guard.rule(ca, lab)], st))
            an_st.append(st['regime'] == 'stressed')
    Phi, a, b = phi_ols(ys, xf, xt, G['G6']['Phi_ab_floor_min'])
    qa = float(q_value(an_rows, an_st)[0]) if an_rows else float('nan')
    qc = float(q_value(ca_rows, an_st)[0]) if ca_rows else float('nan')
    out['E7'] = {'Phi': Phi, 'a': a, 'b': b, 'n_answers': len(ys),
                 'Q_AN': qa, 'Q_Ca_same_states': qc,
                 'Q_AN_minus_Ca': qa - qc, 'n_AN_states': len(an_rows)}

    # E8: negation coherence (reported and flagged)
    ce, pairs = [], []
    for s in cx.A:
        ng = g('E8', s, 'NG')
        if not ng or s not in pc:
            continue
        ks = [x for x in cx.states[s]['labels'] + [BOOK]
              if x in ng and x in pc[s]]
        ce += [abs(pc[s][x] + ng[x] - 1) for x in ks]
        pairs.append(([pc[s][x] for x in ks], [ng[x] for x in ks]))
    rho, prho = neg_coherence(pairs, cx.perms, cx.seed)
    CE = _mean(ce)
    ng0 = P.get('E8|S0|NG')
    ce0 = [abs(s0c[x] + ng0[x] - 1) for x in ng0 if x in s0c] \
        if ng0 and s0c else []
    fl = G['E8_flag']
    out['E8'] = {'CE': CE, 'rho_neg': rho, 'p_rho_neg': prho,
                 'n_states': len(pairs),
                 'flag': bool((_ok(CE) and CE > fl['CE_gt'])
                              or (_ok(rho) and rho <= fl['rho_neg_max'])),
                 'S0': {'CE': _mean(ce0), 'n': len(ce0)}}

    out['E9'] = score_s0(P, cx, h, phi, s0c, ex_O)
    out['q_counts'] = {k: [list(map(float, r)) for r in rows[k]]
                       for k in ('k1', 'k3')}
    return out


def score_s0(P, cx, h, phi, ctrl, ex_O):
    """E9: the real state S0 -- its decisions under k1, k3 and h against
    the rule's, the back-to-back caching ratio, and its nuisance and order
    deviations. Diagnostic only: S0 carries no labels."""
    cache = cx.G.get('caching', {}).get('rho_b2b_lt', CACHE_RHO_FLAG)
    names = cx.s0['names']
    arm = {it['arm']: P.get(i) for i, it in cx.item.items()
           if it['state_id'] == 'S0'}
    reps = [arm[a] for a in sorted(arm) if a.startswith('S0rep') and arm[a]]
    b2b = [arm.get(f'B2B{i}') for i in (1, 2, 3)]
    b2b = [p for p in b2b if p]
    rd = guard.rule(guard.ask_rule(cx.s0['state'], names)[0], names)
    k1 = [guard.rule(p, names) for p in reps]
    k3 = [guard.rule(m, names) for m in
          (mean_p([arm.get(f'S0O{j}{c}') for j in range(3)]) for c in 'ab')
          if m]

    def agree(ds):
        return _mean([d[x] == rd[x] for d in ds for x in d if x in rd])

    def var(ps):
        if len(ps) < 2:
            return float('nan')
        return _mean([float(np.var([p[x] for p in ps], ddof=1))
                      for x in ps[0] if all(x in p for p in ps)])

    vb, vr = var(b2b), var(reps)
    rho = vb / vr if _ok(vb) and _ok(vr) and vr > 0 else float('nan')
    still = _ok(vb) and _ok(vr) and vb == 0 and vr == 0 and \
        _ok(ex_O) and ex_O > 0

    def dev(ps):
        sq = [(p[x] - ctrl[x]) ** 2 for p in ps if p for x in p
              if ctrl and x in ctrl]
        return {'rms': math.sqrt(_mean(sq)) if sq else float('nan'),
                'ex': ex(sq, phi), 'n_answers': len(sq)}

    return {'decisions_k1': k1, 'decisions_k3': k3,
            'decisions_h': (k1[:1] if h == 'k1' else k3[:1]),
            'rule': rd, 'agree_rule_k1': agree(k1),
            'agree_rule_k3': agree(k3),
            'k1_stable': _mean([len({d[x] for d in k1}) == 1 for x in k1[0]])
            if k1 else float('nan'),
            'rho_b2b': rho, 'var_b2b': vb, 'var_reps': vr,
            'caching_flag': bool((_ok(rho) and rho < cache) or still),
            'caching_reason': 'rho_b2b < %g' % cache
            if _ok(rho) and rho < cache else
            ('identical bytes, identical answers; reorderings vary'
             if still else None),
            'nuisance': dev([arm.get('S0V')]),
            'order': dev([arm.get(f'S0O{j}{c}') for j in range(3)
                          for c in 'ab']),
            'n_reps': len(reps), 'n_b2b': len(b2b)}


# ---------------------------------------------------------------- score --

def baselines(cx):
    """The offline stand-ins on the analysis set's core states, from the
    locked items alone (C^a = C^b for a deterministic baseline, so both
    decision sets are the same)."""
    rows = {'Q_rule': [], 'constant_0.5': [], 'skip_nothing_always_hedge': [],
            'criteria_oracle': []}
    for s in cx.A:
        st = cx.states[s]
        lab = st['labels']
        core = cx.item[f'E1|{s}|Ca']['body']['state']
        half = dict({t: 0.5 for t in lab}, **{BOOK: 0.5})
        ds = {'Q_rule': guard.rule(guard.ask_rule(core, lab)[0], lab),
              'constant_0.5': guard.rule(half, lab),
              'skip_nothing_always_hedge': dict({t: 'hold' for t in lab},
                                                **{BOOK: 'hedge'}),
              # decided from the criteria directly: a p-level oracle could
              # sit under the median rule's own blind spot (6 skip names)
              'criteria_oracle': dict(
                  {t: 'skip' if st['conditions'][t] in SKIP_CRITERIA
                   else 'hold' for t in lab},
                  **{BOOK: 'hedge' if st['regime'] == 'stressed'
                     else 'none'})}
        for k, d in ds.items():
            rows[k].append(qrow([d, d], st))
    return {k: float(q_value(v, cx.stressed)[0]) if v else float('nan')
            for k, v in rows.items()}


def analysis_set(models, ops, items, states, dropped_blocks, finals, share):
    """The blocks in which every model passing G1's operational checks has
    at least `share` of its synthetic items final (spec section 7)."""
    blocks = sorted({st['block'] for st in states.values()}
                    - set(dropped_blocks))
    per = {}
    for b in blocks:
        ids = [i for i, it in items.items()
               if it['state_id'] != 'S0' and it['block'] == b]
        per[b] = {m: sum(i in finals[m] for i in ids) / len(ids)
                  for m in models if ids}
    ok = [m for m in models if ops[m]['pass_ops']]
    A = [b for b in blocks if all(_ge(per[b].get(m, 0.0), share)
                                  for m in ok)]
    return A, per


def _clean(o):
    """JSON-safe: numpy scalars to Python, nan/inf to null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def score(run_dir, plan_path=None):
    """Score a run: every metric, gate and test of spec section 2 on the
    analysis set, written to run_dir/summary.json (deterministic bytes)."""
    mp = os.path.join(run_dir, 'run_meta.json')
    meta = _json(mp) if os.path.exists(mp) else {}
    plan_path = plan_of(run_dir, plan_path)
    L = load(plan_path)
    plan, states, s0 = L['plan'], L['states'], L['s0']
    refuse_aliases(models_of(plan))
    shas, lock = check_lock(plan_path, plan, meta)
    items = {it['item_id']: it for it in L['items']}
    G = plan['gates']
    sd = meta.get('scale_down') or {}
    dropped_b = list(sd.get('dropped_blocks') or [])
    dropped_m = set(sd.get('dropped_models') or [])
    models = [m for m in models_of(plan) if m not in dropped_m]
    calls = read_calls(run_dir)
    finals = {m: {} for m in models}
    for c in calls.values():
        if c.get('model') in finals and c.get('item_id') in items:
            finals[c['model']][c['item_id']] = c
    ops = {m: model_ops(finals[m], items, G['G1']) for m in models}
    A, per_block = analysis_set(models, ops, items, states, dropped_b, finals,
                                G['analysis_set']['block_final_share_min'])
    A_states = sorted(s for s in states if states[s]['block'] in A)
    cx = Ctx(plan, states, L['items'], s0, A_states)
    planned_A = [i for i, it in items.items() if it['experiment'] != 'E0'
                 and (it['state_id'] == 'S0' or it['block'] in A)]
    res, holm_in = {}, {}
    for m in models:
        o = ops[m]
        o['completion'] = (sum(i in finals[m] for i in planned_A)
                           / len(planned_A)) if planned_A else float('nan')
        o['G1'] = bool(o['pass_ops']
                       and _ge(o['completion'], G['G1']['completion_min']))
        # only the modal (build, provider)'s valid answers are analysed
        P = {i: pdict(c, items[i]) for i, c in finals[m].items()
             if c.get('valid') and items[i]['experiment'] != 'E0'
             and (str(c.get('answered_as')), str(c.get('provider')))
             == (o['build'], o['provider'])}
        r = score_model(P, cx)
        r['ops'] = o
        # a build dated after S0's night could have seen its outcome
        b = _build_date(o['build'])
        r['E9']['build_after_S0'] = bool(b and b > _next_day(cx.date))
        res[m] = r
        if o['G1']:
            holm_in[f'{m}|G3a'] = r['E4']['p_signflip']
            holm_in[f'{m}|G3b'] = r['E6']['p_signflip']
    H = holm(holm_in, G['holm']['fwer'])
    base = baselines(cx)
    for m, r in res.items():
        hp = {k: H[f'{m}|{k}']['p_holm'] if f'{m}|{k}' in H else None
              for k in ('G3a', 'G3b')}
        r['holm'] = hp
        r['gates'] = gates(r, G, base['Q_rule'], hp)
        r['failed'] = [k for k in GATE_NAMES if not r['gates'][k]]
        e1 = r['E1']
        tb = [r['E4']['lift_E'], r['E5']['lift_M6'], r['E6']['delta_h']]
        r['headline'] = {
            'h': r['h'], 'Q_h': e1['Q_h'], 'Q_k1': e1['Q_k1'],
            'Q_k3': e1['Q_k3'], 'Q_noE': e1['Q_noE'],
            'AUC_comb': e1['AUC_comb'], 'flip_h': e1['flip_h'],
            'lift_E': tb[0], 'lift_M6': tb[1], 'delta_h': tb[2],
            'tiebreak_mean': _mean(tb),
            'median_cost_usd': r['ops']['median_cost_usd']}
    suff = len(A) >= G['analysis_set']['blocks_min']
    summary = {
        'version': plan.get('version'), 'mode': plan.get('mode', 'select'),
        'plan_sha256': shas['plan'], 'lock': lock, 'shas': shas,
        'calls_sha256': _sha_file(os.path.join(run_dir, 'calls.jsonl'))
        if os.path.exists(os.path.join(run_dir, 'calls.jsonl')) else None,
        'models_tested': models, 'dropped_models': sorted(dropped_m),
        'analysis_set': {
            'blocks': A, 'dropped_blocks': dropped_b,
            'sufficient': suff, 'blocks_min': G['analysis_set']['blocks_min'],
            'states': A_states,
            'strata': [states[s]['regime'] for s in A_states],
            'final_share_by_block': per_block},
        'baselines': base,
        'holm': H,
        'models': res}
    summary = _clean(summary)
    with open(os.path.join(run_dir, 'summary.json'), 'w',
              encoding='utf-8') as f:
        f.write(json.dumps(summary, indent=1) + '\n')
    return summary


def _build_date(build):
    m = re.search(r'-(\d{8})$', build or '')
    return m.group(1) if m else None


def _next_day(d):
    t = dt.datetime.strptime(d, '%Y%m%d') + dt.timedelta(days=1)
    return t.strftime('%Y%m%d')


def gates(r, G, q_rule, hp):
    """The eight gates of spec section 3 for one model's metrics."""
    e1, e4, e6 = r['E1'], r['E4'], r['E6']
    pa, pb = hp.get('G3a'), hp.get('G3b')
    return {
        'G1': bool(r['ops']['G1']),
        'G2a': bool(_le(r['E2']['ex_nuis'], G['G2a']['ex_nuis_max'])
                    and _le(r['E2']['rel'], G['G2a']['rel_max'])
                    and _ge(r['noise']['D'], G['G2a']['D_min'])),
        'G2b': bool(_le(r['E3']['ex_ctx'], G['G2b']['ex_ctx_max'])),
        'G3a': bool(_le(e4['E_bar'], G['G3a']['E_bar_max'])
                    and _ge(e4['frac_neg'], G['G3a']['frac_neg_min'])
                    and _ok(pa) and pa < G['G3a']['holm_p_lt']),
        'G3b': bool(_ge(e6['G_bar'], G['G3b']['G_bar_min'])
                    and _ok(e6['delta_h'])
                    and e6['delta_h'] > G['G3b']['delta_h_gt']
                    and _ok(pb) and pb < G['G3b']['holm_p_lt']),
        'G4': bool(_le(e1['flip_h'], G['G4']['flip_h_max'])),
        'G5': bool(_ok(q_rule) and _ge(e1['Q_h'], q_rule
                                       + G['G5']['Q_minus_Q_rule_min'])),
        'G6': bool(_ge(r['E7']['Phi'], G['G6']['Phi_min']))}


# --------------------------------------------------------------- select --

def _num(x, worst):
    return float(x) if _ok(x) else worst


def rank(ids, models, X, metric='Q_h'):
    """Section 3.4's order: higher metric, then exact ties (to 1e-12) broken
    by higher AUC_comb; higher mean(lift_E, lift_M6, delta_h); lower flip_h;
    higher X (unscored below every scored); lower median cost; model id.
    metric=None orders by the tie-break chain alone."""
    def key(m):
        hl = models[m]['headline']
        r12 = lambda v, w: round(_num(v, w), 12)
        head = () if metric is None else (-r12(hl[metric], -math.inf),)
        return head + (-r12(hl['AUC_comb'], -math.inf),
                       -r12(hl['tiebreak_mean'], -math.inf),
                       r12(hl['flip_h'], math.inf),
                       -_num(X.get(m), -math.inf),
                       _num(hl.get('median_cost_usd'), math.inf), m)
    return sorted(ids, key=key)


def choose(summary, plan):
    """Section 3's rule on a summary: pure, so every branch is testable."""
    G = plan['gates']
    I, R = plan['incumbent'], plan['reference']
    models = summary['models']
    refuse_aliases(models_of(plan) + list(models))
    aset = summary['analysis_set']
    if not aset['sufficient']:
        raise InsufficientAnalysisSet(
            f'{len(aset["blocks"])} complete block(s) {aset["blocks"]} < '
            f'{aset["blocks_min"]}: resume the run')
    X = {c['id']: c.get('X') for c in plan['candidates']}
    E = [m for m in models if not models[m]['failed']]
    ranked = rank(E, models, X)
    W = ranked[0] if ranked else None
    strata = aset['strata']
    stressed = [s == 'stressed' for s in strata]
    sel = G['selection']
    B, seed = int(plan['bootstrap_B']), plan['seeds']['analysis']
    w = strat_weights(strata, B, seed)          # one draw, every model

    def rows(m, k=None):
        return models[m]['q_counts'][k or models[m]['h']]

    test = None
    if not E:
        branch, dec, handling = 'none', I, 'k1'
    elif I not in E:
        branch, dec, handling = 'b', W, models[W]['h']
    else:
        branch, dec = 'a', I
        if W != I:
            alpha = sel['alpha'] / (len(E) - 1)
            qw, qi = models[W]['headline']['Q_h'], models[I]['headline']['Q_h']
            lb = paired_bootstrap_lb(rows(W), rows(I), strata, B, seed,
                                     alpha, w=w)
            margin = qw - qi
            sw = bool(_ge(margin, sel['switch_margin_min'])
                      and _ok(lb) and lb > sel['lower_bound_gt'])
            test = {'W': W, 'I': I, 'Q_W': qw, 'Q_I': qi, 'margin': margin,
                    'margin_min': sel['switch_margin_min'],
                    'lower_bound': lb, 'alpha': alpha, 'B': B, 'seed': seed,
                    'switched': sw}
            dec = W if sw else I
        handling = models[dec]['h']
    o = models[dec]['ops'] if dec in models else {}
    pin = None
    if branch != 'none' and _ok(o.get('provider_share')) and \
            o['provider_share'] < 1 and o.get('pin_probe') == 'ok':
        pin = {'order': [o['provider']], 'allow_fallbacks': False}
    decider = {'pin': dec, 'handling': handling, 'version': 'q1',
               'suffix': '#perm3.q1' if handling == 'k3' else '',
               'provider_pin': pin, 'build': o.get('build'),
               'provider': o.get('provider')}
    ro = models[R]['ops'] if R in models else {}
    if R not in models or not models[R]['gates']['G1']:
        rs, note = 'dropped', 'the reference failed G1: its backend is dropped'
    elif dec == R and handling == 'k3':
        rs, note = 'raw', ('the decider is the reference model at #perm3.q1, '
                           'a separate object from the one raw reference call')
    elif dec == R:
        rs, note = 'retired', ('the decider at k1 would be the reference '
                               'itself: the jev backend retires')
    else:
        rs, note = 'raw', 'one raw canonical call, no suffix'
    reference = {'id': R, 'status': rs, 'suffix': '', 'note': note,
                 'build': ro.get('build'), 'provider': ro.get('provider')}

    # report-only sensitivity: never changes the decision above
    pool = E or [m for m in models if _ok(models[m]['headline']['Q_h'])]
    # within a draw, a tie on Q* goes down the same chain (its point values)
    order = rank(pool, models, X, metric=None)
    p1 = {}
    if order:
        Q = np.column_stack([q_value(rows(m), stressed, w) for m in order])
        Q = np.where(np.isfinite(Q), np.round(Q, 12), -np.inf)
        win = np.argmax(Q, axis=1)              # ties: earlier in the order
        p1 = {m: float(np.mean(win == i)) for i, m in enumerate(order)}
    first = lambda xs: xs[0] if xs else None
    sens = {
        'P_rank1': p1, 'P_rank1_pool': 'eligible' if E else 'all scored',
        'winner_Q_k1': first(rank(E, models, X, 'Q_k1')),
        'winner_Q_noE': first(rank(E, models, X, 'Q_noE')),
        'winner_AUC_comb': first(rank(E, models, X, 'AUC_comb')),
        'S0_decisions': {m: (models[m].get('E9') or {}).get('decisions_h')
                         for m in (E or [dec])}}
    return {
        'version': plan.get('version'), 'mode': 'select',
        'analysis_set': {k: aset[k] for k in ('blocks', 'states', 'strata')},
        'eligible': E,
        'failed_gates': {m: models[m]['failed'] for m in models},
        'holm': summary.get('holm', {}),
        'ranking': [{'id': m, 'Q_h': models[m]['headline']['Q_h'],
                     'AUC_comb': models[m]['headline']['AUC_comb'],
                     'tiebreak_mean': models[m]['headline']['tiebreak_mean'],
                     'flip_h': models[m]['headline']['flip_h'],
                     'X': X.get(m),
                     'median_cost_usd':
                         models[m]['headline']['median_cost_usd']}
                    for m in ranked],
        'top': W, 'incumbent': I, 'branch': branch, 'switch_test': test,
        'decider': decider, 'reference': reference,
        'measured_builds': {m: {'build': models[m]['ops'].get('build'),
                                'provider': models[m]['ops'].get('provider'),
                                'modal_share':
                                    models[m]['ops'].get('modal_share')}
                            for m in models},
        'results_note': 'no candidate qualified' if branch == 'none'
        else None,
        'gate_unchanged': {'GATE_NIGHTS': guard.GATE_NIGHTS,
                           'GATE_T': guard.GATE_T},
        'sensitivity': sens}


def iterate_outcome(summary, plan):
    """Section 4's adoption rule for an iteration round (or the holdout):
    the arms are the plan's control and variants, scored like models."""
    it = plan['gates']['iteration']
    models = summary['models']
    ctrl, variants = plan['control'], list(plan['variants'])
    hold = bool(plan.get('holdout'))
    alpha = it['alpha'] if hold else it['alpha'] / max(1, len(variants))
    strata = summary['analysis_set']['strata']
    B, seed = int(plan['bootstrap_B']), plan['seeds']['analysis']
    w = strat_weights(strata, B, seed)
    hc = models[ctrl]['headline']
    rows = lambda m: models[m]['q_counts'][models[m]['h']]
    checks, ok = {}, []
    for v in variants:
        hv = models[v]['headline']
        dq = hv['Q_h'] - hc['Q_h']
        lb = paired_bootstrap_lb(rows(v), rows(ctrl), strata, B, seed, alpha,
                                 w=w)
        c = {'gates': not models[v]['failed'], 'delta_Q': dq,
             'lower_bound': lb, 'alpha': alpha,
             'lb_ok': bool(_ok(lb) and lb > it['lower_bound_gt'])}
        if hold:
            c['ok'] = bool(c['gates'] and c['lb_ok'])
        else:
            c['dq_ok'] = _ge(dq, it['delta_Q_min'])
            c['auc_ok'] = _ge(hv['AUC_comb'] - hc['AUC_comb'],
                              -it['AUC_comb_drop_max'])
            c['phi_ok'] = _ge(models[v]['E7']['Phi'] - models[ctrl]['E7']
                              ['Phi'], -it['Phi_drop_max'])
            c['ok'] = bool(c['gates'] and c['lb_ok'] and c['dq_ok']
                           and c['auc_ok'] and c['phi_ok'])
        checks[v] = c
        if c['ok']:
            ok.append(v)
    best = sorted(ok, key=lambda v: (-models[v]['headline']['Q_h'],
                                     variants.index(v)))
    return {'version': plan.get('version'), 'mode': 'iterate',
            'round': plan.get('round'), 'holdout': hold, 'control': ctrl,
            'variants': variants, 'checks': checks,
            'adopted': best[0] if best else None}


def select(run_dir):
    """Read summary.json, apply section 3 (or section 4 in iterate mode) and
    write the outcome beside it. A summary missing, or scored before the
    last calls landed (a resume), is re-scored first: score is
    deterministic, so this only ever brings it up to date."""
    sp = os.path.join(run_dir, 'summary.json')
    cp = os.path.join(run_dir, 'calls.jsonl')
    cur = _sha_file(cp) if os.path.exists(cp) else None
    if not os.path.exists(sp) or _json(sp).get('calls_sha256') != cur:
        score(run_dir)
    summary = _json(sp)
    plan_path = plan_of(run_dir)
    if _sha_file(plan_path) != summary['plan_sha256']:
        raise ValueError('summary.json was scored against another plan')
    plan = _json(plan_path)
    if plan.get('mode') == 'iterate':
        out = iterate_outcome(summary, plan)
        name = f'iter_r{plan["round"]}_outcome.json' if 'round' in plan \
            else 'iter_holdout_outcome.json'
    else:
        try:
            out = choose(summary, plan)
        except InsufficientAnalysisSet as ex:
            print(f'select: {ex.msg}', file=sys.stderr)
            raise
        name = f'selection_{plan.get("version", "v1")}.json'
    out['summary_sha256'] = _sha_file(sp)
    out['plan_sha256'] = summary['plan_sha256']
    out = _clean(out)
    with open(os.path.join(run_dir, name), 'w', encoding='utf-8') as f:
        f.write(json.dumps(out, indent=1) + '\n')
    return out


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'score':
        s = score(sys.argv[2])
        a = s['analysis_set']
        print(f'blocks {a["blocks"]}, {len(a["states"])} states; '
              f'Q_rule {s["baselines"]["Q_rule"]:.4f}')
        for m, r in s['models'].items():
            q = r['headline']['Q_h']
            print(f'  {m}: h={r["h"]} Q_h={q if q is None else round(q, 4)}'
                  f' failed={",".join(r["failed"]) or "-"}')
    elif len(sys.argv) == 3 and sys.argv[1] == 'select':
        o = select(sys.argv[2])
        print(json.dumps({k: o.get(k) for k in ('branch', 'decider',
                                                'reference', 'adopted')
                          if k in o}, indent=1))
    else:
        sys.exit('usage: battery_score.py score RUN_DIR | select RUN_DIR')
