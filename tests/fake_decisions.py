# Part of qqq-microstructure.
#
# A stand-in for OpenRouter's Decisions API and its GET /api/v1/key, for the
# battery's tests (spec section 8). FakeDecisions is an opener with urllib's
# urlopen signature; what it answers depends on the model id asked, so one
# plan can hold deciders that follow the request's own criteria, ignore it,
# or are noisy, lethargic, order- or context-sensitive, and the operational
# faults (impostor build, provider splits, 429, a 502 storm, 401, no cost,
# slow, a key in an error body). It reads each question the way a model
# must: the name from the backticks in its instructions (ids may be opaque),
# the row from the state, a negated wording from its text. Also: a model
# that answers its first call and then times out on every one (hang), a 408
# on each request's first try (flaky408), a cache that never expires
# (sticky), and an account whose credit runs out (credit_after).
#
# make_test_plan writes a tiny plan + states + items + lock in the
# contract's formats, built here rather than by battery_items so these
# tests stand alone.

import os, io, re, sys, json, math, time, random, shutil, hashlib, threading
import urllib.error
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import guard

S0_FILE = os.path.join(ROOT, 'bench', 'battery', 's0_20261013.json')
ANCHORS_FILE = os.path.join(ROOT, 'bench', 'battery', 'anchors_20261013.json')
BUILD = '20261001'
SENTINEL = 'sk-or-v1-SENTINEL0123456789abcdef'
LEAK = json.dumps({'error': {'message': f'invalid request for key {SENTINEL}',
                             'metadata': {'echo': f'Bearer {SENTINEL}'}}})
BEHAVIOURS = ('oracle', 'const', 'uniform', 'lethargic', 'rule', 'jevlike',
              'posbias', 'joint', 'tickerprior', 'caching', 'sticky',
              'impostor', 'split50', 'split95', 'ratelimit', 'storm',
              'unauth', 'nocost', 'slow', 'leaky', 'nopin', 'partial',
              'hang', 'flaky408')
TICKER = re.compile(r'`([^`]+)`')
CACHE_CALLS = 10           # the caching model remembers a body this many calls


class Crash(BaseException):
    """An injected crash: not an Exception, so nothing in the runner can
    swallow it."""


class Clock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self, t=1000.0):
        self.t, self.slept, self.lock = t, [], threading.Lock()

    def now(self):
        with self.lock:
            return self.t

    def sleep(self, s):
        with self.lock:
            self.t += s
            self.slept.append(s)


class _Resp(io.BytesIO):
    status = 200

    def __init__(self, obj):
        super().__init__(json.dumps(obj).encode())
        self.headers = {}


def _err(url, code, body, headers=None):
    return urllib.error.HTTPError(url, code, 'error', headers or {},
                                  io.BytesIO(body.encode()))


def _h01(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 0xffffffff


def _clip(p):
    return min(1.0, max(0.0, p))


def _graded(x, yes):
    # graded by how far past (x > 1) or short of the line the row sits, and
    # always on the criteria's side of 0.5
    p = 0.5 + 0.45 * math.tanh(2.5 * (x - 1))
    return round(max(p, 0.55) if yes else min(p, 0.45), 3)


# Where 'extreme' starts, in sd of the row's own vol20d. The battery's
# generator draws an extreme move at 3.5-5 sd and a moderate one at 1.5-2.5
# sd, then folds the move into vol20d (a coherent update), so a row shows
# less than it was drawn with: after the update an extreme last_night is
# >= 3.11 sigma_on and a moderate one <= 2.46; an extreme ret5d >= 2.44 sd
# and a moderate one <= 2.36. The lines sit between, so the oracle skips
# exactly the generator's Extreme names (a 3 sd line on ret5d missed four).
EXTREME_NIGHT, EXTREME_RET5D = 2.8, 2.4


def oracle(state, ticker):
    """p(yes) to the canonical (un-negated) question, by its own criteria:
    a name is skipped on earnings tonight or |last_night| >= 2.8 sigma_on
    (sigma_on = 0.5 vol20d) or |ret5d| >= 2.4 vol20d sqrt(5); the book is
    hedged when QQQ's vol20d >= 150 or |last_night| >= 2 sigma_on."""
    if ticker is None:
        r = state.get('QQQ')
        if not r:
            return 0.5
        v = max(r.get('vol20d', 0), 1)
        ln = abs(r.get('last_night', 0))
        return _graded(max(v / 150, ln / v), v >= 150 or ln >= 2 * 0.5 * v)
    r = state.get('names', {}).get(ticker)
    if r is None:
        return 0.5
    if r.get('earnings_tonight'):
        return 0.05
    v = max(r.get('vol20d', 0), 1)
    x = max(abs(r.get('last_night', 0)) / (EXTREME_NIGHT * 0.5 * v),
            abs(r.get('ret5d', 0)) / (EXTREME_RET5D * v * math.sqrt(5)))
    return round(1 - _graded(x, x >= 1), 3)


def read_question(q):
    """(ticker or None for the book, negated?) from a question's text."""
    text = q.get('instructions', '')
    m = TICKER.search(text)
    return (m.group(1) if m else None), \
        ('left out' in text or 'left unhedged' in text)


class FakeDecisions:
    """POST .../decisions and GET .../key. behaviours: {model id: behaviour}
    for ids that do not name one ('fake/<behaviour>' does); anything else
    is the oracle. costs: {model id: $ per call}. Records every request in
    .calls (method, url, headers, data)."""

    def __init__(self, behaviours=None, cost=1e-4, costs=None,
                 limit_remaining=35.39, key_usage=14.61, label=SENTINEL,
                 key_status=200, clock=None, slow_s=30.0, retry_after='7',
                 crash_at=None, delay=0.0, credit_after=None):
        self.behaviours, self.cost, self.costs = behaviours or {}, cost, \
            costs or {}
        self.limit_remaining, self.key_usage, self.label = \
            limit_remaining, key_usage, label
        self.key_status, self.clock, self.slow_s = key_status, clock, slow_s
        self.retry_after, self.crash_at, self.delay = retry_after, crash_at, \
            delay
        self.credit_after = credit_after
        self.lock = threading.Lock()
        self.calls, self.n_post, self.billed = [], 0, 0.0
        self.n_model, self.attempts, self.answered = Counter(), Counter(), \
            Counter()
        self.inflight, self.inflight_model = 0, Counter()
        self.max_inflight, self.max_inflight_model = 0, Counter()
        self.cache = {}

    def behaviour(self, model):
        if model in self.behaviours:
            return self.behaviours[model]
        b = model[5:] if model.startswith('fake/') else None
        return b if b in BEHAVIOURS else 'oracle'

    def posts(self, model=None):
        return [c for c in self.calls if c['method'] == 'POST' and
                (model is None or json.loads(c['data'])['model'] == model)]

    def __call__(self, req, timeout=None):
        with self.lock:
            self.calls.append({'method': req.get_method(),
                               'url': req.full_url,
                               'headers': dict(req.header_items()),
                               'data': req.data})
        if req.get_method() == 'GET':
            return self._key(req)
        return self._decide(req)

    def _key(self, req):
        if self.key_status != 200:
            raise _err(req.full_url, self.key_status,
                       '{"error":{"message":"No auth credentials found"}}')
        with self.lock:
            b = self.billed
        return _Resp({'data': {'label': self.label, 'limit': 50.0,
                               'usage': self.key_usage + b,
                               'limit_remaining': self.limit_remaining - b,
                               'is_free_tier': False,
                               'rate_limit': {'requests': -1,
                                              'interval': '10s'}}})

    def _decide(self, req):
        data, url = req.data, req.full_url
        body = json.loads(data)
        model = body['model']
        beh = self.behaviour(model)
        with self.lock:
            self.n_post += 1
            n = self.n_post
            self.n_model[model] += 1
            c = self.n_model[model]
            # per request object: a retry reuses it, an identical body
            # (Ca and Cb, the S0 repeats) is a new request
            self.attempts[req] += 1
            a = self.attempts[req]
            self.inflight += 1
            self.inflight_model[model] += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.max_inflight_model[model] = max(
                self.max_inflight_model[model], self.inflight_model[model])
        try:
            if self.crash_at is not None and n == self.crash_at:
                raise Crash(f'injected crash at call {n}')
            if self.delay:
                time.sleep(self.delay)
            if self.credit_after is not None and n > self.credit_after:
                raise _err(url, 402, '{"error":{"code":402,"message":'
                                     '"Insufficient credits"}}')
            if beh == 'unauth':
                raise _err(url, 401, '{"error":{"code":401,"message":'
                                     '"No auth credentials found"}}')
            if beh == 'hang' and c > 1:   # answers W, then never again
                if self.clock:
                    self.clock.sleep(self.slow_s)
                raise urllib.error.URLError(TimeoutError('timed out'))
            if beh == 'flaky408' and a == 1:
                raise _err(url, 408, '{"error":{"code":408,"message":'
                                     '"Request timed out"}}')
            if beh == 'storm':
                raise _err(url, 502, '{"error":"upstream"}')
            if beh == 'ratelimit' and a == 1:
                raise _err(url, 429, '{"error":"rate limited"}',
                           {'Retry-After': self.retry_after})
            if beh == 'leaky':
                raise _err(url, 400, LEAK)
            if beh == 'nopin' and 'provider' in body:
                raise _err(url, 400, '{"error":"provider routing is not '
                                     'supported for this model"}')
            if beh == 'slow':
                if self.clock:
                    self.clock.sleep(self.slow_s)
                else:
                    time.sleep(0.001)
            answers = self._answers(beh, model, body, data, c)
            prov = {'split50': 'ProvA' if c % 2 else 'ProvB',
                    'split95': 'ProvB' if c % 20 == 0 else 'ProvA'}.get(
                beh, 'FakeProv')
            who = f'{model}-mini-{BUILD}' if beh == 'impostor' \
                else f'{model}-{BUILD}'
            cost = self.costs.get(model, self.cost)
            usage = {'input_tokens': len(data) // 4,
                     'output_tokens': 3 * len(answers), 'cost': cost}
            if beh == 'nocost':
                del usage['cost']
            with self.lock:
                self.billed += cost
                self.answered[model] += 1
            return _Resp({'id': f'gen-dec-{n}', 'model': who,
                          'provider': prov,
                          'answers': {k: {'type': 'noul', 'noul': p}
                                      for k, p in answers.items()},
                          'usage': usage})
        finally:
            with self.lock:
                self.inflight -= 1
                self.inflight_model[model] -= 1

    def _answers(self, beh, model, body, data, c):
        if beh in ('caching', 'sticky'):
            with self.lock:
                hit = self.cache.get((model, data))
            if hit and (beh == 'sticky' or c - hit[0] <= CACHE_CALLS):
                return dict(hit[1])
        st, qs = body.get('state', {}), body.get('questions', {})
        h = hashlib.sha256(data).hexdigest()
        out = {}
        for pos, (qid, q) in enumerate(qs.items()):
            t, neg = read_question(q)
            row = st.get('names', {}).get(t, {}) if t else {}
            if beh == 'const':
                p = 0.5
            elif beh == 'uniform':
                p = round(random.Random(f'{h}|{qid}').random(), 2)
            elif beh == 'lethargic':      # Perplexity on S0: a narrow band
                p = 0.558 if t is None else round(0.46 + 0.15 * _h01(t), 2)
            elif beh == 'rule':           # guard.ask_rule, read per question
                p = 0.0 if t is None or row.get('earnings_tonight') else 1.0
                p = 1 - p if neg else p
            elif beh == 'tickerprior':    # knows the name, ignores the row
                p = 0.3 if t is None else round(0.1 + 0.8 * _h01(t), 2)
                p = 1 - p if neg else p
            else:
                p = oracle(st, t)
                p = round(1 - p, 3) if neg else p
                if beh in ('jevlike', 'caching', 'sticky'):
                    sd = 0.03 if beh == 'jevlike' else 0.05
                    p = round(_clip(p + random.Random(f'{h}|{qid}|{c}')
                                    .gauss(0, sd)), 2)
                elif beh == 'posbias':
                    p = round(_clip(p - 0.02 * pos), 3)
                elif beh == 'joint':      # pulled to 0.5 when asked alone
                    s = 0.1 * (12 - len(qs)) / 12
                    p = round(p + s if p < 0.5 else p - s, 3)
            out[qid] = p
        if beh == 'partial' and out:
            out.pop(list(out)[-1])
        if beh in ('caching', 'sticky'):
            with self.lock:         # sticky keeps its first answer
                if beh == 'caching' or (model, data) not in self.cache:
                    self.cache[(model, data)] = (c, dict(out))
        return out


# ------------------------------------------------------------ test plans --

NEGATION = {
    'hold': ('Should the long position in `{n}` be left out tonight -- not '
             'bought at today\'s closing auction and not sold at tomorrow\'s '
             'opening auction?'),
    'hold_criteria': {'true': 'skip: earnings tonight or an extreme recent '
                              'move',
                      'false': 'hold: an ordinary night for this name'},
    'hedge': ('Should tonight\'s whole basket be left unhedged -- no short '
              'position in QQQ from today\'s close to tomorrow\'s open?'),
    'hedge_criteria': {'true': 'do not hedge: an ordinary night',
                       'false': 'hedge: elevated risk of a market-wide '
                                'overnight gap'}}


def _body(state, names, **kw):
    b = guard.request_body(state, names, 'm', **kw)
    del b['model']
    return b


def _subset(state, names, keep):
    full = _body(state, names)
    q = {k: v for k, v in full['questions'].items() if k in keep}
    held = [k[5:] for k in q if k.startswith('hold_')]
    return {'state': dict(state, names={n: state['names'][n] for n in held}),
            'questions': q}


def _negated(state, names):
    q = {f'hold_{n}': {'type': 'noul',
                       'instructions': NEGATION['hold'].format(n=n),
                       'criteria': dict(NEGATION['hold_criteria'])}
         for n in names}
    q['hedge_book'] = {'type': 'noul', 'instructions': NEGATION['hedge'],
                       'criteria': dict(NEGATION['hedge_criteria'])}
    return {'state': state, 'questions': q}


def _item(exp, sid, arm, block, body, qmap=None, meta=None):
    assert 'model' not in body
    return {'item_id': f'{exp}|{sid}|{arm}', 'experiment': exp, 'arm': arm,
            'state_id': sid, 'block': block, 'body': body,
            'body_sha256': hashlib.sha256(json.dumps(body).encode())
            .hexdigest(),
            'qmap': qmap or {q: q for q in body['questions']},
            'meta': meta or {}}


def _synthetic(k, regime, anchors, book):
    rng = np.random.default_rng(np.random.SeedSequence(20261013,
                                                       spawn_key=(k,)))
    labels = [str(t) for t in rng.choice(
        sorted(t for t in anchors if t != 'QQQ'), 11, replace=False)]
    conds = ['Earnings', 'ExtremeDown', 'ExtremeUp', 'Moderate'] + \
        ['Ordinary'] * 7
    conds = [conds[i] for i in rng.permutation(11)]
    r5 = math.sqrt(5)
    rows = {}
    for t, c in zip(labels, conds):
        vol = anchors[t] * (1.5 if regime == 'stressed' else 1.0)
        z = rng.uniform(-1, 1, 3)
        ln, avg, ret = z[0] * 0.5 * vol, z[1] * 0.5 * vol / r5, \
            z[2] * vol * r5
        if c == 'ExtremeDown':
            ln = -4 * 0.5 * vol
        elif c == 'ExtremeUp':
            ret = 4 * vol * r5
        elif c == 'Moderate':
            ln = 2 * 0.5 * vol
        rows[t] = {'last_night': int(round(ln)), 'nights5_avg': int(round(avg)),
                   'ret5d': int(round(ret)), 'vol20d': int(round(vol)),
                   'earnings_tonight': c == 'Earnings'}
    aq = anchors['QQQ']
    zq = rng.uniform(-1, 1, 3)
    sign, m = (1 if rng.random() < 0.5 else -1), float(rng.uniform(2, 3))
    qqq = {'calm': {'last_night': int(round(zq[0] * 0.5 * aq)),
                    'nights5_avg': int(round(zq[1] * 0.5 * aq / r5)),
                    'ret5d': int(round(zq[2] * aq * r5)), 'vol20d': int(aq)},
           'stressed': {'last_night': sign * int(round(3 * 0.5 * aq)),
                        'nights5_avg': sign * int(round(0.75 * aq / r5)),
                        'ret5d': sign * int(round(3 * aq * r5)),
                        'vol20d': int(round(aq * m))}}
    state = {'schema': 2, 'date': '20261013', 'book': book,
             'names': {t: rows[t] for t in labels}, 'QQQ': qqq[regime]}
    return {'regime': regime, 'labels': labels,
            'conditions': dict(zip(labels, conds)), 'rows': rows, 'qqq': qqq,
            'stressed_m': m, 'stressed_sign': sign, 'state': state}


def _state_items(k, sid, b, s, paraphrase):
    """A few items per experiment for one synthetic state."""
    st, labels = s['state'], s['labels']
    qids = [f'hold_{t}' for t in labels] + ['hedge_book']
    ordin = [t for t in labels if s['conditions'][t] == 'Ordinary']
    out = [_item('E1', sid, 'Ca', b, _body(st, labels)),
           _item('E1', sid, 'Cb', b, _body(st, labels))]
    for i in (1, 2):
        out.append(_item('E1', sid, f'O{i}', b, _body(
            st, labels, order=guard.question_order(f'battery-v1|{sid}|O{i}',
                                                   qids))))
    ids = {q: 'q' + hashlib.sha256(f'{sid}|V|{q}'.encode()).hexdigest()[:8]
           for q in qids}
    v = dict(st, date='20261014', book=paraphrase)
    out.append(_item('E2', sid, 'V', b, _body(
        v, labels, order=guard.question_order(f'battery-v1|{sid}|V', qids),
        ids=ids), qmap={x: q for q, x in ids.items()},
        meta={'opaque': True, 'date_shift': 1, 'paraphrase': 1}))
    cx = s['cx_type']
    keep = {1: qids[:5] + ['hedge_book'], 2: [qids[0]], 3: ['hedge_book']}[cx]
    out.append(_item('E3', sid, 'CX', b, _subset(st, labels, keep),
                     meta={'cx_type': cx, 'cx_subset': keep}))
    j = ordin[0]
    ef = dict(st, names={t: dict(r, earnings_tonight=True) if t == j else r
                         for t, r in st['names'].items()})
    out.append(_item('E4', sid, 'EF', b, _body(ef, labels),
                     meta={'treated': j}))
    j2, r = ordin[1], st['names'][ordin[1]]
    new = int(round(3 * r['vol20d']))
    d = new - r['last_night']
    xr = dict(r, last_night=new, nights5_avg=r['nights5_avg'] +
              int(round(d / 5)), ret5d=r['ret5d'] + d,
              vol20d=int(round(math.sqrt(r['vol20d'] ** 2 + d * d / 19))))
    xm = dict(st, names={t: xr if t == j2 else rr
                         for t, rr in st['names'].items()})
    out.append(_item('E5', sid, 'XM', b, _body(xm, labels),
                     meta={'treated': j2, 'dose': 3, 'sign': 1}))
    other = 'calm' if s['regime'] == 'stressed' else 'stressed'
    out.append(_item('E6', sid, 'QS', b, _body(dict(st, QQQ=s['qqq'][other]),
                                               labels),
                     meta={'regime': other}))
    if s['e7_kind'] == 'LD':
        perm = {t: labels[(i + 1) % 11] for i, t in enumerate(labels)}
        ld = dict(st, names={t: st['names'][perm[t]] for t in labels})
        out.append(_item('E7', sid, 'LD', b, _body(ld, labels),
                         meta={'perm': perm}))
    else:
        anon = {t: f'S{i + 1:02d}' for i, t in enumerate(labels)}
        an = dict(st, names={anon[t]: st['names'][t] for t in labels})
        body = _body(an, [anon[t] for t in labels])
        out.append(_item('E7', sid, 'AN', b, body,
                         qmap={**{f'hold_{anon[t]}': f'hold_{t}'
                                  for t in labels},
                               'hedge_book': 'hedge_book'},
                         meta={'anon': anon}))
    if s['e8']:
        out.append(_item('E8', sid, 'NG', b, _negated(st, labels),
                         meta={'negated': True}))
    return out


def _s0_items(s0, blocks):
    st, names = s0['state'], s0['names']
    qids = [f'hold_{n}' for n in names] + ['hedge_book']
    canon = _body(st, names)
    out = [_item('E0', 'S0', 'W', 0, canon),
           _item('E0', 'S0', 'PP', 0, dict(canon, provider='__FROM_W__'))]
    for q in (qids[0], 'hedge_book'):
        out.append(_item('E3', 'S0', f'alone_{q}', 1, _subset(st, names, [q])))
    out.append(_item('E8', 'S0', 'NG', min(2, blocks), _negated(st, names),
                     meta={'negated': True}))
    for b in range(1, blocks + 1):
        out.append(_item('E9', 'S0', f'S0rep{b}', b, canon))
    for i in (1, 2, 3):
        out.append(_item('E9', 'S0', f'B2B{i}', min(2, blocks), canon))
    sha = hashlib.sha256(json.dumps(st, sort_keys=True).encode()) \
        .hexdigest()[:12]
    o = guard.question_order(f'20261013|{sha}|0|order-v1|http', qids)
    for x in 'ab':
        out.append(_item('E9', 'S0', f'S0O0{x}', min(3, blocks),
                         _body(st, names, order=o)))
    ids = {q: 'q' + hashlib.sha256(f'S0|V|{q}'.encode()).hexdigest()[:8]
           for q in qids}
    out.append(_item('E9', 'S0', 'S0V', blocks, _body(st, names, ids=ids),
                     qmap={x: q for q, x in ids.items()},
                     meta={'opaque': True}))
    return out


def make_test_plan(tmpdir, candidates, incumbent, reference,
                   states_per_block=1, blocks=4, dispatch_seed=20261015,
                   spend=None, **plan):
    """A tiny contract-format plan (plan_v1.json, states_v1.json,
    items_v1.jsonl, lock_v1.json, and copies of S0 and the anchors) in
    tmpdir; returns the plan's path. candidates: ids or {id, price_in, X}
    dicts; I and R are added (scored) when not among them."""
    d = str(tmpdir)
    os.makedirs(d, exist_ok=True)
    s0 = json.load(open(S0_FILE))
    anchors = json.load(open(ANCHORS_FILE))['vol20d']
    cands = [dict(c) if isinstance(c, dict) else
             {'id': c, 'price_in': 2e-8, 'X': 0.5} for c in candidates]
    for m, x in ((incumbent, 1.0), (reference, 0.799)):
        if m not in [c['id'] for c in cands]:
            cands.append({'id': m, 'price_in': 2e-8, 'X': x})
    book = s0['state']['book']
    paraphrases = [book, 'a long book of these names, bought at the close '
                   'auction and sold at the next open auction; basis points',
                   'each name is held overnight, close auction to open '
                   'auction, equal weight; moves in basis points']
    states, items, k = {}, [], 0
    for b in range(1, blocks + 1):
        for j in range(states_per_block):
            k += 1
            sid = f'S{k:02d}'
            s = _synthetic(k, 'stressed' if k % 2 else 'calm', anchors, book)
            s.update(block=b, cx_type=(k - 1) % 3 + 1,
                     e7_kind='AN' if k % 3 == 0 else 'LD', e8=k % 2 == 1)
            states[sid] = {x: s[x] for x in (
                'block', 'regime', 'labels', 'conditions', 'rows', 'qqq',
                'stressed_m', 'stressed_sign', 'cx_type', 'e7_kind', 'e8',
                'state')}
            items += _state_items(k, sid, b, s, paraphrases[1])
    items += _s0_items(s0, blocks)
    items.sort(key=lambda it: it['block'])
    p = {'version': 'v1', 'mode': 'select', 'incumbent': incumbent,
         'reference': reference, 'candidates': cands,
         'seeds': {'items': 20261013, 'analysis': 20261014,
                   'dispatch': dispatch_seed},
         'blocks': blocks, 'states_per_block': states_per_block,
         'generator': {'test_plan': True, 'stressed_name_vol_scale': 1.5},
         'texts': {'book_paraphrases': paraphrases, 'negation': NEGATION},
         'gates': {'G1': {'valid_rate': 0.98, 'completion': 0.95,
                          'modal_share': 0.9, 'p95_s': 60,
                          'median_cost': 0.01}},
         'bootstrap_B': 10000, 'signflip_draws': 200000,
         'spend': dict({'max_usd': 8, 'max_usd_ceiling': 12,
                        'credit_margin': 10, 'pessimistic_per_call': 0.0012,
                        'price_ref': 3.1e-8}, **(spend or {})),
         'deadline_min': 90, 'timeout_s': 60, 'concurrency': 16,
         'per_model': 3,
         'files': {'states': 'states_v1.json', 'items': 'items_v1.jsonl',
                   's0': 's0_20261013.json',
                   'anchors': 'anchors_20261013.json'},
         'baselines_note': 'a test plan (tests/fake_decisions.py), not a '
                           'declaration'}
    p.update(plan)
    files = {'plan': 'plan_v1.json', 'states': 'states_v1.json',
             'items': 'items_v1.jsonl', 's0': 's0_20261013.json',
             'anchors': 'anchors_20261013.json'}
    with open(os.path.join(d, files['plan']), 'w') as f:
        f.write(json.dumps(p, indent=1) + '\n')
    with open(os.path.join(d, files['states']), 'w') as f:
        f.write(json.dumps(states, indent=1) + '\n')
    with open(os.path.join(d, files['items']), 'w') as f:
        f.write(''.join(json.dumps(it) + '\n' for it in items))
    shutil.copyfile(S0_FILE, os.path.join(d, files['s0']))
    shutil.copyfile(ANCHORS_FILE, os.path.join(d, files['anchors']))
    lock = {k: hashlib.sha256(open(os.path.join(d, f), 'rb').read())
            .hexdigest() for k, f in files.items()}
    with open(os.path.join(d, 'lock_v1.json'), 'w') as f:
        f.write(json.dumps(lock, indent=1) + '\n')
    return os.path.join(d, files['plan'])
