# Part of qqq-microstructure.
#
# The decider battery's runner (RESULTS 27): every committed item, sent to
# every candidate decision model on OpenRouter's Decisions API, one line per
# call in an append-only log. It sends and records; scoring and selection
# read the log (battery_score). Why it is built the way it is:
#   - the bytes sent are the committed bytes: an item's body plus the pinned
#     model, built in one place (send_bytes); for the real state of
#     2026-10-13 they are exactly guard.request_body's, the record's bytes
#   - the key: proxy mode never reads it; env mode holds it in one variable
#     and puts it only in a request header; every string written or printed
#     passes scrub(); /key keeps two floats (its label can hold the key)
#   - money: one pilot call per model measures the cost first, a projection
#     scales the run down once (run_meta, immutable), and no call goes out
#     that could cross the cap
#   - no shopping among re-runs: what runs is what was committed (the plan
#     and its files tracked, nothing uncommitted under src/, tests/,
#     bench/battery/ or the plan's directory), the run directory is one
#     child of <plan dir>/runs/, the log is fsynced per line, re-running the
#     same command resumes it -- only with the bytes it started with, and
#     never once it has been scored -- and a second run directory for the
#     same plan, on disk or in git history, is refused
#   - the API is alpha: 401/403 stop everything; 402 (no credit) stops
#     sending, resumably; 408, 429, 5xx and timeouts back off and get one
#     more pass at the end; a model whose calls keep failing that way is
#     parked so the rest move on; any other 4xx, or an answer that is not
#     every asked id as a noul p from the pinned model and a named provider,
#     is recorded invalid -- never guessed at
#
# Exit codes: 0 done, 2 auth failure, 3 capped, out of credit or incomplete
# (resumable), 4 deadline (resumable), 5 refused.

import os, re, sys, glob, json, math, time, random, hashlib, threading
import subprocess
import datetime as dt
import http.client
import urllib.request, urllib.error
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECISIONS = 'https://openrouter.ai/api/alpha/decisions'
KEY_URL = 'https://openrouter.ai/api/v1/key'
DONE, AUTH, CAPPED, DEADLINE, REFUSED = 0, 2, 3, 4, 5
TRIES = 6
BACKOFF = (1, 2, 4, 8, 16, 32)          # seconds, jittered +-25%
RETRY_AFTER_MAX = 60
# 408 is OpenRouter's 'request timed out': a timeout like any other
TRANSIENT = {408, 429, 502, 503, 504, 529}  # and every other 5xx, timeouts
NO_CREDIT = 402                         # stops sending; the key stays open
NO_USAGE_COST = 0.01                    # an answer (200) with no usage at all
# A model whose BREAKER calls in a row each failed transiently after every
# try is parked for the rest of the pass, so one provider that is down does
# not hold every model at each block boundary (a timed-out call takes ~6.5
# minutes). See _trip for when a parked model's keys are settled.
BREAKER = 3
SCORED = ('summary.json', 'selection_*.json', 'iter_*_outcome.json')
FROM_W = '__FROM_W__'                   # PP's provider: the one W was served by
# sk-... and Bearer ... (spec), plus any sk-or- prefix at all: scrub-check
# flags 'sk-or-' and 'Bearer ', so nothing it could flag may survive scrub()
SECRETS = (re.compile(r'sk-[A-Za-z0-9_-]{8,}'), re.compile(r'Bearer\s+\S*'),
           re.compile(r'sk-or-[A-Za-z0-9_-]*'))
FLAGS = ('sk-or-', 'Bearer ')
B2B = re.compile(r'B2B\d+$')


# ---------------------------------------------------------------- secrecy --

def scrub(s):
    s = str(s)
    for r in SECRETS:
        s = r.sub('***', s)
    return s


def _scrubbed(x):
    if isinstance(x, str):
        return scrub(x)
    if isinstance(x, dict):
        return {scrub(k) if isinstance(k, str) else k: _scrubbed(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_scrubbed(v) for v in x]
    return x


def scrub_check(run_dir):
    """Every file in the run directory that holds 'sk-or-' or 'Bearer '
    (reported by name, never by content). Empty: clean."""
    hits = []
    for d, _, fs in os.walk(run_dir):
        for f in sorted(fs):
            p = os.path.join(d, f)
            with open(p, 'rb') as fh:
                text = fh.read().decode('utf-8', 'replace')
            hits += [f'{os.path.relpath(p, run_dir)}: {k.strip()}'
                     for k in FLAGS if k in text]
    return hits


def say(msg):
    print(scrub(msg), flush=True)


def _refuse(why):
    say(f'refused: {why}')
    return REFUSED


# ------------------------------------------------------------- the items --

def _sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_plan(plan_path):
    """(plan, items, plan_dir), as committed."""
    d = os.path.dirname(os.path.abspath(plan_path))
    with open(plan_path, encoding='utf-8') as f:
        plan = json.load(f)
    with open(os.path.join(d, plan['files']['items']), encoding='utf-8') as f:
        items = [json.loads(ln) for ln in f if ln.strip()]
    return plan, items, d


def models_of(plan):
    """Every model the plan tests: the candidates, with I and R always in."""
    ids = [c['id'] for c in plan['candidates']]
    return ids + [m for m in (plan['incumbent'], plan['reference'])
                  if m not in ids]


def _w_id(items):
    w = [it['item_id'] for it in items
         if it['experiment'] == 'E0' and it['arm'] == 'W']
    return w[0] if len(w) == 1 else None


def problems(plan_path, plan, items, plan_dir):
    """Why these files are not the committed declaration (empty: they are)."""
    out, files = [], plan['files']
    lock = os.path.join(plan_dir,
                        files.get('lock', f'lock_{plan["version"]}.json'))
    if not os.path.exists(lock):
        out.append(f'no lock file {os.path.basename(lock)}')
    else:
        with open(lock, encoding='utf-8') as f:
            lk = json.load(f)
        want = {'plan': plan_path,
                'items': os.path.join(plan_dir, files['items'])}
        want.update({k: os.path.join(plan_dir, files[k])
                     for k in ('states', 's0', 'anchors') if files.get(k)})
        for k, p in want.items():
            if k not in lk:
                if k in ('plan', 'items'):
                    out.append(f'the lock has no {k} sha')
            elif not os.path.exists(p) or _sha(p) != lk[k]:
                out.append(f'{k} does not match the lock')
    seen = set()
    for it in items:
        b = it.get('body', {})
        if 'model' in b:
            out.append(f'{it["item_id"]}: a body names a model')
        if hashlib.sha256(json.dumps(b).encode()).hexdigest() != \
                it.get('body_sha256'):
            out.append(f'{it["item_id"]}: body_sha256 does not match')
        if it['item_id'] in seen:
            out.append(f'{it["item_id"]}: duplicate item id')
        seen.add(it['item_id'])
    for m in models_of(plan):
        if m.startswith('~') or ':free' in m or '|' in m:
            out.append(f'{m}: not a pinned id')
    w = _w_id(items)
    if w is None:
        out.append('the plan needs exactly one E0 W item (the pilot)')
    elif files.get('s0') and os.path.exists(os.path.join(plan_dir,
                                                         files['s0'])):
        # the pilot is the record's own request: the bytes every logged
        # night was asked with (guard.request_body's defaults)
        import guard
        with open(os.path.join(plan_dir, files['s0']), encoding='utf-8') as f:
            s0 = json.load(f)
        body = next(it['body'] for it in items if it['item_id'] == w)
        if send_bytes(body, 'pin') != json.dumps(guard.request_body(
                s0['state'], s0['names'], 'pin')).encode():
            out.append('the W item is not the canonical S0 request')
    return out


def dispatch_order(plan, items, models=None):
    """Every (model, item_id) pair in the order it is sent: blocks in order;
    within a block every pair shuffled with the dispatch seed, one stream per
    block (a block's order never depends on another's); a back-to-back
    triple rides on the slot of its first member (and goes out one member
    at a time, see _pick). Computed over the whole plan, so a scale-down
    only filters it."""
    models = models or models_of(plan)
    by_id = {it['item_id']: it for it in items}
    seed = plan['seeds']['dispatch']
    out = []
    for b in sorted({it['block'] for it in items}):
        pairs = [(m, it['item_id']) for m in models for it in items
                 if it['block'] == b]
        rng = np.random.default_rng(np.random.SeedSequence(seed,
                                                           spawn_key=(b,)))
        pairs = [pairs[i] for i in rng.permutation(len(pairs))]
        out += _back_to_back(pairs, by_id)
    return out


def _back_to_back(pairs, by_id):
    def group(m, it):
        return (m, it['experiment'], it['state_id'])
    members = {}
    for m, i in pairs:
        if B2B.match(by_id[i]['arm']):
            members.setdefault(group(m, by_id[i]), []).append(i)
    out, placed = [], set()
    for m, i in pairs:
        it = by_id[i]
        if not B2B.match(it['arm']):
            out.append((m, i))
        elif group(m, it) not in placed:
            placed.add(group(m, it))
            out += [(m, x) for x in sorted(members[group(m, it)],
                                           key=lambda x: by_id[x]['arm'])]
    return out


def _cut(it, sd):
    # truncation drops a synthetic block; the S0 items dispatched in it stay
    # (they are reported apart from the blocks: the spec's floor counts)
    return it['block'] in sd.get('dropped_blocks', []) and \
        it['state_id'] != 'S0'


def planned(plan, items, scale_down=None):
    """The keys ('model|item_id') this run must settle, after a scale-down:
    a dropped model keeps only its pilot call."""
    sd, w = scale_down or {}, _w_id(items)
    gone = set(sd.get('dropped_models', []))
    return {f'{m}|{it["item_id"]}' for m in models_of(plan) for it in items
            if not _cut(it, sd) and (m not in gone or it['item_id'] == w)}


def send_bytes(body, pin):
    """The one construction of what is sent (contract): the item's state and
    questions with the pinned model between them, then its provider."""
    send = {'state': body['state'], 'model': pin,
            'questions': body['questions']}
    if 'provider' in body:
        send['provider'] = body['provider']
    return json.dumps(send).encode()


def _wants_w(p):
    return p == FROM_W or (isinstance(p, dict) and
                           FROM_W in (p.get('order') or []))


def _fill(p, prov):
    if p == FROM_W:
        return {'order': [prov], 'allow_fallbacks': False}
    return dict(p, order=[prov if x == FROM_W else x for x in p['order']])


# -------------------------------------------------------------- validity --

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and \
        math.isfinite(x)


def _noul(a):
    if not isinstance(a, dict) or a.get('type') != 'noul':
        return None
    return float(a['noul']) if _num(a.get('noul')) else None


def valid_call(pin, status, answers, answered_as, provider):
    """(valid, why not): HTTP 200, every sent id a noul p in [0, 1], answered
    by the pin or a dated build of it, from a named provider."""
    if status != 200:
        return False, f'HTTP {status}'
    bad = [q for q, p in answers.items() if p is None or not 0 <= p <= 1]
    if not answers or bad:
        return False, (f'{len(bad)} of {len(answers)} answers missing, not '
                       f'noul or outside [0, 1]')
    if not isinstance(answered_as, str) or \
            not re.fullmatch(re.escape(pin) + r'(-\d{8})?', answered_as):
        return False, f'answered as {answered_as!r}, not {pin}'
    if not provider:
        return False, 'no provider named'
    return True, None


# ------------------------------------------------------------------ files --

def read_lines(run_dir):
    """Every parseable line of calls.jsonl, in order (a crash can cut the
    last one; it is skipped, and the writer starts a fresh line)."""
    path = os.path.join(run_dir, 'calls.jsonl')
    out = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for ln in f:
                try:
                    rec = json.loads(ln)
                except ValueError:
                    continue
                if isinstance(rec, dict) and 'key' in rec:
                    out.append(rec)
    return out


def read_calls(run_dir):
    """key -> the last final line of that key."""
    return {r['key']: r for r in read_lines(run_dir) if r.get('final')}


class _Log:
    def __init__(self, run_dir):
        path = os.path.join(run_dir, 'calls.jsonl')
        if os.path.exists(path) and os.path.getsize(path):
            with open(path, 'rb') as f:
                f.seek(-1, os.SEEK_END)
                cut = f.read(1) != b'\n'
        else:
            cut = False
        self.f = open(path, 'a', encoding='utf-8', newline='\n')
        if cut:
            self.f.write('\n')

    def write(self, rec):
        self.f.write(json.dumps(_scrubbed(rec)) + '\n')
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_meta(run_dir, meta):
    """run_meta.json, atomically. The scale-down is decided once: a write
    that would change it raises."""
    path = os.path.join(run_dir, 'run_meta.json')
    old = _read_json(path)
    if old and old.get('scale_down') is not None and \
            meta.get('scale_down') != old['scale_down']:
        raise ValueError('the scale-down decision is immutable once written')
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(_scrubbed(meta), indent=1) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(
        timespec='milliseconds').replace('+00:00', 'Z')


def _git(*args):
    try:
        r = subprocess.run(['git', '-C', ROOT, *args], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _rel(path):
    return os.path.relpath(path, ROOT).replace(os.sep, '/').rstrip('/')


def _dirty(skip, plan_dir=None):
    """Changed or untracked paths under src/, tests/, bench/battery/ and the
    plan's directory, except under `skip` (the run directory): what runs
    must be what was committed."""
    where = ['src', 'tests', 'bench/battery']
    if plan_dir is not None and _inside(os.path.abspath(plan_dir), ROOT):
        where.append(_rel(os.path.abspath(plan_dir)))
    s = _git('status', '--porcelain', '--untracked-files=all', '--', *where)
    if s is None:
        return ['(git status failed)']
    mine = _rel(os.path.abspath(skip))
    bad = []
    for ln in s.splitlines():
        p = ln[3:].split(' -> ')[-1].strip().strip('"')
        if not (p + '/').startswith(mine + '/'):
            bad.append(p)
    return sorted(set(bad))


def _inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:                  # another drive
        return False


def declared_files(plan_path, plan, plan_dir):
    """{name: path} of the declaration: the plan, the files it names, and
    the lock beside it."""
    fs = plan['files']
    out = {'plan': os.path.abspath(plan_path)}
    out.update({k: os.path.join(plan_dir, fs[k])
                for k in ('items', 'states', 's0', 'anchors') if fs.get(k)})
    out['lock'] = os.path.join(plan_dir,
                               fs.get('lock', f'lock_{plan["version"]}.json'))
    return out


def declared_shas(plan_path, plan, plan_dir):
    return {k: _sha(p) if os.path.exists(p) else None
            for k, p in declared_files(plan_path, plan, plan_dir).items()}


def tree_problems(plan_path, plan, plan_dir, out_dir=None):
    """Why this checkout may not run this plan (empty: it may). What runs
    must be what was committed: the plan inside this repository; it, every
    file it names and the lock tracked by git; nothing changed or untracked
    under src/, tests/, bench/battery/ or the plan's directory, outside the
    run directory (out_dir; for verify, anything under <plan dir>/runs/);
    and out_dir a direct child of <plan dir>/runs/ -- a run written
    anywhere else could hide a first run from a second, or the run
    directory could swallow src/ out of the check."""
    plan_path = os.path.abspath(plan_path)
    if not _inside(plan_path, ROOT):
        return [f'the plan is not in this repository ({plan_path})']
    runs = os.path.join(plan_dir, 'runs')
    if out_dir is not None and os.path.dirname(out_dir) != runs:
        return [f'the run directory must be directly under {_rel(runs)}/, '
                f'not {_rel(out_dir)}']
    out = []
    rels = [_rel(p) for p in declared_files(plan_path, plan,
                                            plan_dir).values()]
    ls = _git('ls-files', '--', *rels)
    if ls is None:
        out.append('(git ls-files failed)')
    else:
        missing = sorted(set(rels) - set(ls.splitlines()))
        if missing:
            out.append('not committed: ' + ', '.join(missing))
    dirty = _dirty(out_dir if out_dir is not None else runs, plan_dir)
    if dirty:
        out.append('uncommitted changes: ' + ', '.join(dirty[:5])
                   + (', ...' if len(dirty) > 5 else ''))
    return out


def _other_run(out_dir, plan_dir, sha):
    for r in sorted({os.path.dirname(out_dir), os.path.join(plan_dir,
                                                            'runs')}):
        if not os.path.isdir(r):
            continue
        for name in sorted(os.listdir(r)):
            p = os.path.join(r, name)
            if p == out_dir:
                continue
            m = _read_json(os.path.join(p, 'run_meta.json'))
            if m and m.get('plan_sha') == sha:
                return p
    return None


def _run_in_history(plan_dir, sha):
    """Files under <plan dir>/runs/ that held this plan's sha in any commit
    of any ref: a run directory deleted or lost after it was committed."""
    s = _git('log', '--all', '--format=', '--name-only', '-S', sha, '--',
             _rel(os.path.join(plan_dir, 'runs')))
    if s is None:
        return ['(git log failed)']
    return sorted({p for p in s.splitlines() if p.strip()})


def _scored(out_dir):
    """The scorer's outputs already in a run directory: a run that has been
    scored (its results seen) is frozen."""
    return sorted(os.path.basename(p) for pat in SCORED
                  for p in glob.glob(os.path.join(out_dir, pat)))


# ------------------------------------------------------------------ calls --

def _retry_after(ex):
    try:
        s = float((ex.headers or {}).get('Retry-After'))
    except (TypeError, ValueError):
        return None
    return min(max(s, 0.0), RETRY_AFTER_MAX) if math.isfinite(s) else None


def _err_body(ex):
    try:
        return scrub(ex.read(4000).decode('utf-8', 'replace'))[:300]
    except Exception:
        return ''


def _f(x):
    return float(x) if _num(x) else None


class _Run:
    """One session of a run: the pilot, the main dispatch, the retry pass.
    All bookkeeping (log, spend, smoke stop) happens on the calling thread;
    worker threads only make HTTP requests."""

    def __init__(self, plan, items, out_dir, key, auth, max_usd, deadline_at,
                 concurrency, per_model, opener, sleep, now, decisions_url,
                 breaker=None):
        self.plan, self.out, self.key, self.auth = plan, out_dir, key, auth
        self.items_list = items
        self.items = {it['item_id']: it for it in items}
        self.models = models_of(plan)
        self.order = dispatch_order(plan, items, self.models)
        self.w = _w_id(items)
        self.max_usd, self.deadline_at = max_usd, deadline_at
        self.concurrency, self.per_model = concurrency, per_model
        self.opener, self.sleep, self.now = opener, sleep, now
        self.url, self.timeout = decisions_url, plan.get('timeout_s', 60)
        cand = {c['id']: c for c in plan['candidates']}
        self.price = {m: cand.get(m, {}).get('price_in') for m in self.models}
        self.scored = {m: cand.get(m, {}).get('X') is not None
                       for m in self.models}
        self.exempt = {plan['incumbent'], plan['reference']}
        sp = plan['spend']
        # before a model's pilot: its catalogue-scaled pessimistic cost
        self.c0 = {m: sp['pessimistic_per_call'] * (
            self.price[m] / sp['price_ref']
            if self.price[m] is not None and sp.get('price_ref') else 1.0)
            for m in self.models}
        self.lines = read_lines(out_dir)
        self.last = {}
        for ln in self.lines:
            self.last[ln['key']] = ln
        self.spent = sum(self._billed(ln) for ln in self.lines)
        # c_m is measured only from a W that was answered (200): a failed W
        # costs nothing and says nothing about what an answer costs
        self.cm = {}
        for m in self.models:
            ln = self.last.get(f'{m}|{self.w}')
            if ln and ln.get('attempts') and ln.get('http_status') == 200:
                self.cm[m] = self._cost(m, ln)
        self.reserved = 0.0
        self.sd = None
        self.keys = planned(plan, items)
        self.stopped, self.smoke = set(), {}
        self.abort = self.capped = self.late = False
        self.no_credit = self.outage = False
        # the circuit breaker: {model: {'trips', 'others_answered'}} carried
        # across sessions in run_meta; per pass, the parked models and each
        # model's current streak [first failing send, count]
        self.breaker = breaker if breaker is not None else {}
        self.parked, self.streak, self.trip_start = set(), {}, {}
        self.sends, self.valid_sends = 0, []
        self.b2b_busy = set()
        self.stop = threading.Event()
        self.rng = random.Random()
        self.log = None

    # ------------------------------------------------------------ money --
    def _cost(self, m, line):
        """c_m from a line: usage.cost; else the input tokens at the
        catalogue price, doubled; else $0.01 (spec section 7)."""
        if _num(line.get('cost')):
            return float(line['cost'])
        u = line.get('usage') or {}
        tok, price = u.get('input_tokens'), self.price.get(m)
        if _num(tok) and _num(price):
            return tok * price * 2
        return NO_USAGE_COST

    def _billed(self, line):
        # an HTTP error is not billed; an answer that names no cost is, at
        # the estimate (the /key usage cross-checks the sum at the end)
        if not line.get('attempts') or line.get('http_status') != 200:
            return 0.0
        return self._cost(line['model'], line)

    def _est(self, m):
        if m in self.cm:
            return self.cm[m]
        return max([self.c0[m]] + list(self.cm.values()))

    def _gone(self):
        return set(self.sd['dropped_models']) if self.sd else set()

    def _affordable(self):
        # spent + 1.5 max_m c_m <= cap, with every call in flight counted
        # as already spent
        live = [m for m in self.models
                if m not in self.stopped and m not in self._gone()]
        top = max((self._est(m) for m in live), default=0.0)
        return self.spent + self.reserved + 1.5 * top <= self.max_usd + 1e-12

    # ----------------------------------------------------------- the log --
    def _write(self, rec):
        self.log.write(rec)
        self.lines.append(rec)
        self.last[rec['key']] = rec

    def _final(self, k):
        ln = self.last.get(k)
        return bool(ln and ln.get('final'))

    def _rec(self, m, it, qids, **kw):
        r = {'key': f'{m}|{it["item_id"]}', 'model': m,
             'item_id': it['item_id'], 'experiment': it['experiment'],
             'state_id': it['state_id'], 'block': it['block'],
             't_sent_utc': None, 'latency_ms': None, 'http_status': None,
             'attempts': 0, 'answered_as': None, 'provider': None,
             'answers': {q: None for q in qids}, 'usage': None, 'cost': None,
             'valid': False, 'final': False, 'error': None}
        r.update(kw)
        if r['error'] is not None:
            r['error'] = scrub(r['error'])[:200]
        return r

    def _outcome(self, m, it, qids, out, final):
        resp = out['resp'] if isinstance(out['resp'], dict) else {}
        got = resp.get('answers')
        got = got if isinstance(got, dict) else {}
        answers = {q: _noul(got.get(q)) for q in qids}
        who = resp.get('model') if isinstance(resp.get('model'), str) \
            else None
        prov = resp.get('provider') if isinstance(resp.get('provider'), str) \
            else None
        usage = resp.get('usage') if isinstance(resp.get('usage'), dict) \
            else None
        cost = float(usage['cost']) if usage and _num(usage.get('cost')) \
            else None
        ok, why = valid_call(m, out['status'], answers, who, prov)
        err = out['error'] or (None if ok else why)
        st = out['status']
        if it['arm'] == 'PP' and st and 400 <= st < 500 and \
                not out['transient'] and not out['abort'] and \
                not out['credit']:
            err = f'pin unsupported: {err}'       # not held against G1
        settled = (ok or not out['transient'] or final) and \
            not out['abort'] and not out['credit']
        return self._rec(m, it, qids, t_sent_utc=out['t_sent'],
                         latency_ms=out['latency_ms'], http_status=st,
                         attempts=out['attempts'], answered_as=who,
                         provider=prov, answers=answers, usage=usage,
                         cost=cost, valid=ok, final=settled, error=err)

    # ---------------------------------------------------------- the wire --
    def _headers(self, **h):
        if self.key:
            h['Authorization'] = 'Bearer ' + self.key
        return h

    def _http(self, req):
        """One request under the retry policy, on a worker thread. Returns
        what happened; raises nothing but a crash."""
        out = {'attempts': 0, 'status': None, 'resp': None, 'error': None,
               'transient': False, 'abort': False, 'credit': False,
               'latency_ms': None, 't_sent': None}
        for i in range(TRIES):
            if self.stop.is_set():
                break
            out['t_sent'] = out['t_sent'] or _utc()
            out['attempts'] += 1
            pause, t0 = None, self.now()
            try:
                with self.opener(req, timeout=self.timeout) as r:
                    raw = r.read()
                    st = getattr(r, 'status', None) or 200
                out.update(status=st, transient=False, error=None,
                           latency_ms=int(round((self.now() - t0) * 1000)))
                if st != 200:
                    out['error'] = f'HTTP {st}'
                    return out
                try:
                    out['resp'] = json.loads(raw.decode('utf-8'))
                except ValueError:
                    out['error'] = 'unparseable answer'
                return out
            except urllib.error.HTTPError as ex:
                out.update(status=ex.code, latency_ms=None,
                           error=f'HTTP {ex.code}: {_err_body(ex)}')
                if ex.code in (401, 403):
                    out.update(abort=True, transient=False)
                    return out
                if ex.code == NO_CREDIT:        # top up and resume: the
                    out.update(credit=True, transient=False)   # key stays
                    return out                  # open, never invalid
                if ex.code not in TRANSIENT and ex.code < 500:
                    out['transient'] = False
                    return out
                out['transient'] = True
                if ex.code == 429:
                    pause = _retry_after(ex)
            except (urllib.error.URLError, http.client.HTTPException,
                    OSError) as ex:                 # timeouts, resets, DNS
                out.update(status=None, latency_ms=None, transient=True,
                           error=f'{type(ex).__name__}: {ex}')
            if i == TRIES - 1:
                break
            if pause is None:
                pause = BACKOFF[i] * (0.75 + 0.5 * self.rng.random())
            if self.now() + pause > self.deadline_at or self.stop.is_set():
                break
            self.sleep(pause)
        return out

    def _call(self, data):
        return self._http(urllib.request.Request(
            self.url, data=data, method='POST',
            headers=self._headers(**{'Content-Type': 'application/json'})))

    def key_floats(self, url):
        """('auth' | 'failed' | None, {limit_remaining, usage}): the two
        floats of GET /key; the rest -- the label can hold the key itself --
        is dropped unread."""
        out = self._http(urllib.request.Request(url, method='GET',
                                                headers=self._headers()))
        if out['abort']:
            return 'auth', None
        j = out['resp'] if out['status'] == 200 else None
        if not isinstance(j, dict):
            return 'failed', None
        d = j['data'] if isinstance(j.get('data'), dict) else j
        return None, {'limit_remaining': _f(d.get('limit_remaining')),
                      'usage': _f(d.get('usage'))}

    # ------------------------------------------------------- dispatching --
    def _halted(self):
        return self.abort or self.capped or self.late or self.no_credit or \
            self.outage

    def _body(self, m, it):
        body = it['body']
        if not _wants_w(body.get('provider')):
            return body
        wl = self.last.get(f'{m}|{self.w}')
        prov = wl.get('provider') if wl and wl.get('valid') else None
        return dict(body, provider=_fill(body['provider'], prov)) if prov \
            else None

    def _pick(self, q, per):
        best = None
        for m, dq in q.items():
            if not dq or per[m] >= self.per_model:
                continue
            s = self.smoke.get(m)
            if s is not None and s['sent'] >= 2:
                continue                # W failed: two more, then judge
            if m in self.b2b_busy and B2B.match(self.items[dq[0][1]]['arm']):
                # the back-to-back triple is sent on one slot: a member
                # goes out only once the one before it has landed, so a
                # cache that serves identical bytes can show itself
                continue
            if best is None or dq[0][0] < q[best][0][0]:
                best = m
        return best

    def _send(self, ex, flight, per, m, iid, final):
        it = self.items[iid]
        body = self._body(m, it)
        if body is None:                 # PP: W named no provider to pin
            wl = self.last.get(f'{m}|{self.w}')
            self._write(self._rec(m, it, list(it['body']['questions']),
                                  error='no W provider',
                                  final=final or bool(wl and wl['final'])))
            return
        c = self._est(m)
        counted = m in self.smoke
        if counted:
            self.smoke[m]['sent'] += 1
        if B2B.match(it['arm']):
            self.b2b_busy.add(m)
        self.sends += 1
        fut = ex.submit(self._call, send_bytes(body, m))
        flight[fut] = (m, iid, c, list(body['questions']), counted,
                       self.sends)
        per[m] += 1
        self.reserved += c

    def _land(self, flight, per, final):
        done, _ = wait(list(flight), return_when=FIRST_COMPLETED)
        for fut in done:
            m, iid, c, qids, counted, seq = flight.pop(fut)
            per[m] -= 1
            self.reserved -= c
            if B2B.match(self.items[iid]['arm']):
                self.b2b_busy.discard(m)
            out = fut.result()           # an injected crash propagates
            if not out['attempts']:      # stopped before it was sent
                if counted and m in self.smoke:
                    self.smoke[m]['sent'] -= 1
                continue
            line = self._outcome(m, self.items[iid], qids, out,
                                 final or m in self.stopped)
            self._write(line)
            self.spent += self._billed(line)
            if out['abort']:
                self.abort = True
                self.stop.set()
            if out['credit'] and not self.no_credit:
                self.no_credit = True
                self.stop.set()
                say('HTTP 402 (insufficient credits): sending stopped -- '
                    'top up, then resume with the same command')
            if iid == self.w and line['http_status'] == 200:
                self.cm[m] = self._cost(m, line)
            if line['valid']:
                self.valid_sends.append((seq, m))
            if counted and m in self.smoke:
                self._smoke(m, line['valid'], flight)
            if out['transient'] and out['attempts'] >= TRIES:
                st = self.streak.setdefault(m, [seq, 0])
                st[0], st[1] = min(st[0], seq), st[1] + 1
                if st[1] >= BREAKER and m not in self.stopped and \
                        m not in self.parked:
                    self._trip(m, flight, final)
            elif not out['transient']:
                self.streak.pop(m, None)

    def _dispatch(self, pairs, final=False):
        """Send pairs in their order: all of a block's pairs go out before
        any of the next block's; at most `concurrency` in flight and
        `per_model` per model. Stops sending on abort, cap or deadline, and
        returns once the calls in flight have landed."""
        groups = []
        for m, iid in pairs:
            b = self.items[iid]['block']
            if not groups or groups[-1][0] != b:
                groups.append((b, []))
            groups[-1][1].append((m, iid))
        flight, per = {}, Counter()
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            try:
                for _, grp in groups:
                    q = {}
                    for pos, (m, iid) in enumerate(grp):
                        q.setdefault(m, deque()).append((pos, iid))
                    while not self._halted():
                        for m in [m for m in q if m in self.stopped
                                  or m in self.parked]:
                            del q[m]
                        if not any(q.values()):
                            break
                        pick = self._pick(q, per) \
                            if len(flight) < self.concurrency else None
                        if pick is not None and self.now() >= \
                                self.deadline_at:
                            self.late = True
                            break
                        if pick is not None and not self._affordable():
                            if not flight:
                                self.capped = True
                                break
                            pick = None      # wait: a landing may free room
                        if pick is None:
                            if not flight:
                                break
                            self._land(flight, per, final)
                            continue
                        self._send(ex, flight, per, pick,
                                   q[pick].popleft()[1], final)
                    if self._halted():
                        break
                while flight:
                    self._land(flight, per, final)
            except BaseException:
                self.stop.set()
                raise

    # -------------------------------------------------------- smoke stop --
    def _keys_of(self, m):
        keys = self.keys
        return [(i, f'{m}|{i}') for mm, i in self.order
                if mm == m and f'{m}|{i}' in keys]

    def _smoke(self, m, ok, flight=None):
        s = self.smoke[m]
        s['results'].append(ok)
        if ok or len(s['results']) >= 2:
            del self.smoke[m]
            if not ok and not any(s['results']):
                self._stop_model(m, flight or {})

    def _stop_model(self, m, flight, why='smoke stop',
                    note='smoke stop (W and the next two calls invalid)'):
        """The smoke stop (spec E0): W and the next two calls all invalid.
        Every open key of the model is settled invalid, so no resume sends
        it again; the model fails G1. Never applied to I or R. (_trip
        settles a model that stays down the same way.)"""
        self.stopped.add(m)
        busy = {f'{x[0]}|{x[1]}' for x in flight.values()}
        for iid, k in self._keys_of(m):
            ln = self.last.get(k)
            if k in busy or (ln and ln.get('final')):
                continue
            err = why + (f' after {ln["error"]}'
                         if ln and ln.get('error') else '')
            it = self.items[iid]
            self._write(self._rec(m, it, list(it['body']['questions']),
                                  error=err, final=True))
        say(f'{m}: {note}')

    def _trip(self, m, flight, final):
        """BREAKER calls of m in a row failed transiently after every try.
        m is parked for the rest of this pass -- its keys stay open for the
        retry pass or a resume, and the block boundary stops waiting for
        it. Whether other models answered meanwhile (calls sent after m's
        streak began) tells a provider that is down from an API that is.
        A trip in the retry pass, whose results are final anyway, settles
        m's remaining keys 'unavailable' when m has been found down while
        others answered (in this session or an earlier one); the whole API
        being down never settles anything."""
        start = self.streak.pop(m)[0]
        self.parked.add(m)
        self.trip_start[m] = min(start, self.trip_start.get(m, start))
        self.breaker.setdefault(m, {'trips': 0, 'others_answered': False})
        self.breaker[m]['trips'] += 1
        if final and self.others_answered(m):
            self._stop_model(
                m, flight, why='unavailable',
                note=f'unavailable: {BREAKER} calls in a row failed after '
                     f'every retry, again in the retry pass -- its open keys '
                     f'are settled invalid')
        else:
            say(f'{m}: {BREAKER} calls in a row failed after every retry -- '
                f'parked for this pass')

    def others_answered(self, m):
        """Whether m was found down while the API answered: BREAKER valid
        answers from other models to calls sent after the streak that
        tripped m began (in this session, or recorded in an earlier one).
        Counted when asked, so answers that land after the trip count."""
        b = self.breaker.get(m)
        if b is None:
            return False
        if not b['others_answered'] and m in self.trip_start:
            n = sum(1 for seq, x in self.valid_sends
                    if seq > self.trip_start[m] and x != m)
            b['others_answered'] = n >= BREAKER
        return b['others_answered']

    def _smoke_init(self):
        for m in self.models:
            if m in self.exempt or m in self._gone():
                continue
            wk = f'{m}|{self.w}'
            wl = self.last.get(wk)
            if not wl or wl.get('valid') or \
                    all(self._final(k) for _, k in self._keys_of(m)):
                continue
            # a resumed run: the calls already made after the last W
            at = max(i for i, ln in enumerate(self.lines) if ln['key'] == wk)
            seen, res = set(), []
            for ln in self.lines[at + 1:]:
                if ln['model'] != m or ln['key'] == wk or \
                        not ln.get('attempts') or ln['key'] in seen:
                    continue
                seen.add(ln['key'])
                res.append(bool(ln.get('valid')))
                if len(res) == 2:
                    break
            if any(res):
                continue
            if len(res) == 2:
                self._stop_model(m, {})
            else:
                self.smoke[m] = {'sent': len(res), 'results': res}

    # ------------------------------------------------------------ phases --
    def pilot(self, meta):
        """E0 W first, for every model: the cost per call, measured."""
        self._dispatch([(m, i) for m, i in self.order
                        if i == self.w and not self._final(f'{m}|{i}')])
        if meta.get('scale_down') is None and not self._halted():
            self._scale_down(meta)
            write_meta(self.out, meta)
        self.sd = meta.get('scale_down')
        self.keys = planned(self.plan, self.items_list, self.sd)

    def _scale_down(self, meta):
        """Projection sum_m c_m n_m; above 0.8 max_usd drop block 4, then
        unscored models, dearest first, until it fits -- never I or R. A
        model whose W was not answered (an unbilled 4xx, or a transient
        failure not yet retried) is projected at the dearest c_m measured,
        or at its catalogue estimate if no W was answered at all."""
        fill = max(self.cm.values()) if self.cm else None
        cm = {m: self.cm[m] if m in self.cm else
              (fill if fill is not None else self.c0[m])
              for m in self.models}
        lim = 0.8 * self.max_usd

        def proj(sd):
            n = Counter(k.split('|', 1)[0]
                        for k in planned(self.plan, self.items_list, sd))
            return sum(cm[m] * n[m] for m in self.models), sum(n.values())
        sd = {'dropped_blocks': [], 'dropped_models': []}
        p0, n0 = proj(sd)
        if p0 > lim:
            sd['dropped_blocks'] = [self.plan['blocks']]
            for m in sorted((m for m in self.models if not self.scored[m]
                             and m not in self.exempt),
                            key=lambda m: (-cm[m], m)):
                if proj(sd)[0] <= lim:
                    break
                sd['dropped_models'].append(m)
        p1, n1 = proj(sd)
        meta['pilot_costs'] = cm
        meta['pilot_unmeasured'] = sorted(m for m in self.models
                                          if m not in self.cm)
        meta['projection'] = {'usd': p0, 'calls': n0, 'limit_usd': lim,
                              'after_scale_down_usd': p1,
                              'after_scale_down_calls': n1}
        meta['scale_down'] = sd
        say(f'pilot: projected ${p0:.4f} for {n0} calls (limit ${lim:.4f}); '
            f'scale-down: blocks {sd["dropped_blocks"]}, '
            f'models {sd["dropped_models"]} -> ${p1:.4f} for {n1} calls')

    def main(self):
        self._smoke_init()
        keys = self.keys
        self._dispatch([(m, i) for m, i in self.order
                        if i != self.w and f'{m}|{i}' in keys
                        and m not in self.stopped
                        and not self._final(f'{m}|{i}')])
        # every model still open was parked, and none was ever found down
        # while others answered: the API itself is down. The retry pass
        # (final) would settle keys on an outage; stop, resume later.
        open_ = {k.split('|', 1)[0] for k in keys if not self._final(k)} \
            - self.stopped
        if self.parked and open_ and open_ <= self.parked and not any(
                self.others_answered(m) for m in self.parked):
            self.outage = True
            say('every open model failed every retry, none while others '
                'answered: an outage -- resume later')

    def retry_pass(self):
        """One more pass for what failed transiently; whatever it gets is
        final. W first, so a PP can still be pinned. A model parked in the
        main pass is tried again."""
        self.parked.clear()
        self.streak.clear()
        keys = self.keys
        redo = [(m, i) for m, i in self.order if f'{m}|{i}' in keys
                and m not in self.stopped and not self._final(f'{m}|{i}')]
        self._dispatch([p for p in redo if p[1] == self.w], final=True)
        if not self._halted():
            self._dispatch([p for p in redo if p[1] != self.w], final=True)

    def report(self):
        keys = self.keys
        usd = Counter()
        for ln in self.lines:
            usd[ln['model']] += self._billed(ln)
        for m in self.models:
            ks = [k for k in keys if k.split('|', 1)[0] == m]
            fin = sum(self._final(k) for k in ks)
            ok = sum(bool(self.last.get(k, {}).get('valid')) for k in ks)
            say(f'{m}: {fin}/{len(ks)} final, {ok} valid, ${usd[m]:.4f}')
        return [k for k in keys if not self._final(k)]


def run(plan_path, out_dir, auth='proxy', max_usd=None, deadline_min=None,
        concurrency=None, per_model=None, opener=urllib.request.urlopen,
        sleep=time.sleep, now=time.monotonic, key_url=KEY_URL,
        decisions_url=DECISIONS, check_tree=True):
    """Run (or resume) the battery for a plan into out_dir. Exit code per
    spec section 7. check_tree=False skips the checks that what runs is
    what was committed (tree_problems, the git history): for tests on a
    plan outside the repository only -- the command line cannot pass it."""
    plan_path, out_dir = os.path.abspath(plan_path), os.path.abspath(out_dir)
    plan, items, plan_dir = read_plan(plan_path)
    bad = problems(plan_path, plan, items, plan_dir)
    if bad:
        return _refuse('; '.join(bad[:5]))
    sp, psha = plan['spend'], _sha(plan_path)
    shas = declared_shas(plan_path, plan, plan_dir)
    if check_tree:
        bad = tree_problems(plan_path, plan, plan_dir, out_dir)
        if bad:
            return _refuse('; '.join(bad))
    meta = _read_json(os.path.join(out_dir, 'run_meta.json'))
    resume = meta is not None
    if resume and meta.get('plan_sha') != psha:
        return _refuse('this run directory belongs to another plan')
    if resume:
        # the bytes a run started with are the only ones it may go on with
        # (the plan's own sha does not cover its items)
        was = dict(meta.get('shas') or {}, items=meta.get('items_sha'))
        changed = sorted(k for k, v in was.items() if v and shas.get(k) != v)
        if changed:
            return _refuse(f'{", ".join(changed)} changed since this run '
                           f'started: it can only go on with its own bytes')
        done = _scored(out_dir)
        if done:
            return _refuse(f'this run has been scored ({", ".join(done)}): '
                           f'it is frozen')
    else:
        if os.path.exists(os.path.join(out_dir, 'calls.jsonl')):
            return _refuse('a call log without run_meta.json')
        other = _other_run(out_dir, plan_dir, psha)
        if other:
            return _refuse(f'a run directory for this plan already exists '
                           f'({other}): resume it')
        if check_tree:
            seen = _run_in_history(plan_dir, psha)
            if seen:
                return _refuse(f'a run of this plan is in git history '
                               f'({", ".join(seen[:3])}): resume it')
    max_usd = float(sp['max_usd'] if max_usd is None else max_usd)
    cap = sp['max_usd_ceiling'] if resume else sp['max_usd']
    if not 0 < max_usd <= cap:
        return _refuse(f'max_usd {max_usd} outside (0, {cap}]'
                       + ('' if resume else ' (a resume may go to the '
                          f'ceiling {sp["max_usd_ceiling"]})'))
    if auth == 'env':
        key = os.environ.get('OPENROUTER_API_KEY')   # held here, nowhere else
        if not key:
            say('auth failure: OPENROUTER_API_KEY is not set')
            return AUTH
    elif auth == 'proxy':
        key = None                       # the proxy adds it; never read here
    else:
        return _refuse(f'unknown auth {auth!r}')
    deadline_min = plan['deadline_min'] if deadline_min is None \
        else deadline_min
    r = _Run(plan, items, out_dir, key, auth, max_usd,
             now() + 60.0 * deadline_min,
             int(concurrency or plan['concurrency']),
             int(per_model or plan['per_model']), opener, sleep, now,
             decisions_url, breaker=(meta or {}).get('breaker'))
    del key
    err, kb = r.key_floats(key_url)
    if err == 'auth':
        say('auth failure: GET /key answered 401/403')
        return AUTH
    if err:
        return _refuse('the credit check (GET /key) failed')
    floor = max_usd + sp['credit_margin']
    if kb['limit_remaining'] is None or kb['limit_remaining'] < floor:
        return _refuse(f'credit ${kb["limit_remaining"]} is below the floor '
                       f'${floor:.2f} (max_usd + {sp["credit_margin"]})')
    os.makedirs(out_dir, exist_ok=True)
    commit = (_git('rev-parse', 'HEAD') or '').strip() or None
    tree = 'clean' if check_tree else 'not checked'
    if not resume:
        meta = {'plan': _rel(plan_path) if _inside(plan_path, ROOT)
                else plan_path,
                'plan_sha': psha, 'items_sha': shas['items'], 'shas': shas,
                'commit': commit, 'tree': tree,
                'python': sys.version.split()[0], 'numpy': np.__version__,
                'started_utc': _utc(), 'ended_utc': None,
                'key_before': kb, 'key_after': None, 'pilot_costs': None,
                'projection': None, 'scale_down': None, 'status': 'running',
                'spent_usd': r.spent, 'breaker': {}, 'sessions': []}
    session = {'started_utc': _utc(), 'ended_utc': None, 'auth': auth,
               'max_usd': max_usd, 'commit': commit, 'tree': tree,
               'key_before': kb, 'key_after': None,
               'spent_before_usd': r.spent, 'status': 'running'}
    if resume and check_tree and commit and meta.get('commit') and \
            commit != meta['commit']:
        # a resume at another commit is allowed (the run directory itself
        # is committed between sessions), but code changed under it is
        # recorded, and printed
        diff = _git('diff', '--name-only', meta['commit'], commit, '--',
                    'src', 'tests')
        session['code_changed'] = sorted(diff.split()) if diff is not None \
            else ['(git diff failed)']
        if session['code_changed']:
            say('code changed since this run started: '
                + ', '.join(session['code_changed'][:5]))
    meta['sessions'].append(session)
    meta['status'] = 'running'
    write_meta(out_dir, meta)
    r.log = _Log(out_dir)
    try:
        r.pilot(meta)
        if not r._halted():
            r.main()
        if not r._halted():
            r.retry_pass()
    finally:
        r.log.close()
    left = r.report()
    if r.abort:
        status, code = 'auth_failed', AUTH
    elif not left:
        status, code = 'done', DONE
    elif r.no_credit:
        status, code = 'no_credit', CAPPED
    elif r.capped:
        status, code = 'capped', CAPPED
    elif r.late:
        status, code = 'deadline', DEADLINE
    elif r.outage:
        status, code = 'outage', CAPPED
    else:
        status, code = 'incomplete', CAPPED
    for m in r.trip_start:              # recorded for later sessions
        r.others_answered(m)
    ka = None if r.abort else r.key_floats(key_url)[1]
    session.update(ended_utc=_utc(), key_after=ka, status=status,
                   parked=sorted(r.parked),
                   spent_usd=r.spent - session['spent_before_usd'],
                   key_usage_delta=(ka['usage'] - kb['usage'])
                   if ka and ka['usage'] is not None
                   and kb['usage'] is not None else None)
    meta.update(ended_utc=session['ended_utc'], key_after=ka, status=status,
                spent_usd=r.spent, breaker=r.breaker)
    write_meta(out_dir, meta)
    say(f'spent ${r.spent:.4f} of ${max_usd:.2f}; {len(left)} keys open -- '
        f'{status}')
    if r.abort:
        say('auth failure: a call answered 401/403 -- run aborted')
    return code
