# Part of qqq-microstructure.
#
# A Jev-style decision model in SHADOW MODE: it decides, it is graded, it
# never trades.
#
# Jev (TypeSafe AI, Sept 2026) is a "System One" decision model: one call
# carries a `state` and typed `questions`, and returns typed answers -- for a
# yes/no ("noul") question, a probability. Open, locally hosted servers that
# speak the same API exist (see RESULTS 27). This file asks such a model,
# every afternoon before the MOC cutoff:
#
#   per name   "hold {ticker} overnight tonight?"   -> p(yes)
#   the book   "hedge tonight's basket 1:1 with QQQ?" -> p(yes)
#
# appends the answers -- timestamped, with the model version the server
# reports -- to reports/guard_log.csv, and later grades the SHADOW book they
# imply against the frozen book on identical nights.
#
# Why shadow mode, and why forward only (RESULTS 27):
#   - A model built on a language-model backbone has read the internet
#     through its training cutoff, so it has, in effect, seen 2003-2026. Any
#     backtest of its decisions over those years is contaminated, so the only
#     honest test is decisions committed BEFORE the outcome, graded after.
#   - Gating the traded book would change the pre-registered object that the
#     forward test (RESULTS 19) is grading. The frozen book trades; the
#     shadow book is only logged.
#   - On the live nights so far, the losses were the whole basket falling
#     together (2026-09-11: all 11 names -4% to -9%), and earnings drove none
#     of them. A per-name veto has little to work with; the book-level hedge
#     question is the one with evidence behind it.
#
# Integrity rules, enforced, not hoped for:
#   - one decision per night per backend (the decider, the reference and
#     the earnings-rule control); a night is never re-decided
#   - a decision time-stamped at or after 15:50 ET on its night is ignored by
#     the grader (it could have seen the close)
#   - the log is append-only and committed before the close, from the
#     owner's own machine (decide_local.py) -- never from a cloud runner
#   - an answer is stamped when it ARRIVES; one that arrives at or after the
#     cutoff is not logged (a local model on a CPU takes minutes)
#   - one record per (backend, model): a new decider starts its own record and
#     its own 63-night clock, never pooled with the old one
#   - --dry-run asks and prints, writes nothing: for trying a decider by hand
#
# PRE-DECLARED GATE: the shadow book may be proposed for a dated re-freeze only
# if, over at least 63 decided-and-graded nights, its paired nightly
# difference against the frozen book has mean > 0 AND paired t > GATE_T
# (2.46: three records can pass, so 2's 2.5% level is split three ways).
# Otherwise the frozen book stands, and that is a recorded result.
#
# Under the gate line, never instead of it, the grade also prints
# randomization inference (RESULTS 27): a sign-flip p for the mean
# difference, and an activity-matched null -- the same number of skips each
# night on names drawn at random, the same number of hedged nights drawn at
# random -- which tells an edge that comes from how much a record skips and
# hedges (exposure) from one that comes from which names and nights it picks
# (selection). Only the three GATED records (the decider, the reference, the
# control) share the gate's split; any other prints as '[not a gated record]'.
#
# Backends:
#   http   any Jev-compatible endpoint (DECIDER_URL, DECIDER_KEY,
#          DECIDER_MODEL for a dry run). In the record, decide_local.py runs:
#            http  the decider, Perplexity Decider v1.1 27B, via OpenRouter
#            jev   the reference, Jev 1.13 itself, via OpenRouter
#          (or, with --engine local, Kev on a local llama.cpp as 'http')
#   rule   the deterministic baseline any model must beat: skip a name that
#          reports earnings tonight, never hedge
#
#   python src/decide_local.py                   # the nightly run (both)
#   python src/guard.py --dry-run                # try a server by hand
#   python src/guard.py --grade

import os, sys, re, csv, json, math, time, zlib, argparse, hashlib, itertools
import datetime as dt
import urllib.request, urllib.error
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
DAILY = os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv')
GLOG = os.path.join(ROOT, 'reports', 'guard_log.csv')
ET = ZoneInfo('America/New_York')
CUTOFF = (15, 50)                       # NYSE MOC entry cutoff, ET
COLS = ['date', 'decided_et', 'backend', 'model', 'question', 'ticker', 'p',
        'decision', 'latency_ms', 'state_sha']
# The gate: 63 nights and paired t above the threshold. Three records can
# each reach it -- the decider (Perplexity Decider), the reference (Jev) and
# the earnings-rule control -- so the one-sided 2.5% of t > 2 is split three
# ways (Bonferroni, 62 df): t > 2.46. Fixed 2026-10-13, before any of the
# three had a graded night.
GATE_NIGHTS, GATE_T = 63, 2.46
BOOK = 'BOOK'
# PRE-DECLARED DECISION RULE (fixed 2026-10-12, before any decided night had
# an outcome): skip a name only if p(hold) < 0.5 AND it sits at least
# SKIP_MARGIN below the night's median p. A small local model answers with a
# constant lean -- Qwen3.5-0.8B's first run put all 11 names at 0.35-0.41 --
# and a bare p < 0.5 would skip everything every night; the median test
# ignores a shared bias and acts only on a name the model singles out. For a
# calibrated decider (hosted Jev) it reduces to the plain rule. Hedge: p >= 0.5.
SKIP_MARGIN = 0.10


# ---------------------------------------------------------------- the API --

# The questions' wording, by version. 'q1' is the wording every logged night
# so far was asked with; a new version is a new record (its suffix is
# logged with the model), never an edit of q1.
QUESTIONS = {'q1': {
    'hold': ('Should the long position in `{n}` be held overnight tonight '
             '-- bought at today\'s closing auction and sold at '
             'tomorrow\'s opening auction?',
             {'true': 'hold: an ordinary night for this name',
              'false': 'skip: earnings tonight or an extreme recent move'}),
    'hedge': ('Should tonight\'s whole basket be hedged with an equal-dollar '
              'short position in QQQ, held from today\'s close to '
              'tomorrow\'s open?',
              {'true': 'hedge: elevated risk of a market-wide overnight gap',
               'false': 'do not hedge: an ordinary night'})}}


def question_order(seed, qids):
    """A seeded, reproducible order of question ids: sha256 of seed|id. The
    battery and the runtime share it, so the bytes tested are the bytes
    sent (RESULTS 27)."""
    return sorted(qids, key=lambda q: hashlib.sha256(
        f'{seed}|{q}'.encode()).hexdigest())


def request_body(state, names, model, order=None, version='q1', ids=None,
                 provider=None, view=None):
    """The documented POST /v1/systemone body: a state, a model, and typed
    questions keyed by ids we choose (ids are not sent to the model).

    Defaults reproduce the bytes every logged night was asked with (a test
    pins their sha256). order: question ids (hold_{name}, hedge_book) in the
    order to ask them; the state's names follow the same order. ids:
    {question id: id sent}, e.g. opaque ids. provider: OpenRouter's
    provider-routing object. view: a function of the state that returns
    what is sent (the shared state, and its sha, are unchanged)."""
    w = QUESTIONS[version]
    q = {}
    for n in names:
        q[f'hold_{n}'] = {
            'type': 'noul',
            'instructions': w['hold'][0].format(n=n),
            'criteria': dict(w['hold'][1])}
    q['hedge_book'] = {
        'type': 'noul',
        'instructions': w['hedge'][0],
        'criteria': dict(w['hedge'][1])}
    if order is not None:
        assert sorted(order) == sorted(q), 'order must be a permutation'
        q = {k: q[k] for k in order}
        held = [k[5:] for k in order if k.startswith('hold_')]
        state = dict(state, names={n: state['names'][n] for n in held
                                   if n in state['names']})
    if view is not None:
        state = view(state)
    if ids is not None:
        q = {ids.get(k, k): v for k, v in q.items()}
    body = {'state': state, 'model': model, 'questions': q}
    if provider is not None:
        body['provider'] = provider
    return body


def parse_answers(resp, names):
    """{ticker or BOOK: p(yes)} from a /v1/systemone response."""
    a = resp.get('answers', {})
    out = {}
    for n in names:
        x = a.get(f'hold_{n}', {})
        if x.get('type') == 'noul' and x.get('noul') is not None:
            out[n] = float(x['noul'])
    x = a.get('hedge_book', {})
    if x.get('type') == 'noul' and x.get('noul') is not None:
        out[BOOK] = float(x['noul'])
    return out


def ask_http(state, names, url, key=None, model='jev-latest', tries=4,
             opener=urllib.request.urlopen, timeout=None):
    """Any Jev-compatible endpoint: a local llama-server, TypeSafe, or
    OpenRouter's Decisions API. The decider's name is the model the response
    says answered -- versioned on OpenRouter -- and, when the response names
    one, the provider that served it ('model@provider')."""
    body = json.dumps(request_body(state, names, model)).encode()
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    for i in range(tries):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=body, headers=headers,
                                         method='POST')
            # hosted Jev answers in ~0.1 s; a local model on a CPU can take
            # minutes for a dozen questions, so the wait is configurable
            to = timeout or float(os.environ.get('DECIDER_TIMEOUT', 600))
            with opener(req, timeout=to) as r:
                resp = json.loads(r.read().decode())
            who = resp.get('model', model)
            if resp.get('provider'):
                who = f'{who}@{resp["provider"]}'
            return parse_answers(resp, names), who, \
                int((time.time() - t0) * 1000)
        except urllib.error.HTTPError as ex:
            if ex.code in (429, 502, 503, 504, 529) and i < tries - 1:
                time.sleep(2 ** i)              # the documented back-off
                continue
            raise SystemExit(f'decision server refused ({ex.code}): '
                             f'{ex.read().decode(errors="replace")[:300]}')
        except (urllib.error.URLError, TimeoutError) as ex:
            if i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise SystemExit(f'decision server unreachable at {url} '
                             f'({ex}) -- nothing logged for tonight')


def ask_rule(state, names):
    """The deterministic baseline: skip a name that reports between today's
    close and tomorrow's open; never hedge."""
    out = {n: 0.0 if state['names'].get(n, {}).get('earnings_tonight')
           else 1.0 for n in names}
    out[BOOK] = 0.0
    return out, 'rule-earnings-v1', 0


# -------------------------------------------------------------- the state --

STATE_SCHEMA = 2      # 1: raw 5-night series (2026-10-12 only); 2: summaries


def build_state(names, today, download=None, earnings=None):
    """What the model sees, computed in code ("keep math in code" is
    TypeSafe's own advice for jev-1.13, and small local deciders cap the
    prompt near 2,000 tokens): per name, last night's overnight move, the
    5-night overnight mean, the 5-day return, 20-day volatility (all bps,
    rounded) and whether it reports earnings tonight; the same for QQQ. All
    known before today's close."""
    import yfinance as yf
    download = download or yf.download
    v = download([n.replace('.', '-') for n in names + ['QQQ']],
                 period='3mo', interval='1d', auto_adjust=False,
                 actions=False, group_by='ticker', progress=False)
    st = {'schema': STATE_SCHEMA, 'date': today,
          'book': 'long these names overnight: buy at the close auction, '
                  'sell at the next open auction; moves in basis points',
          'names': {}}
    for n in names + ['QQQ']:
        try:
            s = v[n.replace('.', '-')][['Open', 'Close']].dropna()
        except KeyError:
            continue
        s = s[s.index.strftime('%Y%m%d') < today]   # nothing from today
        on = np.log(s.Open / s.Close.shift()).dropna() * 1e4
        cc = np.log(s.Close / s.Close.shift()).dropna() * 1e4
        if len(on) < 5 or len(cc) < 20:
            continue
        row = {'last_night': int(round(on.iloc[-1])),
               'nights5_avg': int(round(on.tail(5).mean())),
               'ret5d': int(round(cc.tail(5).sum())),
               'vol20d': int(round(cc.tail(20).std()))}
        if n == 'QQQ':
            st['QQQ'] = row
        else:
            row['earnings_tonight'] = bool((earnings or {}).get(n))
            st['names'][n] = row
    return st


def earnings_tonight(names, today, lookup=None, tries=3, wait=5):
    """{name: True} for names reporting between today's close and the next
    open, from Yahoo's earnings calendar. A lookup that still fails after
    `tries` is NOT read as "no report" -- the night is refused, because the
    earnings rule is the control and must not silently hold a reporter."""
    import yfinance as yf
    lookup = lookup or (lambda n: yf.Ticker(n.replace('.', '-'))
                        .get_earnings_dates(limit=8))
    lo = pd.Timestamp(today).tz_localize(ET) + pd.Timedelta(hours=15)
    hi = lo + pd.Timedelta(hours=19)            # through the next 10:00
    out, failed = {}, []
    for n in names:
        for i in range(tries):
            try:
                e = lookup(n)
                if e is None:
                    raise ValueError('no calendar returned')
                out[n] = any(lo <= t.tz_convert(ET) <= hi for t in e.index)
                break
            except Exception as ex:
                if i == tries - 1:
                    failed.append(f'{n} ({ex})')
                else:
                    time.sleep(wait * (i + 1))
    if failed:
        raise SystemExit('earnings lookup failed for ' + ', '.join(failed)
                         + ' -- not deciding on an unknown calendar')
    return out


# ---------------------------------------------------------------- the log --

def read_log(path=GLOG):
    return pd.read_csv(path, dtype={'date': str}) if os.path.exists(path) \
        else pd.DataFrame(columns=COLS)


def decide(backend, names, today, now_et, state, path=GLOG, ask=None):
    """Ask once for tonight and append. Refuses a night this backend already
    decided (the decider and the earnings-rule baseline each decide once),
    and a decision at or after the MOC cutoff."""
    log = read_log(path)
    if ((log.date == today) & (log.backend == backend)).any():
        raise SystemExit(f'{today} already decided by the {backend} backend '
                         f'-- a night is never re-decided')
    if now_et.strftime('%Y%m%d') != today:
        raise SystemExit(f'{now_et:%Y-%m-%d} ET is not the decision day '
                         f'{today}')
    if (now_et.hour, now_et.minute) >= CUTOFF:
        raise SystemExit(f'{now_et:%H:%M} ET is at or past the {CUTOFF[0]}:'
                         f'{CUTOFF[1]:02d} cutoff -- too late to decide '
                         f'tonight honestly')
    t0, w0 = time.monotonic(), time.time()
    p, model, ms = ask(state, names)
    # stamped when the answer arrived, not when it was asked: a local model
    # on a CPU can take minutes, and an answer after the cutoff is no answer.
    # The later of the two clocks: monotonic time stops while a machine
    # sleeps, wall time can be stepped; the larger elapsed is the safe one.
    done = now_et + dt.timedelta(seconds=max(time.monotonic() - t0,
                                             time.time() - w0))
    if done.date() != now_et.date() or (done.hour, done.minute) >= CUTOFF:
        raise SystemExit(f'the answer arrived {done:%H:%M} ET, at or past the '
                         f'cutoff -- nothing logged for tonight')
    d = rule(p, names)
    sha = hashlib.sha256(json.dumps(state, sort_keys=True)
                         .encode()).hexdigest()[:12]
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        if new:
            w.writerow(COLS)
        for k in names + [BOOK]:
            if k not in p:
                continue
            w.writerow([today, done.strftime('%Y-%m-%d %H:%M:%S'), backend,
                        model, 'hedge_book' if k == BOOK else 'hold', k,
                        f'{p[k]:.4f}', d[k], ms, sha])
    return p


def basket(paper, today):
    """Tonight's names: the basket registered for the decision day's month.
    Not registered yet (the 1st, before the morning registration) -> None:
    deciding on last month's names would grade against the wrong book."""
    rows = paper[paper.month == f'{today[:4]}-{today[4:6]}']
    return rows.iloc[-1].q5.split(';') if len(rows) else None


def tonight(now=None, download=None, earnings=None):
    """(date, names, state) for tonight, built once so that every backend
    asked tonight -- the decider and the control -- sees the identical state.
    Refuses (SystemExit) rather than decide on a partial picture: no basket
    for this month, or any name or QQQ missing from the market data."""
    today = (now or dt.datetime.now(ET)).strftime('%Y%m%d')
    names = basket(pd.read_csv(PAPER, dtype=str), today)
    if not names:
        raise SystemExit(f'no basket registered for {today[:4]}-{today[4:6]} '
                         f'yet -- not deciding on last month\'s names')
    ern = (earnings or earnings_tonight)(names, today)
    state = build_state(names, today, download=download, earnings=ern)
    missing = [n for n in names if n not in state['names']] + \
        ([] if 'QQQ' in state else ['QQQ'])
    if missing:
        raise SystemExit('no market data for ' + ', '.join(missing) +
                         ' -- not deciding on a partial state')
    return today, names, state


def answer(backend, ask, names, today, state, dry_run=False, path=GLOG,
           now=None):
    """Ask one backend about tonight. A dry run prints and writes nothing;
    otherwise decide() logs it (and refuses a repeat or a late answer)."""
    if dry_run:
        p, model, ms = ask(state, names)
        d = rule(p, names)
        print(f'{today} DRY RUN, nothing written -- {model}, {ms} ms')
        for k in names + [BOOK]:
            if k in p:
                print(f'  {k:6s} {p[k]:.4f}  {d[k]}')
        return p
    p = decide(backend, names, today, now or dt.datetime.now(ET), state,
               path=path, ask=ask)
    print(f'{today} [{backend}]: ' + '  '.join(f'{k} {v:.2f}'
                                              for k, v in p.items())
          + f'  -> {path} (commit before the close)')
    return p


def rule(p, names):
    """The pre-declared rule (RESULTS 27): skip a name only if p < 0.5 AND p
    is at least SKIP_MARGIN below the night's median p(hold); hedge the book
    iff p >= 0.5."""
    held = [p[n] for n in names if n in p]
    med = float(np.median(held)) if held else 0.5
    # 'at least SKIP_MARGIN below', tie-safe: Jev answers in hundredths, so a
    # name can sit exactly 0.10 under the median, and a bare float compare
    # skipped 0.46 under 0.56 but held 0.49 under 0.59 (RESULTS 27)
    d = {n: 'skip' if p[n] < 0.5 and med - p[n] >= SKIP_MARGIN - 1e-9
         else 'hold' for n in names if n in p}
    if BOOK in p:
        d[BOOK] = 'hedge' if p[BOOK] >= 0.5 else 'none'
    return d


# -------------------------------------------------------------- the grade --

# Randomization inference, printed under the gate line (RESULTS 27). Fixed
# 2026-10-13, before any record had a graded night. Each record draws from
# its own stream, keyed by (backend, model), so one record's p never depends
# on which other records are in the log.
GRADE_SEED = 20261013
RAND_B = 20000                  # activity-matched draws (exact when fewer)
FLIP_DRAWS = 100_000            # sign-flip draws above 20 nights (exact below)
RAND_ALPHA = 0.05 / 3           # three gated records, the same split as GATE_T

# The gated records: (backend, regex the logged model must fullmatch). Only
# these three share GATE_T's split; any other record (an older decider, an
# undated build, an undeclared handling suffix such as '#perm3.q1') is graded
# the same way but printed as '[not a gated record]'. Declaring a new decider
# or a suffix for it changes this one line: (pin, suffix), or (pin, suffix,
# False) for a model the API serves under an undated build (the battery's
# selection says which: its decider's 'dated').
DECIDER = ('perplexity/pplx-decider-v1.1-27b', '')


def decider_slot(pin, suffix='', dated=True):
    """The decider's record: any dated build of the pin (or, for a model
    served undated, the pin itself or a dated build of it), from one named
    provider, carrying exactly the declared suffix."""
    build = r'-\d{8}' if dated else r'(-\d{8})?'
    return re.escape(pin) + build + r'@[^@#]+' + re.escape(suffix)


GATED = (('rule', 'rule-earnings-v1'),
         ('http', decider_slot(*DECIDER)),
         ('jev', r'typesafe/jev-1\.13-\d{8}@[^@#]+'))


def gated(backend, model):
    return any(b == backend and re.fullmatch(rx, str(model))
               for b, rx in GATED)


def grade(log, paper, daily, prints):
    """Per decided night: the frozen book (every basket name, unhedged) vs the
    SHADOW book (skipped names left out, flat if all are; minus QQQ when the
    model said hedge), both at official prints. prints:
    {(sym, YYYYMMDD): (open, close)} incl. QQQ. Decisions at/after the cutoff
    are dropped. Each night also keeps its detail for the randomization
    inference: rr (graded name -> overnight bps), kept (the names held) and
    q (QQQ's overnight bps)."""
    ok = log[pd.to_datetime(log.decided_et).dt.strftime('%H%M')
             < f'{CUTOFF[0]:02d}{CUTOFF[1]:02d}']
    ok = ok[pd.to_datetime(ok.decided_et).dt.strftime('%Y%m%d') == ok.date]
    baskets = {r.month: r.q5.split(';') for r in paper.itertuples()}
    days = sorted({d for (_, d) in prints})
    nxt = dict(zip(days[:-1], days[1:]))
    rows = []
    for D, g in ok.groupby('date'):
        Dn = nxt.get(D)
        names = baskets.get(f'{D[:4]}-{D[4:6]}')
        if not Dn or not names or D not in set(daily.date):
            continue

        def r(s):
            a, b = prints.get((s, D)), prints.get((s, Dn))
            return None if a is None or b is None else \
                float(np.log(b[0] / a[1]) * 1e4)
        rr = {n: r(n) for n in names}
        rr = {k: v for k, v in rr.items() if v is not None}
        q = r('QQQ')
        if len(rr) < 5 or q is None:
            continue
        hold = g[g.question == 'hold'].set_index('ticker').decision.to_dict()
        kept = [n for n in rr if hold.get(n, 'hold') == 'hold']
        hedge = (g[g.question == 'hedge_book'].decision == 'hedge').any()
        frozen = float(np.mean(list(rr.values())))
        shadow = float(np.mean([rr[n] for n in kept])) if kept else 0.0
        if hedge:
            shadow -= q
        rows.append(dict(date=D, frozen=frozen, shadow=shadow,
                         diff=shadow - frozen, skipped=len(rr) - len(kept),
                         hedged=bool(hedge),
                         backend=g.backend.iloc[0], model=g.model.iloc[0],
                         rr=rr, kept=kept, q=q))
    return pd.DataFrame(rows)


# the gate, exactly as declared: verdict() prints these lines first, unchanged
def gate_verdict(t):
    if t.empty:
        return 'no decided-and-graded nights yet'
    n, m = len(t), t['diff'].mean()
    sd = t['diff'].std(ddof=1) if n > 1 else float('nan')
    tt = m / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float('nan')
    head = (f'{n} night(s): shadow - frozen {m:+.2f} bps/night, paired t '
            f'{tt:+.2f}; shadow sd {t.shadow.std():.0f} vs frozen '
            f'{t.frozen.std():.0f}; {int(t.skipped.sum())} name-nights '
            f'skipped, {int(t.hedged.sum())} nights hedged')
    if n < GATE_NIGHTS:
        return head + f'\n  gate: not before {GATE_NIGHTS} nights -- no verdict'
    if m > 0 and tt > GATE_T:
        return head + (f'\n  gate: PASSED (t > {GATE_T}) -- the shadow book '
                       f'may be proposed for a dated re-freeze')
    return head + f'\n  gate: not passed (t > {GATE_T} needed) -- the frozen ' \
        f'book stands'


def _stream(backend, model):
    return np.random.default_rng(
        [GRADE_SEED, zlib.crc32(f'{backend}|{model}'.encode())])


def _who(t, backend, model):
    """The record's (backend, model) for its stream: given, else its own."""
    pick = lambda v, c: v if v is not None else \
        (str(t[c].iloc[0]) if c in t and len(t) else '')
    return pick(backend, 'backend'), pick(model, 'model')


def _at_least(T, Ts):
    """#{T* >= T}. The observed assignment is among the draws, summed in
    another order, so a tie must count whatever the float rounding."""
    tol = 1e-9 * (1 + abs(T)) if np.isfinite(T) else 0.0
    return int((Ts >= T - tol).sum())


def signflip_p(diff, backend='', model=''):
    """One-sided p for mean(diff) > 0 when each night's difference is as
    likely + as -: exact over all 2^n signs for n <= 20, else FLIP_DRAWS
    draws from the record's stream, p = (1 + #{T* >= T}) / (1 + draws)."""
    x = np.asarray(diff, float)
    if len(x) == 0:
        return float('nan')
    T = x.sum()                     # the mean's order, for a fixed n
    if len(x) <= 20:
        s = np.zeros(1)
        for v in x:
            s = np.concatenate([s + v, s - v])
        return _at_least(T, s) / len(s)
    rng, hits = _stream(backend, model), 0
    for a in range(0, FLIP_DRAWS, 10_000):
        sg = rng.integers(0, 2, (min(10_000, FLIP_DRAWS - a), len(x))) * 2 - 1
        hits += _at_least(T, sg @ x)
    return (1 + hits) / (1 + FLIP_DRAWS)


def _nights(t):
    """A record as arrays, names padded to the widest night: rr [n, M],
    valid [n, M], observed skips [n, M], q [n], hedged [n]."""
    n, M = len(t), max(len(r) for r in t.rr)
    R, V, S = (np.zeros((n, M)), np.zeros((n, M), bool),
               np.zeros((n, M), bool))
    for i, (rr, kept) in enumerate(zip(t.rr, t.kept)):
        kept = set(kept)
        R[i, :len(rr)] = list(rr.values())
        V[i, :len(rr)] = True
        S[i, :len(rr)] = [x not in kept for x in rr]
    return R, V, S, t.q.to_numpy(float), t.hedged.to_numpy(bool)


def _tstat(skip_sum, hedge, R, V, k, q):
    """Studentized mean of shadow - frozen, one row per assignment:
    skip_sum [D, n] is the bps of the names skipped each night (k of them),
    hedge [D, n] the hedged nights. Arithmetic as in grade()."""
    m, tot = V.sum(1), R.sum(1)
    left = m - k
    kept = np.where(left > 0, (tot - skip_sum) / np.maximum(left, 1), 0.0)
    d = kept - q * hedge - tot / m
    mu, sd = d.mean(-1), d.std(-1, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = mu / (sd / np.sqrt(d.shape[-1]))
    # a constant difference: +-inf by its sign (0 if none), never nan
    return np.where(sd > 0, T, np.where(mu > 0, np.inf,
                                        np.where(mu < 0, -np.inf, 0.0)))


# skip sets per night kept as a table of their bps; a basket so wide that a
# night has more (19+ names) draws by sorting instead
SKIP_TABLE = 50_000


def _skip_table(R, V, k):
    """Per night, the bps of every skip set of the observed size: table
    [n, C] (a night's own sets first, the rest unused) and the counts. The
    table is None past SKIP_TABLE sets on a night."""
    m = V.sum(1)
    sizes = [math.comb(int(a), int(b)) for a, b in zip(m, k)]
    if max(sizes) > SKIP_TABLE:
        return None, sizes
    table = np.zeros((len(m), max(sizes)))
    for i, (mi, ki, si) in enumerate(zip(m, k, sizes)):
        c = np.array(list(itertools.combinations(range(int(mi)), int(ki))),
                     int).reshape(si, int(ki))
        table[i, :si] = R[i, c].sum(1)
    return table, sizes


def _all_skips(table, sizes, B):
    """Every assignment of the nights' skip sets, as skip sums [count, n]
    (the observed one among them), or None if there are more than B."""
    total = math.prod(sizes)
    if table is None or total > B:
        return None
    stride = np.cumprod([1] + sizes[:-1])
    return table[np.arange(len(sizes)),
                 np.arange(total)[:, None] // stride % np.array(sizes)]


def _draw_skips(rng, table, sizes, R, V, k, B):
    """B draws of a uniform skip set of size k[d] on each night d among its
    graded names, as skip sums [B, n]: one of the night's sets, uniformly;
    without a table, the k smallest of iid uniforms."""
    n, M = R.shape
    if table is not None:
        return table[np.arange(n), rng.integers(0, np.array(sizes), (B, n))]
    out, step = np.empty((B, n)), max(1, 2_000_000 // (n * M))
    kth = np.maximum(k - 1, 0)[None, :, None]
    for a in range(0, B, step):
        u = rng.random((min(step, B - a), n, M))
        u[:, ~V] = np.inf               # padding is never a name to skip
        thr = np.take_along_axis(np.sort(u, -1),
                                 np.broadcast_to(kth, u.shape[:2] + (1,)), -1)
        pick = (u <= thr) & (k > 0)[None, :, None]
        out[a:a + len(u)] = (pick * R).sum(-1)
    return out


def _all_hedges(n, H, B):
    """Every set of H hedged nights among n [count, n], or None if more
    than B."""
    if math.comb(n, H) > B:
        return None
    out = np.zeros((math.comb(n, H), n), bool)
    for r, c in enumerate(itertools.combinations(range(n), H)):
        out[r, list(c)] = True
    return out


def _draw_hedges(rng, n, H, B):
    """B draws of a uniform set of H hedged nights among n [B, n]."""
    u = rng.random((B, n))
    return u <= np.sort(u, 1)[:, H - 1:H] if H else np.zeros((B, n), bool)


def randomization(t, B=RAND_B, backend=None, model=None):
    """The activity-matched null: does the record's CHOICE of names and
    nights beat random choices with its own activity? A draw skips, on each
    night, a uniform random set of the observed size from that night's
    graded names, and hedges a uniform random set of H nights, H as
    observed. T* = mean / (sd / sqrt(n)) of the draw's shadow - frozen.
    p_joint draws both; p_skip holds the hedges as observed; p_hedge holds
    the skips as observed. p = #{T* >= T} / count when every assignment can
    be enumerated (count <= B), else (1 + #{T* >= T}) / (B + 1) over B draws
    from the record's stream. 'n/a' where there is no activity to test
    (and for fewer than two nights, where T is undefined)."""
    out = dict(p_joint='n/a', p_skip='n/a', p_hedge='n/a')
    if len(t) < 2 or not {'rr', 'kept', 'q', 'hedged'} <= set(t.columns):
        return out
    R, V, S, q, h = _nights(t)
    n, k, H = len(t), S.sum(1), int(h.sum())
    if not k.any() and not H:
        return out
    seen = (S * R).sum(1)[None]
    T = _tstat(seen, h[None], R, V, k, q)[0]
    table, sizes = _skip_table(R, V, k)
    rng, drawn = _stream(*_who(t, backend, model)), {}

    def draws(what):                # drawn once, in a fixed order of need
        if what not in drawn:
            drawn[what] = _draw_skips(rng, table, sizes, R, V, k, B) \
                if what == 'skip' else _draw_hedges(rng, n, H, B)
        return drawn[what]

    def p(skips, hedges, exact):
        Ts = _tstat(skips, hedges, R, V, k, q)
        hit = _at_least(T, Ts)
        return hit / len(Ts) if exact else (1 + hit) / (B + 1)

    ss, hh = _all_skips(table, sizes, B), _all_hedges(n, H, B)
    if ss is not None and hh is not None and len(ss) * len(hh) <= B:
        out['p_joint'] = p(np.repeat(ss, len(hh), 0),
                           np.tile(hh, (len(ss), 1)), True)
    else:
        out['p_joint'] = p(draws('skip'), draws('hedge'), False)
    if k.any():
        out['p_skip'] = p(ss, h[None], True) if ss is not None else \
            p(draws('skip'), h[None], False)
    if H:
        out['p_hedge'] = p(seen, hh, True) if hh is not None else \
            p(seen, draws('hedge'), False)
    return out


def _fmt(p):
    if isinstance(p, str) or not np.isfinite(p):
        return 'n/a'
    return f'{p:.4f}' if p >= 1e-4 else f'{p:.1e}'


def inference(t, backend=None, model=None):
    """The lines under the gate: what the record did, where its difference
    came from, and whether its choices beat random ones of the same
    activity. Report only: the gate above is the verdict."""
    backend, model = _who(t, backend, model)
    n = len(t)
    out = [f'  activity: {t.skipped.mean():.2f} skips/night, '
           f'{int(t.hedged.sum())} of {n} nights hedged']
    if {'rr', 'kept', 'q'} <= set(t.columns):
        kept = [np.mean([r[x] for x in k]) if len(k) else 0.0
                for r, k in zip(t.rr, t.kept)]
        skip = float(np.mean(np.array(kept) - t.frozen.to_numpy(float)))
        hedge = 0.0 - float(np.mean(t.q.to_numpy(float)
                                    * t.hedged.to_numpy(bool)))
        out.append(f'  diff split: skip part {skip:+.2f}, hedge part '
                   f'{hedge:+.2f} bps/night')
    else:
        out.append('  diff split: n/a (no per-night detail)')
    out.append('  sign-flip p (mean > 0): '
               + _fmt(signflip_p(t['diff'], backend, model)))
    r = randomization(t, backend=backend, model=model)
    out.append('  activity-matched null: '
               + ', '.join(f'{k} {_fmt(v)}' for k, v in r.items()))
    beats = not isinstance(r['p_joint'], str) and r['p_joint'] <= RAND_ALPHA
    out.append('  ' + ('selection beats activity-matched random' if beats
                       else 'any edge is exposure, not selection'))
    return out


def verdict(t, backend=None, model=None):
    """The gate's lines, unchanged, then the randomization inference under
    them. backend and model key the record's random stream (default: the
    record's own)."""
    out = gate_verdict(t)
    if t.empty:
        return out
    return out + '\n' + '\n'.join(inference(t, backend, model))


def report(log, paper, daily, prints):
    """One verdict per record. One record per decider: a change of model,
    build, provider or handling suffix starts its own record (and its own
    63-night gate), it never pools with the old one."""
    out = []
    for (b, m), g in log.groupby(['backend', 'model']):
        tag = '' if gated(b, m) else '[not a gated record] '
        out.append(f'{tag}[{b} {m}] '
                   + verdict(grade(g, paper, daily, prints), b, m))
    return out


def main():
    ap = argparse.ArgumentParser(description='A Jev-style decision model in '
                                 'shadow mode: decides, is graded, never '
                                 'trades.')
    ap.add_argument('--decide', action='store_true',
                    help='retired: the record\'s decisions come only from '
                         'src/decide_local.py (Kev, local, pinned)')
    ap.add_argument('--grade', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='ask and print tonight\'s answers; write nothing '
                         '(any machine, any time -- the record is untouched)')
    ap.add_argument('--backend', choices=['http', 'rule'], default='http')
    a = ap.parse_args()
    if a.decide:
        raise SystemExit('the record\'s decisions come only from '
                         'python src/decide_local.py (it pins the model, the '
                         'runtime and the server); here, use --dry-run')
    paper = pd.read_csv(PAPER, dtype=str)
    if a.dry_run:
        today, names, state = tonight()
        if a.backend == 'http':
            url = os.environ.get('DECIDER_URL')
            if not url:
                raise SystemExit('set DECIDER_URL to a Jev-compatible '
                                 '/v1/systemone endpoint (local server, or '
                                 'TypeSafe with DECIDER_KEY)')
            ask = lambda s, nm: ask_http(
                s, nm, url, os.environ.get('DECIDER_KEY'),
                os.environ.get('DECIDER_MODEL', 'jev-latest'))
        else:
            ask = ask_rule
        answer(a.backend, ask, names, today, state, dry_run=True)
        return
    if a.grade:
        from paper_bench import official_prints
        log = read_log()
        daily = pd.read_csv(DAILY, dtype={'date': str})
        if log.empty:
            print('no decisions logged yet')
            return
        names = sorted({n for q in paper.q5 for n in q.split(';')} | {'QQQ'})
        lo = (pd.Timestamp(log.date.min()) - pd.Timedelta(days=5))
        prints = official_prints(names, lo.strftime('%Y-%m-%d'),
                                 (pd.Timestamp.today() + pd.Timedelta(days=1))
                                 .strftime('%Y-%m-%d'))
        # one record per decider: a change of model starts its own record
        # (and its own 63-night gate), it never pools with the old one;
        # a record outside GATED is printed as '[not a gated record]'
        for line in report(log, paper, daily, prints):
            print(line)


if __name__ == '__main__':
    main()
