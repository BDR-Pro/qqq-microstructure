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
#   - one decision per night per backend (the decider, and the earnings-rule
#     control it is compared with); a night is never re-decided
#   - a decision time-stamped at or after 15:50 ET on its night is ignored by
#     the grader (it could have seen the close)
#   - the log is append-only and committed before the close, from the
#     owner's own machine (decide_local.py: Kev on llama.cpp, 127.0.0.1,
#     pinned by sha256) -- never from a cloud runner or a hosted API
#   - an answer is stamped when it ARRIVES; one that arrives at or after the
#     cutoff is not logged (a local model on a CPU takes minutes)
#   - one record per (backend, model): a new decider starts its own record and
#     its own 63-night clock, never pooled with the old one
#   - --dry-run asks and prints, writes nothing: for trying a decider by hand
#
# PRE-DECLARED GATE: the shadow book may be proposed for a dated re-freeze only
# if, over at least 63 decided-and-graded nights, its paired nightly
# difference against the frozen book has mean > 0 AND paired t > 2.
# Otherwise the frozen book stands, and that is a recorded result.
#
# Backends:
#   http   any Jev-compatible server: DECIDER_URL, DECIDER_KEY if it needs
#          one, DECIDER_MODEL. The record's decider is Kev on a local
#          llama.cpp, started and stopped by decide_local.py (RESULTS 27).
#   rule   the deterministic baseline any model must beat: skip a name that
#          reports earnings tonight, never hedge
#
#   python src/decide_local.py                   # the nightly run (both)
#   python src/guard.py --dry-run                # try a server by hand
#   python src/guard.py --grade

import os, sys, csv, json, time, argparse, hashlib, datetime as dt
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
GATE_NIGHTS, GATE_T = 63, 2.0
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

def request_body(state, names, model):
    """The documented POST /v1/systemone body: a state, a model, and typed
    questions keyed by ids we choose (ids are not sent to the model)."""
    q = {}
    for n in names:
        q[f'hold_{n}'] = {
            'type': 'noul',
            'instructions': (
                f'Should the long position in `{n}` be held overnight tonight '
                f'-- bought at today\'s closing auction and sold at '
                f'tomorrow\'s opening auction?'),
            'criteria': {
                'true': 'hold: an ordinary night for this name',
                'false': 'skip: earnings tonight or an extreme recent move'}}
    q['hedge_book'] = {
        'type': 'noul',
        'instructions': (
            'Should tonight\'s whole basket be hedged with an equal-dollar '
            'short position in QQQ, held from today\'s close to tomorrow\'s '
            'open?'),
        'criteria': {
            'true': 'hedge: elevated risk of a market-wide overnight gap',
            'false': 'do not hedge: an ordinary night'}}
    return {'state': state, 'model': model, 'questions': q}


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
            return parse_answers(resp, names), resp.get('model', model), \
                int((time.time() - t0) * 1000)
        except urllib.error.HTTPError as ex:
            if ex.code in (429, 529) and i < tries - 1:
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
    d = {n: 'skip' if p[n] < 0.5 and p[n] < med - SKIP_MARGIN else 'hold'
         for n in names if n in p}
    if BOOK in p:
        d[BOOK] = 'hedge' if p[BOOK] >= 0.5 else 'none'
    return d


# -------------------------------------------------------------- the grade --

def grade(log, paper, daily, prints):
    """Per decided night: the frozen book (every basket name, unhedged) vs the
    SHADOW book (skipped names left out, flat if all are; minus QQQ when the
    model said hedge), both at official prints. prints:
    {(sym, YYYYMMDD): (open, close)} incl. QQQ. Decisions at/after the cutoff
    are dropped."""
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
                         backend=g.backend.iloc[0], model=g.model.iloc[0]))
    return pd.DataFrame(rows)


def verdict(t):
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
    return head + ('\n  gate: PASSED -- the shadow book may be proposed for a '
                   'dated re-freeze' if m > 0 and tt > GATE_T else
                   '\n  gate: not passed -- the frozen book stands')


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
        # (and its own 63-night gate), it never pools with the old one
        for (b, m), g in log.groupby(['backend', 'model']):
            print(f'[{b} {m}] ' + verdict(grade(g, paper, daily, prints)))


if __name__ == '__main__':
    main()
