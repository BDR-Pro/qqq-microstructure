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
#   - one decision per night per question; a night is never re-decided
#   - a decision time-stamped at or after 15:50 ET on its night is ignored by
#     the grader (it could have seen the close)
#   - the log is append-only and committed by the crontab before the close
#
# PRE-DECLARED GATE: the shadow book may be proposed for a dated re-freeze only
# if, over at least 63 decided-and-graded nights, its paired nightly
# difference against the frozen book has mean > 0 AND paired t > 2.
# Otherwise the frozen book stands, and that is a recorded result.
#
# Backends:
#   http   any Jev-compatible server: DECIDER_URL (e.g. a local LocalJev/Kev
#          server, or https://api.typesafe.ai/v1/systemone), DECIDER_KEY if it
#          needs one, DECIDER_MODEL (default jev-latest)
#   rule   the deterministic baseline any model must beat: skip a name that
#          reports earnings tonight, never hedge
#
#   python src/guard.py --decide                 # before 15:50 ET
#   python src/guard.py --decide --backend rule
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
                'true': 'hold it: no reason in the state to expect tonight to '
                        'be worse for this name than an ordinary night',
                'false': 'skip it tonight: the state shows a specific reason, '
                         'such as an earnings report before tomorrow\'s open '
                         'or an extreme recent move'}}
    q['hedge_book'] = {
        'type': 'noul',
        'instructions': (
            'Should tonight\'s whole basket be hedged with an equal-dollar '
            'short position in QQQ, held from today\'s close to tomorrow\'s '
            'open?'),
        'criteria': {
            'true': 'hedge: the basket is concentrated in one theme or the '
                    'state shows elevated risk of a market-wide overnight gap',
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

def build_state(names, today, download=None, earnings=None):
    """What the model sees: per name, recent overnight and intraday moves,
    20-day volatility and whether it reports earnings tonight; QQQ the same.
    Numbers only, all known before today's close."""
    import yfinance as yf
    download = download or yf.download
    v = download([n.replace('.', '-') for n in names + ['QQQ']],
                 period='3mo', interval='1d', auto_adjust=False,
                 actions=False, group_by='ticker', progress=False)
    st = {'date': today, 'strategy': 'long an 11-name basket overnight, '
          'buy at the close auction, sell at the next open auction',
          'names': {}}
    for n in names + ['QQQ']:
        try:
            s = v[n.replace('.', '-')][['Open', 'Close']].dropna()
        except KeyError:
            continue
        s = s[s.index.strftime('%Y%m%d') < today]   # nothing from today
        on = np.log(s.Open / s.Close.shift()).dropna() * 1e4
        cc = np.log(s.Close / s.Close.shift()).dropna() * 1e4
        row = {'overnight_bps_last5': [round(x, 1) for x in on.tail(5)],
               'close_to_close_bps_last5': [round(x, 1) for x in cc.tail(5)],
               'vol_20d_bps': round(float(cc.tail(20).std()), 1)}
        if n != 'QQQ':
            row['earnings_tonight'] = bool((earnings or {}).get(n))
        (st.setdefault('market', {}) if n == 'QQQ' else st['names'])[n] = row
    return st


def earnings_tonight(names, today):
    """{name: True} for names reporting between today's close and the next
    open, from Yahoo's earnings calendar."""
    import yfinance as yf
    lo = pd.Timestamp(today).tz_localize(ET) + pd.Timedelta(hours=15)
    hi = lo + pd.Timedelta(hours=19)            # through the next 10:00
    out = {}
    for n in names:
        try:
            e = yf.Ticker(n.replace('.', '-')).get_earnings_dates(limit=8)
            ts = [t.tz_convert(ET) for t in (e.index if e is not None else [])]
            out[n] = any(lo <= t <= hi for t in ts)
        except Exception:
            out[n] = False
    return out


# ---------------------------------------------------------------- the log --

def read_log(path=GLOG):
    return pd.read_csv(path, dtype={'date': str}) if os.path.exists(path) \
        else pd.DataFrame(columns=COLS)


def decide(backend, names, today, now_et, state, path=GLOG, ask=None):
    """Ask once for tonight and append. Refuses a night already decided, and
    a decision at or after the MOC cutoff."""
    log = read_log(path)
    if (log.date == today).any():
        raise SystemExit(f'{today} already decided -- a night is never '
                         f're-decided')
    if (now_et.hour, now_et.minute) >= CUTOFF:
        raise SystemExit(f'{now_et:%H:%M} ET is at or past the {CUTOFF[0]}:'
                         f'{CUTOFF[1]:02d} cutoff -- too late to decide '
                         f'tonight honestly')
    p, model, ms = ask(state, names)
    held = [p[n] for n in names if n in p]
    med = float(np.median(held)) if held else 0.5
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
            q = 'hedge_book' if k == BOOK else 'hold'
            yes = p[k] >= 0.5 if k == BOOK else \
                not (p[k] < 0.5 and p[k] < med - SKIP_MARGIN)
            w.writerow([today, now_et.strftime('%Y-%m-%d %H:%M:%S'), backend,
                        model, q, k, f'{p[k]:.4f}',
                        ('hedge' if yes else 'none') if k == BOOK else
                        ('hold' if yes else 'skip'), ms, sha])
    return p


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
    ap.add_argument('--decide', action='store_true')
    ap.add_argument('--grade', action='store_true')
    ap.add_argument('--backend', choices=['http', 'rule'], default='http')
    a = ap.parse_args()
    paper = pd.read_csv(PAPER, dtype=str)
    if a.decide:
        now = dt.datetime.now(ET)
        today = now.strftime('%Y%m%d')
        names = paper.iloc[-1].q5.split(';')
        state = build_state(names, today,
                            earnings=earnings_tonight(names, today))
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
        p = decide(a.backend, names, today, now, state, ask=ask)
        print(f'{today}: ' + '  '.join(f'{k} {v:.2f}' for k, v in p.items())
              + '  -> reports/guard_log.csv (commit before the close)')
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
        for b, g in log.groupby('backend'):
            print(f'[{b}] ' + verdict(grade(g, paper, daily, prints)))


if __name__ == '__main__':
    main()
