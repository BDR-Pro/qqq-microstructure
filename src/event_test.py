# Part of qqq-microstructure.
#
# Does event information that was PUBLIC BEFORE the 15:30 ET decision improve
# the hedged book? Six pre-registered hypotheses on SEC EDGAR filings, one
# placebo, a shuffle control and three leak canaries, tested on the wide
# panel (data/xsec1000, 1999-2026) and then applied to the traded object.
#
# Everything below this header down to "THE TEST" was written before any
# event-conditional return was computed, and is not edited after. The
# research that set it up looked only at counts, timestamps and unsigned gap
# sizes (the mapping fingerprint) -- never at a signed return.
#
# ============================================================================
# DECLARATIONS (pre-registered 2026-10-14)
# ============================================================================
#
# UNIT. Night D = the close of trading day D -> the open of the next trading
# day d. Outcome y(i, D) = on_bps(i, d) - on_bps(QQQ, d) (QQQQ from 2004-12
# to 2011-03): the hedged book's per-name tilt. on_bps is load_panel's
# (log, bps; NaN across missing days, split-ratio gaps and the 40% backstop --
# the frozen path's own mask). Name-nights with a NaN outcome are dropped.
# A second reading, y relative to the equal-weight panel mean of that night,
# is printed beside the first; with night fixed effects the estimate is the
# same by construction.
#
# UNIVERSE. Wide-panel name-nights (top-1000 by monthly dollar volume),
# ETFs/ETNs (the frozen list) and exchange test symbols excluded, and only
# names with a CIK in the research's validated (ticker, year) map for the
# year of night D -- an unmapped name's event status is unknown, so it is
# neither flagged nor in the comparison group. Tiers per xsec_wide: CORE
# (ever ranked <= 150) and never-core.
#
# TIME. tau_D = 15:30 ET (12:30 ET on 13:00-close days). An event e may be
# used for night D only if its SEC acceptance time A(e) < tau_D - 10 min
# (strict). A(e) is the true ET acceptance (filing index "Accepted", or the
# company filing feed's <updated>, which carries its UTC offset); when
# neither was fetched, the JSON wall clock read as ET (an upper bound, 0-10 h
# late, never early); for date-only filings before 2002-07, filingDate 17:30
# ET. First eligible night n(e) = the first trading day D with A(e) <
# tau_D - 10 min. A window of k nights: e is active on nights n(e) ..
# n(e)+k-1 (trading days). Amendments (/A) are other form types and never
# match; a filing listed under several mapped tickers counts once per name.
#
# HYPOTHESES (expected sign of y on flagged name-nights):
#   H1  earnings release expected tonight                              (+)
#       Sources: 8-K with item 2.02 (filed >= 2004-08-23) or old item 12
#       (2003-03-28 .. 2004-08-22). A source's earnings night E: A >= the
#       close on trading day t -> night t; A before 09:30 on t -> the night
#       before t (weekend/holiday times: the night spanning them); A inside a
#       session (09:30 to the close) -> not a source; a source whose bounds
#       leave the night ambiguous -> not a source. Flag night D for the CIK
#       iff (a) D is the first trading day on or after day(E) + 364 calendar
#       days for some source E, and (b) no item-2.02/12 8-K from the CIK has
#       A in [day(D) - 45 calendar days 00:00 ET, tau_D - 10 min).
#       One night (k = 1). Flags start 2004-03.
#   H2  bad-news 8-K items {1.03, 2.05, 2.06, 3.01, 4.01, 4.02},
#       filed >= 2004-08-23; k = 5                                     (-)
#       Secondary, reported apart and outside the Holm family: old items 3
#       and 4 (bankruptcy, auditor change), 1999 .. 2004-08-22; k = 5.
#   H3  NT 10-K or NT 10-Q (late-filing notice), 1999-2026; k = 5       (-)
#   H4  common-equity supply: S-1, S-3, S-3ASR, F-1, F-3, F-3ASR (initial)
#       or 424B1/3/4/5/7, issuer-years with > 12 424B1-7 filings excluded
#       (structured-note shelves), and the primary document's first 20,000
#       text characters match 'shares of (our )?(class [a-z] )?common stock|
#       ordinary shares|american depositary shares' while its cover (first
#       3,000 characters) does not match '% (senior )?notes due|debentures';
#       k = 3. Documents not fetched -> not classified -> not an event.  (-)
#   H5  initial SC 13D / SCHEDULE 13D whose index page lists the mapped CIK
#       as Subject (not only Filed by), 1999-2026; k = 3                 (+)
#   H6  Form 4 (not 4/A) for the mapped issuer with >= 1 non-derivative
#       'P' transaction and >= 1 reporting owner who is a director or
#       officer (SEC Insider Transactions Data Sets, 2006q1-2026q1); k = 3 (+)
#   P0  PLACEBO: 8-K whose items are exactly {5.07} or {5.07, 9.01}, filed
#       >= 2010-02-28 (shareholder-vote results); k = 5. Must show |t| < 2;
#       a significant P0 means the design is biased, not that votes carry
#       information.
#
# STATISTICS. Per hypothesis: beta from y ~ flag with night fixed effects
# (within-night demeaning), standard error two-way clustered by night and
# by name (Cameron-Gelbach-Miller), t = beta / SE; the night-clustered-only
# t is printed beside it. Also printed: flagged name-nights, mean y flagged
# and unflagged, their raw difference. One-sided p in the declared direction
# (normal). Multiple testing: Holm over H1-H6 at family alpha 0.05 (m = 6).
# Breakdowns (information, not tested): era of night D (1999-2000,
# 2001-2009, 2010-2017, 2018-2026) and tier (CORE, never-core).
#
# CONTROLS that must show nothing:
#   - P0, above.
#   - SHUFFLE: every event of a hypothesis moved to a random other night of
#     the same name (uniform over that CIK's test name-nights, seeds 1-5),
#     same k. Pass: |t| < 2 for the seed-1 draw; all five t's printed.
# LEAK CANARIES (H2-H6, P0): the same events placed (a) by filingDate
# (known from 00:00 ET of the filing date) and (b) by the JSON timestamp
# read as UTC. An effect that clears |t| > 2 in the declared direction ONLY
# under a canary placement is lookahead, and is reported as such.
#
# THE BOOK. The traded object: the frozen model (models/xsec_on_lgbm.txt,
# loaded, never refitted) scores each trade month's top-150 universe --
# replicated from the wide panel as the names ranked <= 150 by that month's
# own dollar volume, through xsec_ml.build_table unchanged -- and the top
# 20% by score is the Q5 basket (xsec_ml.portfolio's rule). Night tilt = the
# equal-weight mean of the basket's valid on_bps minus QQQ's. Counterfactual
# per hypothesis: names flagged on night D are skipped and the hedge is
# resized to the long notional actually held (capital re-spread over the
# remaining names; all flagged -> flat, tilt 0). Change per night =
# counterfactual - as-is, over ALL basket nights; mean, t (nights iid), the
# nights it touched, and the book's nightly sd before and after. A second
# reading keeps per-name weights (skipped capital stays in cash). The model
# was trained through 2026-03 on these months, so the LEVEL of the replica's
# tilt is in-sample; the CHANGE is what this measures. Basket names without
# a mapped CIK cannot be flagged; their share is printed.
#
# VERDICT RULES.
#   A hypothesis is SUPPORTED iff Holm rejects it in its declared direction,
#   P0 passes, its seed-1 shuffle passes and no canary flags it as a leak.
#   A veto is worth PROPOSING for the book only if its hypothesis is
#   SUPPORTED and the book change has t > 2 in the improving direction.
#   Nothing here changes the traded book: a veto would be a dated re-freeze
#   with its own forward record (RESULTS 19/20/27).
#
#   python src/event_test.py               # needs data/edgar/events.parquet
#
# ============================================================================

import os, sys, argparse, math, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgar_events as ee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'models', 'xsec_on_lgbm.txt')
OUT = os.path.join(ROOT, 'data', 'edgar', 'event_test_results.csv')
ALPHA = 0.05
SEEDS = (1, 2, 3, 4, 5)
ERAS = [('1999-2000', 1999, 2000), ('2001-2009', 2001, 2009),
        ('2010-2017', 2010, 2017), ('2018-2026', 2018, 2026)]

# id: (label, event class, k nights, declared sign, in the Holm family)
HYP = {
    'H1': ('earnings expected tonight (364-day rule)', 'earn', 1, +1, True),
    'H2': ('bad-news 8-K items', 'bad', 5, -1, True),
    'H3': ('late-filing notice NT 10-K/Q', 'nt', 5, -1, True),
    'H4': ('common-equity supply visible', 'supply', 3, -1, True),
    'H5': ('initial 13D, company is subject', 'sc13d', 3, +1, True),
    'H6': ('insider open-market purchase', 'buy', 3, +1, True),
    'P0': ('PLACEBO vote-results 8-K', 'vote', 5, 0, False),
    'H2old': ('old items 3/4 (secondary)', 'bad_old', 5, -1, False),
}
HOLM = [h for h, v in HYP.items() if v[4]]


# ============================================================================
# THE TEST -- machinery
# ============================================================================

def holm(pvals, alpha=ALPHA):
    """{id: p} -> {id: rejected?} by Holm's step-down procedure."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    out = {k: False for k in pvals}
    for j, k in enumerate(order):
        if pvals[k] <= alpha / (m - j):
            out[k] = True
        else:
            break
    return out


def one_sided_p(t, sign):
    """p of the declared direction under N(0,1); sign 0 -> two-sided."""
    if t is None or not np.isfinite(t):
        return 1.0
    if sign == 0:
        return math.erfc(abs(t) / math.sqrt(2))
    return 0.5 * math.erfc(sign * t / math.sqrt(2))


def fe_twoway(y, f, night, name):
    """y on a 0/1 flag with night fixed effects; SE two-way clustered by
    night and name (CGM: V_night + V_name - V_obs, each with G/(G-1)).
    Returns dict(beta, se, t, t_night, n, n_flag, mean_f, mean_u, diff)."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    night = np.asarray(night)
    name = np.asarray(name)
    n = len(y)
    nf = int(f.sum())
    res = dict(n=n, n_flag=nf, beta=np.nan, se=np.nan, t=np.nan,
               t_night=np.nan,
               mean_f=float(y[f > 0].mean()) if nf else np.nan,
               mean_u=float(y[f == 0].mean()) if nf < n else np.nan)
    res['diff'] = res['mean_f'] - res['mean_u']
    if nf == 0 or nf == n:
        return res
    ni = pd.factorize(night)[0]
    mi = pd.factorize(name)[0]
    cnt = np.bincount(ni)
    ft = f - (np.bincount(ni, f) / cnt)[ni]
    yt = y - (np.bincount(ni, y) / cnt)[ni]
    sxx = float((ft * ft).sum())
    if sxx <= 0:
        return res
    beta = float((ft * yt).sum() / sxx)
    s = ft * (yt - beta * ft)

    def v(codes):
        S = np.bincount(codes, s)
        G = int((np.bincount(codes) > 0).sum())
        return G / max(G - 1, 1) * float((S * S).sum()) / sxx ** 2

    vn, vm = v(ni), v(mi)
    vo = n / max(n - 1, 1) * float((s * s).sum()) / sxx ** 2
    var = vn + vm - vo
    if var <= 0:
        var = max(vn, vm)
    res.update(beta=beta, se=math.sqrt(var), t=beta / math.sqrt(var),
               t_night=beta / math.sqrt(vn) if vn > 0 else np.nan)
    return res


def active_keys(cik, n_pos, k, n_days):
    """Event (cik, first eligible night) -> int64 keys cik*2^20 + night for
    every night of its k-night window inside the calendar."""
    cik = np.asarray(cik, np.int64)
    n_pos = np.asarray(n_pos, np.int64)
    if not len(cik):
        return np.array([], np.int64)
    pos = (n_pos[:, None] + np.arange(k)[None, :]).ravel()
    c = np.repeat(cik, k)
    ok = (pos >= 0) & (pos < n_days)
    return np.unique(c[ok] * (1 << 20) + pos[ok])


def nn_keys(cik, pos):
    return np.asarray(cik, np.int64) * (1 << 20) + np.asarray(pos, np.int64)


def h1_flags(earn, cal):
    """H1's (cik, night) pairs from earnings 8-Ks (see the declaration).
    `earn`: DataFrame cik, A (ns, best-known), lo (ns lower bound),
    exact_or_resolved (bool)."""
    seg_a = cal.segment(earn.A.values)
    seg_l = cal.segment(earn.lo.values)
    resolved = earn.exact.values | (seg_a == seg_l)
    src = earn[resolved & (seg_a % 2 == 1) & (seg_a >= 0)]
    E = (cal.segment(src.A.values) - 1) // 2
    eday = pd.to_datetime(cal.days[E], format='%Y%m%d')
    target = (eday + pd.Timedelta(days=364)).strftime('%Y%m%d').values
    D = cal.on_or_after(target)
    cand = pd.DataFrame({'cik': src.cik.values, 'D': D})
    cand = cand[cand.D < len(cal.days)].drop_duplicates()
    if cand.empty:
        return cand.assign(D=cand.D.astype(int))
    # (b): no earnings 8-K accepted in [day(D) - 45 days 00:00 ET, cut_D)
    dday = pd.to_datetime(cal.days[cand.D.values], format='%Y%m%d')
    lo = ee.as_ns((dday - pd.Timedelta(days=45)).tz_localize(ee.ET))
    hi = cal.cut[cand.D.values]
    keep = np.ones(len(cand), bool)
    by = {c: np.sort(g.values) for c, g in earn.groupby('cik').A}
    for i, (c, l, h) in enumerate(zip(cand.cik.values, lo, hi)):
        a = by.get(c)
        if a is not None and np.searchsorted(a, h, 'left') > \
                np.searchsorted(a, l, 'left'):
            keep[i] = False
    return cand[keep]


def shuffle_events(cik, n_pos, cik_nights, rng):
    """Each event moved to a uniformly random OTHER night of the same name
    (that CIK's test name-nights). Events of a CIK with < 2 nights drop."""
    out_c, out_n = [], []
    for c, n in zip(cik, n_pos):
        pool = cik_nights.get(int(c))
        if pool is None or len(pool) < 2:
            continue
        x = pool[rng.integers(len(pool))]
        while x == n:
            x = pool[rng.integers(len(pool))]
        out_c.append(c)
        out_n.append(x)
    return np.array(out_c, np.int64), np.array(out_n, np.int64)


# ============================================================================
# data
# ============================================================================

def qqq_series(df):
    q = df[df.ticker.isin(['QQQ', 'QQQQ'])][['day', 'ticker', 'on_bps']]
    q = q.sort_values('ticker')            # QQQ before QQQQ if both
    return q.dropna(subset=['on_bps']).drop_duplicates('day') \
        .set_index('day').on_bps


def name_nights(df, cal, amap, tiers):
    """The test universe: one row per (name, night D) with a valid outcome
    and a mapped CIK. `pos` = calendar position of night D."""
    from xsec_backtest import ETF
    d = df[df.on_bps.notna() & ~df.ticker.isin(ETF)][
        ['ticker', 'day', 'on_bps']].copy()
    dpos = pd.Series(np.arange(len(cal.days)), index=cal.days)
    d['pos'] = dpos.reindex(d.day.values).values - 1
    d = d[d.pos >= 0]
    d['year'] = pd.Series(cal.days[d.pos.values]).str[:4].astype(int).values
    m = amap.rename(columns={'yr': 'year'})
    d = d.merge(m, on=['ticker', 'year'], how='left')
    qqq = qqq_series(df)
    d['y'] = d.on_bps - qqq.reindex(d.day.values).values
    ew = df[df.on_bps.notna() & ~df.ticker.isin(ETF)].groupby('day').on_bps \
        .mean()
    d['y_ew'] = d.on_bps - ew.reindex(d.day.values).values
    d['tier'] = d.ticker.map(tiers).fillna('never-core').values
    d['mapped'] = d.cik.notna()
    d = d[d.y.notna()].reset_index(drop=True)
    return d


def event_frame(ev, cls, cal, placement='true'):
    """One row per (cik, filing) of an event class, with the placement's A
    and first eligible night."""
    classes = cls if isinstance(cls, tuple) else (cls,)
    e = ev[ev.event.isin(classes)]
    if 'sc13d' in classes:
        e = e[e.role == 'subject']
    if {'supply', 'supply_x'} & set(classes):
        e = e[e.equity == 1.0]
    e = e.drop_duplicates(['cik', 'accession'])
    if placement == 'true':
        A = ee.as_ns(e.accepted_et)
    elif placement == 'filingdate':
        A = ee.as_ns(pd.DatetimeIndex(pd.to_datetime(e.filing_date))
                     .tz_localize(ee.ET))
    elif placement == 'z_as_utc':
        A = ee.as_ns(pd.DatetimeIndex(pd.to_datetime(
            e.json_raw.str.slice(0, 19), format='%Y-%m-%dT%H:%M:%S'))
            .tz_localize('UTC'))
    else:
        raise ValueError(placement)
    out = pd.DataFrame({'cik': e.cik.values.astype(np.int64), 'A': A,
                        'lo': ee.as_ns(e.accepted_lo_et),
                        'exact': (e.ts_kind == 'exact').values,
                        'ts_kind': e.ts_kind.values})
    out['n'] = cal.first_eligible(out.A.values)
    return out


def flags_for(hid, ev, cal, nn, placement='true', shuffle_seed=None,
              cik_nights=None):
    """0/1 flag per test name-night for hypothesis `hid`, plus the events."""
    label, cls, k, sign = (HYP[hid] if hid in HYP else HYP_PH[hid])[:4]
    e = event_frame(ev, cls, cal, placement)
    if hid == 'H1':
        pairs = h1_flags(e, cal)
        c, n = pairs.cik.values, pairs.D.values
    else:
        c, n = e.cik.values, e.n.values
    if shuffle_seed is not None:
        c, n = shuffle_events(c, n, cik_nights,
                              np.random.default_rng(shuffle_seed))
    keys = active_keys(c, n, k, len(cal.days))
    f = np.isin(nn_keys(nn.cik.values, nn.pos.values), keys)
    return f.astype(np.int8), e


# ============================================================================
# the book
# ============================================================================

def book_table(df, booster):
    """Q5 basket name-nights of the replicated top-150 traded object:
    columns ticker, day (the row whose on_bps is the night), pos, on_bps."""
    from xsec_wide import monthly_rank
    from xsec_ml import build_table, FEATS
    rk = monthly_rank(df).rename('rank').reset_index()
    core = df.merge(rk[rk['rank'] <= 150][['month', 'ticker']],
                    on=['month', 'ticker'])
    t = build_table(core)
    t['score'] = booster.predict(t[FEATS])
    bym = {m: g[['ticker', 'day', 'on_bps']] for m, g in core.groupby('month')}
    rows = []
    for T, g in t.groupby('tmonth'):
        k = len(g) // 5
        if k < 5:
            continue
        q5 = set(g.nlargest(k, 'score').ticker)
        sub = bym[T]
        rows.append(sub[sub.ticker.isin(q5)].assign(tmonth=T, k=k))
    b = pd.concat(rows, ignore_index=True)
    return b, t


def book_eval(b, flag, qqq):
    """Nightly tilt as-is and with flagged names skipped. `flag` aligns
    with b's rows. Returns per-night frame."""
    b = b.assign(fl=flag.astype(bool))
    v = b[b.on_bps.notna()]
    g = v.groupby('day')
    asis = g.on_bps.mean()
    n_all = g.size()
    keep = v[~v.fl].groupby('day').on_bps
    kmean = keep.mean().reindex(asis.index)
    kn = keep.size().reindex(asis.index).fillna(0)
    q = qqq.reindex(asis.index)
    tilt = asis - q
    # re-spread: mean of the names held minus QQQ; nothing held -> flat
    cf = (kmean - q).where(kn > 0, 0.0)
    # cash: per-name weights kept, skipped capital idle
    cash = ((kmean - q) * kn / n_all).where(kn > 0, 0.0)
    out = pd.DataFrame({'tilt': tilt, 'cf': cf, 'cash': cash,
                        'n': n_all, 'skipped': n_all - kn}).dropna(
        subset=['tilt'])
    return out


def t_of(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))


# ============================================================================
# POST-HOC AMENDMENTS -- written 2026-10-14 AFTER the real run (13:01Z), in
# answer to an independent review of it. NOT EVIDENCE: each was chosen after
# the declared results had been seen. They print BELOW the declared output,
# which is unchanged, and no declared verdict reads them.
#   A1 (review finding 1) a universe known before the decision: the name
#      must be in the wide-panel file of the last complete month before day
#      D (month(D) - 1), as xsec_live's lagged rule; the declared universe
#      used month(d)'s own dollar volume, which includes the event's
#      after-the-fact volume. Residual exit conditioning stays: a name that
#      leaves the top 1000 has no outcome rows.
#   A2 (finding 6) name AND night fixed effects (an exact two-way
#      projection, TwoWay), the same CGM clustering, and the shuffle re-run
#      under it. P0 cannot detect name-composition bias (votes hit almost
#      every firm), so it is no check on this.
#   A3 (finding 7) the outcome with the 40% backstop nights restored:
#      load_panel(keep_backstop=True) for the OUTCOME only (never features),
#      gaps beyond 10,000 bps dropped as split errors, the rest clipped at
#      +-BACKSTOP; the hedge leg stays on the default series.
#   A4 (finding 3) H4 with the structured-shelf count known at acceptance
#      (edgar_events.n424_known_at_acceptance -> class 'supply_x'): H4a.
#   A5 (finding 2) the canary map from ticker evidence alone
#      (edgar_events.evidence_map), restricted to CIKs whose filings were
#      read; and H2/H3 split by whether the CIK stops filing within 3 years
#      of night D (information: it uses the future, as the map did).
#   A6 (finding 9) leak inflation: a canary placement whose t beats the true
#      placement's by more than 2 in the declared direction.
#   A7 (findings 4, 5, 8, 10) the book: per-name attribution of each veto's
#      change (largest name's share, t without it, t clustered by name); the
#      cash reading's level term; the replica rebuilt the way xsec_live
#      trades (the T-1 top-150 list with no month-T membership, scored with
#      xsec_live.features; night D traded with the basket of D's month); and
#      the pre-registered region's sha256, checked every run and logged.
#   ALL = A1 + A2 + A3 together, with H4a for H4: the post-hoc best estimate.
#   Small cells (< 200 flagged name-nights, < 30 book nights) are marked.
# ============================================================================

import hashlib, subprocess, datetime as _dt

# sha256 of every line above the rule that opens "THE TEST -- machinery", as
# it stood for the real run of 2026-10-14 13:01Z (file mtime 11:18:34Z).
PREREG_SHA256 = \
    '05c682bc325da3fedc796c756ec809fb0886bbce0936ab187b4d83d413264593'
RUNLOG = os.path.join(ROOT, 'data', 'edgar', 'event_test_runs.log')
OUT_PH = os.path.join(ROOT, 'data', 'edgar', 'event_test_posthoc.csv')
SMALL_FLAGS = 200
SMALL_NIGHTS = 30
SPLIT_ERR_BPS = 1e4

# post-hoc variants: id -> (label, event classes, k, sign)
HYP_PH = {'H4a': ('equity supply, shelf count known at acceptance',
                  ('supply', 'supply_x'), 3, -1)}
HOLM_PH = ['H1', 'H2', 'H3', 'H4a', 'H5', 'H6']


def prereg_sha256(path=None):
    """sha256 of the pre-registered region: every line above the '# ===='
    rule that opens 'THE TEST -- machinery'."""
    L = open(path or os.path.abspath(__file__)).read().split('\n')
    i = next(i for i, l in enumerate(L)
             if l.startswith('# THE TEST -- machinery'))
    return hashlib.sha256(('\n'.join(L[:i - 1]) + '\n').encode()).hexdigest()


def log_run(mode, sha, path=RUNLOG):
    """One ledger line per run: UTC time, mode, the region's hash, whether
    it matches the pin, the whole file's hash and git HEAD. data/edgar is
    git-ignored, so this is a local record; pin it in git before any new
    real run (commit the declaration, or this hash, first)."""
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        head = '?'
    whole = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()) \
        .hexdigest()
    line = (f'{_dt.datetime.now(_dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}\t{mode}'
            f'\tprereg={sha}\t{"MATCH" if sha == PREREG_SHA256 else "EDITED"}'
            f'\tfile={whole}\tgit={head}\n')
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').write(line)
    except OSError:
        pass
    return line


def _cgm(s, sxx, ni, mi):
    """Two-way (night, name) clustered variance of a slope from its score
    contributions s (CGM, each piece with G/(G-1)); also the night-only."""
    def v(codes):
        S = np.bincount(codes, s)
        G = int((np.bincount(codes) > 0).sum())
        return G / max(G - 1, 1) * float((S * S).sum()) / sxx ** 2
    n = len(s)
    vn, vm = v(ni), v(mi)
    vo = n / max(n - 1, 1) * float((s * s).sum()) / sxx ** 2
    var = vn + vm - vo
    return (var if var > 0 else max(vn, vm)), vn


class TwoWay:
    """A2: the exact two-way (night + name) fixed-effects projection. Nights
    are absorbed by demeaning; the name effects solve (diag(n_name) -
    C' diag(1/n_night) C) a = the name sums of the night-demeaned vector,
    C the night x name incidence (at most one row per name-night). The
    matrix is inverted once per universe; each projection is then two
    bincounts and one matrix-vector product."""

    def __init__(self, night, name):
        self.ni = pd.factorize(np.asarray(night))[0]
        self.mi = pd.factorize(np.asarray(name))[0]
        N, M = int(self.ni.max()) + 1, int(self.mi.max()) + 1
        self.M = M
        self.cn = np.bincount(self.ni, minlength=N).astype(float)
        C = np.zeros((N, M))
        np.add.at(C, (self.ni, self.mi), 1.0)
        A = np.diag(C.sum(0)) - (C.T / self.cn) @ C
        del C
        A += 1.0 / M              # pins the constant (A's null space is 1)
        try:
            self.Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.Ainv = np.linalg.pinv(A)

    def __call__(self, v):
        v = np.asarray(v, float)
        w = v - (np.bincount(self.ni, v) / self.cn)[self.ni]
        a = self.Ainv @ np.bincount(self.mi, w, minlength=self.M)
        an = a[self.mi]
        return w - (an - (np.bincount(self.ni, an) / self.cn)[self.ni])


def fe_name_night(y, f, tw, yt=None):
    """fe_twoway's estimate with name AND night fixed effects (A2), the
    same two-way clustering. `yt` = tw(y) when already computed."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    n, nf = len(y), int(f.sum())
    res = dict(n=n, n_flag=nf, beta=np.nan, se=np.nan, t=np.nan,
               t_night=np.nan,
               mean_f=float(y[f > 0].mean()) if nf else np.nan,
               mean_u=float(y[f == 0].mean()) if nf < n else np.nan)
    res['diff'] = res['mean_f'] - res['mean_u']
    if nf == 0 or nf == n:
        return res
    ft = tw(f)
    yt = tw(y) if yt is None else yt
    sxx = float((ft * ft).sum())
    if sxx <= 1e-9:
        return res
    beta = float((ft * yt).sum() / sxx)
    var, vn = _cgm(ft * (yt - beta * ft), sxx, tw.ni, tw.mi)
    res.update(beta=beta, se=math.sqrt(var), t=beta / math.sqrt(var),
               t_night=beta / math.sqrt(vn) if vn > 0 else np.nan)
    return res


def lagged_member(nn, df, cal):
    """A1: is the name in the wide-panel file of the month before night D's
    month -- the last complete month at the decision (xsec_live's rule)?"""
    months = np.sort(df.month.unique())
    prev = dict(zip(months[1:], months[:-1]))
    have = df[['month', 'ticker']].drop_duplicates()
    have = set(zip(have.month.values, have.ticker.values))
    dd = pd.Series(cal.days[np.asarray(nn.pos, int)])
    pm = (dd.str[:4] + '-' + dd.str[4:6]).map(prev).values
    return np.array([(m, t) in have for m, t in zip(pm, nn.ticker.values)],
                    bool)


def clip_outcome(on_keep, backstop_bps, split_bps=SPLIT_ERR_BPS):
    """A3: backstop-kept overnight gaps; beyond +-split_bps they are split
    errors (dropped), the rest clipped at +-backstop_bps with sign kept."""
    x = np.asarray(on_keep, float)
    return np.where(np.abs(x) > split_bps, np.nan,
                    np.clip(x, -backstop_bps, backstop_bps))


def leak_inflation(t_canary, t_true, sign):
    """A6: a canary placement beats the true one by > 2 t in the declared
    direction (and clears 2 itself) -- lookahead inflating a real-time
    effect, which the declared 'only under a canary' rule cannot see."""
    if sign == 0 or not np.isfinite(t_canary) or not np.isfinite(t_true):
        return False
    return sign * t_canary > 2 and sign * (t_canary - t_true) > 2


def small(n, lim):
    return '  small-n' if n < lim else ''


def book_prepare(b, cal, amap):
    """Book rows -> + pos (night D), year, cik, mapped, key (cik, night)."""
    b = b.copy()
    if 'pos' not in b:
        dpos = pd.Series(np.arange(len(cal.days)), index=cal.days)
        b['pos'] = dpos.reindex(b.day.values).values - 1
        b = b[b.pos >= 0]
    b = b.reset_index(drop=True)
    b['year'] = pd.Series(cal.days[b.pos.values.astype(int)]).str[:4] \
        .astype(int).values
    b = b.merge(amap.rename(columns={'yr': 'year'}), on=['ticker', 'year'],
                how='left')
    b['mapped'] = b.cik.notna()
    b['bk'] = nn_keys(b.cik.fillna(0).astype(np.int64).values,
                      b.pos.values.astype(np.int64))
    return b


def book_contrib(b, fl, qqq):
    """Exact per-name split of a veto's nightly change (re-spread reading):
    skipping s of n names moves the night by sum_j (m - x_j) / (n - s), m
    the as-is basket mean and x_j a skipped name's on_bps; when all n are
    skipped (flat) -tilt is split equally. Returns (per-night frame of
    book_eval, change per night, rows day / ticker / c)."""
    fl = np.asarray(fl, bool)
    r = book_eval(b, fl, qqq)
    chg = r.cf - r.tilt
    v = b.assign(fl=fl)
    v = v[v.on_bps.notna() & v.fl & v.day.isin(r.index)]
    if not len(v):
        return r, chg, pd.DataFrame({'day': [], 'ticker': [], 'c': []})
    m = b[b.on_bps.notna()].groupby('day').on_bps.mean()
    d = v.day.values
    n = r.n.reindex(d).values.astype(float)
    s = r.skipped.reindex(d).values.astype(float)
    flat = n == s
    c = np.where(flat, -r.tilt.reindex(d).values / s,
                 (m.reindex(d).values - v.on_bps.values) /
                 np.where(flat, 1.0, n - s))
    return r, chg, pd.DataFrame({'day': d, 'ticker': v.ticker.values,
                                 'c': c})


def book_diag(b, fl, qqq):
    """A7: a veto's book change with its concentration: the largest
    contributing name (in the direction of the total), its share, the
    change and t without that name's flags, and t clustered by name (each
    hit night assigned to the name with its largest |contribution|; nights
    not hit are singletons). Also the cash reading's level term: cash chg =
    re-spread chg - mean(skip share x kept tilt), and that kept tilt is the
    in-sample level, not the veto."""
    fl = np.asarray(fl, bool)
    r, chg, c = book_contrib(b, fl, qqq)
    hit = r.skipped > 0
    out = dict(chg=chg.mean(), t=t_of(chg), nights=int(hit.sum()),
               skipped=int(r.skipped.sum()),
               chg_hit=chg[hit].mean() if hit.any() else np.nan,
               sd_after=r.cf.std(), cash=(r.cash - r.tilt).mean(),
               level=float(((r.skipped / r.n) * r.cf).where(hit, 0).mean()),
               top=None, share=np.nan, chg_ex=np.nan, t_ex=np.nan,
               t_name=np.nan)
    if not len(c):
        return out
    tot = float(c.c.sum())
    by = c.groupby('ticker').c.sum()
    top = (by * (1.0 if tot >= 0 else -1.0)).idxmax()
    out.update(top=top, share=by[top] / tot if tot else np.nan)
    _, chg2, _ = book_contrib(b, fl & (b.ticker.values != top), qqq)
    out.update(chg_ex=chg2.mean(), t_ex=t_of(chg2))
    lead = c.loc[c.c.abs().groupby(c.day.values).idxmax().values] \
        .set_index('day').ticker
    x = chg.values.astype(float)
    e = x - x.mean()
    L = lead.reindex(chg.index).values
    codes = np.empty(len(x), np.int64)
    isn = pd.isna(L)
    codes[~isn] = pd.factorize(L[~isn])[0]
    codes[isn] = codes[~isn].max() + 1 + np.arange(isn.sum()) if (~isn).any() \
        else np.arange(isn.sum())
    S = np.bincount(codes, e)
    G = len(S)
    var = G / max(G - 1, 1) * float((S * S).sum()) / len(x) ** 2
    out['t_name'] = x.mean() / math.sqrt(var) if var > 0 else np.nan
    return out


def book_table_lagged(df, booster, cal):
    """A7: the replica the way xsec_live trades. Trade month T's universe =
    the names ranked <= 150 by month T-1's dollar volume (no month-T
    membership); features = xsec_live.features over months T-12..T-1 of
    the top-150 rows (build_table's own aggregation), ranked across that
    universe; Q5 = the top 20% by the frozen booster. Night D is traded
    with the basket of D's month (live trades the last night of T-1 with
    T-1's basket); its outcome is the wide panel's next row, so a name that
    leaves the top 1000 cannot be priced (residual exit conditioning).
    Rows: ticker, day (the outcome row), pos (night D), on_bps, tmonth, k."""
    from xsec_wide import monthly_rank
    from xsec_live import month_aggs, features
    from xsec_ml import rank_pm, FEATS
    from xsec_backtest import ETF, MIN_NAMES
    rk = monthly_rank(df).rename('rank').reset_index()
    core = df.merge(rk[rk['rank'] <= 150][['month', 'ticker']],
                    on=['month', 'ticker'])
    agg = month_aggs(core)
    months = sorted(core.month.unique())
    uni = {m: set(g) for m, g in core.groupby('month')['ticker']}
    dpos = pd.Series(np.arange(len(cal.days)), index=cal.days)
    w = df[['ticker', 'day', 'on_bps'] +
           (['on_c'] if 'on_c' in df else [])].copy()
    w['pos'] = dpos.reindex(w.day.values).values - 1
    w = w[w.pos >= 0]
    dd = pd.Series(cal.days)
    w['dmonth'] = (dd.str[:4] + '-' + dd.str[4:6]).values[w.pos.values]
    byD = {m: g for m, g in w.groupby('dmonth')}
    rows = []
    for i in range(13, len(months)):
        T = months[i]
        live = features(agg, months[i - 12:i], uni[months[i - 1]] - ETF)
        if len(live) < MIN_NAMES:
            continue
        for f in FEATS:
            live[f] = rank_pm(live[f], pd.Series(0, index=live.index))
        live['score'] = booster.predict(live[FEATS])
        k = len(live) // 5
        if k < 5 or T not in byD:
            continue
        q5 = set(live.nlargest(k, 'score').ticker)
        g = byD[T]
        rows.append(g[g.ticker.isin(q5)].drop(columns='dmonth')
                    .assign(tmonth=T, k=k))
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# report
# ============================================================================

def fmt(r):
    return (f'{r["n_flag"]:>9,} {r["mean_f"]:>+8.2f} {r["mean_u"]:>+7.2f} '
            f'{r["diff"]:>+8.2f} {r["beta"]:>+8.2f} {r["t"]:>+6.2f} '
            f'{r["t_night"]:>+6.2f}')


HDR = (f'{"flagged":>9} {"mean_f":>8} {"mean_u":>7} {"diff":>8} '
       f'{"beta_FE":>8} {"t_2w":>6} {"t_nt":>6}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/xsec1000')
    ap.add_argument('--events', default=ee.EVENTS)
    ap.add_argument('--no-book', action='store_true')
    ap.add_argument('--no-posthoc', action='store_true',
                    help='skip the post-hoc amendments (A1-A7)')
    ap.add_argument('--dry-run', action='store_true',
                    help='machinery check: every outcome replaced by seeded '
                         'noise, so nothing about real returns is seen')
    ap.add_argument('--allow-edited-prereg', action='store_true',
                    help='run for real although the pre-registered region '
                         'no longer matches its pinned hash')
    a = ap.parse_args()
    t0 = time.time()
    sha = prereg_sha256()
    ok = sha == PREREG_SHA256
    print(f'pre-registered region sha256 {sha} -- '
          f'{"matches the pin" if ok else "DOES NOT MATCH the pin " + PREREG_SHA256}')
    if not ok and not a.dry_run and not a.allow_edited_prereg:
        raise SystemExit('the declarations were edited after the real run; '
                         'refusing a real run (--allow-edited-prereg to '
                         'override, and say so in the report)')
    print('run ledger: ' + log_run('dry-run' if a.dry_run else 'real', sha)
          .strip())
    from xsec_backtest import load_panel
    path = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    ev = pd.read_parquet(a.events)
    amap = ee.load_map()
    df = load_panel(path)
    nm = df[['ticker', 'month', 'dollar_vol']]
    tiers = ee.tier_of_tickers(nm)
    bars = df[df.ticker.isin(['SPY', 'QQQ', 'QQQQ'])].groupby('day').bars.max()
    days = np.sort(df.day.unique())
    years = range(int(days[0][:4]), int(days[-1][:4]) + 1)
    cal = ee.Calendar(days, ee.rule_half_days(years) |
                      set(bars[bars < 300].index))
    nn_all = name_nights(df, cal, amap, tiers)
    nn = nn_all[nn_all.mapped].reset_index(drop=True)
    if a.dry_run:
        print('\n*** DRY RUN: outcomes are N(0, 100) noise -- machinery '
              'only, no real result ***')
        nn['y'] = np.random.default_rng(0).normal(0, 100, len(nn))
        nn['y_ew'] = nn.y
    nn['name_id'] = pd.factorize(nn.ticker)[0]
    nn['cik'] = nn.cik.astype(np.int64)
    print(f'\nEVENT TEST on {len(nn_all):,} wide-panel name-nights with a valid '
          f'outcome (ETFs out); {len(nn):,} ({len(nn) / len(nn_all):.1%}) '
          f'have a mapped CIK and form the test universe; '
          f'{nn.ticker.nunique():,} names, {nn.cik.nunique():,} CIKs, '
          f'{nn.pos.nunique():,} nights; half days {int(cal.half.sum())}')
    print(f'outcome y = on_bps(name) - on_bps(QQQ/QQQQ), bps; decision '
          f'15:30 ET (12:30 half days), events need A < tau - 10 min')
    era = pd.cut(nn.year, [0, 2000, 2009, 2017, 2100],
                 labels=[e[0] for e in ERAS]).astype(str).values
    cik_nights = {int(c): np.unique(g.values) for c, g in
                  nn.groupby('cik').pos}

    results, flags = {}, {}
    print(f'\n{"":<6}{"hypothesis":<40}{"sign":>5} {HDR}   p(1s)  events '
          f'(exact%)')
    for hid, (label, cls, k, sign, _) in HYP.items():
        f, e = flags_for(hid, ev, cal, nn)
        flags[hid] = f
        r = fe_twoway(nn.y.values, f, nn.pos.values, nn.name_id.values)
        r['p'] = one_sided_p(r['t'], sign)
        r['events'] = len(e)
        r['exact'] = float(e.exact.mean()) if len(e) else np.nan
        results[hid] = r
        print(f'{hid:<6}{label:<40}{"+-0"[[1, -1, 0].index(sign)]:>5} '
              f'{fmt(r)}  {r["p"]:.4f}  {len(e):>6,} ({r["exact"]:.0%})')
    rej = holm({h: results[h]['p'] for h in HOLM})
    print(f'\nHolm over {", ".join(HOLM)} at family alpha {ALPHA}: ' +
          '  '.join(f'{h} {"REJECT" if rej[h] else "keep"}' for h in HOLM))
    p0 = results['P0']
    p0_pass = not (abs(p0['t']) >= 2)
    print(f'P0 placebo: t {p0["t"]:+.2f} -> {"PASS" if p0_pass else "FAIL"} '
          f'(must be |t| < 2)')
    print('\nsecond outcome, y relative to the equal-weight panel mean '
          '(raw means; the FE estimate is identical):')
    for hid in HYP:
        f = flags[hid].astype(bool)
        if f.any():
            print(f'  {hid:<6} flagged {nn.y_ew.values[f].mean():+8.2f}  '
                  f'unflagged {nn.y_ew.values[~f].mean():+7.2f}  diff '
                  f'{nn.y_ew.values[f].mean() - nn.y_ew.values[~f].mean():+8.2f}')

    print(f'\nBY ERA (night FE within the era) -- information; "small-n" = '
          f'< {SMALL_FLAGS} flagged name-nights, not to be quoted:')
    print(f'  {"":<6}{"era":<11} {HDR}')
    for hid in HYP:
        for lab, _, _ in ERAS:
            m = era == lab
            r = fe_twoway(nn.y.values[m], flags[hid][m], nn.pos.values[m],
                          nn.name_id.values[m])
            if r['n_flag']:
                print(f'  {hid:<6}{lab:<11} {fmt(r)}'
                      f'{small(r["n_flag"], SMALL_FLAGS)}')
    print(f'\nBY TIER -- information:')
    print(f'  {"":<6}{"tier":<11} {HDR}')
    for hid in HYP:
        for lab in ('CORE', 'never-core'):
            m = nn.tier.values == lab
            r = fe_twoway(nn.y.values[m], flags[hid][m], nn.pos.values[m],
                          nn.name_id.values[m])
            if r['n_flag']:
                print(f'  {hid:<6}{lab:<11} {fmt(r)}'
                      f'{small(r["n_flag"], SMALL_FLAGS)}')

    print(f'\nSHUFFLE CONTROL: events moved to a random other night of the '
          f'same name (seeds {SEEDS[0]}-{SEEDS[-1]}); must show |t| < 2 '
          f'(seed 1 binding)')
    shuf_ok = {}
    for hid in HYP:
        ts = []
        for s in SEEDS:
            f, _ = flags_for(hid, ev, cal, nn, shuffle_seed=s,
                             cik_nights=cik_nights)
            ts.append(fe_twoway(nn.y.values, f, nn.pos.values,
                                nn.name_id.values)['t'])
        shuf_ok[hid] = not (abs(ts[0]) >= 2)
        print(f'  {hid:<6} t by seed ' + ' '.join(f'{x:+5.2f}' for x in ts) +
              f'   -> {"pass" if shuf_ok[hid] else "FAIL"}')

    print(f'\nLEAK CANARIES (same events, leaky placements): an effect only '
          f'under a canary is lookahead [x = canary beta / true beta; '
          f'"LEAK INFLATION" = post-hoc rule A6, canary t beats the true t '
          f'by > 2 in the declared direction]')
    print(f'  {"":<6}{"placement":<12} {HDR}')
    leak = {h: False for h in HYP}
    inflation = {h: [] for h in HYP}
    for hid, (label, cls, k, sign, _) in HYP.items():
        if hid == 'H1':
            continue
        for pl in ('filingdate', 'z_as_utc'):
            f, _ = flags_for(hid, ev, cal, nn, placement=pl)
            r = fe_twoway(nn.y.values, f, nn.pos.values, nn.name_id.values)
            hit = sign != 0 and sign * r['t'] > 2 and \
                not (sign * results[hid]['t'] > 2)
            leak[hid] |= hit
            infl = leak_inflation(r['t'], results[hid]['t'], sign)
            if infl:
                inflation[hid].append(pl)
            ratio = r['beta'] / results[hid]['beta'] \
                if results[hid]['beta'] else np.nan
            print(f'  {hid:<6}{pl:<12} {fmt(r)}' +
                  ('   <- LEAK SIGNATURE' if hit else '') +
                  f'   x{ratio:+.1f}' +
                  ('  <- LEAK INFLATION (post-hoc A6)' if infl else ''))
    for pl in ('true', 'filingdate'):
        e = event_frame(ev, 'effect', cal, pl)
        keys = active_keys(e.cik.values, e.n.values, 3, len(cal.days))
        f = np.isin(nn_keys(nn.cik.values, nn.pos.values), keys)
        r = fe_twoway(nn.y.values, f, nn.pos.values, nn.name_id.values)
        print(f'  {"EFFECT":<6}{pl:<12} {fmt(r)}   (k=3; no declared sign)')

    print('\nVERDICTS (as declared):')
    supported = {}
    for hid in HOLM:
        supported[hid] = rej[hid] and p0_pass and shuf_ok[hid] and \
            not leak[hid]
        why = [] if supported[hid] else \
            [x for x, ok in (('Holm keeps', rej[hid]), ('P0 fails', p0_pass),
                             ('shuffle fails', shuf_ok[hid]),
                             ('leak canary', not leak[hid])) if not ok]
        print(f'  {hid}: {"SUPPORTED" if supported[hid] else "not supported"}'
              + (f' ({", ".join(why)})' if why else ''))

    rows = [dict(id=h, **{k: v for k, v in results[h].items()})
            for h in HYP]
    book = None
    if not a.no_book:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=MODEL)
        b, t = book_table(df, booster)
        qqq = qqq_series(df)
        if a.dry_run:
            rng = np.random.default_rng(1)
            b['on_bps'] = np.where(b.on_bps.notna(),
                                   rng.normal(0, 150, len(b)), np.nan)
            qqq = pd.Series(rng.normal(0, 100, len(qqq)), index=qqq.index)
        dpos = pd.Series(np.arange(len(cal.days)), index=cal.days)
        b['pos'] = dpos.reindex(b.day.values).values - 1
        b = b[b.pos >= 0].reset_index(drop=True)
        b['year'] = pd.Series(cal.days[b.pos.values]).str[:4].astype(int) \
            .values
        b = b.merge(amap.rename(columns={'yr': 'year'}), on=['ticker', 'year'],
                    how='left')
        mapped = b.cik.notna().values
        bk = nn_keys(b.cik.fillna(0).astype(np.int64).values, b.pos.values)
        base = book_eval(b, np.zeros(len(b), bool), qqq)
        print(f'\nTHE BOOK: frozen-model Q5 of the replicated top-150, '
              f'{t.tmonth.nunique()} trade months {t.tmonth.min()}..'
              f'{t.tmonth.max()}, {len(base):,} nights, '
              f'{base.n.mean():.1f} names/night; {mapped.mean():.1%} of '
              f'basket name-nights have a mapped CIK')
        print(f'  as-is tilt {base.tilt.mean():+.2f} bps/night (t '
              f'{t_of(base.tilt):+.2f}, sd {base.tilt.std():.1f}) -- in-sample '
              f'level, read the change')
        print(f'  {"veto":<22}{"nights hit":>11}{"names skip":>11}'
              f'{"chg/night":>10}{"t":>7}{"chg|hit":>9}{"sd after":>9}'
              f'{"cash chg":>9}{"t*":>7}{"level*":>8}')
        books, books_keys, books_fl = {}, {}, {}
        combo = ('negative vetoes together', None, 0, -1, False)
        for hid, (label, cls, k, sign, _) in list(HYP.items()) + \
                [('H2+H3+H4', combo)]:
            if hid == 'H2+H3+H4':
                keys = np.concatenate([books_keys[h] for h in
                                       ('H2', 'H3', 'H4')])
            else:
                e = event_frame(ev, cls, cal)
                if hid == 'H1':
                    p = h1_flags(e, cal)
                    keys = active_keys(p.cik.values, p.D.values, k,
                                       len(cal.days))
                else:
                    keys = active_keys(e.cik.values, e.n.values, k,
                                       len(cal.days))
                books_keys[hid] = keys
            fl = np.isin(bk, keys) & mapped
            books_fl[hid] = fl
            r = book_eval(b, fl, qqq)
            chg = r.cf - r.tilt
            cchg = r.cash - r.tilt
            hit = r.skipped > 0
            books[hid] = dict(chg=chg.mean(), t=t_of(chg),
                              nights=int(hit.sum()),
                              skipped=int(r.skipped.sum()),
                              chg_hit=chg[hit].mean() if hit.any() else np.nan,
                              sd_after=r.cf.std(), cash=cchg.mean(),
                              t_cash=t_of(cchg))
            level = float(((r.skipped / r.n) * r.cf).where(hit, 0).mean())
            x = books[hid]
            print(f'  {hid + " " + label[:16]:<22}{x["nights"]:>11,}'
                  f'{x["skipped"]:>11,}{x["chg"]:>+10.3f}{x["t"]:>+7.2f}'
                  f'{x["chg_hit"]:>+9.2f}{x["sd_after"]:>9.1f}'
                  f'{x["cash"]:>+9.3f}{x["t_cash"]:>+7.2f}{level:>+8.3f}'
                  f'{small(x["nights"], SMALL_NIGHTS)}')
        print('  * cash chg = chg/night - level, level = mean(skip share x '
              'kept in-sample tilt): the cash t mostly measures the in-sample '
              'level, not the veto -- read chg/night (post-hoc note A7)')
        print('\nBOOK VERDICTS (as declared: propose only if SUPPORTED and '
              'the change has t > 2 in the improving direction):')
        for hid in HOLM:
            x = books[hid]
            prop = supported[hid] and x['t'] > 2
            print(f'  {hid}: {"PROPOSE (a dated re-freeze, not done here)" if prop else "do not propose"}'
                  f'  (change {x["chg"]:+.3f} bps/night, t {x["t"]:+.2f})')
        for r_ in rows:
            r_.update({f'book_{k}': v for k, v in books.get(r_['id'],
                                                           {}).items()})
        book = dict(b=b, qqq=qqq, keys=books_keys, fl=books_fl,
                    booster=booster)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nresults -> {OUT}   ({time.time() - t0:.0f} s)')
    if not a.no_posthoc:
        decl = dict(results=results, flags=flags, supported=supported,
                    rej=rej, shuf_ok=shuf_ok, leak=leak, inflation=inflation)
        posthoc(a, path, df, ev, amap, tiers, cal, nn_all, nn, decl, book)
        print(f'\ntotal {time.time() - t0:.0f} s')


def _row(r):
    return f'{r["beta"]:>+7.2f} {r["t"]:>+6.2f}'


def posthoc(a, path, df, ev, amap, tiers, cal, nn_all, nn, decl, book):
    """The A1-A7 amendments (see the block above PREREG_SHA256). NOT
    EVIDENCE: written after the declared results were seen."""
    from xsec_backtest import load_panel, BACKSTOP
    print('\n' + '=' * 78)
    print('POST-HOC AMENDMENTS A1-A7 -- written after the real run, in answer '
          'to a review.\nNOT EVIDENCE: chosen after the declared results were '
          'seen. The declared output above\nis unchanged and no declared '
          'verdict reads anything below.')
    print('=' * 78)
    t0 = time.time()
    hyps = list(HYP) + list(HYP_PH)
    sgn = {h: (HYP[h] if h in HYP else HYP_PH[h])[3] for h in hyps}
    n_x = int((ev.event == 'supply_x').sum())
    print(f'A4: {n_x:,} supply_x rows in events.parquet (filings the declared '
          f'year-total shelf rule excluded but whose count known at acceptance '
          f'is <= {ee.STRUCTURED_MAX}); '
          f'{int(((ev.event == "supply_x") & (ev.equity == 1.0)).sum()):,} '
          f'classified as common equity')
    rng = np.random.default_rng(7)
    noise = (lambda d: d.assign(y=rng.normal(0, 100, len(d)))) \
        if a.dry_run else (lambda d: d)
    qd = qqq_series(df)

    # ---- universes ------------------------------------------------------
    nn = nn.copy()
    nn['lag'] = lagged_member(nn, df, cal)
    dk = load_panel(path, keep_backstop=True)
    same = len(dk) == len(df) and (dk.ticker.values == df.ticker.values).all() \
        and (dk.day.values == df.day.values).all()
    if not same:
        raise SystemExit('keep_backstop panel rows do not align with the '
                         'default panel')
    on_c = clip_outcome(dk.on_bps.values, BACKSTOP * 1e4)
    restored = int((np.isnan(df.on_bps.values) & np.isfinite(on_c)).sum())
    split_err = int((np.abs(dk.on_bps.values) > SPLIT_ERR_BPS).sum())
    del dk
    dfc = df[['ticker', 'day', 'month']].assign(on_bps=on_c)
    nnc = name_nights(dfc, cal, amap, tiers)
    nnc['y'] = nnc.on_bps - qd.reindex(nnc.day.values).values
    nnc = nnc[nnc.mapped & nnc.y.notna()].reset_index(drop=True)
    nnc['cik'] = nnc.cik.astype(np.int64)
    nnc['lag'] = lagged_member(nnc, df, cal)
    nnc = noise(nnc)
    nall = nnc[nnc.lag].reset_index(drop=True)
    nlag = nn[nn.lag].reset_index(drop=True)
    for x in (nnc, nall, nlag):
        x['name_id'] = pd.factorize(x.ticker)[0]
    # A5: the ticker-evidence map, restricted to CIKs whose filings were read
    emp = ee.EVIDENCE_MAP
    emap = pd.read_parquet(emp) if os.path.exists(emp) else \
        ee.build_evidence_map(df[['ticker', 'month', 'dollar_vol']])
    evc = set(ev.cik.unique())
    em = emap[emap.cik.isin(evc)][['ticker', 'yr', 'cik']] \
        .rename(columns={'yr': 'year', 'cik': 'cik_e'})
    nne = nn_all.drop(columns=['cik']).merge(em, on=['ticker', 'year'],
                                            how='inner')
    nne = nne.rename(columns={'cik_e': 'cik'}).astype({'cik': np.int64})
    nne = noise(nne.reset_index(drop=True))
    nne['name_id'] = pd.factorize(nne.ticker)[0]
    both = nne.merge(nn[['ticker', 'pos', 'cik']], on=['ticker', 'pos'],
                     how='left', suffixes=('', '_r'))
    print(f'\nuniverses (name-nights): declared {len(nn):,} | A1 lagged '
          f'(in the month(D)-1 file) {len(nlag):,} ({len(nlag) / len(nn):.1%}'
          f') | A3 backstop restored {len(nnc):,} (+{len(nnc) - len(nn):,}; '
          f'{restored:,} panel nights restored, {split_err:,} split errors '
          f'> {SPLIT_ERR_BPS:,.0f} bps dropped, clip +-{BACKSTOP * 1e4:,.0f} '
          f'bps) | ALL = A1+A3 {len(nall):,} | A5 evidence map {len(nne):,} '
          f'({both.cik_r.notna().mean():.1%} also in the research map, '
          f'{(both.cik == both.cik_r).mean():.1%} same CIK; the evidence map '
          f'gives {len(emap):,} ticker-years, {emap.cik.isin(evc).mean():.1%}'
          f' with filings read)')
    print(f'  A1: share of flagged vs unflagged declared name-nights whose name '
          f'was NOT in the month(D)-1 file (entered on month-local volume):')
    for h in HYP:
        f = decl['flags'][h].astype(bool)
        if f.any():
            print(f'    {h:<6} flagged {(~nn.lag.values[f]).mean():6.2%}   '
                  f'unflagged {(~nn.lag.values[~f]).mean():6.2%}')

    tw_d = TwoWay(nn.pos.values, nn.name_id.values)
    tw_a = TwoWay(nall.pos.values, nall.name_id.values)
    yt_d, yt_a = tw_d(nn.y.values), tw_a(nall.y.values)
    cn_d = {int(c): np.unique(g.values) for c, g in nn.groupby('cik').pos}
    cn_a = {int(c): np.unique(g.values) for c, g in nall.groupby('cik').pos}

    def est(d, f):
        return fe_twoway(d.y.values, f, d.pos.values, d.name_id.values)

    res = {}
    print(f'\nPOST-HOC PANEL (beta_FE bps, t two-way clustered):')
    print(f'  {"":<6}{"declared":>15}{"A2 name+night":>15}{"A1 lagged":>15}'
          f'{"A3 restored":>15}{"A5 evid. map":>15}{"ALL=A1+A2+A3":>23}'
          f'{"flagged":>9}')
    for h in hyps:
        fd = flags_for(h, ev, cal, nn)[0]
        r_dec = est(nn, fd)
        r_a2 = fe_name_night(nn.y.values, fd, tw_d, yt_d)
        r_a1 = est(nlag, flags_for(h, ev, cal, nlag)[0])
        r_a3 = est(nnc, flags_for(h, ev, cal, nnc)[0])
        r_a5 = est(nne, flags_for(h, ev, cal, nne)[0])
        fa = flags_for(h, ev, cal, nall)[0]
        r_all = fe_name_night(nall.y.values, fa, tw_a, yt_a)
        r_all['p'] = one_sided_p(r_all['t'], sgn[h])
        res[h] = dict(declared=r_dec, A2=r_a2, A1=r_a1, A3=r_a3, A5=r_a5,
                      ALL=r_all)
        print(f'  {h:<6}{_row(r_dec):>15}{_row(r_a2):>15}{_row(r_a1):>15}'
              f'{_row(r_a3):>15}{_row(r_a5):>15}{_row(r_all):>15}'
              f'  p {r_all["p"]:.4f}{r_all["n_flag"]:>9,}'
              f'{small(r_all["n_flag"], SMALL_FLAGS)}')
    rej = holm({h: res[h]['ALL']['p'] for h in HOLM_PH})
    p0 = res['P0']['ALL']
    p0_pass = not (abs(p0['t']) >= 2)
    print(f'  Holm (ALL) over {", ".join(HOLM_PH)}: ' +
          '  '.join(f'{h} {"REJECT" if rej[h] else "keep"}' for h in HOLM_PH)
          + f'   | P0 (ALL) t {p0["t"]:+.2f} -> '
          f'{"pass" if p0_pass else "FAIL"} (P0 cannot see name-composition '
          f'bias: votes hit almost every firm)')
    print('  name effect of flagged minus unflagged rows (night-demeaned name '
          'means; what A2 removes):')
    yn = nn.y.values - (np.bincount(tw_d.ni, nn.y.values) / tw_d.cn)[tw_d.ni]
    ne = (np.bincount(tw_d.mi, yn) / np.bincount(tw_d.mi))[tw_d.mi]
    print('    ' + '  '.join(
        f'{h} {ne[decl["flags"][h] > 0].mean() - ne[decl["flags"][h] == 0].mean():+.2f}'
        for h in HYP if decl['flags'][h].any()))

    print(f'\nA2 SHUFFLE under name+night FE (seeds {SEEDS[0]}-{SEEDS[-1]}; '
          f'seed 1 binding), declared universe | ALL universe:')
    shuf = {}
    for h in hyps:
        td, ta = [], []
        for s in SEEDS:
            f, _ = flags_for(h, ev, cal, nn, shuffle_seed=s, cik_nights=cn_d)
            td.append(fe_name_night(nn.y.values, f, tw_d, yt_d)['t'])
            f, _ = flags_for(h, ev, cal, nall, shuffle_seed=s,
                             cik_nights=cn_a)
            ta.append(fe_name_night(nall.y.values, f, tw_a, yt_a)['t'])
        shuf[h] = not (abs(ta[0]) >= 2)
        print(f'  {h:<6}' + ' '.join(f'{x:+5.2f}' for x in td) + '  | ' +
              ' '.join(f'{x:+5.2f}' for x in ta) +
              f'   -> {"pass" if shuf[h] else "FAIL"}')

    print('\nA4/A6 canaries for H4a (declared estimator and universe):')
    t_true = res['H4a']['declared']['t']
    infl = dict(decl['inflation'])
    leak = dict(decl['leak'])
    infl['H4a'], leak['H4a'] = [], False
    for pl in ('filingdate', 'z_as_utc'):
        r = est(nn, flags_for('H4a', ev, cal, nn, placement=pl)[0])
        sig = sgn['H4a'] * r['t'] > 2 and not (sgn['H4a'] * t_true > 2)
        leak['H4a'] |= sig
        if leak_inflation(r['t'], t_true, sgn['H4a']):
            infl['H4a'].append(pl)
        print(f'  H4a   {pl:<12} {fmt(r)}' +
              ('   <- LEAK SIGNATURE' if sig else '') +
              ('   <- LEAK INFLATION' if pl in infl['H4a'] else ''))
    print('A6 leak inflation (canary t beats the true t by > 2 in the declared '
          'direction): ' + (', '.join(f'{h} ({"/".join(v)})' for h, v in
                                      infl.items() if v) or 'none'))

    # A5 information: by the CIK's later survival (uses the future, as the
    # map's candidates did)
    ent = pd.read_parquet(os.path.join(ee.EDGAR, 'entities.parquet'),
                          columns=['cik', 'last'])
    lastf = ent.set_index('cik')['last']
    ly = pd.to_numeric(nn.cik.map(lastf).astype(object).str[:4],
                       errors='coerce').fillna(9999).values
    died = ly <= nn.year.values + 3
    print(f'\nA5 information -- by whether the mapped CIK stops filing within '
          f'3 years of night D (dying CIKs are {died.mean():.1%} of the '
          f'declared universe; the map under-covers them, see the coverage '
          f'table):')
    for h in ('H2', 'H3', 'H2old', 'H4a'):
        f = flags_for(h, ev, cal, nn)[0]
        rd = fe_twoway(nn.y.values[died], f[died], nn.pos.values[died],
                       nn.name_id.values[died])
        rs = fe_twoway(nn.y.values[~died], f[~died], nn.pos.values[~died],
                       nn.name_id.values[~died])
        print(f'  {h:<6} dying {rd["beta"]:+8.2f} (t {rd["t"]:+5.2f}, '
              f'{rd["n_flag"]:,} flagged{small(rd["n_flag"], SMALL_FLAGS)})'
              f'   surviving {rs["beta"]:+7.2f} (t {rs["t"]:+5.2f}, '
              f'{rs["n_flag"]:,})')

    print('\nPOST-HOC VERDICTS (NOT EVIDENCE): Holm on ALL in the declared '
          'direction, P0 (ALL) passes, the name-FE shuffle (ALL, seed 1) '
          'passes, no leak signature or inflation, and the evidence-map '
          'canary (A5) has the declared sign:')
    sup = {}
    for h in HOLM_PH:
        base_h = 'H4' if h == 'H4a' else h
        m_ok = np.sign(res[h]['A5']['beta']) == sgn[h]
        checks = (('Holm keeps', rej[h]), ('P0 fails', p0_pass),
                  ('shuffle fails', shuf[h]),
                  ('leak', not leak.get(h, leak.get(base_h, False))),
                  ('inflation', not infl.get(h)),
                  ('map canary sign', m_ok))
        sup[h] = all(ok for _, ok in checks)
        why = [x for x, ok in checks if not ok]
        r = res[h]['ALL']
        print(f'  {h:<4} ALL {r["beta"]:+6.2f} bps (t {r["t"]:+5.2f}): '
              f'{"supported (post-hoc)" if sup[h] else "not supported"}'
              + (f' ({", ".join(why)})' if why else ''))

    rows = []
    for h in hyps:
        for v, r in res[h].items():
            rows.append(dict(id=h, variant=v, **{k: r.get(k) for k in
                        ('n', 'n_flag', 'beta', 'se', 't', 't_night', 'mean_f',
                         'mean_u', 'diff', 'p')}))
    if book is None:
        pd.DataFrame(rows).to_csv(OUT_PH, index=False)
        print(f'\npost-hoc results -> {OUT_PH}   ({time.time() - t0:.0f} s)')
        return

    # ---- the book -------------------------------------------------------
    b, qqq, booster = book['b'], book['qqq'], book['booster']
    print(f'\nA7 THE BOOK, declared replica -- concentration of each veto\'s '
          f'change (re-spread reading). t_name: nights clustered by the name '
          f'that drove them. ex-top: the largest contributor\'s flags '
          f'removed.')
    hdr = (f'  {"veto":<10}{"nights":>7}{"chg/night":>10}{"t":>7}'
           f'{"t_name":>8}{"top name":>10}{"share":>7}{"chg ex-top":>11}'
           f'{"t":>7}{"level":>8}')
    print(hdr)
    bd = {}
    for h in list(HYP) + ['H2+H3+H4']:
        x = book_diag(b, book['fl'][h], qqq)
        bd[h] = x
        print(f'  {h:<10}{x["nights"]:>7,}{x["chg"]:>+10.3f}{x["t"]:>+7.2f}'
              f'{x["t_name"]:>+8.2f}{str(x["top"]):>10}{x["share"]:>7.0%}'
              f'{x["chg_ex"]:>+11.3f}{x["t_ex"]:>+7.2f}{x["level"]:>+8.3f}'
              f'{small(x["nights"], SMALL_NIGHTS)}')
    # the lagged replica, the object xsec_live trades
    df['on_c'] = on_c                  # outcome column only, never a feature
    bl = book_table_lagged(df, booster, cal)
    df.drop(columns='on_c', inplace=True)
    if a.dry_run:
        r_ = np.random.default_rng(3)
        bl['on_bps'] = np.where(bl.on_bps.notna(), r_.normal(0, 150, len(bl)),
                                np.nan)
        bl['on_c'] = bl.on_bps
    bl = book_prepare(bl, cal, amap)
    nights = bl.day.nunique()
    q5n = bl.groupby('tmonth').ticker.nunique()
    keys = dict(book['keys'])
    e4 = event_frame(ev, HYP_PH['H4a'][1], cal)
    keys['H4a'] = active_keys(e4.cik.values, e4.n.values, 3, len(cal.days))
    keys['H2+H3+H4'] = np.concatenate([keys[h] for h in ('H2', 'H3', 'H4')])
    keys['H2+H3+H4a'] = np.concatenate([keys[h] for h in ('H2', 'H3', 'H4a')])
    base = book_eval(bl, np.zeros(len(bl), bool), qqq)
    basec = book_eval(bl.assign(on_bps=bl.on_c), np.zeros(len(bl), bool), qqq)
    print(f'\nA7 THE BOOK as xsec_live trades it (T-1 top-150 list, no month-T '
          f'membership, night D with the basket of D\'s month): '
          f'{bl.tmonth.nunique()} months, {nights:,} nights, '
          f'{q5n.mean():.1f} Q5 names/month, {base.n.mean():.1f} priced '
          f'names/night, {bl.mapped.mean():.1%} mapped; as-is tilt '
          f'{base.tilt.mean():+.2f} (in-sample level), with A3 outcomes '
          f'{basec.tilt.mean():+.2f}')
    print(hdr + f'{"A3 chg":>9}{"t":>7}')
    bk = {}
    for h in list(HYP) + ['H4a', 'H2+H3+H4', 'H2+H3+H4a']:
        fl = np.isin(bl.bk.values, keys[h]) & bl.mapped.values
        x = book_diag(bl, fl, qqq)
        rc = book_eval(bl.assign(on_bps=bl.on_c), fl, qqq)
        cc = rc.cf - rc.tilt
        x.update(chg_c=cc.mean(), t_c=t_of(cc))
        bk[h] = x
        print(f'  {h:<10}{x["nights"]:>7,}{x["chg"]:>+10.3f}{x["t"]:>+7.2f}'
              f'{x["t_name"]:>+8.2f}{str(x["top"]):>10}{x["share"]:>7.0%}'
              f'{x["chg_ex"]:>+11.3f}{x["t_ex"]:>+7.2f}{x["level"]:>+8.3f}'
              f'{x["chg_c"]:>+9.3f}{x["t_c"]:>+7.2f}'
              f'{small(x["nights"], SMALL_NIGHTS)}')
    print('\nPOST-HOC BOOK VERDICTS (NOT EVIDENCE): propose only if supported '
          '(post-hoc) AND on the lagged replica the change has t > 2, '
          't_name > 2 and t > 2 without its largest name:')
    for h in HOLM_PH:
        x = bk[h]
        ok = sup[h] and x['t'] > 2 and x['t_name'] > 2 and x['t_ex'] > 2
        print(f'  {h:<4} {"would propose" if ok else "do not propose"}  '
              f'(chg {x["chg"]:+.3f} bps/night, t {x["t"]:+.2f}, t_name '
              f'{x["t_name"]:+.2f}, ex-{x["top"]} t {x["t_ex"]:+.2f})')
    for h, x in bk.items():
        rows.append(dict(id=h, variant='book_lagged', **{
            k: x.get(k) for k in ('nights', 'chg', 't', 't_name', 'top',
                                  'share', 'chg_ex', 't_ex', 'level', 'chg_c',
                                  't_c')}))
    for h, x in bd.items():
        rows.append(dict(id=h, variant='book_declared', **{
            k: x.get(k) for k in ('nights', 'chg', 't', 't_name', 'top',
                                  'share', 'chg_ex', 't_ex', 'level')}))
    pd.DataFrame(rows).to_csv(OUT_PH, index=False)
    print(f'\npost-hoc results -> {OUT_PH}   ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
