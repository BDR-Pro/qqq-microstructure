# Part of qqq-microstructure.
#
# "Would leverage make the traded book perform better, in which world, at
#  which capital, and at what tail cost?"
#
# The book is the one orders.py trades: long the frozen model's Q5 basket
# overnight (buy MOC, sell MOO), unhedged. RESULTS 19 rejected leverage on the
# v2 book at a flat 5%/yr. This asks again for THIS book, at the owner's
# capital, with IBKR's actual financing, commissions and margin rules, in the
# same five worlds and the same one-year fan as roi_year.py.
#
# RESEARCH ONLY. The project rule (RESULTS 19) is no live money until the
# pre-registered forward test passes. A GO verdict rules out "no edge"; it
# does not establish the as-backtested world.
#
# DECISION RULE -- declared here, before any run:
#   Leverage L > 1 is worth considering only if, at the live-volatility
#   stress, it RAISES the median one-year return of the half-the-edge world
#   AND does NOT LOWER that world's p5, AND its P(margin call) is below 1% in
#   every world at both volatilities. Otherwise 1x stands.
#   Why these three: half the edge is the world a GO verdict leaves most
#   plausible; the live stress is the volatility the basket is actually
#   showing; and a liquidated account cannot wait for its edge.
#
# Method, and why each choice:
#   - Worlds, bootstrap, horizon and live-volatility stress are roi_year's,
#     imported rather than copied: roi_year.worlds / scaled / load / prices,
#     and mc_risk.paths with SEED 7 and 21-day blocks. The bootstrap draws
#     night INDICES, so each resampled night carries its own calendar gap and
#     a weekend night brings its weekend's interest with it.
#   - Account: FIXED notional, as in roi_year (orders.py --capital). The book
#     is sized once at L x capital through paper_bench.sizing (whole shares,
#     the order code's own sizing), and P&L adds rather than compounds. Costs
#     are paper_bench.night_cost on the book actually SENT that night: when
#     the Reg T cap below shrinks the book, its commissions are re-priced from
#     a curve of night_cost at smaller sizings of the same basket (per-order
#     minimums kept), not charged at the full book's rate. The per-order
#     minimum is spread over more notional as L rises -- the one channel
#     where leverage helps a small account, shown in the account table.
#   - E[net] in the one-year tables is each fan's OWN mean night (mean
#     one-year P&L / capital / 252). A start-of-year closed form (debit fixed,
#     no cut-back) overstated the levered rows at the live stress by 1-1.5
#     bps/night; it is printed once beside the decision world for comparison.
#   - Financing: IBKR Pro's blended tiers on the ACTUAL debit, i.e. exposure
#     minus equity at the close. The debit grows on losing paths and shrinks
#     on winning ones, and a 1x book borrows too once it is down by more than
#     its idle cash. Tiers: BM+1.5% on the first $100k, BM+1.0% to $1M,
#     BM+0.75% to $50M, BM+0.5% beyond (IBKR margin-rates page, 2026-10-13).
#     BM 3.88% = EFFR on 2026-10-09 (NY Fed), so 5.38% on a small debit; the
#     Fed hiked in 2026-09, so no cuts are assumed. --rate R charges a flat R%
#     instead (RESULTS 19 used 5).
#   - Day count: IBKR charges Actual/360 on ending SETTLED cash. Under T+1 the
#     morning MOO sale and the evening MOC buy settle on the same day, so while
#     the book trades nightly the debit never closes, and interest accrues on
#     every calendar day even though the account is flat all day. Each night
#     is charged the calendar days from the previous trading day to its open
#     day (Friday -> Monday = 3; holidays add), from the real calendar of the
#     series being resampled. Settlement-date accounting would hang the
#     weekend on Thursday's buy instead. Totals over any run of nights are
#     identical; this way the weekend's interest rides with the weekend's
#     (riskier) return. A gap over 5 days (Sandy's 2012 closure is the
#     longest in 2003-26) is a data hole and is capped.
#   - Reg T: overnight exposure at most 2x equity (50% initial; IBKR's
#     end-of-day SMA check). The account is flat when the MOC order goes in,
#     so IBKR's order check binds every night: exposure = min(the sized
#     notional, 2 x equity). After a loss a 2x account is therefore cut back
#     rather than rolled forward on SMA netting. The netting exists, but an
#     account that leans on it is a different, riskier account.
#   - Maintenance, checked at the open: if equity at the open (after the
#     night's move, commissions and interest) is below the house maintenance
#     rate times the position's value, the account is liquidated. IBKR does
#     not call; it liquidates in real time, after the auction, with market
#     orders. Treated conservatively: an extra 100 bps of the position's value
#     lost to the liquidator, and the path stops trading for the rest of the
#     year. Rate: 25% with 9+ names held (IBKR's WhatIf on the October basket,
#     2026-10-13), and its concentration surcharges for fewer (28.1% at 8,
#     45.1% at 5, 75.3% at 3; an unmeasured count takes the next lower
#     measured one; 1-2 names 100%). Below $2,000 equity no position opens.
#   - Kelly: a COMPOUNDING variant (exposure reset to f x equity every close),
#     used only in the Kelly section. f* maximises mean log growth over the
#     bootstrapped nights with commissions and financing included, rather than
#     mu/sigma^2. Two f*: the textbook one, with no margin rules (a reference
#     only: anything whose exposure is over the account's ceiling is marked
#     NOT ATTAINABLE and its growth is not printed), and the FEASIBLE one,
#     searched only where exposure <= 2.0x equity (Reg T) and with the
#     maintenance liquidation applied inside the growth sum, as simulate()
#     applies it. With $110k+ a portfolio-margin f* is added (exposure <=
#     1/25.9% = 3.86x, 23.5% maintenance). The closed form (mu - c)/sigma^2 is
#     printed beside them as a check; it now includes the fixed per-order
#     charge as a corner (0 when no exposure out-earns it). Commissions and
#     interest are held at their starting-capital fractions there, the
#     textbook approximation.
#   - Benchmarks, same money: QQQ held (roi_year's row); a QLD-like 2x daily
#     QQQ, held: 2 x the daily return less EFFR + 1.75%/yr per calendar day.
#     That is QLD's measured all-in drag against 2x QQQ total return,
#     2025-10..2026-10, expense ratio included. QQQ overnight-only at 1x and
#     2x runs through the same account model as the book (sized,
#     commissioned, financed, margined). The QQQ rows are never
#     volatility-stressed (roi_year's convention).
#   - Two data modes. --real (authoritative; the owner's machine):
#     roi_year.load(), the walk-forward Q5 nights and the panel's QQQ/SPY
#     factors up to the frozen cutoff. --proxy (any machine): a FACTOR PROXY
#     built from REAL QQQ and SPY overnight returns (Yahoo, split-adjusted,
#     not dividend-adjusted), 2003-01-02 .. the cutoff month:
#        q5 = a + 0.69 QQQ_on + 0.73 SPY_on + e
#     The betas are RESULTS 21's ON row. e is iid Student-t(4), scaled to the
#     residual sd that row implies: alpha 4.13 at residual Sharpe 1.26 gives
#     4.13 x sqrt(252) / 1.26 = 52.0 bps. a is set so the gross mean is
#     RESULTS 26's +10.50 (21's +8.50 plus its assumed 2 bps cost). The proxy
#     carries the index's real crashes and volatility clustering, but not the
#     basket's own: its residual has no clustering and no sector unwinds, so it
#     understates tails the real series has. A constant alpha would make the
#     proxy's "recent era" the BEST world (it inherits only the index's strong
#     2016+ nights), the opposite of the real data: RESULTS 26's real 2016+
#     row tied half the edge on median at $25k and $100k, with a wider left
#     tail. So the proxy's alpha steps at 2016: the 2016+ nights' mean is
#     pinned to the half-the-edge world's, the earlier nights' raised to keep
#     the overall gross at +10.50. Only that mean is calibrated; the row's
#     spread is whatever the index's 2016+ nights and the iid residual give
#     (the real row's 1x p5 was -21% vs -17% for half the edge at $25k).
#     Every recent-era row a proxy run prints is tagged '*' with that note,
#     and every table title carries the PROXY banner.
#   - The live nights (reports/xsec_paper_daily.csv) are replayed at each L as
#     a realised check: a few dozen nights, one path, a standard error of
#     ~35 bps/night on the mean. It checks plumbing and sign; it is not
#     evidence.
#
# Known and NOT modelled:
#   - IBKR's daily Exposure Fee on concentrated, stress-heavy accounts.
#   - House margin raised, or Soft Edge Margin withdrawn, without notice.
#   - Names over 1% of shares outstanding losing marginability.
#   - A cross printing above the 15:45 sizing price, which takes "2.0x" over
#     the SMA line (operate near 1.9x).
#   - Taxes: short-term gains, wash sales, margin interest as investment
#     interest.
#   - Dividends: every row omits them, so the QLD-like row is conservative by
#     ~2x the yield.
#   - Monthly posting of interest: it is charged into equity daily, slightly
#     conservative.
#   - Portfolio margin (2.5x, 3x) is not simulated in the fans. It needs $110k
#     NLV to open; below $100k it blocks margin-increasing trades, and every
#     nightly re-entry is one; its concentration add-ons put this basket at
#     25.9% initial. It appears only as a Kelly line at $110k+.
#
# Modelled since the first review (2026-10-13):
#   - Credit interest on idle cash (E - exposure, when positive): IBKR Pro
#     pays BM - 0.5% on the balance above $10k, scaled by min(1, NAV/$100k),
#     Actual/360 on calendar days. A fixed-notional 1x book keeps its gains as
#     cash, so at $100k it holds >$10k idle on about half of all nights; the
#     first draft said idle cash "stays below $10k", which was false there,
#     and leaving the credit out biased the comparison slightly against 1x.
#     The fans' "fin $/yr" is interest paid NET of this credit.
#
#   python src/leverage.py --capital 25000 --plan tiered --proxy
#   python src/leverage.py --capital 100000 --plan tiered          # --real
#   options: --rate R (flat %/yr instead of the tiers)  --bm 3.88
#            --paths N  --prices CSV

import os, sys, json, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_risk import paths, BLOCK, SEED
import roi_year as ry
import paper_bench as pb

H = ry.H
GRID = (1.0, 1.25, 1.5, 2.0)
BM = 3.88
TIERS = ((100e3, 1.50), (1e6, 1.00), (50e6, 0.75), (np.inf, 0.50))
DAYCOUNT = 360
REGT = 2.0
MIN_EQUITY = 2000.0
HOUSE = ((9, 0.25), (8, 0.281), (5, 0.451), (3, 0.753), (0, 1.0))
ETF_MAINT = 0.25            # QQQ, Reg T 50/25 (IBKR WhatIf 2026-10-13)
LIQ_SLIP = 100.0            # bps of the position's value, on a liquidation
MAX_GAP = 5
QLD_DRAG = 1.75             # %/yr over EFFR, QLD vs 2x QQQ TR, measured
CALL_MAX = 0.01
DECIDE = 'half the edge'
ROBUST = ('as backtested', 'half the edge', 'recent era (2016+)',
          'no edge (beta only)')
KELLY_PATHS = 2000
FMAX = 6.0
PM_MIN = 110e3              # IBKR: NLV to open portfolio margin
PM_INIT = 0.259             # this basket's PM initial (IBKR WhatIf 2026-10-13)
PM_MAINT = 0.235            # ... and maintenance
CREDIT_SPREAD = 0.50        # IBKR Pro pays BM - 0.5% on idle USD
CREDIT_FREE = 10e3          # ... nothing on the first $10k
CREDIT_NAV = 100e3          # ... and NAV/100k of the rate below $100k NAV
CURVE_PTS = 48              # sizings in each account's commission curve
RECENT = 'recent era (2016+)'
# the proxy's 2016+ nights keep this fraction of the selection edge: RESULTS
# 26's real 2016+ row tied the half-the-edge world on median at both capitals
PROXY = dict(b_qqq=0.69, b_spy=0.73, alpha=4.13, rsharpe=1.26, gross=10.50,
             df=4, start='20030102', recent_edge=0.5)
PROXY_TAG = '*'
PROXY_RECENT_NOTE = (
    "  * proxy: only the 2016+ world's MEAN is calibrated -- pinned to the "
    "half-the-edge world's, where\n    RESULTS 26's real 2016+ row landed "
    "(median tie at $25k and $100k). Its spread is not calibrated:\n    it "
    "is the index's 2016+ nights plus the iid residual (the real row's 1x p5 "
    "was -21% vs -17% for\n    half the edge at $25k, -17% vs -13% at $100k)."
    " Being half the edge by construction, it is not\n    independent "
    "evidence about the edge.")
BANNER = '*** PROXY: factor model, not the real book ***'
BANNER_LIVE = ('*** PROXY run -- but this table replays the REAL live nights, '
               'not the factor model ***')
PM_LABEL = ('2.5x and 3.0x: not allowed under Reg T; portfolio margin only, '
            'and concentration add-ons apply (this basket: 25.9% initial)')


def interest(debit, bm=BM, flat=None):
    """$ per year on a debit. Default: IBKR Pro's blended tiers, each slice of
    the debit at its own tier's rate. flat: one rate (%/yr) on all of it."""
    d = np.maximum(np.asarray(debit, float), 0.0)
    if flat is not None:
        return d * flat / 100
    out = np.zeros_like(d)
    lo = 0.0
    for hi, spread in TIERS:
        out = out + np.clip(d - lo, 0.0, hi - lo) * (bm + spread) / 100
        lo = hi
    return out


def credit(cash, equity, bm=BM):
    """$ per year IBKR Pro pays on idle cash: BM - 0.5% on the balance above
    $10k, times min(1, NAV / $100k). Negative cash is a debit, not this."""
    c = np.maximum(np.asarray(cash, float) - CREDIT_FREE, 0.0)
    scale = np.clip(np.asarray(equity, float) / CREDIT_NAV, 0.0, 1.0)
    return c * max(bm - CREDIT_SPREAD, 0.0) / 100 * scale


def day_gaps(days, prev=None):
    """Calendar days each night is financed, for nights dated by their OPEN
    day (the panel's and the proxy's convention): days since the previous
    trading day. prev: the trading day before days[0] if known, else the
    previous weekday."""
    t = pd.to_datetime(pd.Index(days, dtype=str), format='%Y%m%d')
    p0 = pd.Timestamp(prev) if prev is not None else t[0] - pd.offsets.BDay(1)
    d = pd.DatetimeIndex([p0]).append(t)
    return np.minimum(np.asarray((d[1:] - d[:-1]).days), MAX_GAP)


def buy_day_gaps(dates):
    """The same for nights dated by their BUY day (the live log): days to the
    next trading day; the last night runs to the next weekday."""
    t = pd.to_datetime(pd.Index(dates, dtype=str), format='%Y%m%d')
    nxt = t[1:].append(pd.DatetimeIndex([t[-1] + pd.offsets.BDay(1)]))
    return np.minimum(np.asarray((nxt - t).days), MAX_GAP)


def house_maint(held):
    for n, m in HOUSE:
        if held >= n:
            return m
    return HOUSE[-1][1]


def cost_curve(names, px, notional, plan, k=CURVE_PTS):
    """(deployed $, commissions $) for the same basket sized by
    paper_bench.sizing at k targets from 2% of `notional` up to it, so a book
    the Reg T cap shrinks is charged what ITS orders cost (per-order minimums
    included). Increasing in deployed $; the last node is the full book."""
    pts = {}
    for t in notional * np.geomspace(0.02, 1.0, k):
        b, _ = pb.sizing(names, px, t)
        d = float(b.notional.sum())
        if d > 0:
            pts[d] = float(pb.night_cost(plan, b))
    xp = np.array(sorted(pts))
    return xp, np.array([pts[x] for x in xp])


def account(names, px, capital, L, plan, maint=None):
    """The book orders.py would send at notional L x capital, its cost, and
    the cost of the smaller books the same basket sizes to (cost_curve)."""
    book, _ = pb.sizing(names, px, L * capital)
    notional = float(book.notional.sum())
    held = int((book.shares > 0).sum())
    cost = float(pb.night_cost(plan, book))
    xp, fp = cost_curve(names, px, L * capital, plan)
    if notional > 0:
        assert np.isclose(xp[-1], notional) and np.isclose(fp[-1], cost)
    return dict(L=L, notional=notional, dep=notional / capital, held=held,
                names=len(names), cost=cost, cost_bps=cost / capital * 1e4,
                cost_traded=cost / notional * 1e4 if notional else np.nan,
                maint=house_maint(held) if maint is None else maint,
                curve=(xp, fp))


def night_cost_at(acct, G):
    """Commissions $ for a night whose exposure is G: the full book's cost
    when nothing cut it back, else the curve (an account without a curve,
    e.g. a synthetic one in a test, pays its fixed cost)."""
    G = np.asarray(G, float)
    cv = acct.get('curve')
    if cv is None or len(cv[0]) == 0:
        return np.full(G.shape, float(acct['cost']))
    xp, fp = cv
    return np.where(G >= acct['notional'], float(acct['cost']),
                    np.interp(G, xp, fp))


def simulate(R, gaps, capital, acct, rate, cap=REGT, credit=None, log=None):
    """Every path's year at once. R: (n, h) simple returns of the book;
    gaps: (n, h) calendar days; rate: debit $ -> $/yr; credit: (cash $,
    equity $) -> $/yr paid on idle cash, or None for none. Returns equity in
    $ (n, h+1), which paths were liquidated, and net financing per path
    (interest paid minus credit received). log: a dict to fill with per-path
    'comm', 'interest' and 'credit' $."""
    R = np.atleast_2d(R)
    gaps = np.atleast_2d(gaps)
    n, h = R.shape
    N, m = acct['notional'], acct['maint']
    E = np.full(n, float(capital))
    eq = np.empty((n, h + 1))
    eq[:, 0] = E
    on = np.ones(n, bool)
    called = np.zeros(n, bool)
    paid, got, comm = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(h):
        on &= E >= MIN_EQUITY
        G = np.where(on, np.minimum(N, cap * E), 0.0)
        f = np.where(on, rate(G - E) * gaps[:, t] / DAYCOUNT, 0.0)
        cr = (np.zeros(n) if credit is None else
              credit(E - G, E) * gaps[:, t] / DAYCOUNT)
        c = np.where(on, night_cost_at(acct, G), 0.0)
        mv = G * (1 + R[:, t])
        E = E + G * R[:, t] - f + cr - c
        call = on & (E < m * mv)
        E = E - np.where(call, LIQ_SLIP / 1e4 * mv, 0.0)
        called |= call
        on &= ~call
        paid += f
        got += cr
        comm += c
        eq[:, t + 1] = E
    if log is not None:
        log.update(comm=comm, interest=paid, credit=got)
    return eq, called, paid - got


def summary(eq, called, fin, capital):
    """The fan's statistics. 'night' is the fan's own mean night: mean
    one-year P&L / capital / nights, in bps."""
    ret = eq[:, -1] / capital - 1
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    return dict(p5=np.percentile(ret, 5), p25=np.percentile(ret, 25),
                p50=np.percentile(ret, 50), p75=np.percentile(ret, 75),
                p95=np.percentile(ret, 95), loss=(ret < 0).mean(),
                bad=(ret < -0.20).mean(), dd50=np.percentile(dd, 50),
                dd5=np.percentile(dd, 5), mcall=called.mean(),
                ruin=(eq[:, -1] <= 0).mean(),
                call_loss=float(np.median(ret[called])) if called.any()
                else np.nan, fin=float(np.median(fin)),
                night=float(ret.mean()) / (eq.shape[1] - 1) * 1e4)


def expected_night(r, acct, capital, gap_mean, rate):
    """A CLOSED FORM, in bps of capital per night: the full book held all
    year at the START-of-year debit (notional - capital), with no Reg T
    cut-back -- roi_year's 'night' less the interest. Linear, so it matches
    the fan's mean only where nothing is path-dependent (1x, low volatility);
    at the live stress it overstates the levered rows. The tables print the
    fan's own mean (summary's 'night'); this is kept as the comparison."""
    mu = np.expm1(np.asarray(r, float) / 1e4).mean()
    fin = float(rate(acct['notional'] - capital)) * gap_mean / DAYCOUNT
    return (acct['notional'] * mu - acct['cost'] - fin) / capital * 1e4


def draw(r, gaps, n, k=1.0, block=BLOCK, seed=SEED):
    """mc_risk.paths on night indices: the same draws roi_year.fan makes for
    the same seed, with each night's calendar gap carried along."""
    rs = ry.scaled(r, k)
    idx = paths(np.arange(len(rs)), n, H, block,
                np.random.default_rng(seed)).astype(int)
    return np.expm1(rs[idx] / 1e4), np.asarray(gaps, float)[idx], rs


def fan(r, gaps, acct, capital, n, rate, k=1.0, block=BLOCK, credit=None):
    """One account's one-year fan. 'night' is the fan's own mean night;
    'night0' the start-of-year closed form (expected_night)."""
    R, G, rs = draw(r, gaps, n, k, block)
    s = summary(*simulate(R, G, capital, acct, rate, credit=credit), capital)
    s['night0'] = expected_night(rs, acct, capital, float(np.mean(gaps)), rate)
    return s


def held(r, n):
    """A held, unlevered-by-us position (QQQ, QLD-like): roi_year's own fan."""
    s = ry.fan(r, 1.0, 0.0, n, np.random.default_rng(SEED), compound=True)
    s.update(mcall=np.nan, fin=np.nan, ruin=0.0, call_loss=np.nan)
    return s


def qld_like(cc, gaps, bm=BM):
    """2x the daily QQQ return less EFFR + 1.75%/yr per calendar day."""
    x = 2 * np.expm1(np.asarray(cc, float) / 1e4) - \
        (bm + QLD_DRAG) / 100 * np.asarray(gaps, float) / 365.25
    return 1e4 * np.log1p(x)


def growth(R, gaps, f, book, fin_day, xcap=None, maint=None):
    """Mean log growth per night at target leverage f. book(f) -> (exposure,
    commissions) as fractions of equity; fin_day(exposure) -> interest per
    calendar day, likewise.
    The margin rules, off by default (the textbook f*):
      xcap   the initial-margin ceiling on exposure (Reg T 2.0; portfolio
             margin 1/0.259): a book above it cannot be opened, -inf.
      maint  the maintenance rate: a night whose equity at the open is below
             maint x the position's value is a liquidation, and pays
             simulate()'s LIQ_SLIP on that value. (At exposure >= 1/maint
             every night is one: the account is in deficit at entry.)"""
    x_f, c = book(f)
    if xcap is not None and x_f > xcap + 1e-12:
        return -np.inf
    x = x_f * R - c - fin_day(x_f) * gaps
    if maint is not None and x_f > 0:
        mv = x_f * (1 + R)
        x = x - np.where(1 + x < maint * mv, LIQ_SLIP / 1e4 * mv, 0.0)
    if (x <= -1).any():
        return -np.inf
    return float(np.log1p(x).mean())


def kelly(R, gaps, book, fin_day, fmax=FMAX, xcap=None, maint=None):
    """Growth-optimal constant leverage: argmax of mean log growth over the
    given nights, on a 0.05 grid refined to 0.01. Returns (f*, g per night).
    f is the TARGET; the exposure is what book(f) actually deploys, so an
    empty whole-share book earns and costs nothing. xcap / maint: growth()'s
    margin rules; with them this is the FEASIBLE f*."""
    R, gaps = np.ravel(R), np.ravel(gaps)
    g_ = lambda f: growth(R, gaps, f, book, fin_day, xcap, maint)
    coarse = np.round(np.arange(0, fmax + 1e-9, 0.05), 4)
    g = [g_(f) for f in coarse]
    f0 = coarse[int(np.argmax(g))]
    fine = np.round(np.arange(max(0.0, f0 - 0.05), min(fmax, f0 + 0.05) + 1e-9,
                              0.01), 4)
    g = [g_(f) for f in fine]
    i = int(np.argmax(g))
    return float(fine[i]), float(g[i])


def kelly_closed(R, gaps, slope, c_day, c_fixed=0.0):
    """The first-order condition (mu - k - c)/sigma^2, in units of EXPOSURE:
    k the marginal commission per unit of exposure, c the interest per unit
    borrowed per night. Below 1x nothing is borrowed: mu/sigma^2 capped at 1.
    c_fixed is the part of the nightly commission that does not scale with
    exposure (the per-order minimums). It cannot move an interior optimum,
    but it can make NOT trading better than any exposure -- a corner a
    first-order condition cannot see -- so the growth at the FOC's x is
    checked against 0 and the answer is 0 when it does not beat it."""
    R = np.ravel(R)
    mu, s2 = R.mean() - slope, R.var()
    c = c_day * np.mean(gaps)
    if (mu - c) / s2 > 1:
        x = (mu - c) / s2
    else:
        x = float(np.clip(mu / s2, 0, 1))
    g = x * mu - c * max(0.0, x - 1) - c_fixed - x * x * s2 / 2
    return float(x) if x > 0 and g > 0 else 0.0


def decide(live_tab, tabs, grid=GRID, world=DECIDE):
    """The header's pre-declared rule. live_tab: {world: {L: summary}} at the
    live-volatility stress; tabs: every book table (both volatilities).
    Returns (the L > 1 that pass, {L: reasons it fails})."""
    base = live_tab[world][1.0]
    why = {}
    for L in grid:
        if L == 1.0:
            continue
        s, w = live_tab[world][L], []
        if not s['p50'] > base['p50']:
            w.append(f'half-edge median {s["p50"] * 100:+.1f}% vs 1x '
                     f'{base["p50"] * 100:+.1f}%: median not raised')
        if s['p5'] < base['p5']:
            w.append(f'half-edge p5 {s["p5"] * 100:+.1f}% vs 1x '
                     f'{base["p5"] * 100:+.1f}%: p5 lowered')
        worst = max(t[x][L]['mcall'] for t in tabs for x in t if L in t[x])
        if not worst < CALL_MAX:
            w.append(f'P(margin call) up to {worst:.1%} (limit {CALL_MAX:.0%})')
        why[L] = w
    return [L for L, w in why.items() if not w], why


def title(text, proxy, live_data=False):
    print()
    if proxy:
        print(BANNER_LIVE if live_data else BANNER)
    print(f'== {text} ==')


HEAD = (f'{"":<26}{"E[net]":>8}{"p5":>8}{"MEDIAN":>9}{"p95":>8}'
        f'{"P(loss)":>8}{"P<-20%":>8}{"P(call)":>8}{"dip p50":>8}'
        f'{"dip 1/20":>9}{"median $":>10}{"fin $/yr":>9}')


def line(label, s, capital, note=''):
    pc = '--' if np.isnan(s['mcall']) else f'{s["mcall"] * 100:.1f}%'
    fn = '--' if np.isnan(s['fin']) else f'{s["fin"]:,.0f}'
    return (f'{label:<26}{s["night"]:>+6.1f}bp{s["p5"] * 100:>+7.1f}%'
            f'{s["p50"] * 100:>+8.1f}%{s["p95"] * 100:>+7.1f}%'
            f'{s["loss"] * 100:>7.0f}%{s["bad"] * 100:>7.0f}%{pc:>8}'
            f'{s["dd50"] * 100:>7.0f}%{s["dd5"] * 100:>8.0f}%'
            f'{s["p50"] * capital:>+10,.0f}{fn:>9}{note}')


def wname(name, proxy):
    """A world's printed name: a proxy run tags its recent era (see
    PROXY_RECENT_NOTE)."""
    return name + (PROXY_TAG if proxy and name == RECENT else '')


def book_table(text, ws, gw, accts, bench, capital, n, rate, proxy, k=None,
               credit=None):
    title(text, proxy)
    print(HEAD)
    out = {}
    for name, r in ws.items():
        kk = 1.0 if k is None else k(r)
        print(wname(name, proxy)
              + (f'   (deviations x{kk:.2f})' if kk != 1.0 else ''))
        out[name] = {}
        for a in accts:
            s = fan(r, gw[name], a, capital, n, rate, k=kk, credit=credit)
            out[name][a['L']] = s
            b = out[name][GRID[0]]
            dom = (a['L'] > 1 and s['p50'] < b['p50'] and s['p5'] < b['p5'])
            print(line(f'  {a["L"]:.2f}x  ({a["dep"]:.2f} deployed)', s,
                       capital, '  dominated by 1x' if dom else ''))
    print('  -- the same money, instead:')
    for name, s in bench.items():
        print(line(f'  {name}', s, capital))
    if proxy and RECENT in ws:
        print(PROXY_RECENT_NOTE)
    d = out.get(DECIDE)
    if d:
        print('  E[net] = each fan\'s own mean night (mean one-year P&L / '
              'capital / 252: the cut-back after losses,\n  the moving debit, '
              're-priced commissions and the idle-cash credit all in; held '
              'rows: roi_year\'s mean\n  night). The start-of-year closed form '
              '(debit fixed, no cut-back) would print, for '
              f'{DECIDE}:\n  '
              + ', '.join(f'{L:.2f}x {d[L]["night0"]:+.1f} (fan '
                          f'{d[L]["night"]:+.1f})' for L in d)
              + '.  "fin $/yr" is interest paid net of credit received.')
    calls = [(w, L, s) for w, row in out.items() for L, s in row.items()
             if s['mcall'] > 0]
    worst = min(float(np.min(ry.scaled(r, 1.0 if k is None else k(r))))
                for r in ws.values())
    if calls:
        for w, L, s in calls:
            print(f'  margin calls: {w} at {L:.2f}x on {s["mcall"]:.1%} of '
                  f'paths; median year on those paths '
                  f'{s["call_loss"] * 100:+.0f}%; equity <= 0 on '
                  f'{s["ruin"]:.1%}')
    else:
        m = accts[-1]['maint']
        print(f'  no path breached maintenance at an open: at {REGT:.0f}x '
              f'equity and {m:.0%} maintenance it takes a one-night basket drop '
              f'of\n  {(1 / REGT - m) / (1 - m):.0%}; the worst resampled '
              f'night here is {np.expm1(worst / 1e4) * 100:+.1f}%.')
    return out


def kelly_section(ws, gw, names, px, capital, plan, rate, k, n, proxy,
                  maint=HOUSE[0][1]):
    cache = {}

    def book(f):
        f = round(float(f), 4)
        if f <= 0:
            return 0.0, 0.0
        if f not in cache:
            b, _ = pb.sizing(names, px, f * capital)
            cache[f] = (b.notional.sum() / capital,
                        pb.night_cost(plan, b) / capital)
        return cache[f]

    def fin_day(x):
        return interest_at(rate, max(0.0, x - 1) * capital) / capital / DAYCOUNT

    (x1, c1), (x2, c2) = book(1.0), book(2.0)
    slope = (c2 - c1) / (x2 - x1)
    fixed = c1 - slope * x1
    pm = capital >= PM_MIN
    ceiling = 1 / PM_INIT if pm else REGT

    title(f'KELLY -- the compounding variant (exposure reset to f x equity '
          f'every close; commissions and interest in). f* maximises mean log '
          f'growth over {n:,} bootstrapped years; g = compounded growth per '
          f'year', proxy)
    vols = [('backtest volatility', None)] + ([('live volatility', k)] if k
                                              else [])
    cols = [('f* free', 9), ('(mu-c)/s2', 10), ('f* RegT', 8), ('g 1x', 7),
            ('g 2x', 7), ('g RegT', 8)] + \
        ([('f* PM', 7), ('g PM', 7)] if pm else [])
    wd = sum(w for _, w in cols) + 4
    print(f'{"":<24}' + ''.join(f'{v:<{wd}}' for v, _ in vols))
    print(f'{"":<24}' + (''.join(f'{c:>{w}}' for c, w in cols) + '    ')
          * len(vols))
    res = {}
    ann = lambda g: np.expm1(H * g) * 100
    for name, r in ws.items():
        cells = ''
        for v, kf in vols:
            kk = 1.0 if kf is None else kf(r)
            R, G, _ = draw(r, gw[name], n, kk)
            Rr, Gr = np.ravel(R), np.ravel(G)
            f, gf = kelly(R, G, book, fin_day)
            fr, gr = kelly(R, G, book, fin_day, xcap=REGT, maint=maint)
            cf = kelly_closed(R, G, slope, fin_day(2.0), fixed)
            g1 = growth(Rr, Gr, 1.0, book, fin_day, REGT, maint)
            g2 = growth(Rr, Gr, 2.0, book, fin_day, REGT, maint)
            reach = book(f)[0] <= ceiling + 1e-12
            cells += (f'{f:>8.2f}{" " if reach else "!"}{cf:>10.2f}'
                      f'{fr:>8.2f}{ann(g1):>+6.0f}%{ann(g2):>+6.0f}%'
                      f'{ann(gr):>+7.0f}%')
            row = dict(f=f, g=gf, reach=reach, x=book(f)[0], f_regt=fr,
                       x_regt=book(fr)[0], g_regt=gr, g1=g1, g2=g2, closed=cf)
            if pm:
                fp, gp = kelly(R, G, book, fin_day, xcap=1 / PM_INIT,
                               maint=PM_MAINT)
                cells += f'{fp:>7.2f}{ann(gp):>+6.0f}%'
                row.update(f_pm=fp, g_pm=gp)
            cells += '    '
            res[(name, v)] = row
        print(f'{wname(name, proxy):<24}{cells}')
    xmax = max(r['x'] for r in res.values())
    print(f'  f* free: no margin rules -- a textbook reference, NOT a result. '
          f'"!" = not attainable here: its\n  exposure (up to {xmax:.2f}x '
          f'equity) is over the ceiling, and its growth is not printed. '
          f'Reg T caps\n  overnight exposure at {REGT:.1f}x equity; portfolio '
          f'margin needs ${PM_MIN / 1e3:.0f}k NLV and still caps this basket at '
          f'{1 / PM_INIT:.2f}x\n  ({PM_INIT:.1%} initial); and above '
          f'{1 / maint:.1f}x a {maint:.0%} maintenance line is breached at '
          f'entry. Half of an f* free\n  above 4 is still over the Reg T '
          f'ceiling.')
    print(f'  f* RegT: the FEASIBLE optimum -- exposure <= {REGT:.1f}x equity, '
          f'and a night that breaches {maint:.0%}\n  maintenance at the open '
          f'pays the liquidation slippage. g 1x / g 2x / g RegT are growth '
          f'under those\n  rules. ' +
          (f'f* PM / g PM: portfolio margin, exposure <= {1 / PM_INIT:.2f}x, '
           f'{PM_MAINT:.1%} maintenance.' if pm else
           f'No portfolio-margin column: ${capital:,.0f} is below the '
           f'${PM_MIN / 1e3:.0f}k IBKR needs to open it.'))
    print('  All f* are TARGETS (the grid\'s units); the whole-share book '
          'deploys 87-97% of them, and the Reg T\n  test is on what it '
          'deploys. f* 0 = do not trade: no exposure beats the commissions. '
          '(mu-c)/s2 is\n  the first-order check in units of exposure, with '
          f'the marginal commission per unit ({slope * 1e4:.2f} bps),\n  the '
          f'interest per unit borrowed, and the fixed per-order part '
          f'({fixed * 1e4:.2f} bps/night) as a corner: 0 when\n  no exposure '
          'out-earns it.')
    if proxy and RECENT in ws:
        print(PROXY_RECENT_NOTE)
    return res


def interest_at(rate, debit):
    return float(np.asarray(rate(debit)).item())


def tail_check(r, g, accts, capital, n, rate, k, proxy, label, credit=None):
    title(f'TAIL CHECK -- {DECIDE}, {label}: longer blocks keep months-long '
          f'regimes together', proxy)
    print(f'{"":<8}' + ''.join(f'{f"{b}-day blocks":^30}' for b in (21, 63, 126)))
    print(f'{"":<8}' + f'{"p5":>8}{"dip 1/20":>11}{"P(call)":>11}' * 3)
    for a in accts:
        cells = ''
        for b in (21, 63, 126):
            s = fan(r, g, a, capital, n, rate, k=k, block=b, credit=credit)
            cells += (f'{s["p5"] * 100:>+7.1f}%{s["dd5"] * 100:>+10.0f}%'
                      f'{s["mcall"] * 100:>10.1f}%')
        print(f'  {a["L"]:.2f}x {cells}')
    if proxy:
        print('  (the proxy\'s residual is iid, so only the index factor carries '
              'regimes here: longer blocks\n   move it little. On the real '
              'series they deepened the tails by up to ~3 points, RESULTS 26.)')


def replay(daily, accts, qaccts, capital, rate, proxy, credit=None):
    d = daily.dropna(subset=['q5_on_bps']).sort_values('date')
    gaps = buy_day_gaps(d.date)[None, :]
    title(f'LIVE REPLAY -- the {len(d)} graded nights {d.date.iloc[0]}..'
          f'{d.date.iloc[-1]} at each L (official prints, these costs and '
          f'this financing)', proxy, live_data=True)
    print(f'{"":<22}{"return":>8}{"P&L $":>10}{"max dip":>9}{"net bp/nt":>11}'
          f'{"comm $":>9}{"fin $":>8}  margin call')
    rows = [('book', 'q5_on_bps', a) for a in accts] + \
           [('QQQ overnight', 'qqq_on_bps', a) for a in qaccts]
    out = {}
    for lab, col, a in rows:
        R = np.expm1(d[col].values.astype(float) / 1e4)[None, :]
        lg = {}
        eq, called, fin = simulate(R, gaps, capital, a, rate, credit=credit,
                                   log=lg)
        e = eq[0]
        dd = (e / np.maximum.accumulate(e) - 1).min()
        net = (e[-1] - capital) / capital / len(d) * 1e4
        tag = f'{lab} {a["L"]:.2f}x'
        out[tag] = dict(ret=e[-1] / capital - 1, comm=float(lg['comm'][0]),
                        fin=float(fin[0]), called=bool(called[0]))
        print(f'{tag:<22}{(e[-1] / capital - 1) * 100:>+7.1f}%'
              f'{e[-1] - capital:>+10,.0f}{dd * 100:>+8.1f}%{net:>+10.1f}'
              f'{lg["comm"][0]:>9,.0f}{fin[0]:>8,.0f}  '
              f'{"YES" if called[0] else "no"}')
    print('  comm $: what the account paid, re-priced on any night the Reg T '
          'cap cut the book back; fin $: interest\n  net of idle-cash credit.')
    sd = float(d.q5_on_bps.std())
    print(f'  {len(d)} nights is one short path: the mean night carries a '
          f'standard error of ~{sd / np.sqrt(len(d)):.0f} bps. This checks '
          f'plumbing and sign, not the edge.')
    return out


def selection_alpha(q5, F):
    """roi_year.run's alpha: the gross Q5 night on QQQ (+SPY) overnight."""
    from factor import regress
    cols = ['mkt_on'] + (['spy_on'] if 'spy_on' in F else [])
    j = pd.concat([q5.rename('y'), F[cols]], axis=1).dropna()
    r = regress(j.y, j[cols])
    return float(r['alpha']), cols, r


def run(q5, F, cut, names, px, daily, capital, plan, n, proxy=False,
        gaps=None, flat=None, bm=BM, kelly_paths=KELLY_PATHS):
    """q5: gross nightly log-bps dated by open day; F: mkt_on, mkt_cc (and
    spy_on) likewise; daily: the live log (date, q5_on_bps, qqq_on_bps);
    gaps: calendar days per open day (else derived from the dates)."""
    rate = lambda d: interest(d, bm, flat)
    cred = lambda cash, eq: credit(cash, eq, bm)
    if gaps is None:
        cal = sorted(set(q5.index) | set(F.index))
        gaps = pd.Series(day_gaps(cal), index=cal)
    alpha, cols, reg = selection_alpha(q5, F)
    qqq_cc = F.mkt_cc.dropna()
    qqq_cc = qqq_cc[qqq_cc.index >= q5.index.min()]
    qqq_on = F.mkt_on.dropna()
    qqq_on = qqq_on[qqq_on.index >= q5.index.min()]
    g_q5 = gaps.reindex(q5.index).fillna(1).values
    g_cc = gaps.reindex(qqq_cc.index).fillna(1).values
    g_on = gaps.reindex(qqq_on.index).fillna(1).values
    ws = ry.worlds(q5, alpha)
    gw = {w: g_q5 for w in ws}
    gw[RECENT] = g_q5[np.asarray(q5.index >= ry.RECENT)]
    for w in ws:
        assert len(gw[w]) == len(ws[w]), w
    accts = [account(names, px, capital, L, plan) for L in GRID]
    qaccts = [account(['QQQ'], px, capital, L, plan, maint=ETF_MAINT)
              for L in (1.0, 2.0)]
    gm = float(np.mean(g_q5))

    if proxy:
        print(BANNER)
    print('LEVERAGE, RESEARCH ONLY -- the year AFTER a GO verdict, not this '
          'year. The project\'s rule\n(RESULTS 19) is no live money until '
          'the forward test passes; GO does not establish the\n'
          'as-backtested world.')
    print('\nDECISION RULE (declared in the header before any run): L > 1 '
          'only if, at the live-volatility\nstress, it RAISES the half-the-'
          'edge median AND does not LOWER its p5, AND P(margin call)\n< 1% '
          'in every world at both volatilities. Otherwise 1x stands.\n')
    src = ('PROXY book: a + 0.69 QQQ_on + 0.73 SPY_on + t(4) residual (sd '
           f'{PROXY["alpha"] * np.sqrt(252) / PROXY["rsharpe"]:.1f}; a steps '
           'down at 2016 so its 2016+ world ties half the edge), real '
           'Yahoo overnights' if proxy else
           'traded book: long the frozen model\'s Q5 overnight')
    print(f'{src}, {q5.index.min()}..{q5.index.max()} ({len(q5):,} nights; '
          f'history stops at the {cut[:4]}-{cut[4:]} cutoff)')
    print(f'  gross {q5.mean():+.2f} bps/night, sd {q5.std():.0f}; selection '
          f'alpha {alpha:+.2f} on {"+".join(cols)} (betas '
          + ' / '.join(f'{v:+.2f}' for v in reg['betas'].values())
          + f', R2 {reg["r2"]:.2f}); censoring -{ry.CENS}')
    if flat is None:
        cap_ = lambda x: f'${x / 1e6:,.0f}M' if x >= 1e6 else f'${x / 1e3:,.0f}k'
        tiers = ', '.join(f'{bm + s:.2f}% to {cap_(hi)}' for hi, s in TIERS[:3])
        how = f'IBKR Pro blended tiers on the actual debit, BM {bm:.2f}%: {tiers}'
    else:
        how = f'a flat {flat:.2f}%/yr on the actual debit (--rate)'
    print(f'  financing: {how}; Actual/360 on calendar days (mean '
          f'{gm:.3f} per night,\n  so {interest_at(rate, 1e4) / 1e4 / DAYCOUNT * gm * 1e4:.2f} '
          f'bps/night per unit borrowed on a small debit); idle cash earns '
          f'{max(bm - CREDIT_SPREAD, 0):.2f}% above\n  ${CREDIT_FREE / 1e3:.0f}k'
          f', times min(1, NAV/${CREDIT_NAV / 1e3:.0f}k)')

    title(f'ACCOUNTS at ${capital:,.0f} on {plan}: the book sized at L x '
          f'capital (paper_bench.sizing, whole shares)', proxy)
    print(f'{"":>7}{"notional":>10}{"deployed":>10}{"names":>7}{"maint":>7}'
          f'{"comm $/nt":>11}{"bp equity":>10}{"bp traded":>10}{"debit $":>9}'
          f'{"rate":>7}{"fin bp/nt":>10}{"fin $/yr":>9}')
    for a in accts + qaccts:
        D = max(0.0, a['notional'] - capital)
        yr = interest_at(rate, D)
        lab = f'{a["L"]:.2f}x' + (' QQQ' if a['names'] == 1 else '')
        nm = f'{a["held"]}/{a["names"]}'
        print(f'{lab:>9}{a["notional"]:>8,.0f}{a["dep"]:>9.2f}x'
              f'{nm:>7}{a["maint"]:>7.0%}'
              f'{a["cost"]:>11.2f}{a["cost_bps"]:>10.2f}{a["cost_traded"]:>10.2f}'
              f'{D:>9,.0f}{(yr / D * 100 if D else 0):>6.2f}%'
              f'{yr * gm / DAYCOUNT / capital * 1e4:>10.2f}'
              f'{yr * 365.25 / DAYCOUNT:>9,.0f}')
    a1, a2 = accts[0], accts[-1]
    f2 = interest_at(rate, max(0, a2['notional'] - capital)) * gm / DAYCOUNT \
        / capital * 1e4
    extra = a2['dep'] - a1['dep']
    dc = a2['cost_bps'] - a1['cost_bps']
    print(f'  commissions per dollar TRADED fall from {a1["cost_traded"]:.2f} '
          f'bps at 1x to {a2["cost_traded"]:.2f} at {a2["L"]:.0f}x: the '
          f'per-order minimum spread over more notional,\n  the one channel '
          f'where leverage helps a small account. The extra {extra:.2f} units '
          f'of exposure add {dc:.2f} bps/night\n  of commissions and '
          f'{f2:.2f} of interest: {(dc + f2) / extra:.2f} bps per extra unit, '
          f'against {a1["cost_traded"]:.2f} per unit for the first.')
    print(f'  {PM_LABEL}. Not simulated in the fans'
          + (f': IBKR needs ${PM_MIN / 1e3:.0f}k NLV to open portfolio margin.'
             if capital < PM_MIN else ' (a Kelly line only; see the header).'))

    bench = {'QQQ held, 1x': held(qqq_cc.values, n),
             'QLD-like 2x daily, held': held(qld_like(qqq_cc.values, g_cc, bm),
                                             n)}
    for a in qaccts:
        bench[f'QQQ overnight, {a["L"]:.0f}x'] = fan(qqq_on.values, g_on, a,
                                                    capital, n, rate,
                                                    credit=cred)
    tabs = [book_table(f'ONE YEAR ({H} nights), backtest volatility, '
                       f'{BLOCK}-day blocks, fixed notional; returns on '
                       f'${capital:,.0f}', ws, gw, accts, bench, capital, n,
                       rate, proxy, credit=cred)]
    live = pd.Series(daily.q5_on_bps, dtype=float).dropna()
    k = None
    if len(live) >= ry.MIN_LIVE:
        lsd = float(live.std())
        rng = np.random.default_rng(SEED)
        boot = [rng.choice(live.values, len(live)).std() for _ in range(2000)]
        lo, hi = np.percentile(boot, [5, 95])
        k = lambda r: max(1.0, lsd / float(np.std(r)))
        tabs.append(book_table(
            f'ONE YEAR at the LIVE basket\'s volatility ({len(live)} nights: '
            f'sd {lsd:.0f} bps, 90% CI [{lo:.0f}, {hi:.0f}]) -- a stress, not '
            f'a forecast', ws, gw, accts, bench, capital, n, rate, proxy, k=k,
            credit=cred))
        print('  (book rows widened to the live sd, arithmetic mean held, as '
              'roi_year does; the QQQ rows keep\n   their own volatility)')
    else:
        print(f'\n(only {len(live)} live nights -- no volatility stress yet)')
    dw = DECIDE
    tail_check(ws[dw], gw[dw], accts, capital, n, rate,
               1.0 if k is None else k(ws[dw]), proxy,
               'live volatility' if k else 'backtest volatility', credit=cred)
    kres = kelly_section(ws, gw, names, px, capital, plan, rate, k,
                         min(n, kelly_paths), proxy, maint=accts[-1]['maint'])

    title('ROBUST L -- from the fixed-notional medians above (grid '
          + ' / '.join(f'{L:g}' for L in GRID) + ')', proxy)
    picks = {}
    rset = ', '.join(wname(w, proxy) for w in ROBUST)
    for t, v in zip(tabs, ('backtest vol', 'live vol')):
        worst = {L: min(t[w][L]['p50'] for w in ROBUST) for L in GRID}
        Lr = max(GRID, key=lambda L: worst[L])
        wl = min(ROBUST, key=lambda w: t[w][Lr]['p50'])
        picks[v] = Lr
        print(f'  maximin over {{{rset}}}, {v}: {Lr:.2f}x (worst-world '
              f'median {worst[Lr] * 100:+.1f}%, {wname(wl, proxy)}; at 1x '
              f'{worst[1.0] * 100:+.1f}%)')
    if proxy:
        print(PROXY_RECENT_NOTE)
    if k:
        t = tabs[1][DECIDE]
        Lh = max(GRID, key=lambda L: t[L]['p50'])
        print(f'  best half-the-edge median at live vol: {Lh:.2f}x (median '
              f'{t[Lh]["p50"] * 100:+.1f}%, p5 {t[Lh]["p5"] * 100:+.1f}%; 1x '
              f'{t[1.0]["p50"] * 100:+.1f}% / {t[1.0]["p5"] * 100:+.1f}%)')

    rep = replay(daily, accts, qaccts, capital, rate, proxy, credit=cred)

    title('WHAT THIS SAYS', proxy)
    print('  rule (declared before the run): L > 1 only if, at the live-'
          'volatility stress, it raises the\n  half-the-edge median AND does '
          'not lower its p5, AND P(margin call) < 1% in every world.')
    if k is None:
        passing = []
        print('  the live-volatility stress needs '
              f'{ry.MIN_LIVE}+ live nights, so the rule cannot pass yet.')
    else:
        passing, why = decide(tabs[1], tabs)
        t = tabs[1][DECIDE]
        pre = f'  {DECIDE}, live stress:'
        print(f'{pre}{"median":>8}{"p5":>8}{"worst P(call)":>15}')
        for L in GRID:
            mc = max(tb[w][L]['mcall'] for tb in tabs for w in tb)
            v = '' if L == 1.0 else ('  PASSES' if not why[L] else
                                     '  fails: ' + '; '.join(
                                         x.split(': ')[-1] for x in why[L]))
            lab = f'{L:.2f}x'
            print(f'{lab:>{len(pre)}}{t[L]["p50"] * 100:>+7.1f}%'
                  f'{t[L]["p5"] * 100:>+7.1f}%{mc * 100:>14.1f}%{v}')
    print(f'  => {"consider " + ", ".join(f"{L:.2f}x" for L in passing) if passing else "1x stands"}'
          f' at ${capital:,.0f}.')
    if k is not None and not passing:
        t = tabs[1][DECIDE]
        dm = (t[GRID[-1]]['p50'] - t[1.0]['p50']) * 100
        dp = (t[GRID[-1]]['p5'] - t[1.0]['p5']) * 100
        print(f'  The median-only picks above favour leverage; the p5 clause '
              f'is what vetoes it. At the live\n  stress {GRID[-1]:.0f}x buys '
              f'{dm:+.1f} points of half-edge median for {dp:+.1f} of p5.')
    qo, ql = bench.get('QQQ overnight, 2x'), bench['QLD-like 2x daily, held']
    print(f'  The levered index, same money (unstressed): QQQ overnight 2x '
          f'median {qo["p50"] * 100:+.1f}% / p5 {qo["p5"] * 100:+.1f}%;'
          f'\n  QLD-like 2x held {ql["p50"] * 100:+.1f}% / '
          f'{ql["p5"] * 100:+.1f}%.')
    print(f'  Kelly (compounding growth), FEASIBLE f* -- exposure <= '
          f'{REGT:.1f}x equity under Reg T, maintenance applied:')
    for v in ('backtest volatility', 'live volatility'):
        rows = [(w, kres[(w, v)]) for w in ws if (w, v) in kres]
        if not rows:
            continue
        lever = [wname(w, proxy) for w, r in rows if r['f_regt'] > 1]
        win2 = [wname(w, proxy) for w, r in rows if r['g2'] > r['g1']]
        print(f'    {v}: ' + ', '.join(f'{wname(w, proxy)} {r["f_regt"]:.2f}'
                                       for w, r in rows))
        print(f'      f* > 1 in {len(lever)} of {len(rows)} worlds'
              + (f' ({", ".join(lever)})' if lever else '')
              + f'; 2x out-compounds 1x in {len(win2)} of {len(rows)}'
              + (f' ({", ".join(win2)})' if win2 else '') + '.')
    far = [(w, v, r) for (w, v), r in kres.items() if not r['reach']]
    if far:
        w, v, r = max(far, key=lambda x: x[2]['f'])
        print(f'    The no-margin textbook f* reaches {r["f"]:.2f} '
              f'({wname(w, proxy)}, {v}): NOT attainable at '
              f'${capital:,.0f} -- Reg T caps exposure at {REGT:.1f}x'
              + ('' if capital >= PM_MIN else
                 f', portfolio margin needs ${PM_MIN / 1e3:.0f}k') + '.')
    print('    Kelly is growth, not the decision: the rule above reads the '
          'fixed-notional fan\'s median AND p5\n    at the live stress, and '
          'the p5 clause is the one that binds.')
    if 'qqq_on_bps' in daily and len(live) >= 10:
        j = daily[['q5_on_bps', 'qqq_on_bps']].astype(float).dropna()
        beta = float(np.cov(j.q5_on_bps, j.qqq_on_bps)[0, 1]
                     / j.qqq_on_bps.var())
        print(f'  live overnight beta to QQQ {beta:.2f} (backtest ~1.42): '
              f'at {a2["L"]:.0f}x ({a2["dep"]:.2f} deployed) the account '
              f'carries ~{beta * a2["dep"]:.1f} units of overnight-QQQ beta.')
    if proxy:
        print('  PROXY: this is a factor model of the book, not the book. Re-'
              'run with --real on the owner\'s\n  machine before reading any '
              'number above as the book\'s answer.')
    return dict(tabs=tabs, bench=bench, accts=accts, kelly=kres,
                passing=passing, picks=picks, replay=rep)


def frozen_cutoff():
    mp = os.path.join(ry.ROOT, 'models', 'xsec_lgbm.json')
    if not os.path.exists(mp):
        raise SystemExit('models/xsec_lgbm.json missing -- cannot locate the '
                         'frozen cutoff')
    return json.load(open(mp))['last_tmonth'].replace('-', '')[:6]


def yahoo_bars(cut):
    import yfinance as yf
    end = (pd.Timestamp(f'{cut[:4]}-{cut[4:]}-01') + pd.offsets.MonthBegin(1)
           + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    v = yf.download(['QQQ', 'SPY'], start='2002-12-01', end=end,
                    interval='1d', auto_adjust=False, actions=False,
                    group_by='ticker', progress=False)
    bars = {s: v[s][['Open', 'Close']].dropna() for s in ('QQQ', 'SPY')}
    if min(len(b) for b in bars.values()) < 5000:
        raise SystemExit('Yahoo returned too little QQQ/SPY history for the '
                         'proxy; retry')
    return bars


def proxy_inputs(bars, cut, seed=SEED):
    """(q5, F, gaps): the factor proxy from REAL QQQ/SPY daily bars. bars:
    {'QQQ': df, 'SPY': df} with Open and Close (split-adjusted, not
    dividend-adjusted), indexed by date."""
    leg = {}
    for s in ('QQQ', 'SPY'):
        b = bars[s][['Open', 'Close']].dropna()
        b = b[(b.Open > 0) & (b.Close > 0)]
        leg[s] = b.set_axis(pd.DatetimeIndex(b.index).strftime('%Y%m%d'))
    q, s = leg['QQQ'], leg['SPY']
    gaps = pd.Series(day_gaps(q.index), index=q.index)
    F = pd.DataFrame({'mkt_on': 1e4 * np.log(q.Open / q.Close.shift()),
                      'mkt_cc': 1e4 * np.log(q.Close / q.Close.shift()),
                      'spy_on': 1e4 * np.log(s.Open / s.Close.shift())})
    F = F.dropna()
    F = F[(F.index >= PROXY['start']) & (F.index.str[:6] <= cut)]
    e = np.random.default_rng(seed).standard_t(PROXY['df'], len(F))
    sd = PROXY['alpha'] * np.sqrt(252) / PROXY['rsharpe']
    e = (e - e.mean()) / e.std() * sd
    base = PROXY['b_qqq'] * F.mkt_on + PROXY['b_spy'] * F.spy_on + e
    q5 = base - base.mean() + PROXY['gross']
    # The alpha steps at 2016 (see the header): the 2016+ nights' gross mean
    # is pinned so that roi_year's recent-era world (q5[2016+] - CENS) has the
    # half-the-edge world's mean (q5 - CENS - recent_edge x (alpha - CENS)),
    # and the earlier nights take the rest so the overall gross stays put.
    # alpha is the OLS intercept selection_alpha() will regress; it barely
    # moves with the split, so a few fixed-point rounds settle it.
    post = np.asarray(F.index >= ry.RECENT)
    if post.any() and (~post).any():
        X = np.column_stack([np.ones(len(F)), F.mkt_on.values,
                             F.spy_on.values])
        n, n_post = len(F), int(post.sum())
        for _ in range(50):
            alpha = float(np.linalg.lstsq(X, q5.values, rcond=None)[0][0])
            target = PROXY['gross'] - PROXY['recent_edge'] * (alpha - ry.CENS)
            d_post = target - base[post].mean()
            d_pre = ((n * PROXY['gross'] - n_post * target) / (n - n_post)
                     - base[~post].mean())
            new = base + np.where(post, d_post, d_pre)
            done = np.allclose(new.values, q5.values, rtol=0, atol=1e-10)
            q5 = new
            if done:
                break
    return q5.rename('q5'), F, gaps.reindex(F.index)


def load_real():
    if not os.path.exists(os.path.join(ry.DATA, 'xsec_ml_daily.csv')):
        raise SystemExit('data/xsec_ml_daily.csv is not on this machine (it is '
                         'gitignored and lives on the owner\'s). Run --real '
                         'there, or --proxy here for the factor proxy.')
    try:
        return ry.load()
    except SystemExit as ex:
        raise SystemExit(f'{ex}\n--real needs the walk-forward panel too; run '
                         f'it on the owner\'s machine, or use --proxy.')


def main():
    ap = argparse.ArgumentParser(description='Would leverage make the traded '
                                 'book perform better? One-year fans per world '
                                 'and per L, with IBKR financing and margin.')
    ap.add_argument('--capital', type=float, default=25000)
    ap.add_argument('--plan', choices=['fixed', 'tiered'], default='tiered')
    m = ap.add_mutually_exclusive_group()
    m.add_argument('--real', dest='proxy', action='store_false',
                   help='the walk-forward series (default; owner\'s machine)')
    m.add_argument('--proxy', dest='proxy', action='store_true',
                   help='factor proxy from real QQQ/SPY overnights (Yahoo)')
    ap.set_defaults(proxy=False)
    ap.add_argument('--rate', type=float, default=None,
                    help='flat %%/yr on the debit instead of IBKR Pro\'s '
                         'blended tiers (RESULTS 19 used 5.0)')
    ap.add_argument('--bm', type=float, default=BM,
                    help='IBKR USD benchmark (EFFR), %%/yr; try 3.0 and 5.0')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--prices', default=None,
                    help='offline: CSV of SYM,price instead of Yahoo')
    a = ap.parse_args()
    if a.proxy:
        cut = frozen_cutoff()
        q5, F, gaps = proxy_inputs(yahoo_bars(cut), cut)
    else:
        q5, F, cut = load_real()
        gaps = None
    names = pd.read_csv(pb.PAPER, dtype=str).iloc[-1].q5.split(';')
    px = ry.prices(names + ['QQQ'], a.prices)
    daily = pd.read_csv(pb.DAILY, dtype={'date': str})
    run(q5, F, cut, names, px, daily, a.capital, a.plan, a.paths,
        proxy=a.proxy, gaps=gaps, flat=a.rate, bm=a.bm)


if __name__ == '__main__':
    main()
