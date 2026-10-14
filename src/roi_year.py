# Part of qqq-microstructure.
#
# "So in a year, what does the ROI look like?"
#
# For the book orders.py trades -- since 2026-10-14 the QQQ-hedged book: long
# the frozen model's Q5 basket overnight (buy MOC, sell MOO) and short QQQ 1:1
# against it (sell MOC, cover MOO); --hedge 1, the default (below) -- at a
# given capital and commission plan, NEXT TO the two things the same money
# could do instead: hold QQQ, or hold QQQ overnight only. --hedge 0 is the
# former long-only book of RESULTS 26, no longer traded, kept for comparison.
#
# Read it as the year AFTER a GO verdict. The project's own rule (RESULTS 19,
# 25) is no live money until the forward test passes; the first weak read is
# ~2026-11, and the first high-power one is ~2027-08. Today the answer is $0
# live, with paper run at the intended size.
#
# Not one number: which world we are in is what the forward test has not yet
# decided, so the answer is a fan per world.
#   as backtested   the walk-forward Q5 nights, less RESULTS 15's measured
#                   censoring bias (1.5 bps/night)
#   half the edge   the selection alpha halved -- the order of decay found
#                   after an anomaly is published
#   recent era      only the nights since 2016, resampled as they happened
#   no edge         selection alpha removed: the basket's overnight-market
#                   beta, which still earns the 2003-26 overnight premium --
#                   NOT a floor
#   nothing         zero gross drift: the costs and the volatility alone.
#                   This is the floor.
#
# Method, and why each choice (an adversarial review, 2026-10-12):
#   - Each world is a circular block bootstrap (mc_risk.paths) of REAL
#     measured nights, mean-shifted. No invented dynamics. History stops at
#     the frozen training cutoff: the forward months are the test, not the
#     input.
#   - The selection alpha is RESULTS 21's: a regression of the gross Q5 night
#     on QQQ and SPY overnight (factor.py). A QQQ-only alpha came out about
#     0.9 bps lower and flattered the pessimistic rows.
#   - The account trades a FIXED notional (orders.py --capital), so nightly
#     P&L adds up rather than compounds. Compounding overstated the good tail
#     by up to ~19 points.
#   - The live-volatility stress scales the deviations and keeps the
#     ARITHMETIC mean. Keeping the log mean would have handed the basket
#     (k^2 - 1) sigma^2 / 2 of free return for being riskier.
#   - 21-day blocks average away months-long theme unwinds, so the tails are
#     re-run with 63- and 126-day blocks.
#   - Known and NOT modelled: taxes (every gain is short-term, and nightly
#     re-buys make most losing nights wash sales in a US taxable account);
#     interest on idle cash; dividends (live receives them; they would add);
#     a missed order (orders.py now refuses to double up); and the
#     cross-sectional Jensen term. The series is the mean of the names' log
#     returns; an equal-dollar basket earns ~0.5-1.8 bps/night more, which
#     is conservative.
#
# The hedged book, --hedge RATIO (added 2026-10-14, when the owner switched
# the traded book to it): the same basket long overnight AND short RATIO x
# the long notional actually bought in QQQ overnight (sell MOC, cover MOO).
# --hedge 1 is the 1:1 dollar hedge, RESULTS 21's NEU: alpha +3.79 at t 5.25,
# total overnight beta 0.42 against the long book's 1.42. It is the default,
# read from orders.HEDGE, so the default run is the book orders.py trades.
# --hedge 0 is the former long-only book above, and its output is
# byte-identical to what this script printed before the flag existed (a test
# pins it) -- including its first line, which still calls that book the
# traded one: it was, for RESULTS 26.
#   - Traded series: q5 - RATIO x QQQ overnight (the panel's mkt_on, the
#     factor the alpha is regressed on), on the nights both exist, cut at the
#     same frozen cutoff. At RATIO 1 this is the forward test's paper-log
#     metric, xsec_live's tilt_bps. The selection alpha does not move: QQQ
#     overnight is one of the regressors, so subtracting it changes only that
#     beta.
#   - Worlds: the same five, built from the hedged series. The censoring bias
#     stays whole: it sits on the long leg, and the QQQ leg is never
#     censored (RESULTS 15). No edge is the hedged series' beta-only mean:
#     ~0.42 of overnight beta, about +0.3 bps/night (RESULTS 21).
#   - The series is a difference of log returns, as the tilt is. The book's
#     exact two-leg P&L is higher by about (cov(Q5, QQQ) - var(QQQ)) / 1e4,
#     ~0.1-0.2 bps/night at backtest volatility, so this is conservative, in
#     the same direction as the Jensen note above.
#   - Costs: the basket's, plus one QQQ short (SELL MOC) and one cover (BUY
#     OPG) a night on round(RATIO x the long notional / QQQ price) shares,
#     priced by paper_bench.order_cost (hedge_leg below). The SEC fee and the
#     FINRA TAF fall on the short sale, as on any sale. Both legs use today's
#     fee schedule, as the long-only book always has; capital.py prices 2027's
#     (TAF on), ~0.01 bps/night more.
#   - Carry on the short, per trading night, on 365/252 calendar days: under
#     T+1 the evening short and the morning cover settle on the same day, so
#     the settled short never closes while the book trades nightly (the same
#     argument as leverage.py's debit).
#       borrow     0.40%/yr on the short's value, Actual/360 (IBKR's public
#                  QQQ feed at the 2026-10-12 close; 13-week range 0.3-0.5)
#       proceeds   IBKR Pro pays nothing on the first $100k of short-sale
#                  proceeds and BM - 1.25% (2.63%) from $100k to $1M, scaled
#                  by NAV/$100k below that NAV (IBKR pricing pages, read
#                  2026-10-14). Credited only where IBKR pays it: a short of
#                  $100k or less gets nothing. Above $1M nothing is credited.
#       dividend   the short owes QQQ's dividends (0.414%/yr trailing, to
#                  2026-10-12). The price-only series credits the short with
#                  the ex-date drop instead, so it is charged back here. The
#                  basket's own dividends stay left out, as above.
#     Interest on idle cash stays unmodelled, as for the long-only book.
#   - Reg T: long + short <= 2x equity at IBKR's end-of-day check, with no
#     offset between a stock basket and QQQ, so a 1:1 book at --capital needs
#     about --capital of equity. The run prints the equity to hold at
#     leverage.py's 1.9x operating line and refuses a book above 2x. It also
#     needs a margin account: a cash account or an IRA cannot short.
#   - Live stress: the live hedged series, q5 - RATIO x QQQ from
#     reports/xsec_paper_daily.csv (tilt_bps itself at RATIO 1).
#   - Benchmarks unchanged: QQQ held, and QQQ overnight only.
#
#   python src/roi_year.py --capital 500000 --plan tiered     # the traded book
#   python src/roi_year.py --capital 25000 --hedge 0          # RESULTS 26's

import os, sys, json, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_risk import paths, BLOCK, SEED
import paper_bench as pb
from orders import HEDGE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CENS = 1.5            # RESULTS 15: long-only censoring bias, bps/night
RECENT = '20160101'
H = 252
MIN_LIVE = 20         # live nights needed before the volatility stress runs

# the hedged book's short (header, "The hedged book"); %/yr unless noted
BORROW = 0.40         # QQQ borrow fee, IBKR public feed, 2026-10-12
BM = 3.88             # IBKR benchmark, EFFR 2026-10-09 (leverage.py's BM)
SHORT_SPREAD = 1.25   # IBKR Pro pays BM - 1.25% on short-sale proceeds ...
SHORT_FREE = 100e3    # ... nothing on the first $100k of them ...
SHORT_TOP = 1e6       # ... and the tier above $1M is not credited here
SHORT_NAV = 100e3     # ... scaled by NAV / $100k below that NAV
QQQ_DIV = 0.414       # QQQ trailing-12-month yield to 2026-10-12, owed
DAYS = 365 / 252      # calendar days financed per trading night
REGT = 2.0            # long + short <= 2x equity, IBKR's end-of-day check
REGT_OP = 1.9         # leverage.py's operating line under it


def worlds(q5, alpha):
    """{name: gross nightly log-bps series}, each a mean shift of a real
    series. alpha: the gross selection alpha (RESULTS 21's method)."""
    edge = alpha - CENS                       # the selection alpha that is real
    return {
        'as backtested': q5.values - CENS,
        'half the edge': q5.values - CENS - 0.5 * edge,
        'recent era (2016+)': q5[q5.index >= RECENT].values - CENS,
        'no edge (beta only)': q5.values - alpha,
        'nothing (zero drift)': zero_drift(q5.values),
    }


def hedged(q5, qqq_on, ratio):
    """The hedged book's nights: q5 - ratio x QQQ overnight, in log bps, on
    the days both exist, in date order (at ratio 1, xsec_live's tilt)."""
    j = pd.concat([q5.rename('q5'), qqq_on.rename('qqq')], axis=1).dropna()
    return (j.q5 - ratio * j.qqq).sort_index().rename(q5.name)


def hedge_leg(plan, notional, price, day=None):
    """(shares, $) for one night's hedge: a SELL MOC short and a BUY OPG
    cover of round(notional / price) shares -- nearest, not down, since
    always rounding down under-hedges for good. order_cost keys the Nasdaq
    auction fee by side as if BUY were the MOC and SELL the MOO; for the
    hedge it is the other way round, and the two rates just swap, so the
    pair's total is exact."""
    q = int(round(notional / price)) if notional > 0 and price else 0
    return q, (pb.order_cost(plan, q, price, 'SELL', day) +
               pb.order_cost(plan, q, price, 'BUY', day))


def carry(short, nav, borrow=BORROW, bm=BM, spread=SHORT_SPREAD):
    """$ per trading night a short of `short` dollars carries, in an account
    of `nav`: the borrow fee paid, IBKR's interest on the proceeds received
    (only on the slice it pays on), and QQQ's dividend owed."""
    days = DAYS / 360
    tier = min(max(short - SHORT_FREE, 0.0), SHORT_TOP - SHORT_FREE)
    return dict(borrow=short * borrow / 100 * days,
                credit=tier * max(bm - spread, 0.0) / 100 * days
                * min(1.0, nav / SHORT_NAV),
                div=short * QQQ_DIV / 100 / H)


def live_series(daily, ratio):
    """The live nights the volatility stress is scaled to: the basket's own
    (q5_on_bps) for the long-only book, the hedged night otherwise."""
    if not ratio:
        return daily.q5_on_bps
    if ratio == 1:
        return daily.tilt_bps
    return daily.q5_on_bps - ratio * daily.qqq_on_bps


def zero_drift(r):
    """The series with zero ARITHMETIC mean: removing only the log mean
    would leave sigma^2/2 (~0.6 bps/night) of return from volatility alone,
    and a floor must not earn anything."""
    d = np.asarray(r, float) - np.mean(r)
    return d - 1e4 * np.log(np.exp(d / 1e4).mean())


def scaled(r, k):
    """Scale deviations by k and hold the ARITHMETIC (simple-return) mean
    exactly, so more volatility costs variance drag instead of adding
    return."""
    if k == 1.0:
        return np.asarray(r, float)
    r = np.asarray(r, float)
    mu = r.mean()
    rk = mu + k * (r - mu)
    return rk + 1e4 * np.log(np.exp(r / 1e4).mean() / np.exp(rk / 1e4).mean())


def fan(r, deployed, cost_bps, n, rng, k=1.0, block=BLOCK, h=H,
        compound=False):
    """One year for an account. r: gross nightly log-bps. Fixed notional by
    default (orders.py --capital): P&L in fractions of capital adds up.
    compound=True is for a HELD position (QQQ buy-and-hold)."""
    p = paths(scaled(r, k), n, h, block, rng)
    if compound:
        eq = np.exp(np.cumsum(deployed * p - cost_bps, axis=1) / 1e4)
    else:
        eq = 1 + np.cumsum(deployed * np.expm1(p / 1e4) - cost_bps / 1e4,
                           axis=1)
    eq0 = np.hstack([np.ones((n, 1)), eq])
    ret = eq[:, -1] - 1
    dd = (eq0 / np.maximum.accumulate(eq0, axis=1) - 1).min(axis=1)
    rs = scaled(r, k)
    night = (deployed * np.expm1(rs / 1e4).mean() - cost_bps / 1e4) * 1e4
    return dict(night=night, p5=np.percentile(ret, 5),
                p25=np.percentile(ret, 25), p50=np.percentile(ret, 50),
                p75=np.percentile(ret, 75), p95=np.percentile(ret, 95),
                loss=(ret < 0).mean(), bad=(ret < -0.20).mean(),
                dd50=np.percentile(dd, 50), dd5=np.percentile(dd, 5))


HEAD = (f'{"":<24}{"E[net]":>8}{"p5":>7}{"p25":>7}{"MEDIAN":>8}{"p75":>7}'
        f'{"p95":>7}{"P(loss)":>8}{"P<-20%":>8}{"dip p50":>8}{"dip 1/20":>9}'
        f'{"median $":>10}')


def line(name, s, capital):
    return (f'{name:<24}{s["night"]:>+6.1f}bp' +
            ''.join(f'{s[q] * 100:>+6.0f}%' for q in ('p5', 'p25'))
            + f'{s["p50"] * 100:>+7.0f}%' +
            ''.join(f'{s[q] * 100:>+6.0f}%' for q in ('p75', 'p95')) +
            f'{s["loss"] * 100:>7.0f}%{s["bad"] * 100:>7.0f}%'
            f'{s["dd50"] * 100:>7.0f}%{s["dd5"] * 100:>8.0f}%'
            f'{s["p50"] * capital:>+10,.0f}')


def table(title, ws, bench, deployed, cost_bps, capital, n, k=None):
    print(f'\n{title}\n{HEAD}')
    out = {}
    for name, r in ws.items():
        kk = 1.0 if k is None else k(r)
        out[name] = fan(r, deployed, cost_bps, n, np.random.default_rng(SEED),
                        k=kk)
        print(line(name, out[name], capital))
    print('  -- the same money, instead:')
    for name, (r, dep, cost, comp) in bench.items():
        s = fan(r, dep, cost, n, np.random.default_rng(SEED), compound=comp)
        out[name] = s
        print(line(name, s, capital))
    return out


def load():
    """(q5, factors, cutoff) from the user's own data: the walk-forward Q5
    series and the panel's stitched QQQ/SPY factors, both cut at the frozen
    training cutoff."""
    mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
    if not os.path.exists(mp):
        raise SystemExit('models/xsec_lgbm.json missing -- cannot locate the '
                         'frozen cutoff, and without it the forward months '
                         'would leak into the history')
    cut = json.load(open(mp))['last_tmonth'].replace('-', '')[:6]
    ml = pd.read_csv(os.path.join(DATA, 'xsec_ml_daily.csv'),
                     dtype={'day': str}).set_index('day')
    q5 = ml.mlon_q5.dropna()
    from xsec_backtest import load_panel
    from factor import factors
    F = factors(load_panel())
    return q5[q5.index.str[:6] <= cut], F[F.index.str[:6] <= cut], cut


def prices(want, path=None):
    if path:
        px = dict(pd.read_csv(path, header=None).values)
    else:
        import yfinance as yf
        v = yf.download([x.replace('.', '-') for x in want], period='5d',
                        interval='1d', auto_adjust=False, actions=False,
                        group_by='ticker', progress=False)
        px = {}
        for x in want:
            try:
                s = v[x.replace('.', '-')]['Close'].dropna()
                if len(s):
                    px[x] = float(s.iloc[-1])
            except KeyError:
                pass
    missing = [x for x in want if not px.get(x)]
    if missing:
        raise SystemExit(f'no price for {missing} -- a missing price would '
                         f'shrink the book for a whole simulated year; retry, '
                         f'or pass --prices')
    return px


def hedge_book(plan, book, px, capital, ratio, day=None, **rates):
    """The hedge on a sized long book, $ per night: QQQ shares short against
    ratio x the long notional actually bought (not --capital), the hedge
    leg's commissions and fees, and the short's carry (carry()'s keys)."""
    long_ = float(book.notional.sum())
    q, leg = hedge_leg(plan, ratio * long_, px['QQQ'], day)
    short = q * px['QQQ']
    return dict(long=long_, shares=q, short=short, leg=leg,
                gross=(long_ + short) / capital,
                **carry(short, capital, **rates))


def run(q5, F, cut, names, px, live, capital, plan, n, hedge=0.0):
    """hedge: the QQQ short as a multiple of the long notional (0 = the
    long-only book, its output unchanged)."""
    from factor import regress
    if hedge < 0:
        raise SystemExit('--hedge is the short QQQ multiple: 0 or more')
    y = hedged(q5, F.mkt_on, hedge) if hedge else q5
    cols = ['mkt_on'] + (['spy_on'] if 'spy_on' in F else [])
    j = pd.concat([y.rename('y'), F[cols]], axis=1).dropna()
    alpha = float(regress(j.y, j[cols])['alpha'])
    qqq_cc = F.mkt_cc.dropna()
    qqq_cc = qqq_cc[qqq_cc.index >= y.index.min()]
    qqq_on = F.mkt_on.dropna()
    qqq_on = qqq_on[qqq_on.index >= y.index.min()]
    book, _ = pb.sizing(names, px, capital)
    deployed = book.notional.sum() / capital
    if deployed <= 0:
        raise SystemExit('nothing deployable at this capital')
    cost = pb.night_cost(plan, book) / capital * 1e4
    qbook, _ = pb.sizing(['QQQ'], px, capital)
    q_dep = qbook.notional.sum() / capital
    q_cost = pb.night_cost(plan, qbook) / capital * 1e4
    if hedge:
        h = hedge_book(plan, book, px, capital, hedge)
        if h['gross'] > REGT:
            raise SystemExit(f'long + short = {h["gross"]:.2f}x capital, over '
                             f'Reg T\'s {REGT:.0f}x end-of-day line -- lower '
                             f'--hedge')
        bps = lambda usd: usd / capital * 1e4
        leg = bps(h['leg'])
        fin = bps(h['borrow'] - h['credit'] + h['div'])
        basket = cost
        cost = basket + leg + fin

    print('THE YEAR AFTER A GO VERDICT -- not this year. The project\'s rule is '
          'no live money until\nthe forward test passes (RESULTS 19/25: first '
          'weak read ~2026-11, high-power ~2027-08).\nToday: $0 live; run paper '
          'at the size you intend to trade.\n')
    book_is = ('long the frozen model\'s Q5 overnight' if not hedge else
               f'long the frozen model\'s Q5 overnight, short {hedge:g}x its '
               f'notional in QQQ overnight')
    print(f'traded book: {book_is}, '
          f'{y.index.min()}..{y.index.max()} ({len(y):,} nights; history '
          f'stops at the {cut[:4]}-{cut[4:]} training cutoff)')
    print(f'  gross {y.mean():+.2f} bps/night; selection alpha {alpha:+.2f} '
          f'(regressed on {"+".join(cols)}, RESULTS 21\'s method); censoring '
          f'-{CENS}')
    if not hedge:
        print(f'  account: ${capital:,.0f} on {plan}, {deployed:.0%} deployed '
              f'(whole shares), costs {cost:.2f} bps/night = '
              f'${pb.night_cost(plan, book) * H:,.0f}/yr, a certain '
              f'-{cost * H / 100:.1f}%/yr before any return')
    else:
        print(f'  account: ${capital:,.0f} on {plan}, {deployed:.0%} deployed '
              f'long (whole shares), short {h["shares"]} QQQ = '
              f'${h["short"]:,.0f} ({h["short"] / h["long"]:.3f} of the long)')
        print(f'  costs {basket + leg:.2f} bps/night (basket {basket:.2f}, QQQ '
              f'leg {leg:.2f}) + carry {fin:+.2f} (borrow '
              f'{bps(h["borrow"]):+.2f}, short-proceeds credit '
              f'{-bps(h["credit"]) + 0.0:+.2f}, QQQ dividend owed '
              f'{bps(h["div"]):+.2f}) = {cost:.2f} bps/night = '
              f'${cost / 1e4 * capital * H:,.0f}/yr, a certain '
              f'{-cost * H / 100:+.1f}%/yr before any return')
        print(f'  Reg T: long + short = {h["gross"]:.2f}x capital (limit '
              f'{REGT:.0f}x at IBKR\'s end-of-day check; a stock basket does '
              f'not offset QQQ) -> hold >= '
              f'${(h["long"] + h["short"]) / REGT_OP:,.0f} of equity '
              f'({REGT_OP}x), in a margin account (not cash, not IRA)')

    ws = worlds(y, alpha)
    bench = {'QQQ, bought and held': (qqq_cc.values, 1.0, 0.0, True),
             'QQQ, overnight only': (qqq_on.values, q_dep, q_cost, False)}
    out = table(f'ONE YEAR ({H} nights), backtest volatility, 21-day blocks; '
                f'returns on ${capital:,.0f}', ws, bench, deployed, cost,
                capital, n)

    print('\n  tail check, as backtested: longer blocks keep months-long '
          'regimes together')
    for b in (21, 63, 126):
        s = fan(ws['as backtested'], deployed, cost, n,
                np.random.default_rng(SEED), block=b)
        print(f'    {b:>3}-day blocks: p5 {s["p5"] * 100:+.0f}%   P(<-20%) '
              f'{s["bad"] * 100:.0f}%   1-in-20 dip {s["dd5"] * 100:.0f}%')

    live = pd.Series(live).dropna()
    if len(live) < MIN_LIVE:
        print(f'\n(only {len(live)} live nights -- no volatility stress yet)')
    else:
        lsd = float(live.std())
        rng = np.random.default_rng(SEED)
        boot = [rng.choice(live.values, len(live)).std() for _ in range(2000)]
        lo, hi = np.percentile(boot, [5, 95])
        whose = ('LIVE basket\'s' if not hedge else
                 f'LIVE hedged book\'s (q5 - {hedge:g} x QQQ)')
        table(f'ONE YEAR at the {whose} volatility ({len(live)} '
              f'nights: sd {lsd:.0f} bps, 90% CI [{lo:.0f}, {hi:.0f}]) -- a '
              f'stress, not a forecast', ws, bench, deployed, cost,
              capital, n, k=lambda r: max(1.0, lsd / float(np.std(r))))
        if not hedge:
            print('  (each basket row is widened to the live sd; the QQQ rows '
                  'keep their own volatility,\n   so the basket\'s extra beta '
                  'shows up as risk the alternatives do not carry)')
        else:
            print('  (each hedged row is widened to the live hedged sd; the '
                  'QQQ rows keep their own volatility,\n   so what the hedge '
                  'leaves -- the basket\'s one-theme risk and its beta above '
                  'the\n   hedge -- shows up as risk the alternatives do not '
                  'carry)')
    print('\nThe forward test decides which ROW is true. Until it does, the '
          'honest answer is the\nspread between the rows -- and whether any '
          'row beats the two QQQ lines.')
    return out


def args(argv=None):
    ap = argparse.ArgumentParser(description='One-year ROI fan for the '
                                 'traded book (the QQQ-hedged one) at a '
                                 'capital and plan, next to holding QQQ; '
                                 '--hedge 0 for the former long-only book.')
    ap.add_argument('--capital', type=float, default=25000)
    ap.add_argument('--plan', choices=['fixed', 'tiered'], default='tiered')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--prices', default=None,
                    help='offline: CSV of SYM,price instead of Yahoo')
    ap.add_argument('--hedge', type=float, default=HEDGE[1], metavar='RATIO',
                    help='short RATIO x the long notional in QQQ overnight; '
                         'default: the traded book\'s declared ratio '
                         '(orders.HEDGE, 1 = the 1:1 dollar hedge, RESULTS 21 '
                         'NEU); 0 = the former long-only book (RESULTS 26)')
    return ap.parse_args(argv)


def main():
    a = args()
    if a.hedge < 0:
        raise SystemExit('--hedge is the short QQQ multiple: 0 or more')
    q5, F, cut = load()
    names = pd.read_csv(pb.PAPER, dtype=str).iloc[-1].q5.split(';')
    px = prices(names + ['QQQ'], a.prices)
    live = live_series(pd.read_csv(pb.DAILY, dtype={'date': str}), a.hedge)
    run(q5, F, cut, names, px, live, a.capital, a.plan, a.paths, a.hedge)


if __name__ == '__main__':
    main()
