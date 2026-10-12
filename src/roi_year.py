# Part of qqq-microstructure.
#
# "So in a year, what does the ROI look like?"
#
# For the book orders.py actually trades -- long the frozen model's Q5 basket
# overnight (buy MOC, sell MOO), unhedged, nothing else -- at a given capital
# and commission plan, NEXT TO the two things the same money could do instead:
# hold QQQ, or hold QQQ overnight only.
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
#   python src/roi_year.py --capital 25000 --plan tiered

import os, sys, json, argparse
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mc_risk import paths, BLOCK, SEED
import paper_bench as pb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
CENS = 1.5            # RESULTS 15: long-only censoring bias, bps/night
RECENT = '20160101'
H = 252
MIN_LIVE = 20         # live nights needed before the volatility stress runs


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


def run(q5, F, cut, names, px, live, capital, plan, n):
    from factor import regress
    cols = ['mkt_on'] + (['spy_on'] if 'spy_on' in F else [])
    j = pd.concat([q5.rename('y'), F[cols]], axis=1).dropna()
    alpha = float(regress(j.y, j[cols])['alpha'])
    qqq_cc = F.mkt_cc.dropna()
    qqq_cc = qqq_cc[qqq_cc.index >= q5.index.min()]
    qqq_on = F.mkt_on.dropna()
    qqq_on = qqq_on[qqq_on.index >= q5.index.min()]
    book, _ = pb.sizing(names, px, capital)
    deployed = book.notional.sum() / capital
    if deployed <= 0:
        raise SystemExit('nothing deployable at this capital')
    cost = pb.night_cost(plan, book) / capital * 1e4
    qbook, _ = pb.sizing(['QQQ'], px, capital)
    q_dep = qbook.notional.sum() / capital
    q_cost = pb.night_cost(plan, qbook) / capital * 1e4

    print('THE YEAR AFTER A GO VERDICT -- not this year. The project\'s rule is '
          'no live money until\nthe forward test passes (RESULTS 19/25: first '
          'weak read ~2026-11, high-power ~2027-08).\nToday: $0 live; run paper '
          'at the size you intend to trade.\n')
    print(f'traded book: long the frozen model\'s Q5 overnight, '
          f'{q5.index.min()}..{q5.index.max()} ({len(q5):,} nights; history '
          f'stops at the {cut[:4]}-{cut[4:]} training cutoff)')
    print(f'  gross {q5.mean():+.2f} bps/night; selection alpha {alpha:+.2f} '
          f'(regressed on {"+".join(cols)}, RESULTS 21\'s method); censoring '
          f'-{CENS}')
    print(f'  account: ${capital:,.0f} on {plan}, {deployed:.0%} deployed '
          f'(whole shares), costs {cost:.2f} bps/night = '
          f'${pb.night_cost(plan, book) * H:,.0f}/yr, a certain '
          f'-{cost * H / 100:.1f}%/yr before any return')

    ws = worlds(q5, alpha)
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
        table(f'ONE YEAR at the LIVE basket\'s volatility ({len(live)} '
              f'nights: sd {lsd:.0f} bps, 90% CI [{lo:.0f}, {hi:.0f}]) -- a '
              f'stress, not a forecast', ws, bench, deployed, cost,
              capital, n, k=lambda r: max(1.0, lsd / float(np.std(r))))
        print('  (each basket row is widened to the live sd; the QQQ rows keep '
              'their own volatility,\n   so the basket\'s extra beta shows up '
              'as risk the alternatives do not carry)')
    print('\nThe forward test decides which ROW is true. Until it does, the '
          'honest answer is the\nspread between the rows -- and whether any '
          'row beats the two QQQ lines.')
    return out


def main():
    ap = argparse.ArgumentParser(description='One-year ROI fan for the '
                                 'traded book at a capital and plan, next to '
                                 'holding QQQ.')
    ap.add_argument('--capital', type=float, default=25000)
    ap.add_argument('--plan', choices=['fixed', 'tiered'], default='tiered')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--prices', default=None,
                    help='offline: CSV of SYM,price instead of Yahoo')
    a = ap.parse_args()
    q5, F, cut = load()
    names = pd.read_csv(pb.PAPER, dtype=str).iloc[-1].q5.split(';')
    px = prices(names + ['QQQ'], a.prices)
    live = pd.read_csv(pb.DAILY, dtype={'date': str}).q5_on_bps
    run(q5, F, cut, names, px, live, a.capital, a.plan, a.paths)


if __name__ == '__main__':
    main()
