# Part of qqq-microstructure.
#
# "So in a year, what does the ROI look like?"
#
# For the book orders.py actually trades -- long the frozen model's Q5 basket
# overnight (buy MOC, sell MOO), unhedged, nothing else -- at a given capital
# and commission plan. Not one number: which world we are in is exactly what
# the forward test (RESULTS 19) has not decided yet, so the answer is a fan per
# world.
#
#   as backtested     the walk-forward Q5 series, less RESULTS 15's measured
#                     censoring bias (1.5 bps/night)
#   half the edge     the selection alpha halved -- the order of decay the
#                     literature finds after an anomaly is published
#   recent era        only the days since 2016, the post-publication regime,
#                     resampled as they happened
#   no edge           the selection alpha removed entirely: what is left is
#                     the overnight-market beta the basket carries anyway --
#                     still the historical overnight premium, so NOT a floor
#   nothing           zero gross drift: the costs and the volatility alone.
#                     This is the floor.
#
# Each world is a circular block bootstrap (mc_risk.paths, 21-day blocks) of
# real measured nights, mean-shifted -- no invented dynamics. On top, what an
# account does to it: only the deployed fraction earns (whole shares), and the
# nightly commissions and fees come from paper_bench's IBKR model at this
# capital and plan. The whole table is then repeated at the volatility the
# live basket has actually shown, which so far is far higher than the
# backtest's (RESULTS 19).
#
# History stops at the frozen training cutoff: the forward months are the
# test, not the input.
#
#   python src/roi_year.py --capital 25000 --plan tiered
#   python src/roi_year.py --capital 100000 --plan tiered --prices p.csv

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


def alpha_beta(q5, qqq):
    """OLS of the gross Q5 night on the QQQ overnight: (alpha, beta)."""
    j = pd.concat([q5, qqq], axis=1, join='inner').dropna()
    X = np.column_stack([np.ones(len(j)), j.iloc[:, 1].values])
    a, b = np.linalg.lstsq(X, j.iloc[:, 0].values, rcond=None)[0]
    return float(a), float(b)


def worlds(q5, alpha):
    """{name: gross nightly series (bps)}, each a mean-shifted real series."""
    edge = alpha - CENS                       # the selection alpha that is real
    return {
        'as backtested': q5.values - CENS,
        'half the edge': q5.values - CENS - 0.5 * edge,
        'recent era (2016+)': q5[q5.index >= RECENT].values - CENS,
        'no edge (beta only)': q5.values - alpha,
        'nothing (zero drift)': q5.values - q5.values.mean(),
    }


def fan(r, deployed, cost_bps, k, n, rng, h=H):
    """One-year outcomes for an account: deployed x (resampled night,
    volatility scaled by k about its mean) - nightly cost in bps of capital."""
    mu = r.mean()
    p = paths(mu + k * (r - mu), n, h, BLOCK, rng)
    net = deployed * p - cost_bps
    eq = np.exp(np.cumsum(net, axis=1) / 1e4)
    ret = eq[:, -1] - 1
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    return dict(night=deployed * mu - cost_bps, p5=np.percentile(ret, 5),
                p25=np.percentile(ret, 25), p50=np.percentile(ret, 50),
                p75=np.percentile(ret, 75), p95=np.percentile(ret, 95),
                loss=(ret < 0).mean(), bad=(ret < -0.20).mean(),
                dd50=np.percentile(dd, 50), dd5=np.percentile(dd, 5))


def table(title, ws, deployed, cost_bps, k, n, seed, capital):
    print(f'\n{title}')
    print(f'{"world":<22}{"net/night":>10}{"p5":>8}{"p25":>8}{"MEDIAN":>8}'
          f'{"p75":>8}{"p95":>8}{"P(loss)":>9}{"P(<-20%)":>9}'
          f'{"maxDD p50":>10}{"bad 5%":>8}')
    out = {}
    for name, r in ws.items():
        s = fan(r, deployed, cost_bps, k, n, np.random.default_rng(seed))
        out[name] = s
        print(f'{name:<22}{s["night"]:>+8.2f}bp' +
              ''.join(f'{s[q] * 100:>+7.0f}%' for q in
                      ('p5', 'p25', 'p50', 'p75', 'p95')) +
              f'{s["loss"] * 100:>8.0f}%{s["bad"] * 100:>8.0f}%'
              f'{s["dd50"] * 100:>9.0f}%{s["dd5"] * 100:>7.0f}%')
    m = out['as backtested']
    print(f'  in dollars on ${capital:,.0f}, as backtested: median '
          f'{m["p50"] * capital:+,.0f}, bad 5% {m["p5"] * capital:+,.0f}, '
          f'good 5% {m["p95"] * capital:+,.0f}')
    return out


def main():
    ap = argparse.ArgumentParser(description='One-year ROI fan for the '
                                 'traded book at a capital and plan.')
    ap.add_argument('--capital', type=float, default=25000)
    ap.add_argument('--plan', choices=['fixed', 'tiered'], default='tiered')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--prices', default=None,
                    help='offline: CSV of SYM,price instead of Yahoo')
    a = ap.parse_args()

    ml = pd.read_csv(os.path.join(DATA, 'xsec_ml_daily.csv'),
                     dtype={'day': str}).set_index('day')
    bt = pd.read_csv(os.path.join(DATA, 'xsec_daily.csv'),
                     dtype={'day': str}).set_index('day')
    q5 = ml.mlon_q5.dropna()
    mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
    cut = json.load(open(mp)).get('last_tmonth', '9999-99').replace('-', '') \
        if os.path.exists(mp) else '999999'
    q5 = q5[q5.index.str[:6] <= cut[:6]]
    alpha, beta = alpha_beta(q5, bt.qqq_on)

    paper = pd.read_csv(pb.PAPER, dtype=str)
    names = paper.iloc[-1].q5.split(';')
    if a.prices:
        px = dict(pd.read_csv(a.prices, header=None).values)
    else:
        import yfinance as yf
        v = yf.download([x.replace('.', '-') for x in names], period='5d',
                        interval='1d', auto_adjust=False, actions=False,
                        group_by='ticker', progress=False)
        px = {}
        for x in names:
            try:
                s = v[x.replace('.', '-')]['Close'].dropna()
                if len(s):
                    px[x] = float(s.iloc[-1])
            except KeyError:
                pass
    book, _ = pb.sizing(names, px, a.capital)
    deployed = book.notional.sum() / a.capital
    cost = pb.night_cost(a.plan, book) / a.capital * 1e4

    live = pd.read_csv(pb.DAILY, dtype={'date': str})
    k_live = float(live.q5_on_bps.std() / q5[q5.index >= RECENT].std())

    print(f'traded book: long the frozen model\'s Q5 overnight, '
          f'{q5.index.min()}..{q5.index.max()} ({len(q5):,} nights, history '
          f'ends at the {cut[:4]}-{cut[4:6]} training cutoff)')
    print(f'  gross {q5.mean():+.2f} bps/night = selection alpha {alpha:+.2f} '
          f'+ {beta:.2f} x QQQ overnight; censoring correction -{CENS}')
    print(f'  account: ${a.capital:,.0f}, {a.plan}, {deployed:.0%} deployed '
          f'(whole shares), costs {cost:.2f} bps/night '
          f'(${pb.night_cost(a.plan, book) * H:,.0f}/yr)')
    ws = worlds(q5, alpha)
    table(f'ONE YEAR ({H} nights), at the backtest\'s own volatility',
          ws, deployed, cost, 1.0, a.paths, SEED, a.capital)
    table(f'ONE YEAR at the LIVE basket\'s volatility so far ({k_live:.1f}x '
          f'the 2016+ backtest; {len(live)} nights -- a stress, not a '
          f'forecast)', ws, deployed, cost, k_live, a.paths, SEED, a.capital)
    print('\nRead it this way: the forward test decides which ROW you are in. '
          'Until it does,\nthe honest one-year answer is the spread between '
          'the rows, not any one median.\nNot modelled: taxes (short-term '
          'gains), interest on idle cash, dividends (live\nreceives them; '
          'they would add), and any execution failure (a missed MOC night).')


if __name__ == '__main__':
    main()
