# Part of qqq-microstructure.
#
# The question this file answers, verbatim, because it is the right one:
#
#   "Given the returns I've actually measured, what range of outcomes could
#    reasonably happen, and what future result would convince me that the
#    backtest's edge is real?"
#
# Monte Carlo is used here for exactly two things it is good at, and nothing
# it is not: no simulated prices, no strategy search, no invented dynamics.
# Every path below is a circular BLOCK bootstrap of the measured daily series
# (21-day blocks, preserving volatility clustering and within-month
# autocorrelation), so the simulation contains only what the data contains.
#
#   A. THE RANGE. For each book configuration and leverage, 10,000 alternate
#      five-year paths: the distribution of CAGR, max drawdown, and bad months
#      that the measured return process could produce. History gave one path;
#      this is the fan around it. Financing at --rate on borrowed fraction.
#
#   B. THE VERDICT, PRE-REGISTERED. For the two forward metrics being
#      accumulated (the replay's ON long/short, and the paper log's Q5-minus-
#      QQQ tilt), thresholds computed BEFORE the forward months arrive:
#        GO   = the 95th percentile of the no-edge world (same series,
#               demeaned): a forward mean above it is <5% likely if the edge
#               is zero.
#        KILL = the 5th percentile of the as-measured world: a forward mean
#               below it is <5% likely if the backtest's edge is real.
#      Plus the power of each test and the horizon at which the two worlds
#      separate. Commit this output; when the months land, the verdict is
#      mechanical.
#
# Inputs are the CSVs the other scripts already write (run stack_v2.py,
# xsec_ml.py and xsec_backtest.py first): data/stack_daily.csv,
# data/xsec_ml_daily.csv, data/xsec_daily.csv. Deterministic (seed 7).
#
#   python src/mc_risk.py [--rate 5.0] [--paths 10000]

import os, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCK = 21
SEED = 7
HORIZON_A = 1260                      # five years for the range
HORIZONS_B = [63, 126, 252, 504]      # 3m / 6m / 12m / 24m for the verdict


def paths(r, n, h, block, rng):
    """Circular block bootstrap: n paths of length h from measured series r."""
    r = np.asarray(r, float)
    nb = int(np.ceil(h / block))
    start = rng.integers(0, len(r), size=(n, nb))
    take = (start[:, :, None] + np.arange(block)[None, None, :]) % len(r)
    return r[take].reshape(n, -1)[:, :h]


def range_table(name, r, n, rate, rng):
    p = paths(r, n, HORIZON_A, BLOCK, rng)
    print(f'\n{name}: {len(r):,} measured days -> {n:,} five-year paths')
    print(f'{"lev":>5}{"CAGR p5":>9}{"p50":>7}{"p95":>7}{"maxDD p50":>11}'
          f'{"p95":>7}{"worst mo p95":>14}{"P(mo<-20%)":>12}{"P(5y loss)":>12}')
    for lev in (1.0, 1.5, 2.0, 3.0):
        d = lev * p - (lev - 1) * rate / 252 * 100
        eq = np.cumprod(1 + d / 1e4, axis=1)
        cagr = eq[:, -1] ** (252 / HORIZON_A) - 1
        dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
        mo = d[:, :HORIZON_A - HORIZON_A % 21].reshape(len(d), -1, 21).sum(2) / 1e4
        wmo = mo.min(axis=1)
        print(f'{lev:>5.1f}{np.percentile(cagr, 5)*100:>8.1f}%'
              f'{np.percentile(cagr, 50)*100:>6.1f}%'
              f'{np.percentile(cagr, 95)*100:>6.1f}%'
              f'{np.percentile(dd, 50)*100:>10.1f}%'
              f'{np.percentile(dd, 5)*100:>6.1f}%'
              f'{np.percentile(wmo, 5)*100:>13.1f}%'
              f'{(mo < -0.20).any(axis=1).mean()*100:>11.1f}%'
              f'{(cagr < 0).mean()*100:>11.1f}%')


def verdict_table(name, r, n, rng):
    r = np.asarray(r, float)
    r0 = r - r.mean()
    print(f'\n{name}: measured {r.mean():+.1f} bps/day over {len(r):,} days')
    print(f'{"horizon":>9}{"GO >":>8}{"P(pass|real)":>14}{"KILL <":>9}'
          f'{"P(kill|none)":>14}   (mean bps/day over the window)')
    sep = None
    for h in HORIZONS_B:
        m1 = paths(r, n, h, BLOCK, rng).mean(axis=1)
        m0 = paths(r0, n, h, BLOCK, rng).mean(axis=1)
        go, kill = np.percentile(m0, 95), np.percentile(m1, 5)
        print(f'{h//21:>7}mo{go:>8.1f}{(m1 > go).mean()*100:>13.0f}%'
              f'{kill:>9.1f}{(m0 < kill).mean()*100:>13.0f}%')
        if sep is None and go < kill:
            sep = h
    print(f'  worlds fully separate (GO < KILL) at: '
          + (f'{sep//21} months' if sep else
             f'beyond {HORIZONS_B[-1]//21} months -- until then a result '
             f'between KILL and GO is simply more waiting'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rate', type=float, default=5.0,
                    help='annual %% financing on the levered fraction')
    ap.add_argument('--paths', type=int, default=10000)
    ap.add_argument('--allow-forward', action='store_true',
                    help='proceed even if the input series extends past the '
                         'frozen training cutoff (thresholds would then '
                         'contain forward months -- normally a mistake)')
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = os.path.join(ROOT, 'data')

    sp = os.path.join(d, 'stack_daily.csv')
    if not os.path.exists(sp):
        raise SystemExit('missing data/stack_daily.csv -- run stack_v2.py first')
    L = pd.read_csv(sp, dtype={'day': str}).set_index('day')

    print('=' * 22 + ' A. THE RANGE ' + '=' * 22)
    print(f'(block bootstrap of measured dailies, {BLOCK}-day blocks, '
          f'financing {a.rate}%/yr on leverage)')
    for name, cols in (('v2  (ON+MOM)', ['ON', 'MOM']),
                       ('v2n (NEU+MOM)', ['NEU', 'MOM']),
                       ('v2o (ON+MOM+OPT)', ['ON', 'MOM', 'OPT'])):
        if all(c in L for c in cols):
            r = L[cols].dropna().sum(axis=1).values
            if len(r) > 500:
                range_table(name, r, a.paths, a.rate, rng)

    print('\n' + '=' * 20 + ' B. THE VERDICT ' + '=' * 20)
    print('(thresholds pre-registered before the forward months arrive; '
          'commit this output)')
    ml = pd.read_csv(os.path.join(d, 'xsec_ml_daily.csv'),
                     dtype={'day': str}).set_index('day')
    # the thresholds are a PRE-REGISTRATION: they must be computed from the
    # series that ends at the frozen training cutoff, never from one that a
    # routine re-run of xsec_ml.py after xsec_extend has extended (audit 2026-09)
    mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
    if os.path.exists(mp):
        import json
        cut = json.load(open(mp)).get('last_tmonth', '9999-99').replace('-', '')
        last = ml.mlon_ls.dropna().index.max()
        if last[:6] > cut[:6]:
            msg = (f'xsec_ml_daily.csv runs to {last} but the frozen models '
                   f'were trained through {cut[:4]}-{cut[4:6]}: the input '
                   f'contains forward months. Re-run xsec_ml.py --through '
                   f'{cut[:4]}-{cut[4:6]} first.')
            if not a.allow_forward:
                raise SystemExit('REFUSED: ' + msg + '  (--allow-forward overrides)')
            print('WARNING: ' + msg)
    verdict_table('replay metric: ON long/short (gross)',
                  ml.mlon_ls.dropna().values, a.paths, rng)
    bt = pd.read_csv(os.path.join(d, 'xsec_daily.csv'),
                     dtype={'day': str}).set_index('day')
    tilt = (ml.mlon_q5 - bt.qqq_on).dropna()
    verdict_table('paper-log metric: Q5 minus QQQ overnight (gross)',
                  tilt.values, a.paths, rng)

    print('\nThe question all of this answers: given the returns actually '
          'measured, the tables\nabove are the range of outcomes that could '
          'reasonably happen -- and a forward mean\nabove GO is the future '
          'result that would make the edge hard to dismiss, while one\nbelow '
          'KILL is the result that would make the backtest hard to believe.')


if __name__ == '__main__':
    main()
