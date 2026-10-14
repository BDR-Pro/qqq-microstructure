# Part of qqq-microstructure.
#
# Non-binding readouts of the paper log, declared 2026-10-14 (RESULTS 29)
# before the nights they will read. The forward test's verdict is RESULTS 19's
# replay and nothing here changes it. These answer a narrower question: how
# much does each night of the paper log tell us, and can it tell us more?
#
# A night's information is capped by its noise. The live basket is one theme,
# eleven times, with an overnight beta to QQQ near 2.4 (RESULTS 19), so most
# of a raw night is the market. Subtracting the market's share with a beta
# that was NOT fitted on the nights being read removes noise without adding
# information leakage (a control variate). Two such betas:
#
#   fixed      1.42, the ON book's total overnight beta in the backtest
#              (RESULTS 21: 0.69 on QQQ + 0.73 on SPY; QQQ stands in for both
#              here, the paper log carries QQQ only). Declared today.
#   trailing   the OLS slope of the basket on QQQ over the previous 60 graded
#              nights, strictly before the night it adjusts; none until 20
#              prior nights exist. RESULTS 19 proposed this one.
#
# A residual is a DIFFERENT quantity from the tilt: it estimates the selection
# return beyond what the market explains, so it is compared with the
# backtest's alpha, not with the tilt.
#
# Per readout: nights, mean, sd, t, and the nights a t of 2 would need at its
# backtest reference edge (after RESULTS 15's 1.5 bps censoring):
#   raw        the basket            +10.50 gross (RESULTS 26)  -> +9.00
#   tilt       basket - QQQ           +6.40 gross (RESULTS 19)  -> +4.90
#   residuals  basket - beta * QQQ    +6.13 gross alpha (RESULTS 26 / 21's
#                                      method)                   -> +4.63
#
#   python src/xsec_readout.py         # prints, and writes reports/xsec_readout.csv

import os, datetime as dt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, 'reports', 'xsec_paper_daily.csv')
OUT = os.path.join(ROOT, 'reports', 'xsec_readout.csv')
BETA_FIXED = 1.42              # declared 2026-10-14, RESULTS 21
TRAIL, TRAIL_MIN = 60, 20
CENS = 1.5
REF = {'raw': 10.50 - CENS, 'tilt': 6.40 - CENS,
       'resid_fixed': 6.13 - CENS, 'resid_trailing': 6.13 - CENS}


def trailing_beta(q5, qqq, n=TRAIL, min_n=TRAIL_MIN):
    """beta_t from nights strictly before t (never the night itself)."""
    out = np.full(len(q5), np.nan)
    for t in range(len(q5)):
        lo = max(0, t - n)
        x, y = qqq[lo:t], q5[lo:t]
        if t - lo >= min_n and np.var(x) > 0:
            out[t] = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    return out


def series(daily):
    d = daily.sort_values('date')
    q5 = d.q5_on_bps.astype(float).to_numpy()
    q = d.qqq_on_bps.astype(float).to_numpy()
    b = trailing_beta(q5, q)
    return {'raw': q5, 'tilt': q5 - q, 'resid_fixed': q5 - BETA_FIXED * q,
            'resid_trailing': (q5 - b * q)[~np.isnan(b)]}


def stats(x, ref):
    n = len(x)
    if n < 2:
        return dict(nights=n, mean=np.nan, sd=np.nan, t=np.nan,
                    nights_for_t2=np.nan)
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    return dict(nights=n, mean=m, sd=sd, t=m / (sd / np.sqrt(n)),
                nights_for_t2=(2 * sd / ref) ** 2)


def readout(daily, run_date=None):
    run_date = run_date or dt.date.today().isoformat()
    rows = [dict(run_date=run_date, readout=k, ref_edge=REF[k],
                 **stats(x, REF[k])) for k, x in series(daily).items()]
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(DAILY):
        raise SystemExit('no reports/xsec_paper_daily.csv yet -- nothing to read')
    r = readout(pd.read_csv(DAILY, dtype={'date': str}))
    print('NON-BINDING readouts of the paper log (the verdict is RESULTS 19\'s '
          'replay):')
    print(f'  {"readout":<16}{"nights":>7}{"mean":>8}{"sd":>7}{"t":>7}'
          f'{"ref edge":>10}{"nights for t=2":>16}')
    for x in r.itertuples():
        print(f'  {x.readout:<16}{x.nights:>7}{x.mean:>+8.1f}{x.sd:>7.0f}'
              f'{x.t:>+7.2f}{x.ref_edge:>+10.2f}{x.nights_for_t2:>16,.0f}')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.to_csv(OUT, index=False, float_format='%.4f')
    print(f'-> {OUT} (commit with reports/)')


if __name__ == '__main__':
    main()
