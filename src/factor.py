# Part of qqq-microstructure.
#
# Phase 14, the question that decides whether any of this is worth trading:
# is ON+MOM REAL ALPHA, or QQQ/SPY beta in a costume? Every strategy series
# is regressed on the market factors it could be secretly replicating, and
# what survives the regression -- the intercept -- is the only return that is
# actually yours.
#
# Factors, built fresh from the panel (not from the cost-netted stack series),
# each in the SAME session the strategy trades so a real exposure cannot hide
# in a timing mismatch:
#   mkt_on   = QQQ overnight  log(open_t / close_{t-1})   -- the overnight book's factor
#   spy_on   = SPY overnight                              -- broad-market overnight
#   mkt_cc   = QQQ close-to-close                         -- the intraday/day book's factor
#   mkt_id   = QQQ intraday   log(close_t / open_t)       -- MOM's factor
#
# Reported per strategy: alpha (bps/day and %/mo), its t-stat with NEWEY-WEST
# HAC standard errors (lag 5 -- returns here are autocorrelated and a plain
# OLS t overstates significance, exactly the Phase 12 warning), beta on each
# factor, R^2, and the residual Sharpe (the Sharpe of the factor-hedged
# stream: alpha plus residual, the intercept kept). Two built-in
# falsification checks print first, both true by the non-overlapping-session
# structure and NOT by any fitted parameter:
#   - an overnight book (ON) regressed on QQQ INTRADAY must show beta ~ 0,
#   - the intraday book (MOM) regressed on QQQ OVERNIGHT must show beta ~ 0.
# If either fires a real beta, one session is leaking into the other -- a
# look-ahead. (NEU's beta on mkt_on is an OUTPUT to read, not a check: it is
# ON minus a 1.0-unit QQQ hedge, so its residual beta is exactly RESULTS 20's
# under-hedge finding, not zero.)
# Validated on planted truth before real data: a series built as
# 1.4*mkt_on + 6bps + noise recovers beta 1.40 and alpha 6.0 with the planted
# residual Sharpe, and the HAC t on an AR(1) series is correctly smaller than
# the OLS t (see RESULTS).
#
# Needs data/stack_daily.csv (run stack_v2.py) and the panel (xsec_backtest).
#
#   python src/factor.py

import os
import numpy as np, pandas as pd
from xsec_backtest import load_panel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NW_LAG = 5


def nw_se(X, resid):
    """Newey-West HAC covariance of OLS coefficients (Bartlett, lag NW_LAG)."""
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])           # lag 0
    for L in range(1, NW_LAG + 1):
        w = 1 - L / (NW_LAG + 1)
        Xu = X * resid[:, None]
        G = Xu[L:].T @ Xu[:-L]
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv


def regress(y, F):
    """OLS of y on [const, F]; return alpha, betas, HAC t of alpha, R2, and
    residual Sharpe = Sharpe of the factor-hedged stream (alpha + residual,
    i.e. y with only the factor exposure removed -- the intercept is KEPT,
    or the 'residual Sharpe' is trivially zero). y/F in bps/day, aligned."""
    X = np.column_stack([np.ones(len(y)), F.values])
    b, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    resid = y.values - X @ b
    cov = nw_se(X, resid)
    t_alpha = b[0] / np.sqrt(cov[0, 0])
    ss = 1 - resid.var() / y.values.var()
    hedged = y.values - X[:, 1:] @ b[1:]          # alpha + resid, factors out
    rsharpe = hedged.mean() / hedged.std() * np.sqrt(252)
    return dict(alpha=b[0], t_alpha=t_alpha,
                betas=dict(zip(F.columns, b[1:])), r2=ss,
                resid_sharpe=rsharpe, n=len(y))


def factors(df):
    def leg(tk, kind):
        g = df[df.ticker == tk].sort_values('day').set_index('day')
        if kind == 'on':
            return np.log(g.open / g.close.shift(1)) * 1e4
        if kind == 'cc':
            return np.log(g.close / g.close.shift(1)) * 1e4
        return np.log(g.close / g.open) * 1e4                    # intraday
    F = pd.DataFrame({
        'mkt_on': leg('QQQ', 'on'), 'mkt_cc': leg('QQQ', 'cc'),
        'mkt_id': leg('QQQ', 'id')})
    spy = df[df.ticker == 'SPY']
    if len(spy):
        F['spy_on'] = leg('SPY', 'on')
    return F


def show(name, y, F, cols):
    use = F[cols].dropna()
    j = pd.concat([y.rename('y'), use], axis=1).dropna()
    if len(j) < 250:
        print(f'  {name:<16} too few common days ({len(j)})')
        return
    r = regress(j.y, j[cols])
    bs = '  '.join(f'{k} {v:+.2f}' for k, v in r['betas'].items())
    print(f'  {name:<16} alpha {r["alpha"]:+6.2f} bps/d '
          f'(t_HAC {r["t_alpha"]:+5.2f}, {r["alpha"]*21/100:+.2f}%/mo)   '
          f'R2 {r["r2"]:.2f}   residSharpe {r["resid_sharpe"]:+.2f}')
    print(f'  {"":16} betas: {bs}')
    return r


def main():
    d = os.path.join(ROOT, 'data')
    sp = os.path.join(d, 'stack_daily.csv')
    if not os.path.exists(sp):
        raise SystemExit('missing data/stack_daily.csv -- run stack_v2.py first')
    L = pd.read_csv(sp, dtype={'day': str}).set_index('day')
    F = factors(load_panel())
    have_spy = 'spy_on' in F

    S = {}
    for k in ('ON', 'NEU', 'MOM'):
        if k in L:
            S[k] = L[k].dropna()
    if 'ON' in L and 'MOM' in L:
        S['ON+MOM (v2)'] = L[['ON', 'MOM']].dropna().sum(axis=1)
    if 'NEU' in L and 'MOM' in L:
        S['NEU+MOM (v2n)'] = L[['NEU', 'MOM']].dropna().sum(axis=1)

    print('== session-orthogonality checks (beta ~0 by construction, not fit) ==')
    print('  ON vs QQQ-intraday -- overnight book cannot load on the day:')
    if 'ON' in S:
        show('ON|mkt_id', S['ON'], F, ['mkt_id'])
    print('  MOM vs QQQ-overnight -- intraday book cannot load on the night:')
    if 'MOM' in S:
        show('MOM|mkt_on', S['MOM'], F, ['mkt_on'])

    on_cols = ['mkt_on'] + (['spy_on'] if have_spy else [])
    print(f'\n== overnight books vs overnight market'
          f'{" (QQQ+SPY)" if have_spy else " (QQQ)"} ==')
    for k in ('ON', 'NEU', 'ON+MOM (v2)', 'NEU+MOM (v2n)'):
        if k in S:
            show(k, S[k], F, on_cols)

    print('\n== MOM vs QQQ intraday ==')
    if 'MOM' in S:
        show('MOM', S['MOM'], F, ['mkt_id'])

    print('\n== everything vs the full factor set (overnight + intraday) ==')
    full = ['mkt_on', 'mkt_id'] + (['spy_on'] if have_spy else [])
    for k in S:
        show(k, S[k], F, full)

    print('\nalpha is the return no combination of these factors reproduces; '
          'a positive\nalpha at t_HAC>2 with a healthy residual Sharpe is the '
          'evidence that ON+MOM is\nits own source, not repackaged market '
          'exposure. Read betas for what it IS made of.')


if __name__ == '__main__':
    main()
