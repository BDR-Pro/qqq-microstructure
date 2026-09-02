# Part of qqq-microstructure.
#
# A first-class backtest for the OPTION structures, with the same metric and
# significance discipline the equity books get -- and two modes kept strictly
# apart, because they answer different questions and only one is tradeable:
#
#   --source real  (QQQ only)  reads data/opra_daily.parquet: the structures
#       priced at MEASURED bid/ask on real OPRA chains (opra_value.py, the
#       2023-04-> record). This is the truthful backtest -- it already carries
#       the bid/ask friction and the real implied-vol richness. This mode
#       upgrades RESULTS 18 from a per-day EV line to a full equity curve,
#       Sharpe/Sortino/maxDD, per-year, tails, and a block-bootstrap CI.
#
#   --source model (any panel name)  prices the structure each day with
#       Black-Scholes on the underlying's own bars, settling at the realised
#       close. It has NO measured chain, so two things a real seller lives on
#       are absent BY CONSTRUCTION and stated on every run:
#         - no bid/ask: entry is at the mid, so it flatters every structure;
#         - IV = trailing realised vol OF THE 10:30->CLOSE WINDOW x --iv-mult,
#           annualised by the window's own length. At mult 1.0 the structure is
#           priced at the window's realised variance, i.e. ZERO variance-risk
#           premium by construction, so the default run measures DIRECTIONAL
#           P&L only; to model the premium assert --iv-mult > 1 and own it.
#           (An earlier version used full-day close-to-close vol here, which
#           already sold rich; fixed after the 2026-09 audit.)
#       Model mode answers "does the directional/vol thesis have legs on this
#       name"; it does NOT answer "is it tradeable." Only real mode does.
#
# Structures (re-struck each day by moneyness, matching RESULTS 8/18):
#   spread  momentum-direction credit spread: up-signal -> put credit spread
#           (sell 0.5% OTM, buy 1.5% OTM); down-signal -> the call mirror
#   condor  iron condor, sell 0.5% strangle / buy 1.0% wings
#   long    long ATM in the signal direction (a debit; RESULTS 18 found it dead)
# Signal is the sign of the first hour, log(p60/open) -- the same signal MOM
# and the OPT leg use. Held 0DTE: enter ~10:30 (p60), settle at the close.
#
# Options P&L does not compound like an equity return (the capital at risk is
# the structure's max loss, not the notional), so this reports an ADDITIVE
# cumulative-P&L curve in bps of spot and its drawdown, plus $/contract -- not
# a CAGR that would divide by the wrong denominator.
#
# Validated on planted truth: on a synthetic underlying with zero drift a
# fair (mult 1.0) credit spread backtests to ~0 before commissions; selling
# rich (mult>1) turns it positive by the premium; a planted upward drift makes
# the put credit spread profit directionally (see RESULTS).
#
#   python src/optbacktest.py                              # real, QQQ, spread
#   python src/optbacktest.py --struct condor
#   python src/optbacktest.py --source model --sym QQQ --iv-mult 1.2
#   python src/optbacktest.py --source model --sym AAPL   # directional only

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel, era_table
from optionscan import bs, intrinsic
from mc_risk import paths, BLOCK, SEED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF = 5.5 / 6.5                      # 10:30->close is this fraction of a session
LEGS_N = {'spread': 2, 'condor': 4, 'long': 1}


def struct_legs(kind, S, up):
    if kind == 'long':
        return [(S, 'C' if up else 'P', 1.0)]
    if kind == 'condor':
        return [(S * 0.995, 'P', -1.0), (S * 0.99, 'P', 1.0),
                (S * 1.005, 'C', -1.0), (S * 1.01, 'C', 1.0)]
    # spread: momentum-direction credit spread
    if up:
        return [(S * 0.995, 'P', -1.0), (S * 0.985, 'P', 1.0)]
    return [(S * 1.005, 'C', -1.0), (S * 1.015, 'C', 1.0)]


def model_series(sym, kind, iv_mult, iv_fix, comm, df=None):
    df = load_panel() if df is None else df
    g = df[df.ticker == sym].sort_values('day').set_index('day')
    if len(g) < 250:
        raise SystemExit(f'{sym}: only {len(g)} panel days -- too few')
    # "fair" IV for the 10:30->close WINDOW must be the vol of that window's
    # own return, annualised by its own length: the first version used the
    # full close-to-close vol (overnight + first hour included) and so sold
    # ~1.1-1.4x rich at mult 1.0 while claiming a zero variance premium
    # (audit 2026-09, B1#8). Causal: trailing 20 windows, shifted one day.
    wret = np.log(g.close / g.p60.where(g.p60 > 0))
    rv = wret.rolling(20).std().shift(1) * np.sqrt(252.0 / SF)
    S0 = g.p60.where(g.p60 > 0, g.open)                             # 10:30 entry
    T = SF / 252.0
    out = {}
    for day, s0, st, o, p60, v in zip(g.index, S0, g.close, g.open,
                                      g.p60, rv):
        if not (s0 > 0 and st > 0 and o > 0 and p60 > 0) or not np.isfinite(v):
            continue
        sig = np.log(p60 / o)
        if sig == 0:
            continue
        iv = (iv_fix if iv_fix else v) * iv_mult
        if iv <= 0:
            continue
        legs = struct_legs(kind, s0, sig > 0)
        credit = -sum(q * bs(s0, k, T, iv, 0.0, kd) for k, kd, q in legs)
        pnl = (credit + intrinsic(legs, st)) / s0 * 1e4
        cbps = comm * len(legs) / (s0 * 100) * 1e4
        out[day] = pnl - cbps
    return pd.Series(out).dropna(), df


def real_series(kind, comm):
    p = os.path.join(ROOT, 'data', 'opra_daily.parquet')
    if not os.path.exists(p):
        raise SystemExit('no data/opra_daily.parquet -- run opra_value.py '
                         '(needs the pulled OPRA chains), or use --source model')
    od = pd.read_parquet(p)
    od['day'] = od.day.astype(str)
    od = od.set_index('day').sort_index()
    col = {'spread': 'ev_sprd', 'condor': 'ev_cond', 'long': 'ev_long'}[kind]
    cbps = comm * LEGS_N[kind] / (od.spot * 100) * 1e4
    return (od[col] - cbps).dropna()


def report(r, label, spot_med):
    r = r.astype(float)
    v = r.values
    n = len(v)
    eq = np.cumsum(v)                                   # additive, bps of spot
    dd = (eq - np.maximum.accumulate(eq)).min()
    down = np.sqrt(np.mean(np.minimum(v, 0) ** 2))      # downside deviation
    sh = v.mean() / v.std() * np.sqrt(252) if v.std() else np.nan
    so = v.mean() / down * np.sqrt(252) if down else np.nan
    dollar = v.mean() / 1e4 * spot_med * 100
    print(f'\n{label}: {n} days  {r.index[0]}..{r.index[-1]}')
    print(f'  mean {v.mean():+.2f} bps/day  (t {v.mean()/(v.std()/np.sqrt(n)):+.2f})'
          f'  ${dollar:+.2f}/contract/day')
    print(f'  Sharpe {sh:.2f}  Sortino {so:.2f}  win {(v>0).mean()*100:.0f}%  '
          f'cumP&L {eq[-1]/100:+.2f}%-of-spot  maxDD {dd/100:.2f}%-of-spot')
    w = np.sort(v)[:5]
    print(f'  worst 5 days: {"  ".join(f"{x:+.0f}" for x in w)} bps  '
          f'(defined-risk floor caps these)')
    yr = r.groupby(r.index.str[:4]).mean()
    print('  by year: ' + '  '.join(f'{y} {m:+.1f}' for y, m in yr.items()))
    rng = np.random.default_rng(SEED)
    m = paths(v, 4000, n, BLOCK, rng).mean(axis=1)
    print(f'  block-bootstrap mean CI: p5 {np.percentile(m,5):+.2f}  '
          f'p50 {np.percentile(m,50):+.2f}  p95 {np.percentile(m,95):+.2f}  '
          f'P(mean>0) {(m>0).mean()*100:.0f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['real', 'model'], default='real')
    ap.add_argument('--struct', choices=['spread', 'condor', 'long'],
                    default='spread')
    ap.add_argument('--sym', default='QQQ', help='underlying (model mode)')
    ap.add_argument('--iv-mult', type=float, default=1.0,
                    help='model IV = realised vol x this (1.0 = sell at fair, '
                         'no variance premium)')
    ap.add_argument('--iv', type=float, default=None,
                    help='model: fixed annualized IV instead of realised')
    ap.add_argument('--comm', type=float, default=0.65,
                    help='$/contract/side, entry legs')
    a = ap.parse_args()

    if a.source == 'real':
        r = real_series(a.struct, a.comm)
        spot_med = 400.0
        try:
            od = pd.read_parquet(os.path.join(ROOT, 'data',
                                              'opra_daily.parquet'))
            spot_med = float(od.spot.median())
        except Exception:
            pass
        label = f'REAL chains  QQQ {a.struct}  (measured bid/ask, comm ${a.comm})'
        note = ('measured premiums and friction -- this is the tradeable '
                'number.')
    else:
        r, df = model_series(a.sym, a.struct, a.iv_mult, a.iv, a.comm)
        spot_med = float(df[df.ticker == a.sym].close.median())
        label = (f'MODEL  {a.sym} {a.struct}  IV='
                 + (f'{a.iv:.0%} fixed' if a.iv else f'realised x{a.iv_mult:g}')
                 + f'  (comm ${a.comm})')
        note = ('MID fills, no bid/ask; ' + (
            'IV=realised x1.0 so the variance premium is ZERO -- this is '
            'DIRECTIONAL P&L only, not a tradeable edge.'
            if a.iv_mult == 1.0 and not a.iv else
            f'IV set {a.iv_mult:g}x realised -- the premium is your assumption, '
            'not a measurement.'))
    if len(r) < 60:
        raise SystemExit(f'only {len(r)} days -- too few to report')

    print('=' * 70)
    print(f'OPTION BACKTEST -- {label}')
    print('caveat:', note)
    print('=' * 70)
    report(r, a.struct, spot_med)
    if len(r) > 400:
        print('\n  by 4-year era (mean bps/day):')
        era_table(r.index, r.values)
    print('\nreal mode = tradeable truth (measured chains); model mode = a '
          'thesis explorer.\nOnly a structure that survives REAL mode AND '
          "RESULTS 19's forward clock is fundable.")


if __name__ == '__main__':
    main()
