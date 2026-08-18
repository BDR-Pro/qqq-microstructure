# Part of qqq-microstructure.
#
# Backtest of the one validated signal: the sign of the first N minutes, held to the
# close, traded in QQQ shares.
#
# The point of the parameter sweep here is ROBUSTNESS, not optimisation. If the edge
# only exists at one entry time it is an artefact of having looked at that entry time.
# A real effect should show up across a band of nearby choices, and the honest number to
# quote afterwards is the average over the band -- not the best cell in it.
#
# Costs: QQQ NBBO is ~1 tick on a ~$500 underlying, so a round trip crossing the spread
# is ~0.34 bps. Entering and exiting on the close auction is cheaper in practice; 0.34
# is the conservative choice.

import os, sys, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, 'data', 'daily.parquet')
ENTRIES = [5, 10, 15, 30, 60, 120]          # minutes after the open
COST_BPS = 0.34


def load():
    df = pd.read_parquet(DAILY).dropna(subset=['oc_bps']).reset_index(drop=True)
    df['date'] = pd.to_datetime(df.day, format='%Y%m%d')
    df['year'] = df.day.str[:4]
    return df


def series(df, mins, cost=COST_BPS, min_sig=0.0):
    """Daily P&L in bps for entry at `mins`, held to the close."""
    sig = df[f'ret{mins}_bps'].values
    y = np.log(df['close'].values / df[f'p{mins}'].values) * 1e4
    pos = np.sign(sig) * (np.abs(sig) >= min_sig)
    r = pos * y - cost * (pos != 0)
    return pd.Series(r, index=df.index), pos


def stats(r, label=''):
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return None
    eq = (1 + r / 1e4).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1)
    yrs = n / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    downside = r[r < 0].std()
    return dict(
        label=label, n=n, bps_day=r.mean(),
        t=r.mean() / (r.std() / np.sqrt(n)),
        sharpe=r.mean() / r.std() * np.sqrt(252),
        sortino=r.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
        cagr_pct=cagr * 100, total_pct=(eq.iloc[-1] - 1) * 100,
        maxdd_pct=dd.min() * 100,
        calmar=cagr * 100 / abs(dd.min() * 100) if dd.min() < 0 else np.nan,
        hit_pct=(r > 0).mean() * 100,
        pf=r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else np.nan,
        best=r.max(), worst=r.min(), exposure=100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cost', type=float, default=COST_BPS)
    a = ap.parse_args()
    df = load()
    print(f'QQQ momentum backtest | {len(df)} days  {df.day.iloc[0]} .. {df.day.iloc[-1]}')
    print(f'signal: sign of the first N minutes, held to the close, shares')
    print(f'cost:   {a.cost:.2f} bps round trip\n')

    print('=== ROBUSTNESS: is 60 minutes special, or does the band work? ===')
    print(f'{"entry":>7}{"bps/day":>10}{"t":>7}{"Sharpe":>8}{"CAGR%":>8}'
          f'{"maxDD%":>9}{"Calmar":>8}{"hit%":>7}')
    rows = []
    for m in ENTRIES:
        r, _ = series(df, m, a.cost)
        s = stats(r, f'{m}m')
        rows.append((m, s))
        print(f'{m:>6}m{s["bps_day"]:>10.2f}{s["t"]:>7.2f}{s["sharpe"]:>8.2f}'
              f'{s["cagr_pct"]:>8.2f}{s["maxdd_pct"]:>9.2f}{s["calmar"]:>8.2f}'
              f'{s["hit_pct"]:>7.1f}')
    band = [s['bps_day'] for m, s in rows if 10 <= m <= 120]
    print(f'\n  band mean (10m-120m): {np.mean(band):.2f} bps/day, '
          f'{sum(1 for b in band if b > 0)}/{len(band)} positive')
    print('  Quote the band, not the best cell. The best cell is a number you chose.')

    print('\n=== YEARLY, across the band ===')
    print(f'{"year":>7}' + ''.join(f'{str(m)+"m":>9}' for m in ENTRIES))
    for y, g in df.groupby('year'):
        cells = []
        for m in ENTRIES:
            r, _ = series(g.reset_index(drop=True), m, a.cost)
            cells.append(f'{r.mean():>9.2f}')
        print(f'{y:>7}' + ''.join(cells))

    print('\n=== COST SENSITIVITY (60m entry) ===')
    print(f'{"cost bps":>10}{"bps/day":>10}{"CAGR%":>9}{"Sharpe":>8}')
    for c in (0.0, 0.34, 0.7, 1.0, 2.0, 5.0):
        r, _ = series(df, 60, c)
        s = stats(r)
        print(f'{c:>10.2f}{s["bps_day"]:>10.2f}{s["cagr_pct"]:>9.2f}{s["sharpe"]:>8.2f}')

    print('\n=== SIGNAL-STRENGTH FILTER (60m entry) — trade only when the move is big ===')
    print(f'{"min |sig|":>11}{"days traded":>13}{"bps/day":>10}{"bps/trade":>11}{"Sharpe":>8}')
    for ms in (0, 10, 20, 30, 50, 80):
        r, pos = series(df, 60, a.cost, min_sig=ms)
        s = stats(r)
        nt = int((pos != 0).sum())
        bpt = r[pos != 0].mean() if nt else np.nan
        print(f'{ms:>11}{nt:>13}{s["bps_day"]:>10.2f}{bpt:>11.2f}{s["sharpe"]:>8.2f}')

    print('\n=== HEADLINE (60m entry, full sample) ===')
    r, _ = series(df, 60, a.cost)
    s = stats(r)
    for k in ('n', 'bps_day', 't', 'sharpe', 'sortino', 'cagr_pct', 'total_pct',
              'maxdd_pct', 'calmar', 'hit_pct', 'pf', 'best', 'worst'):
        print(f'  {k:<12}{s[k]:>12.2f}')

    print('\n=== LEVERAGE (what it costs to chase ROI) ===')
    print('  drawdown scales with leverage; ruin is a hard floor, not a bad quarter')
    print(f'{"leverage":>10}{"CAGR%":>9}{"maxDD%":>10}{"worst day%":>12}')
    for L in (1, 2, 3, 5, 10):
        rl = r * L
        sl = stats(rl)
        print(f'{L:>9}x{sl["cagr_pct"]:>9.2f}{sl["maxdd_pct"]:>10.2f}'
              f'{sl["worst"]/100:>12.2f}')

    eq = (1 + r / 1e4).cumprod()
    out = pd.DataFrame({'day': df.day, 'ret_bps': r.values, 'equity': eq.values})
    p = os.path.join(ROOT, 'data', 'momentum_equity.csv')
    out.to_csv(p, index=False)
    print(f'\nequity curve -> {os.path.relpath(p, ROOT)}')


if __name__ == '__main__':
    main()
