# Part of qqq-microstructure.
#
# Does the sell-premium overlay have a season? RESULTS 8 priced the 0DTE
# structures against an ASSUMED IV and found the whole ranking pivots on it:
# long calls break even at ~14.8% annualised, short-vol structures need more
# than fair, and the single measured day (RESULTS 9, 2026-07-15) printed 12.5%
# -- below break-even, no variance risk premium at all. Before spending money on
# the full OPRA pull, this script answers the gating question for free: how
# often is QQQ 0DTE-proxy IV above the reference lines, and how does implied
# compare to what the panel then realised?
#
# The proxy: ^VIX1D (CBOE 1-day SPX vol, live since 2022-05 -- the 0DTE era's
# own gauge) scaled by the measured QQQ/SPY realised-vol ratio from the panel.
# Two stated approximations: SPX-vs-SPY is ignored, and the index closes at
# 16:00 while the trade decision is at 10:30. One ground-truth check exists and
# is printed whenever the date is in range: the scaled proxy on 2026-07-15
# against the 12.5% measured from the option chain itself.
#
# Reference lines, all pre-declared from RESULTS 8/9, no bands invented:
#   12.5  the one measured day            14.8  long-call break-even (§8)
#   16.0  the assumption §8's short-wing numbers were computed at
#
#   pip install yfinance
#   python src/iv_regime.py

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import load_panel

LINES = [12.5, 14.8, 16.0]
H = np.sqrt(5.5 / (6.5 * 252))          # 10:30 -> close horizon, in year^0.5


def daily(df, names):
    q = df[df.ticker.isin(names)].sort_values(['day', 'ticker']) \
          .groupby('day').first()
    q = q[q.p60 > 0]
    return pd.DataFrame({'id': np.log(q.close / q.open) * 1e4,
                         'ec': np.log(q.close / q.p60) * 1e4}, index=q.index)


def main():
    argparse.ArgumentParser().parse_args()
    df = load_panel()
    qqq = daily(df, {'QQQ', 'QQQQ'})
    spy = daily(df, {'SPY'})
    j = qqq.join(spy, rsuffix='_s').dropna()
    ratio = j.id.std() / j.id_s.std()
    print(f'\nQQQ/SPY realised-vol ratio (panel, {len(j):,} common days): '
          f'{ratio:.3f}')

    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit('pip install yfinance')
    d0 = pd.to_datetime(qqq.index.min())
    d1 = pd.to_datetime(qqq.index.max()) + pd.Timedelta(days=2)
    v = yf.download(['^VIX1D', '^VXN'], start=d0.date(), end=d1.date(),
                    auto_adjust=False, actions=False, group_by='ticker',
                    progress=False)
    out = {}
    for tk in ('^VIX1D', '^VXN'):
        try:
            s = v[tk]['Close'].dropna()
            s.index = s.index.strftime('%Y%m%d')
            out[tk] = s
        except KeyError:
            print(f'{tk}: no data returned')
    if '^VIX1D' not in out:
        raise SystemExit('no ^VIX1D history -- nothing to gate on')

    proxy = (out['^VIX1D'] * ratio).reindex(qqq.index).dropna()
    print(f'0DTE proxy = ^VIX1D x {ratio:.3f}: {len(proxy):,} days '
          f'{proxy.index[0]} .. {proxy.index[-1]}')

    gt = '20260715'
    if gt in proxy.index:
        print(f'ground truth: proxy on 2026-07-15 = {proxy[gt]:.1f}% vs 12.5% '
              f'measured from the chain (RESULTS 9)')
    else:
        print('ground-truth day 2026-07-15 not in proxy range')

    qs = proxy.quantile([.1, .25, .5, .75, .9])
    print(f'\nproxy distribution: p10 {qs.iloc[0]:.1f}  p25 {qs.iloc[1]:.1f}  '
          f'median {qs.iloc[2]:.1f}  p75 {qs.iloc[3]:.1f}  p90 {qs.iloc[4]:.1f}')
    for x in LINES:
        print(f'  days above {x:>4}%: {(proxy > x).mean()*100:5.1f}%')

    # implied vs realised over the 10:30 -> close horizon the structures trade
    r = qqq.ec.reindex(proxy.index)
    print(f'\n{"year":>6}{"mean IV%":>10}{">14.8%":>9}{"impl sd":>9}'
          f'{"real sd":>9}{"impl/real":>11}   (10:30->close, bps)')
    for y, g in proxy.groupby(proxy.index.str[:4]):
        rr = r.loc[g.index].dropna()
        impl = (g / 100 * H * 1e4).mean()
        print(f'{y:>6}{g.mean():>10.1f}{(g > 14.8).mean()*100:>8.0f}%'
              f'{impl:>9.0f}{rr.std():>9.0f}{impl/rr.std():>11.2f}')
    print('  impl/real > 1 means sellers were being paid more than the risk '
          'realised;\n  < 1 means premium was cheap and RESULTS 9\'s long-call '
          'ranking applies.')

    if '^VXN' in out:
        vxn = out['^VXN'].reindex(qqq.index).dropna()
        print(f'\n^VXN (30-day, context only -- term structure makes it a loose '
              f'0DTE guide):\n  {len(vxn):,} days; by 4-year era, mean:')
        for e, g in vxn.groupby((vxn.index.str[:4].astype(int) // 4) * 4):
            print(f'    {e}-{e+3}: {g.mean():5.1f}%')


if __name__ == '__main__':
    main()
