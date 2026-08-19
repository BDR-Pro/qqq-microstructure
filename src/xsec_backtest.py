# Part of qqq-microstructure.
#
# Signals #4 and #5: the cross-section. Two pre-declared published effects on the
# top-150-per-month panel built by xsec_extract.py. One pass each, no sweeps.
#
#   A. Cross-sectional momentum (Jegadeesh & Titman 1993): rank by the trailing
#      12-to-2-month return (skip the most recent month -- short-term reversal),
#      long the top quintile, short the bottom, equal weight, hold one month.
#   B. Overnight persistence (Lou, Polk & Skouras 2019; Aboody et al. 2018): rank
#      by the prior month's mean overnight return, hold top-minus-bottom quintile
#      close-to-open only, ranking refreshed monthly.
#
# Methodology decisions that carry the statistics:
#
#   - Universe lag. Month M's top-150 list is only knowable at M's end, so month T
#     is traded on the list from file T-1. A name must also appear in file T to be
#     priceable; that conditioning (still-in-top-150 next month) is a mild
#     liquidity survivorship and is MEASURED and printed as attrition, not hidden.
#   - Splits are handled by exclusion, not adjustment. Prices are unadjusted, so a
#     split is a giant fake overnight gap. Any overnight gap that (a) spans a
#     missing day, (b) matches a standard split ratio (3:2, 4:3, 2:1 .. 50:1, or
#     inverse) within 1.5% in log space, or (c) exceeds +/-40% (backstop) is
#     NaN'd. Every multi-day return in this file is a sum of VALID daily log
#     returns, so a level break on an excluded day cannot leak into any window.
#     Momentum additionally requires presence in every formation month. Detected
#     splits are printed, and known splits (NVDA 10:1 2024-06-10, AAPL 4:1 and
#     TSLA 5:1 2020-08-31, AMZN and GOOGL 20:1 2022, QQQ 2:1 2000-03-20) are
#     verified whenever those ticker-days are in the panel. The backstop also
#     drops real >40% crashes; for both L/S signals that is conservative (the
#     dropped days are mostly the short leg's profit), and the count is printed.
#   - Dividends land as negative overnight price moves that the holder actually
#     receives, so both signals are UNDERSTATED, B most (its long leg holds the
#     high-dividend-capture window). Same convention as overnight_study.py.
#   - ETFs/ETNs are excluded by a frozen list (index, sector, country, bond,
#     commodity, volatility, levered, single-stock, crypto, HOLDRs). Residual
#     unlisted funds may survive; the per-month exclusion count is printed.
#   - B's short leg assumes borrow on ~30 liquid names every night; the deployable
#     form is the long tilt (top quintile instead of QQQ), reported separately.
#
#   python src/xsec_backtest.py            # needs data/xsec/ from xsec_extract.py

import os, glob, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XSEC = os.path.join(ROOT, 'data', 'xsec')

MIN_NAMES = 40          # skip a month if fewer eligible names than this
LOOKBACK = 11           # formation months T-12 .. T-2
MIN_CC_DAYS = 10        # valid close-close days per formation month
MIN_ON_DAYS = 12        # valid overnight obs in the ranking month for B
SPLIT_TOL = 0.015       # log-space tolerance around a candidate ratio
BACKSTOP = np.log(1.40) # any |overnight| beyond this is dropped

_r = [1.5, 4 / 3] + list(range(2, 9)) + [10, 12, 15, 20, 25, 30, 40, 50]
SPLIT_LOGR = np.array([np.log(x) for x in _r] + [-np.log(x) for x in _r])

KNOWN_SPLITS = [('NVDA', '20240610', 0.10), ('AAPL', '20200831', 0.25),
                ('TSLA', '20200831', 0.20), ('AMZN', '20220606', 0.05),
                ('GOOGL', '20220718', 0.05), ('GOOG', '20220718', 0.05),
                ('SHOP', '20220629', 0.10), ('NVDA', '20210720', 0.25),
                ('QQQ', '20000320', 0.50)]

ETF = set('''
SPY QQQ QQQQ IWM DIA MDY IJR IJH IVV VOO VTI RSP ONEQ
XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY
XBI XOP XRT XHB XME KRE KBE KWEB SMH SOXX OIH IBB ITB IYR VNQ GDX GDXJ
EEM EFA EWJ EWZ EWY EWT EWA EWG EWU EWH EWC EWM FXI INDA RSX TUR ASHR MCHI
TLT IEF SHY TBT TMF TMV AGG BND LQD HYG JNK EMB TIP SHV BIL GOVT
GLD SLV IAU USO UCO SCO UNG BOIL KOLD DBC PDBC UUP FXE FXY
VXX UVXY SVXY VIXY VXZ SVIX UVIX XIV ZIV VIXM
TQQQ SQQQ QLD QID PSQ SH SDS SSO UPRO SPXU SPXL SPXS SDOW UDOW DXD DDM
TZA TNA URTY SRTY UWM TWM FAS FAZ ERX ERY YINN YANG
SOXL SOXS LABU LABD NUGT DUST JNUG JDST
TSLL TSLS TSLQ TSLZ NVDL NVDU NVDD NVDS MSTU MSTX MSTZ CONL
AAPU AAPD AMZU AMZD MSFU MSFD GGLL GGLS METU METD AMDL AMDS PLTU PLTD
GBTC ETHE BITO BITI IBIT FBTC ARKB BITB ETHA BITX BITU SBIT MSTY
ARKK ARKG ARKW ARKQ ARKF
BBH HHH PPH RTH SWH TTH BDH IAH UTH WMH
'''.split())


def stats(r, lab):
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 50 or r.std() == 0:
        print(f'  {lab:<18} n={n} (too short)')
        return
    sh = r.mean() / r.std() * np.sqrt(252)
    t = r.mean() / (r.std() / np.sqrt(n))
    eq = (1 + pd.Series(r) / 1e4).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = (eq.iloc[-1] ** (252 / n) - 1) * 100
    print(f'  {lab:<18} {r.mean():>+6.2f} bps/d  t={t:>5.2f}  Sharpe {sh:5.2f}  '
          f'CAGR {cagr:6.2f}%  maxDD {dd*100:6.1f}%  = {((1+cagr/100)**(1/12)-1)*100:+.2f}%/mo')


def era_table(days, r):
    d = pd.DataFrame({'y': [int(x[:4]) for x in days], 'r': r}).dropna()
    d['era'] = (d.y // 4) * 4
    for e, g in d.groupby('era'):
        print(f'    {e}-{e+3}: {g.r.mean():+7.2f} bps/day  ({len(g)} days)')


def load_panel():
    files = sorted(glob.glob(os.path.join(XSEC, '*.parquet')))
    if not files:
        raise SystemExit(f'no files in {XSEC} -- run xsec_extract.py first')
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[(df.open > 0) & (df.close > 0)]
    df = df.sort_values(['ticker', 'day']).reset_index(drop=True)

    cal = np.sort(df.day.unique())
    df['dpos'] = df.day.map({d: i for i, d in enumerate(cal)})
    g = df.groupby('ticker')
    df['prev_close'] = g['close'].shift()
    adjacent = (df.dpos - g['dpos'].shift()) == 1

    on = np.log(df.open.values / df.prev_close.values)
    near = np.abs(on[:, None] - SPLIT_LOGR[None, :]).min(1)
    is_split = adjacent.values & (near < SPLIT_TOL) & (np.abs(on) > np.log(1.25))
    too_big = adjacent.values & ~is_split & (np.abs(on) > BACKSTOP)
    ok = adjacent.values & ~is_split & (np.abs(on) <= BACKSTOP)

    df['on_bps'] = np.where(ok, on * 1e4, np.nan)
    df['id_bps'] = np.log(df.close.values / df.open.values) * 1e4
    df['cc_bps'] = df.on_bps + df.id_bps

    n_gap = int((~adjacent & df.prev_close.notna()).sum())
    print(f'panel: {len(files)} months  {len(df):,} ticker-days  '
          f'{df.ticker.nunique()} names  {cal[0]} -> {cal[-1]}')
    print(f'overnight exclusions: {int(is_split.sum())} split-ratio gaps, '
          f'{int(too_big.sum())} >40% backstop, {n_gap} across missing days')

    sp = df[is_split]
    if len(sp):
        big = sp.reindex(np.abs(np.log(sp.open / sp.prev_close)).sort_values()
                         .index[::-1]).head(12)
        for _, r_ in big.iterrows():
            print(f'    split {r_.ticker:<6} {r_.day}  x{r_.open / r_.prev_close:.3f}')

    print('known-split verification (only rows whose ticker-day is in the panel):')
    flag = dict(zip(zip(df.ticker, df.day), is_split))
    gap = dict(zip(zip(df.ticker, df.day), on))
    seen = False
    for tk, day, ratio in KNOWN_SPLITS:
        if (tk, day) not in gap or np.isnan(gap[(tk, day)]):
            continue
        seen = True
        g_ = gap[(tk, day)]
        if flag[(tk, day)] and abs(g_ - np.log(ratio)) < 3 * SPLIT_TOL:
            print(f'    OK   {tk:<6} {day}  x{np.exp(g_):.3f} detected')
        elif abs(g_) < 0.05:
            print(f'    --   {tk:<6} {day}  no gap (source pre-adjusted?)')
        else:
            print(f'    MISS {tk:<6} {day}  gap x{np.exp(g_):.3f} NOT flagged')
    if not seen:
        print('    (none of the known split dates are in this panel)')
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    df = load_panel()

    months = sorted(df.month.unique())
    uni = {m: set(g) for m, g in df.groupby('month')['ticker']}
    agg = df.groupby(['ticker', 'month']).agg(
        cc_sum=('cc_bps', 'sum'), cc_n=('cc_bps', 'count'),
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count')).to_dict('index')
    bym = {m: g[['ticker', 'day', 'cc_bps', 'on_bps', 'id_bps']]
           for m, g in df.groupby('month')}

    attr = [1 - len(uni[months[i - 1]] & uni[months[i]]) / len(uni[months[i - 1]])
            for i in range(1, len(months))]
    n_etf = np.mean([len(uni[m] & ETF) for m in months])
    print(f'universe attrition month-to-month: mean {np.mean(attr)*100:.1f}%  '
          f'worst {np.max(attr)*100:.1f}%   ETFs excluded/month: {n_etf:.0f}')

    # ---- A. cross-sectional momentum: rank on months T-12..T-2, hold month T ----
    A, sizes, prev5, prev1, to5, to1 = [], [], None, None, [], []
    for i in range(LOOKBACK + 1, len(months)):
        T = months[i]
        base = (uni[months[i - 1]] & uni[T]) - ETF
        look = months[i - 1 - LOOKBACK:i - 1]          # T-12 .. T-2
        elig, sig = [], []
        for tk in base:
            rows = [agg.get((tk, m)) for m in look]
            if any(r is None or r['cc_n'] < MIN_CC_DAYS for r in rows):
                continue
            elig.append(tk)
            sig.append(sum(r['cc_sum'] for r in rows))
        if len(elig) < MIN_NAMES:
            continue
        order = np.argsort(sig)
        k = len(elig) // 5
        q1 = {elig[j] for j in order[:k]}
        q5 = {elig[j] for j in order[-k:]}
        sub = bym[T]
        day = sub.groupby('day')
        out = pd.DataFrame({
            'q1': sub[sub.ticker.isin(q1)].groupby('day')['cc_bps'].mean(),
            'q5': sub[sub.ticker.isin(q5)].groupby('day')['cc_bps'].mean(),
            'ew': sub[sub.ticker.isin(set(elig))].groupby('day')['cc_bps'].mean()})
        A.append(out)
        sizes.append(len(elig))
        if prev5 is not None:
            to5.append(1 - len(q5 & prev5) / max(len(prev5), 1))
            to1.append(1 - len(q1 & prev1) / max(len(prev1), 1))
        prev5, prev1 = q5, q1

    print(f'\nA. cross-sectional momentum (12-2, monthly, quintiles of top-150)')
    if not A:
        print('  not enough history (needs 13 consecutive months)')
    else:
        A = pd.concat(A).dropna()
        print(f'  {len(A)} days, {len(sizes)} months, eligible names/month '
              f'mean {np.mean(sizes):.0f}')
        ls = (A.q5 - A.q1).values
        stats(ls, 'L/S Q5-Q1')
        stats((A.q5 - A.ew).values, 'long tilt Q5-EW')
        stats(A.ew.values, 'EW universe')
        to = np.mean(to5 + to1) if to5 else 0
        print(f'  turnover {to*100:.0f}%/month per leg;  L/S net of costs '
              f'(one-way c bps, 4 crossings x turnover):')
        for c in (2.5, 5, 10):
            net = ls.mean() - to * 4 * c / 21
            print(f'    c={c:>4}: {net:+.2f} bps/day')
        print('  by 4-year era (L/S):')
        era_table(A.index, ls)
        mo = pd.DataFrame({'m': [d[:6] for d in A.index], 'r': ls}).groupby('m')['r'].sum()
        worst = mo.nsmallest(5)
        print('  worst 5 months (momentum-crash check): '
              + '  '.join(f'{m} {v/100:+.1f}%' for m, v in worst.items()))

    # ---- B. overnight persistence: rank on month T-1 overnight, hold T nights ----
    B = []
    for i in range(1, len(months)):
        T = months[i]
        base = (uni[months[i - 1]] & uni[T]) - ETF
        elig, sig = [], []
        for tk in base:
            r_ = agg.get((tk, months[i - 1]))
            if r_ is None or r_['on_n'] < MIN_ON_DAYS or np.isnan(r_['on_mean']):
                continue
            elig.append(tk)
            sig.append(r_['on_mean'])
        if len(elig) < MIN_NAMES:
            continue
        order = np.argsort(sig)
        k = len(elig) // 5
        q1 = {elig[j] for j in order[:k]}
        q5 = {elig[j] for j in order[-k:]}
        sub = bym[T]
        out = pd.DataFrame({
            'q1': sub[sub.ticker.isin(q1)].groupby('day')['on_bps'].mean(),
            'q5': sub[sub.ticker.isin(q5)].groupby('day')['on_bps'].mean(),
            'q5id': sub[sub.ticker.isin(q5)].groupby('day')['id_bps'].mean(),
            'q1id': sub[sub.ticker.isin(q1)].groupby('day')['id_bps'].mean(),
            'qqq': sub[sub.ticker.isin({'QQQ', 'QQQQ'})].groupby('day')['on_bps'].mean()})
        B.append(out)

    print(f'\nB. overnight persistence (rank on prior-month overnight, hold '
          f'close->open, quintiles)')
    if not B:
        print('  not enough history')
    else:
        B = pd.concat(B)
        ls = (B.q5 - B.q1).dropna().values
        stats(ls, 'L/S Q5-Q1')
        stats(B.q5.dropna().values, 'Q5 overnight')
        if B.qqq.notna().sum() > 200:
            d2 = B.dropna(subset=['q5', 'qqq'])
            stats((d2.q5 - d2.qqq).values, 'Q5 - QQQ (tilt)')
        tug = (B.q5id - B.q1id).dropna()
        print(f'  tug-of-war check: Q5-Q1 INTRADAY {tug.mean():+.2f} bps/day '
              f'(t={tug.mean()/(tug.std()/np.sqrt(len(tug))):.2f}; '
              f'LPS predicts negative)')
        print(f'  costs: L/S pays 4 crossings/day -> break-even one-way cost '
              f'{np.nanmean(ls)/4:.2f} bps; net at c=1/2.5/5: '
              + '  '.join(f'{np.nanmean(ls)-4*c:+.2f}' for c in (1, 2.5, 5)))
        print('  by 4-year era (L/S):')
        d3 = (B.q5 - B.q1).dropna()
        era_table(d3.index, d3.values)

    if isinstance(A, pd.DataFrame) and isinstance(B, pd.DataFrame):
        j = pd.DataFrame({'a': A.q5 - A.q1, 'b': B.q5 - B.q1}).dropna()
        if len(j) > 100:
            print(f'\ncorrelation(momentum L/S, overnight L/S) = '
                  f'{np.corrcoef(j.a, j.b)[0, 1]:+.3f}   ({len(j)} days)')
        out = pd.DataFrame({'mom_ls': A.q5 - A.q1, 'mom_ew': A.ew,
                            'on_ls': B.q5 - B.q1, 'on_q5': B.q5, 'qqq_on': B.qqq})
        out.index.name = 'day'
        p = os.path.join(ROOT, 'data', 'xsec_daily.csv')
        out.to_csv(p, float_format='%.3f')
        print(f'daily series -> {p}')


if __name__ == '__main__':
    main()
