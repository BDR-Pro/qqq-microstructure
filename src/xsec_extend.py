# Part of qqq-microstructure.
#
# The HuggingFace dataset stalled at 2026-03, five months behind, and with it
# the entire forward test: no new panel months, no xsec_replay verdicts, no
# options-day valuation past March. This rebuilds missing months in the
# extractor's exact schema from Yahoo official daily bars -- the source RESULTS
# 15b validated against the panel at corr 0.99 on 208k name-days.
#
# What Yahoo can and cannot provide, stated:
#   - Daily open/high/low/close/volume: official, full coverage. The ON book's
#     entire data need.
#   - The universe: Yahoo cannot rank every US ticker, so each month's top-150
#     is chosen from the union of the last 12 HF months' members (~250 names)
#     re-ranked by the new month's dollar volume, drawn only from HF-sourced
#     months (Yahoo months are not fed back into the pool). A genuinely new
#     entrant is missed until the HF source catches up; when it does,
#     xsec_extract.py re-extracts and OVERWRITES any month marked src='yahoo'.
#   - Intraday snapshots: p60 from 60-minute bars (Yahoo keeps ~2 years); p15
#     and p30 from 15-minute bars (Yahoo keeps ~60 days). Months older than the
#     15m retention get p15/p30 = open with a loud FABRICATED warning -- for
#     those months the on15 floor degrades to the open exit and the fh feature
#     reads neutral after rank-transform. p60, the feature that matters, holds.
#   - Provenance: rows carry src='yahoo'; HF-derived rows have no src column.
#
# Only COMPLETE months are built (a partial file would be frozen by resumption).
# Validate before trusting, the house way: --check rebuilds a month the HF panel
# already has and prints the universe overlap and per-row price differences
# against ground truth, writing nothing.
#
#   python src/xsec_extend.py --check 2026-02     # grade Yahoo vs HF, no writes
#   python src/xsec_extend.py                     # build all missing complete months

import os, glob, argparse, datetime as dt
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XSEC = os.path.join(ROOT, 'data', 'xsec')
TOP_N, CAND_MONTHS = 150, 12


def month_files():
    return sorted(glob.glob(os.path.join(XSEC, '*.parquet')))


def candidates(before):
    """Names eligible to be re-ranked into month `before`: the union of the
    last CAND_MONTHS HF-sourced months. Yahoo-built months are EXCLUDED from
    the pool on purpose -- feeding them back made the pool a closed, monotone-
    shrinking cohort (audit 2026-09, B1#4). When no HF month remains in the
    window the pool falls back to all files, loudly."""
    fs = [f for f in month_files() if os.path.basename(f)[:7] < before][-CAND_MONTHS:]
    tks, hf = set(), 0
    for f in fs:
        d = pd.read_parquet(f, columns=['ticker'] + (
            ['src'] if 'src' in pd.read_parquet(f).columns else []))
        if 'src' in d and (d.src == 'yahoo').all():
            continue
        hf += 1
        tks.update(d.ticker.unique())
    if not tks:
        print(f'WARNING: no HF-sourced month in the last {CAND_MONTHS} -- the '
              f'candidate pool is a closed Yahoo cohort; new entrants cannot '
              f'be scored until the HF source resumes (re-extract overwrites '
              f'Yahoo months)')
        for f in fs:
            tks.update(pd.read_parquet(f, columns=['ticker']).ticker.unique())
    return sorted(tks)


def fetch(yf, tickers, d0, d1, interval):
    v = yf.download(tickers, start=d0, end=d1, interval=interval,
                    auto_adjust=False, actions=False, group_by='ticker',
                    progress=False, threads=True)
    out = {}
    for tk in tickers:
        try:
            s = v[tk].dropna(how='all')
        except KeyError:
            continue
        if len(s):
            out[tk] = s
    return out


def snap(h, day, hh, mm):
    """Close of the intraday bar that STARTS at hh:mm ET on `day`."""
    if h is None:
        return np.nan
    try:
        idx = h.index
        m = (idx.strftime('%Y%m%d') == day) & (idx.hour == hh) & (idx.minute == mm)
        v = h.Close[m]
        return float(v.iloc[0]) if len(v) else np.nan
    except Exception:
        return np.nan


def build_month(m):
    import yfinance as yf
    cand = candidates(m)
    ym = {tk: tk.replace('.', '-') for tk in cand}
    d0 = f'{m}-01'
    nx = dt.date(int(m[:4]) + (m[5:7] == '12'), int(m[5:7]) % 12 + 1, 1)
    daily = fetch(yf, list(ym.values()), d0, nx.isoformat(), '1d')
    # class-share mapping fallback: BRK.B -> BRK-B is right, but CMCS.A wants
    # CMCSA on Yahoo; retry dot-dropped for dotted names the dash form missed
    miss = [tk for tk in cand if ym[tk] not in daily and '.' in tk]
    if miss:
        alt = fetch(yf, [tk.replace('.', '') for tk in miss], d0,
                    nx.isoformat(), '1d')
        for tk in miss:
            if tk.replace('.', '') in alt:
                ym[tk] = tk.replace('.', '')
                daily[ym[tk]] = alt[ym[tk]]
    dv = {tk: float((daily[y].Volume * daily[y].Close).sum())
          for tk, y in ym.items() if y in daily}
    top = sorted(dv, key=dv.get, reverse=True)[:TOP_N]
    print(f'  {m}: {len(dv)} candidates priced, top {len(top)} kept')

    have15 = (dt.date.today() - dt.date(int(m[:4]), int(m[5:7]), 1)).days < 55
    h60 = {tk: v for tk, v in fetch(yf, [ym[t] for t in top], d0,
                                    nx.isoformat(), '60m').items()}
    h15 = fetch(yf, [ym[t] for t in top], d0, nx.isoformat(), '15m') \
        if have15 else {}
    if not have15:
        print(f'  {m}: beyond Yahoo 15m retention -- p15/p30 FABRICATED as open '
              f'(on15 degrades to the open exit for this month)')

    rows = []
    for tk in top:
        y = ym[tk]
        for ts, r in daily[y].iterrows():
            day = ts.strftime('%Y%m%d')
            p60 = snap(h60.get(y), day, 9, 30)
            p15 = snap(h15.get(y), day, 9, 30) if have15 else np.nan
            p30 = snap(h15.get(y), day, 9, 45) if have15 else np.nan
            o = float(r.Open)
            rows.append((tk, day, m, o, float(r.Close), float(r.High),
                         float(r.Low), float(r.Volume * r.Close), 390,
                         p15 if np.isfinite(p15) else o,
                         p30 if np.isfinite(p30) else o,
                         p60 if np.isfinite(p60) else o, 'yahoo'))
    df = pd.DataFrame(rows, columns=['ticker', 'day', 'month', 'open', 'close',
                                     'high', 'low', 'dollar_vol', 'bars',
                                     'p15', 'p30', 'p60', 'src'])
    df = df[(df.open > 0) & (df.close > 0)]
    n60 = (df.p60 != df.open).mean()
    print(f'  {m}: {len(df):,} rows, {df.ticker.nunique()} names, '
          f'p60 real on {n60 * 100:.0f}% of rows')
    return df


def check(m):
    p = os.path.join(XSEC, f'{m}.parquet')
    if not os.path.exists(p):
        raise SystemExit(f'{m} is not in the HF panel -- pick a month that is')
    hf = pd.read_parquet(p)
    yh = build_month(m)
    u_hf, u_yh = set(hf.ticker), set(yh.ticker)
    print(f'\ncheck {m} against the HF panel:')
    print(f'  universe overlap: {len(u_hf & u_yh)}/{len(u_hf)} of HF top-150 '
          f'recovered ({len(u_hf & u_yh) / len(u_hf) * 100:.0f}%)')
    j = hf.merge(yh, on=['ticker', 'day'], suffixes=('_hf', '_yh'))
    for c in ('open', 'close', 'p60'):
        d = (np.log(j[f'{c}_yh'] / j[f'{c}_hf']) * 1e4).abs()
        big = j.loc[d > 1000, 'ticker'].nunique()
        print(f'  |{c} diff|: median {d.median():.1f} bps  p95 '
              f'{d.quantile(.95):.1f} bps  ({len(j):,} rows'
              + (f'; {big} name(s) look split-adjusted -- Yahoo folds splits '
                 f'into all pre-split history, the panel is raw; the seam is '
                 f'what the split detector is for)' if big else ')'))
    print('  read: overlap >=90% and median/p95 of a few bps mean the '
        'extension can be trusted;\n  p60 diffs include the '
        'minute-bar-vs-60m-bar convention gap and run higher.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', default=None, metavar='YYYY-MM',
                    help='rebuild an existing HF month from Yahoo and diff it; '
                         'writes nothing')
    a = ap.parse_args()
    if a.check:
        check(a.check)
        return
    fs = month_files()
    if not fs:
        raise SystemExit('no panel in data/xsec -- nothing to extend')
    last = os.path.basename(fs[-1])[:7]
    cur = dt.date.today().strftime('%Y-%m')
    todo, y, mo = [], int(last[:4]), int(last[5:7])
    while True:
        y, mo = y + (mo == 12), mo % 12 + 1
        m = f'{y:04d}-{mo:02d}'
        if m >= cur:
            break
        todo.append(m)
    print(f'panel ends {last}; current month {cur}; '
          f'{len(todo)} complete month(s) to build: {todo}')
    for m in todo:
        df = build_month(m)
        if df.ticker.nunique() < 100:
            print(f'  {m}: only {df.ticker.nunique()} names -- NOT writing '
                  f'(source problem?)')
            continue
        df.to_parquet(os.path.join(XSEC, f'{m}.parquet'), index=False)
        print(f'  {m}: written')
    print('done -- xsec_replay.py and opra_value.py will pick the new months up')


if __name__ == '__main__':
    main()
