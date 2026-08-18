# Part of qqq-microstructure. Set QQQ_OPRA_ZIP to your OPRA batch download.
#
# Loads an OPRA `cbbo-1m` batch job for QQQ options and measures the one number the
# whole 0DTE question turns on: what implied vol the market actually charged at entry,
# against the realised move that followed.
#
# Two things this discovered that are worth stating up front:
#
#   1. The `definition` schema is not required. A parent-symbology request embeds the
#      OSI symbol mappings in the DBN metadata, and OSI encodes expiry, right and
#      strike directly ("QQQ   260715C00718000" -> 2026-07-15, call, $718).
#
#   2. The underlying is not required either. Put-call parity recovers it from the chain
#      alone: S = K + C - P. So an OPRA pull does NOT have to overlap the equity archive
#      to be usable -- it carries its own spot.
#
# Usage:
#   python src/opra_load.py                    # summarise every day in the zip
#   python src/opra_load.py --save             # write data/opra_daily.parquet

import os, sys, zipfile, argparse
from math import sqrt
import numpy as np, pandas as pd, databento as db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP = os.environ.get('QQQ_OPRA_ZIP', os.path.join(
    os.path.expanduser("~"), "OneDrive", "المستندات",
    "Trading", "OPRA-20260818-A3GELRSYU5.zip"))
WORK = os.path.join(ROOT, 'data', '_work_opra')
OUT = os.path.join(ROOT, 'data', 'opra_daily.parquet')

OPEN_UTC, ENTRY_UTC, CLOSE_UTC = '13:31', '14:30', '20:00'   # 09:31 / 10:30 / 16:00 ET
SESSION_FRAC = 5.5 / 6.5


def parse_chain(path):
    """Decode one OPRA day, returning the 0DTE chain with strike/right parsed."""
    df = db.DBNStore.from_file(path).to_df(map_symbols=True)
    s = df.symbol.astype(str)
    df['exp'] = s.str[6:12]
    df['cp'] = s.str[12]
    df['strike'] = pd.to_numeric(s.str[13:21], errors='coerce') / 1000.0
    day = str(df.index[0].date())
    z = df[df.exp == day.replace('-', '')[2:]]          # 0DTE only
    z = z[(z.bid_px_00 > 0) & (z.ask_px_00 > 0) & (z.ask_px_00 >= z.bid_px_00)].copy()
    z['mid'] = (z.bid_px_00 + z.ask_px_00) / 2
    z['sprd'] = z.ask_px_00 - z.bid_px_00
    return day, z


def snapshot(z, hhmm):
    day = z.index[0].strftime('%Y-%m-%d')
    t = pd.Timestamp(f'{day} {hhmm}:00+00:00')
    g = z[z.index == t]
    c = g[g.cp == 'C'].set_index('strike').sort_index()
    p = g[g.cp == 'P'].set_index('strike').sort_index()
    k = c.index.intersection(p.index)
    return c.loc[k], p.loc[k]


def spot_from_parity(c, p):
    """S = K + C - P, read at the strike where the call and put are closest in value."""
    if len(c) < 5:
        return np.nan
    par = c.index.values + c['mid'].values - p['mid'].values
    return float(par[np.argmin(np.abs(c['mid'].values - p['mid'].values))])


def measure_day(path):
    day, z = parse_chain(path)
    co, po = snapshot(z, OPEN_UTC)
    ce, pe = snapshot(z, ENTRY_UTC)
    cc, pc = snapshot(z, CLOSE_UTC)
    s_open, s_entry, s_close = (spot_from_parity(co, po), spot_from_parity(ce, pe),
                                spot_from_parity(cc, pc))
    if not np.isfinite([s_open, s_entry, s_close]).all():
        return None
    i = int(np.argmin(np.abs(ce.index.values - s_entry)))
    K = float(ce.index.values[i])
    call, put = float(ce['mid'].values[i]), float(pe['mid'].values[i])
    strad = call + put
    iv = strad / (0.7979 * s_entry) / sqrt(SESSION_FRAC / 252)
    near = ce[(ce.index > s_entry * 0.98) & (ce.index < s_entry * 1.02)]
    sig = 1.0 if s_entry > s_open else -1.0
    return dict(
        day=day, strikes=int(z.strike.nunique()), minutes=int(z.index.nunique()),
        spot_open=s_open, spot_entry=s_entry, spot_close=s_close,
        first60_bps=(s_entry / s_open - 1) * 1e4,
        entry_close_bps=(s_close / s_entry - 1) * 1e4,
        aligned_bps=(s_close / s_entry - 1) * 1e4 * sig,
        atm_strike=K, straddle=strad, straddle_bps=strad / s_entry * 1e4,
        iv_pct=iv * 100,
        atm_spread=float(ce['sprd'].values[i] + pe['sprd'].values[i]),
        spread_pct_of_prem=float((near.sprd / near['mid']).median() * 100),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', action='store_true')
    a = ap.parse_args()
    if not os.path.exists(ZIP):
        raise SystemExit(f'not found: {ZIP}\nset QQQ_OPRA_ZIP')
    os.makedirs(WORK, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.dbn.zst'))
    print(f'{len(names)} day(s) in {os.path.basename(ZIP)}\n')
    rows = []
    for n in names:
        tmp = os.path.join(WORK, os.path.basename(n))
        with zipfile.ZipFile(ZIP) as z, open(tmp, 'wb') as fo:
            fo.write(z.read(n))
        try:
            r = measure_day(tmp)
        finally:
            os.remove(tmp)
        if r is None:
            continue
        rows.append(r)
        print(f"  {r['day']}  spot {r['spot_open']:.2f} -> {r['spot_entry']:.2f} -> "
              f"{r['spot_close']:.2f}   first60 {r['first60_bps']:+.0f} bps   "
              f"aligned {r['aligned_bps']:+.0f} bps")
        print(f"     ATM ${r['atm_strike']:.0f}  straddle ${r['straddle']:.2f} "
              f"({r['straddle_bps']:.0f} bps)  IV {r['iv_pct']:.1f}%  "
              f"near-money spread {r['spread_pct_of_prem']:.1f}% of premium")
    if not rows:
        raise SystemExit('no usable days')
    t = pd.DataFrame(rows)
    print(f"\n=== {len(t)} day(s) ===")
    print(f"  mean 0DTE IV at entry      {t.iv_pct.mean():.1f}%")
    print(f"  mean straddle cost         {t.straddle_bps.mean():.0f} bps of spot")
    print(f"  mean |aligned move|        {t.aligned_bps.abs().mean():.0f} bps")
    print(f"  long straddle won on       {(t.aligned_bps.abs() > t.straddle_bps).sum()}"
          f"/{len(t)} days")
    if a.save:
        t.to_parquet(OUT, index=False)
        print(f'\n-> {os.path.relpath(OUT, ROOT)}')
    if len(t) < 30:
        print('\n  One or a few days cannot establish the IV level. The whole point of')
        print('  matching the equity archive range is to get the distribution of IV,')
        print('  not a single draw from it.')


if __name__ == '__main__':
    main()
