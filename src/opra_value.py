# Part of qqq-microstructure.
#
# The RESULTS 8 EV tables, finally against MEASURED premiums. Reads the entry
# slices opra_pull.py buys (the QQQ chain 10:25-10:35 ET, one parquet per day),
# takes the minute closest to 10:30 ET (DST-correct), recovers spot by put-call
# parity and IV from the ATM straddle (opra_load.py's machinery, without its
# fixed-UTC constants), and values RESULTS 8's structures in the momentum
# direction at the prices a real order faces -- ASK when buying, BID when
# selling. The signal (sign of QQQ's first hour) and the settlement (the 16:00
# close) come from the equity panel, so option data is only needed at entry.
#
# Structures, pre-declared from RESULTS 8, no sweeps:
#   long ATM      buy the ATM call (signal up) / put (signal down) at the ask
#   spread        sell the 0.5%-OTM put at the bid, buy the 1.5%-OTM put at the
#                 ask (mirrored to calls when the signal is down)
#   condor        sell both 0.5% wings at the bid, buy both 1.0% wings at the ask
#
# The gate is RESULTS 8/17's line, 14.8% annualised: buy-side structures are
# expected to pay below it, sell-side above it. Both cells are printed for every
# structure -- reporting, not selection. Commissions default to $0.65 per
# contract per side, charged on entry legs only (expiry is not a trade;
# assignment fees on ITM shorts are ignored and said so).
#
# Days pulled after the equity panel's last month have no signal or settlement
# and are skipped with a count -- extend the panel (xsec_extract.py) to value
# them.
#
#   python src/opra_value.py [--comm 0.65]

import os, glob, argparse, math, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
from xsec_backtest import load_panel, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPRA = os.path.join(ROOT, 'data', 'opra')
ET = ZoneInfo('America/New_York')
SF = 5.5 / 6.5
BE = 14.8                                   # RESULTS 8 break-even, annualised %


def load_day(path):
    df = pd.read_parquet(path)
    if not len(df):
        return None, None
    tcol = next(c for c in df.columns
                if pd.api.types.is_datetime64_any_dtype(df[c]))
    df = df.set_index(tcol)
    s = df.symbol.astype(str)
    df['exp'] = s.str[6:12]
    df['cp'] = s.str[12]
    df['strike'] = pd.to_numeric(s.str[13:21], errors='coerce') / 1000.0
    day = os.path.basename(path)[:8]
    z = df[(df.exp == day[2:]) & (df.bid_px_00 > 0) & (df.ask_px_00 > 0)
           & (df.ask_px_00 >= df.bid_px_00)].copy()
    return day, z


def entry_chain(day, z):
    d = dt.date(int(day[:4]), int(day[4:6]), int(day[6:8]))
    tgt = pd.Timestamp(dt.datetime.combine(d, dt.time(10, 30), ET)
                       .astimezone(dt.timezone.utc))
    ts = z.index.unique()
    t = ts[np.argmin(np.abs((ts - tgt).to_numpy()))]
    g = z[z.index == t]
    c = g[g.cp == 'C'].groupby('strike')[['bid_px_00', 'ask_px_00']].first()
    p = g[g.cp == 'P'].groupby('strike')[['bid_px_00', 'ask_px_00']].first()
    k = c.index.intersection(p.index)
    c, p = c.loc[k].sort_index(), p.loc[k].sort_index()
    c['mid'] = (c.bid_px_00 + c.ask_px_00) / 2
    p['mid'] = (p.bid_px_00 + p.ask_px_00) / 2
    return c, p


def value_day(day, z, q):
    c, p = entry_chain(day, z)
    if len(c) < 5:
        return None
    par = c.index.values + c['mid'].values - p['mid'].values
    spot = float(par[np.argmin(np.abs(c['mid'].values - p['mid'].values))])
    ks = c.index.values
    near = lambda x: float(ks[np.argmin(np.abs(ks - x))])
    Ka = near(spot)
    iv = (c.mid[Ka] + p.mid[Ka]) / (0.7979 * spot) / math.sqrt(SF / 252) * 100
    sgn = 1 if q['p60'] >= q['open'] else -1
    ST = q['close']

    if sgn > 0:
        ev_long = (max(ST - Ka, 0) - c.ask_px_00[Ka]) / spot * 1e4
        k1, k2 = near(spot * 0.995), near(spot * 0.985)
        cr = p.bid_px_00[k1] - p.ask_px_00[k2]
        ev_sprd = (cr - (max(k1 - ST, 0) - max(k2 - ST, 0))) / spot * 1e4
    else:
        ev_long = (max(Ka - ST, 0) - p.ask_px_00[Ka]) / spot * 1e4
        k1, k2 = near(spot * 1.005), near(spot * 1.015)
        cr = c.bid_px_00[k1] - c.ask_px_00[k2]
        ev_sprd = (cr - (max(ST - k1, 0) - max(ST - k2, 0))) / spot * 1e4
    ps, cs, pl, cl = (near(spot * 0.995), near(spot * 1.005),
                      near(spot * 0.99), near(spot * 1.01))
    crc = (p.bid_px_00[ps] + c.bid_px_00[cs]
           - p.ask_px_00[pl] - c.ask_px_00[cl])
    pay = (-(max(ps - ST, 0) - max(pl - ST, 0))
           - (max(ST - cs, 0) - max(ST - cl, 0)))
    ev_cond = (crc + pay) / spot * 1e4
    return dict(day=day, iv=iv, spot=spot, sig=sgn,
                diag_bps=(spot / q['p60'] - 1) * 1e4,
                ev_long=ev_long, ev_sprd=ev_sprd, ev_cond=ev_cond)


def cell(t, col, mask):
    v = t.loc[mask, col]
    if len(v) < 3:
        return f'{"--":>18}'
    tt = v.mean() / (v.std() / np.sqrt(len(v))) if v.std() > 0 else np.nan
    return f'{v.mean():>+8.1f} t={tt:>4.1f} {(v > 0).mean() * 100:3.0f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comm', type=float, default=0.65,
                    help='$ per contract per side, entry legs only')
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(OPRA, '*.parquet')))
    if not files:
        raise SystemExit('no files in data/opra -- run opra_pull.py first')
    df = load_panel()
    qq = df[df.ticker.isin({'QQQ', 'QQQQ'})].sort_values(['day', 'ticker']) \
           .groupby('day').first()
    qq = qq[qq.p60 > 0][['open', 'p60', 'close']]

    rows, empty, nopanel = [], 0, 0
    for f in files:
        day, z = load_day(f)
        if z is None or not len(z):
            empty += 1
            continue
        if day not in qq.index:
            nopanel += 1
            continue
        r = value_day(day, z, qq.loc[day])
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    if not len(t):
        raise SystemExit('no valuable days')
    print(f'\n{len(t)} days valued ({t.day.min()} .. {t.day.max()});  '
          f'{empty} empty files, {nopanel} beyond the equity panel')
    print(f'parity spot vs panel p60: |diff| mean '
          f'{t.diag_bps.abs().mean():.1f} bps  p95 '
          f'{t.diag_bps.abs().quantile(.95):.1f} bps   (cross-source check)')

    qs = t.iv.quantile([.1, .5, .9])
    print(f'\nMEASURED 0DTE IV at 10:30: p10 {qs.iloc[0]:.1f}  median '
          f'{qs.iloc[1]:.1f}  p90 {qs.iloc[2]:.1f}   days above {BE}%: '
          f'{(t.iv > BE).mean() * 100:.1f}%   (the iv_regime proxy said 63.3%)')

    hi = t.iv > BE
    print(f'\nEV per day, bps of spot, GROSS (mean, t, win%); '
          f'gate = measured IV vs {BE}:')
    print(f'{"structure":<12}{"all days":>20}{"IV<=14.8":>20}{"IV>14.8":>20}')
    for col, lab in (('ev_long', 'long ATM'), ('ev_sprd', 'spread'),
                     ('ev_cond', 'condor')):
        print(f'{lab:<12}{cell(t, col, t.index >= 0)}{cell(t, col, ~hi)}'
              f'{cell(t, col, hi)}')
    legs = {'ev_long': 1, 'ev_sprd': 2, 'ev_cond': 4}
    cb = {k: a.comm * v / (t.spot.mean() * 100) * 1e4 for k, v in legs.items()}
    print(f'commissions at ${a.comm}/contract/side (entry legs; assignment '
          f'fees ignored):\n  ' + '  '.join(
              f'{k[3:]} -{v:.2f} bps' for k, v in cb.items()))

    print(f'\n{"year":>6}{"mean IV":>9}{">14.8%":>8}{"long":>8}{"spread":>8}'
          f'{"condor":>8}')
    for y, g in t.groupby(t.day.str[:4]):
        print(f'{y:>6}{g.iv.mean():>9.1f}{(g.iv > BE).mean() * 100:>7.0f}%'
              f'{g.ev_long.mean():>8.1f}{g.ev_sprd.mean():>8.1f}'
              f'{g.ev_cond.mean():>8.1f}')

    print('\nfull-period daily series, gross:')
    for col, lab in (('ev_long', 'long ATM'), ('ev_sprd', 'spread'),
                     ('ev_cond', 'condor')):
        stats(t[col].values, lab)
    print('\nworst 5 days (the RESULTS 8 tail check, at real prices):')
    for col, lab in (('ev_sprd', 'spread'), ('ev_cond', 'condor')):
        w = t.nsmallest(5, col)
        print(f'  {lab:<8}' + '  '.join(
            f'{r.day} {getattr(r, col):+.0f}' for r in w.itertuples()))

    p = os.path.join(ROOT, 'data', 'opra_daily.parquet')
    t.to_parquet(p, index=False)
    print(f'\ndaily rows -> {p}')


if __name__ == '__main__':
    main()
