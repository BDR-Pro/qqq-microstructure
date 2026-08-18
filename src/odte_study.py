# Part of qqq-microstructure.
#
# 0DTE feasibility on QQQ: open a position at the bell, close it at the bell.
#
# There is no options data in this archive -- no strikes, no premiums, no implied vol.
# So this does not price a single option. It tests the two things the underlying can
# settle on its own, both of which sit upstream of any option:
#
#   1. DIRECTION. Buying a 0DTE call or put is a leveraged, cost-laden bet on the
#      open-to-close move. Trading the shares is the same bet with the leverage and the
#      premium stripped out -- a strict upper bound on the option version. If the share
#      trade has no edge, no option structure built on it can have one either.
#
#   2. THE PRICE OF VOLATILITY. A long straddle wins when the realised move exceeds the
#      premium. Under Black-Scholes the expected absolute move is ~0.798*S*sigma and an
#      ATM straddle costs ~0.798*S*IV, so the trade is a pure bet on IV vs realised vol.
#      We can measure realised exactly, and solve for the break-even IV.
#
# The power calculation comes first on purpose. With 515 days, most of what a daily
# strategy could plausibly earn is smaller than the noise in the sample.

import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, 'data', 'daily.parquet')

# What it costs to trade the shares round trip, in bps. QQQ NBBO is ~1 tick.
SHARE_COST_BPS = 0.34


def power(df):
    sd = df.oc_bps.std()
    n = len(df)
    se = sd / np.sqrt(n)
    print('=== STATISTICAL POWER -- read this before any result below ===')
    print(f'  days                          {n}')
    print(f'  open-to-close sd              {sd:.1f} bps/day')
    print(f'  standard error of the mean    {se:.2f} bps/day')
    print(f'  2-sigma detectable edge       {2*se:.2f} bps/day')
    ann = 2 * se * np.sqrt(252) / sd * np.sqrt(252) / np.sqrt(252)
    sharpe_min = (2 * se) / sd * np.sqrt(252)
    print(f'  -> minimum detectable Sharpe  {sharpe_min:.2f} annualised')
    print(f'  A Sharpe 1.0 strategy earns {sd/np.sqrt(252):.2f} bps/day, which is '
          f'{(sd/np.sqrt(252))/se:.2f} sigma here.')
    print('  Anything weaker than the figure above is invisible in this sample: absence')
    print('  of evidence, not evidence of absence.\n')


def distribution(df):
    a = df.oc_bps.abs()
    print('=== OPEN-TO-CLOSE MOVE DISTRIBUTION ===')
    print(f'  mean {df.oc_bps.mean():+.2f} bps   sd {df.oc_bps.std():.1f} bps   '
          f'up days {(df.oc_bps>0).mean()*100:.1f}%')
    print(f'  mean |move| {a.mean():.1f} bps   median |move| {a.median():.1f} bps')
    for p in (10, 25, 50, 75, 90, 95, 99):
        print(f'    p{p:<3} |move| {np.percentile(a,p):>7.1f} bps '
              f'= {np.percentile(a,p)/1e4*100:.2f}%')
    print(f'  intraday range  mean {df.range_bps.mean():.1f} bps  '
          f'median {df.range_bps.median():.1f} bps')
    print(f'  -> the average day travels {df.range_bps.mean()/a.mean():.2f}x further than '
          f'it finishes from where it started\n')


def straddle(df):
    """Break-even implied vol for a long ATM 0DTE straddle held open to close."""
    a = df.oc_bps.abs() / 1e4
    e_abs = a.mean()
    # E|move| = 0.798 * sigma  =>  the realised vol the market must be underpricing
    sigma_eff = e_abs / 0.7979
    straddle_cost = 0.7979 * sigma_eff          # by construction, the fair premium
    print('=== LONG 0DTE STRADDLE: WHAT WOULD HAVE TO BE TRUE ===')
    print(f'  realised E|open-to-close move|      {e_abs*100:.3f} % of spot')
    print(f'  implied daily vol that fair-prices  {sigma_eff*100:.3f} %')
    print(f'  annualised                          {sigma_eff*np.sqrt(252)*100:.1f} %')
    print(f'  ATM straddle at that vol            {straddle_cost*100:.3f} % of spot')
    print(f'  -> a long straddle breaks even only if 0DTE IV prints below '
          f'{sigma_eff*np.sqrt(252)*100:.1f}% annualised,')
    print(f'     BEFORE the bid-ask on two option legs and before commissions.')
    win = (a > straddle_cost).mean()
    print(f'  at exactly fair value it wins {win*100:.1f}% of days and loses the '
          f'premium on the rest.')
    print('  Whether real 0DTE IV sits above or below this cannot be answered from this')
    print('  archive -- it needs an OPRA options feed. See the note at the end.\n')


def directional(df):
    """Is the open-to-close move predictable from what you know at the bell?"""
    print('=== DIRECTIONAL EDGE, SHARES ONLY (upper bound for any long-option version) ===')
    # only what is observable at or before the entry moment
    entries = {
        '09:30 (bell)': ([ 'gap_bps', 'prev_oc_bps', 'prev_rv', 'dow'], 'oc_bps', None),
        '09:45 (15m)':  (['ret15_bps', 'range15_bps', 'gap_bps', 'prev_oc_bps', 'prev_rv', 'dow'],
                         None, 15),
        '10:30 (60m)':  (['ret60_bps', 'range60_bps', 'gap_bps', 'prev_oc_bps', 'prev_rv', 'dow'],
                         None, 60),
    }
    cut = int(len(df) * 0.6)
    for label, (feats, tgt, mins) in entries.items():
        d = df.copy()
        if tgt is None:                      # entry later: target is entry -> close
            d['y'] = np.log(d['close'] / d[f'p{mins}']) * 1e4
        else:
            d['y'] = d[tgt]
        d = d.dropna(subset=feats + ['y'])
        tr, te = d.iloc[:cut], d.iloc[cut:]
        print(f'\n  entry {label}   n={len(d)}  target sd {d.y.std():.1f} bps')
        print(f'    {"feature":<14}{"IC all":>9}{"IC train":>10}{"IC test":>9}')
        for f in feats:
            ic  = d[[f, 'y']].corr().iloc[0, 1]
            ict = tr[[f, 'y']].corr().iloc[0, 1]
            ice = te[[f, 'y']].corr().iloc[0, 1]
            flag = '  <-- flips' if ict * ice < 0 else ''
            print(f'    {f:<14}{ic:>9.4f}{ict:>10.4f}{ice:>9.4f}{flag}')
        # simplest possible tradable rule per feature: sign, sized flat
        print(f'    {"rule":<14}{"train bps/d":>13}{"test bps/d":>12}{"test hit%":>11}')
        for f in feats:
            if f == 'dow':
                continue
            s = np.sign(tr[f].values)
            mtr = np.mean(s * tr.y.values) - SHARE_COST_BPS
            s2 = np.sign(te[f].values)
            mte = np.mean(s2 * te.y.values) - SHARE_COST_BPS
            hit = np.mean(np.sign(s2 * te.y.values) > 0) * 100
            print(f'    sign({f[:9]:<9}){mtr:>13.2f}{mte:>12.2f}{hit:>10.1f}%')


def main():
    if not os.path.exists(DAILY):
        raise SystemExit('run: python src/daily_bars.py 1')
    df = pd.read_parquet(DAILY).dropna(subset=['oc_bps'])
    print(f'QQQ daily spine: {len(df)} days  {df.day.iloc[0]} .. {df.day.iloc[-1]}\n')
    power(df)
    distribution(df)
    straddle(df)
    directional(df)
    print('\n=== WHAT THIS CANNOT ANSWER ===')
    print('  Every number above describes the underlying. The 0DTE question that actually')
    print('  decides P&L -- what the options cost -- needs an OPRA feed with per-strike')
    print('  quotes. Databento sell it as OPRA.PILLAR. Until that exists here, any 0DTE')
    print('  backtest would be inventing its single most important input.')


if __name__ == '__main__':
    main()
