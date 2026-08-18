# Part of qqq-microstructure.
#
# The one signal in the daily data that does not fall apart out of sample: intraday
# momentum. The first hour's direction predicts the rest of the session. This is a
# documented effect (Gao, Han, Li & Zhou, "Market Intraday Momentum", JFE 2018) rather
# than something found by sifting this archive, which is the main reason to take it
# seriously at a sample size this small.
#
# Three parts:
#   1. Validate it properly -- walk-forward, per year, with t-stats.
#   2. Build the CONDITIONAL distribution of the entry-to-close move given the signal.
#      That distribution is the thing you price options against.
#   3. Turn it into a strike table: for each strike offset, the probability of finishing
#      in the money and the expected payoff, so any real option chain can be compared
#      against it directly.
#
# Strike selection is not a separate problem. It is this comparison: your estimate of
# where the underlying actually lands, against what the chain charges for each landing
# zone. You need OPRA quotes for the second half; this file supplies the first.

import os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, 'data', 'daily.parquet')
SHARE_COST_BPS = 0.34
ENTRY_MIN = 60          # signal measured over the first hour, enter at 10:30


def load():
    df = pd.read_parquet(DAILY).dropna(subset=['oc_bps']).reset_index(drop=True)
    df['year'] = df.day.str[:4]
    df['sig'] = df[f'ret{ENTRY_MIN}_bps']
    df['y'] = np.log(df['close'] / df[f'p{ENTRY_MIN}']) * 1e4     # entry -> close, bps
    return df.dropna(subset=['sig', 'y'])


def validate(df):
    print('=== 1. IS THE MOMENTUM SIGNAL REAL? ===')
    print(f'  signal: sign of the first {ENTRY_MIN} minutes; hold to the close\n')
    r = np.sign(df.sig.values) * df.y.values - SHARE_COST_BPS
    n = len(r)
    t = r.mean() / (r.std() / np.sqrt(n))
    sharpe = r.mean() / r.std() * np.sqrt(252)
    print(f'  full sample  n={n}  mean {r.mean():+.2f} bps/day  t={t:.2f}  '
          f'Sharpe {sharpe:.2f}  hit {np.mean(r>0)*100:.1f}%')

    print('\n  by year (a real effect should not live in one of them):')
    print(f'    {"year":<7}{"n":>5}{"bps/day":>10}{"t":>7}{"Sharpe":>8}{"hit%":>7}')
    for y, g in df.groupby('year'):
        rr = np.sign(g.sig.values) * g.y.values - SHARE_COST_BPS
        tt = rr.mean() / (rr.std() / np.sqrt(len(rr))) if len(rr) > 5 else np.nan
        print(f'    {y:<7}{len(rr):>5}{rr.mean():>10.2f}{tt:>7.2f}'
              f'{rr.mean()/rr.std()*np.sqrt(252):>8.2f}{np.mean(rr>0)*100:>7.1f}')

    print('\n  walk-forward (expanding window, sign fitted on the past only):')
    cuts = [int(n * f) for f in (0.4, 0.55, 0.7, 0.85)]
    for c in cuts:
        tr, te = df.iloc[:c], df.iloc[c:]
        d = np.sign(np.corrcoef(tr.sig, tr.y)[0, 1])       # direction learned in-sample
        rr = d * np.sign(te.sig.values) * te.y.values - SHARE_COST_BPS
        print(f'    train {len(tr):>4}d -> test {len(te):>4}d   '
              f'{rr.mean():+.2f} bps/day  t={rr.mean()/(rr.std()/np.sqrt(len(rr))):.2f}  '
              f'dir={"+" if d>0 else "-"}')

    print('\n  by signal strength (does more move mean more follow-through?):')
    q = pd.qcut(df.sig.abs(), 5, labels=False)
    print(f'    {"quintile":<10}{"|sig| bps":>11}{"bps/day":>10}{"hit%":>7}')
    for k in range(5):
        m = q == k
        rr = np.sign(df.sig.values[m]) * df.y.values[m] - SHARE_COST_BPS
        print(f'    {k:<10}{df.sig.abs()[m].mean():>11.1f}{rr.mean():>10.2f}'
              f'{np.mean(rr>0)*100:>7.1f}')
    return r


def conditional(df):
    print('\n\n=== 2. CONDITIONAL DISTRIBUTION OF THE ENTRY-TO-CLOSE MOVE ===')
    print('  signed so positive = in the direction the signal points\n')
    s = np.sign(df.sig.values) * df.y.values
    u = df.y.values
    print(f'  {"":<22}{"signal-aligned":>16}{"unconditional":>16}')
    print(f'  {"mean":<22}{s.mean():>16.2f}{u.mean():>16.2f}  bps')
    print(f'  {"sd":<22}{s.std():>16.2f}{np.abs(u).std():>16.2f}  bps')
    print(f'  {"P(move > 0)":<22}{np.mean(s>0)*100:>15.1f}%{np.mean(u>0)*100:>15.1f}%')
    print(f'  {"E|move|":<22}{np.abs(s).mean():>16.2f}{np.abs(u).mean():>16.2f}  bps')
    print()
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f'    p{p:<3} {np.percentile(s,p):>+9.1f} bps = {np.percentile(s,p)/100:>+6.2f} %')


def strikes(df):
    """P(finish ITM) and expected payoff per strike offset -- the strike-selection table."""
    print('\n\n=== 3. STRIKE TABLE -- what the underlying actually does ===')
    print('  Offsets are from spot AT ENTRY, in the direction the signal points.')
    print('  A long call at offset +0.3% pays off when the aligned move exceeds +0.3%.')
    print('  "fair premium" is the expected payoff: pay less than this and you have edge,')
    print('  pay more and you do not. Compare it against the real chain.\n')
    s = np.sign(df.sig.values) * df.y.values / 100.0        # percent, signal-aligned
    spot = df[f'p{ENTRY_MIN}'].mean()
    print(f'  (mean spot at entry ${spot:.2f}; premia below are % of spot and $/share)')
    print(f'  {"offset":>8}{"P(ITM)":>9}{"E[payoff]":>12}{"fair prem":>11}'
          f'{"$ prem":>9}{"P(2x)":>8}')
    for off in (-1.0, -0.5, -0.3, -0.15, 0.0, 0.15, 0.3, 0.5, 1.0, 1.5):
        pay = np.maximum(s - off, 0.0)          # long call struck `off` above entry
        itm = np.mean(s > off)
        fair = pay.mean()
        p2 = np.mean(pay > 2 * fair) if fair > 0 else 0.0
        print(f'  {off:>+7.2f}%{itm*100:>8.1f}%{fair:>11.3f}%{fair:>10.3f}%'
              f'{fair/100*spot:>9.3f}{p2*100:>7.1f}%')
    print('\n  The same table for the opposite side (selling the wing you think is safe):')
    print(f'  {"offset":>8}{"P(finish beyond)":>19}{"E[payout if sold]":>20}')
    for off in (-2.0, -1.5, -1.0, -0.75, -0.5):
        pay = np.maximum(off - s, 0.0)          # short put struck `off` below entry
        print(f'  {off:>+7.2f}%{np.mean(s < off)*100:>18.1f}%{pay.mean():>19.3f}%')


def main():
    if not os.path.exists(DAILY):
        raise SystemExit('run: python src/daily_bars.py 1')
    df = load()
    print(f'QQQ daily spine: {len(df)} days  {df.day.iloc[0]} .. {df.day.iloc[-1]}')
    print(f'entry {ENTRY_MIN}m after the open, exit at the close\n')
    validate(df)
    conditional(df)
    strikes(df)
    print('\n\n=== HOW TO PICK THE STRIKE, ONCE YOU HAVE THE CHAIN ===')
    print('  1. Take the conditional distribution above (or a version of it fitted only')
    print('     on data before the trading day, to avoid look-ahead).')
    print('  2. For every strike in the 0DTE chain, compute')
    print('        EV = integral[ payoff(S_T) x p_realised(S_T) ] - premium_you_pay')
    print('     using the ASK when buying and the BID when selling, never the mid.')
    print('  3. Discard strikes whose bid-ask exceeds a set fraction of the premium --')
    print('     for 0DTE wings that filter removes most of the chain.')
    print('  4. Rank the survivors by EV, and by EV divided by worst-case loss.')
    print('  That is the whole method: your distribution against their prices. Nothing')
    print('  in it can be evaluated without OPRA quotes.')


if __name__ == '__main__':
    main()
