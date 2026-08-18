# Part of qqq-microstructure.
#
# Ranks 0DTE option structures on QQQ by EXPECTED VALUE, not by payoff shape.
#
# A payoff calculator (optionsprofitcalculator.com and friends) answers "what do I make
# if the stock finishes at X". It cannot answer "should I put this on", because its
# probability model is a lognormal at whatever IV you type in -- which assumes the
# options are fairly priced and therefore that every structure has zero edge. That is
# the assumption under test, so it cannot be an input.
#
# This file replaces that assumption with the empirical distribution measured from 515
# days of QQQ tick data, aligned to the intraday-momentum signal, and prices every
# structure against it:
#
#     EV = E_empirical[payoff at 16:00] - what the market charges you at entry
#
# The market side is Black-Scholes at an assumed IV, because this archive has no options
# data. That makes IV the one number here you should replace with reality: pass --iv
# from a live chain, or better, wire in OPRA quotes and use ask-when-buying,
# bid-when-selling per leg.
#
# Structures needing more than one expiry -- calendars, diagonals, PMCC -- cannot be
# evaluated on a single-session distribution and are excluded on purpose.

import os, sys, argparse
from math import sqrt, log, exp, erf
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, 'data', 'daily.parquet')
ENTRY_MIN = 60
SESSION_FRAC = 5.5 / 6.5          # 10:30 -> 16:00 as a share of the trading day
CONTRACT = 100                    # shares per contract


def N(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(kind, k_pct, sigma_pct):
    """Price in % of spot. Strike is k_pct above spot (negative = below). r=0."""
    S, K, s = 1.0, 1.0 + k_pct / 100.0, sigma_pct / 100.0
    if s <= 0:
        return max(S - K, 0.0) * 100 if kind == 'c' else max(K - S, 0.0) * 100
    d1 = (log(S / K) + s * s / 2) / s
    d2 = d1 - s
    v = S * N(d1) - K * N(d2) if kind == 'c' else K * N(-d2) - S * N(-d1)
    return v * 100.0


def payoff(legs, s):
    """Structure payoff at expiry, % of spot, for terminal move `s` (% of spot)."""
    out = np.zeros_like(s)
    for kind, k, q in legs:
        out += q * (np.maximum(s - k, 0.0) if kind == 'c' else np.maximum(k - s, 0.0))
    return out


def cost(legs, sigma_pct, half_spread_pct):
    """What you pay (positive) or receive (negative), % of spot, crossing the spread."""
    c = 0.0
    for kind, k, q in legs:
        mid = bs(kind, k, sigma_pct)
        c += q * (mid + half_spread_pct if q > 0 else mid - half_spread_pct)
    return c


# Structures. Coordinates are SIGNAL-ALIGNED: +x% means x% in the direction the
# momentum signal points, so "long call" is a continuation bet regardless of side.
def build(w=0.5, ww=1.0):
    return {
        # --- basic, directional with the signal
        'Long call ATM':            [('c', 0.0, 1)],
        f'Long call +{w}% OTM':     [('c', w, 1)],
        'Long put ATM (fade)':      [('p', 0.0, 1)],
        f'Long put -{w}% OTM':      [('p', -w, 1)],
        f'Naked short put -{w}%':   [('p', -w, -1)],
        f'Naked short call +{w}%':  [('c', w, -1)],
        # --- spreads
        f'Bull call spread 0/+{w}': [('c', 0.0, 1), ('c', w, -1)],
        f'Put credit spread -{w}/-{ww}': [('p', -w, -1), ('p', -ww, 1)],
        f'Call credit spread +{w}/+{ww}': [('c', w, -1), ('c', ww, 1)],
        f'Bear put spread 0/-{w}':  [('p', 0.0, 1), ('p', -w, -1)],
        f'Ratio backspread +{w}':   [('c', 0.0, -1), ('c', w, 2)],
        # --- non-directional
        'Long straddle ATM':        [('c', 0.0, 1), ('p', 0.0, 1)],
        'Short straddle ATM':       [('c', 0.0, -1), ('p', 0.0, -1)],
        f'Long strangle +/-{w}':    [('c', w, 1), ('p', -w, 1)],
        f'Short strangle +/-{w}':   [('c', w, -1), ('p', -w, -1)],
        f'Iron condor {w}/{ww}':    [('p', -w, -1), ('p', -ww, 1),
                                     ('c', w, -1), ('c', ww, 1)],
        f'Call butterfly 0/{w}/{ww}': [('c', 0.0, 1), ('c', w, -2), ('c', ww, 1)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iv', type=float, default=16.0, help='0DTE IV, annualised %%')
    ap.add_argument('--half-spread', type=float, default=0.01,
                    help='half the bid-ask per leg, in dollars')
    ap.add_argument('--spot', type=float, default=None, help='spot price for $ figures')
    ap.add_argument('--aligned', action='store_true', default=True)
    ap.add_argument('--unconditional', action='store_true',
                    help='ignore the momentum signal (for the non-directional structures)')
    ap.add_argument('--width', type=float, default=0.5)
    ap.add_argument('--wing', type=float, default=1.0)
    a = ap.parse_args()

    if not os.path.exists(DAILY):
        raise SystemExit('run: python src/daily_bars.py 1')
    df = pd.read_parquet(DAILY).dropna(subset=['oc_bps'])
    y = np.log(df['close'].values / df[f'p{ENTRY_MIN}'].values) * 1e4
    s_aligned = np.sign(df[f'ret{ENTRY_MIN}_bps'].values) * y / 100.0   # %
    s = (y / 100.0) if a.unconditional else s_aligned
    spot = a.spot or float(df[f'p{ENTRY_MIN}'].mean())
    sigma = a.iv / sqrt(252) * sqrt(SESSION_FRAC)          # % over the session
    hs_pct = a.half_spread / spot * 100.0

    print(f'0DTE structure ranking | QQQ | {len(s)} days | entry {ENTRY_MIN}m after open, '
          f'exit at close')
    print(f'distribution: {"unconditional" if a.unconditional else "momentum-aligned"} '
          f'(mean {s.mean()*100:+.1f} bps, sd {s.std()*100:.1f} bps)')
    print(f'market: IV {a.iv:.1f}% -> session sigma {sigma:.3f}% of spot | '
          f'half-spread ${a.half_spread:.3f}/leg | spot ${spot:.2f}\n')

    rows = []
    for name, legs in build(a.width, a.wing).items():
        c = cost(legs, sigma, hs_pct)
        pay = payoff(legs, s)
        ev = pay.mean() - c
        pnl = pay - c
        # bound the tail on a grid wide enough to catch the naked wings
        grid = np.linspace(-15, 15, 6001)
        gp = payoff(legs, grid) - c
        rows.append(dict(
            name=name, cost_pct=c, cost_usd=c / 100 * spot * CONTRACT,
            ev_pct=ev, ev_usd=ev / 100 * spot * CONTRACT,
            win=(pnl > 0).mean() * 100,
            worst_usd=gp.min() / 100 * spot * CONTRACT,
            best_usd=gp.max() / 100 * spot * CONTRACT,
            worst_seen=pnl.min() / 100 * spot * CONTRACT))
    t = pd.DataFrame(rows).sort_values('ev_usd', ascending=False)
    t['ror'] = np.where(t.worst_usd < 0, t.ev_usd / -t.worst_usd * 100, np.nan)

    print(f'{"structure":<30}{"debit(+)":>10}{"EV $/ctr":>10}{"win%":>7}'
          f'{"max loss":>11}{"max gain":>11}{"EV/risk%":>10}')
    print('-' * 89)
    for _, r in t.iterrows():
        ml = '-inf' if r.worst_usd < -1e5 else f'{r.worst_usd:,.0f}'
        mg = 'inf' if r.best_usd > 1e5 else f'{r.best_usd:,.0f}'
        ror = '' if not np.isfinite(r.ror) else f'{r.ror:>9.1f}'
        print(f'{r["name"]:<30}{r.cost_usd:>10.0f}{r.ev_usd:>10.2f}{r.win:>7.1f}'
              f'{ml:>11}{mg:>11}{ror:>10}')
    print('-' * 89)
    print('debit(+) = you pay, (-) = you receive. EV is per contract (100 shares),')
    print('after crossing the bid-ask on every leg. "max loss" is the theoretical bound,')
    print('not the worst day in the sample.\n')

    print('=== BREAK-EVEN IV: the level at which each structure has zero EV ===')
    print('  Above this, short-vol structures win and long-vol structures lose.')
    print(f'  {"structure":<30}{"break-even IV":>15}{"to profit":>12}{"@%.0f%%" % a.iv:>10}')
    for name, legs in build(a.width, a.wing).items():
        pay = payoff(legs, s).mean()

        def ev_at(iv):
            return pay - cost(legs, iv / sqrt(252) * sqrt(SESSION_FRAC), hs_pct)

        # EV rises with IV for short-vol structures and falls for long-vol ones, so
        # bracket first and bisect on the sign change rather than assuming a direction.
        lo, hi = 1.0, 120.0
        elo, ehi = ev_at(lo), ev_at(hi)
        if elo * ehi > 0:
            print(f'  {name:<30}{"no crossing":>15}{"":>12}')
            continue
        for _ in range(80):
            m = (lo + hi) / 2
            em = ev_at(m)
            if em * elo <= 0:
                hi, ehi = m, em
            else:
                lo, elo = m, em
        be = (lo + hi) / 2
        # which side of the break-even pays, for THIS structure
        tag = 'need HIGHER' if ev_at(be + 1) > 0 else 'need LOWER'
        edge = 'edge' if ev_at(a.iv) > 0 else 'no edge'
        print(f'  {name:<30}{be:>14.1f}%{tag:>12}{edge:>10}')

    print('\n=== HOW TO USE THIS WITH A LIVE CHAIN ===')
    print('  1. Read the real 0DTE IV off the chain at 10:30 and re-run with --iv.')
    print('  2. Anything whose break-even IV sits far from the real IV is the trade;')
    print('     structures near it are coin flips after commissions.')
    print('  3. optionsprofitcalculator.com is the right tool for the NEXT step: paste')
    print('     the winner in with real strikes and premiums to see the payoff shape,')
    print('     margin and breakevens. It will not rank structures for you, because it')
    print('     has no view on where the stock actually lands. This does.')
    print('  4. Then check the max-loss column against your account, not against the EV.')


if __name__ == '__main__':
    main()
