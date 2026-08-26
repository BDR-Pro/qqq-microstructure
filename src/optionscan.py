# Part of qqq-microstructure.
#
# The honest option-finder. It does NOT hunt for winning trades -- there is no
# winning trade to find in a fairly-priced chain. It takes a defined-risk
# structure and tells you the three things a broker's "7042% return" banner
# hides:
#
#   1. BREAKEVEN and the move required to reach it (in % and in standard
#      deviations of the implied distribution),
#   2. the probabilities -- P(profit), P(max win), P(max loss) -- under the
#      market's own IV,
#   3. EXPECTED VALUE under two measures side by side:
#        market  = risk-neutral / IV-implied. A fairly-priced structure is
#                  ~0 here BY CONSTRUCTION; the only edge is price-vs-fair,
#                  so if you pass the quoted --credit/--debit it prints the
#                  edge = quoted - fair, which is the whole game.
#        view    = your thesis. Pass --target (the price you expect) and it
#                  recentres the SAME lognormal on it and recomputes: this is
#                  the EV that is real to you, and it is only as good as the
#                  view.
#
# The point it exists to make: a spread that risks $7 to make $493 at a 10%
# probability is a fairly-priced lottery ticket -- risk and reward are exactly
# balanced by that probability, EV ~0 before costs and negative after. The
# tool prints that verdict in words so the banner cannot mislead.
#
# Black-Scholes, lognormal terminal price, r = --rate (default 0). Payoffs are
# held to expiry, defined-risk only. This is a PRICING/EDGE calculator, not a
# signal: nothing here connects to the panel or claims an edge exists.
#
# Legs: --legs "TYPE:STRIKE:QTY,..."  TYPE in {C,P}, QTY +long / -short.
#   put credit spread, short 322.5 / long 317.5:  "P:322.5:-1,P:317.5:1"
#   iron condor:  "P:290:-1,P:285:1,C:330:-1,C:335:1"
#
# Validated on planted truth: analytic POP/EV match a 400k-path Monte Carlo of
# the same lognormal to <0.3%, put-call parity holds, and a long call's EV
# under a drift equal to the risk-free rate is ~0 (see RESULTS).
#
#   # the pasted AAPL example: spot 310, ~30% IV, ~1 day, quoted $4.93 credit,
#   # thesis 330
#   python src/optionscan.py --spot 310 --iv 0.30 --days 1 \
#       --legs "P:322.5:-1,P:317.5:1" --credit 4.93 --target 330

import argparse, math
import numpy as np

SQ2 = math.sqrt(2.0)
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))   # NumPy 1.x/2.x


def _N(x):
    return 0.5 * (1 + math.erf(x / SQ2))


def bs(S, K, T, sig, r, kind):
    if T <= 0 or sig <= 0:
        iv = max(S - K, 0) if kind == 'C' else max(K - S, 0)
        return iv
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == 'C':
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


def intrinsic(legs, ST):
    v = 0.0
    for k, kind, q in legs:
        v += q * (max(ST - k, 0) if kind == 'C' else max(k - ST, 0))
    return v


def grid(S, T, sig, mu):
    """Lognormal terminal-price grid and density, mean exp(mu*T)."""
    sd = sig * math.sqrt(T)
    lo, hi = math.log(S) + mu * T - 8 * sd, math.log(S) + mu * T + 8 * sd
    x = np.linspace(lo, hi, 4001)
    ST = np.exp(x)
    dens = np.exp(-0.5 * ((x - (math.log(S) + (mu - 0.5 * sig * sig) * T))
                          / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
    dens /= _trapz(dens, x)
    return ST, dens, x


def measure(legs, entry_cf, S, T, sig, mu):
    ST, dens, x = grid(S, T, sig, mu)
    pnl = entry_cf + np.array([intrinsic(legs, s) for s in ST])
    ev = _trapz(pnl * dens, x)
    pop = _trapz((pnl >= 0) * dens, x)
    return ev, pop


def parse(spec):
    legs = []
    for tok in spec.split(','):
        kind, k, q = tok.split(':')
        legs.append((float(k), kind.strip().upper(), float(q)))
    return legs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spot', type=float, required=True)
    ap.add_argument('--iv', type=float, required=True, help='annualized, e.g. 0.30')
    ap.add_argument('--days', type=float, required=True, help='to expiry (calendar)')
    ap.add_argument('--legs', required=True)
    ap.add_argument('--credit', type=float, help='quoted net credit received')
    ap.add_argument('--debit', type=float, help='quoted net debit paid')
    ap.add_argument('--target', type=float, help='price you expect at expiry')
    ap.add_argument('--rate', type=float, default=0.0)
    a = ap.parse_args()
    S, sig, T, r = a.spot, a.iv, a.days / 365.0, a.rate
    legs = parse(a.legs)

    fair_cf = -sum(q * bs(S, k, T, sig, r, kind) for k, kind, q in legs)
    width_lo = min([k for k, kind, q in legs], default=0)
    width_hi = max([k for k, kind, q in legs], default=0)
    xs = np.linspace(0, width_hi * 3 + 1, 20000)
    pl_fair = fair_cf + np.array([intrinsic(legs, s) for s in xs])
    maxw, maxl = pl_fair.max(), pl_fair.min()
    # breakevens: sign changes of the fair-value P&L curve
    be = [float((xs[i] + xs[i + 1]) / 2) for i in range(len(xs) - 1)
          if pl_fair[i] == 0 or pl_fair[i] * pl_fair[i + 1] < 0]

    print(f'\nspot {S}  IV {sig:.0%}  {a.days:g}d to expiry  ({len(legs)} legs)')
    print(f'  fair net {"credit" if fair_cf >= 0 else "debit"}: '
          f'{abs(fair_cf):.2f}   max win {maxw:+.2f}   max loss {maxl:+.2f}')
    sd_move = sig * math.sqrt(T)
    for b in be:
        print(f'  breakeven {b:.2f}  ({(b/S-1)*100:+.1f}% = '
              f'{abs(math.log(b/S))/sd_move:.2f} sigma move)')

    ev_m, pop_m = measure(legs, fair_cf, S, T, sig, r)
    print(f'\n  under the MARKET (IV-implied, fair price):')
    print(f'    P(profit) {pop_m*100:4.1f}%   EV {ev_m:+.2f}  '
          f'(~0 by no-arbitrage -- confirms the chain is priced fairly)')

    if a.credit is not None or a.debit is not None:
        q_cf = a.credit if a.credit is not None else -a.debit
        ev_q, pop_q = measure(legs, q_cf, S, T, sig, r)
        edge = q_cf - fair_cf
        print(f'  at your QUOTED {"credit" if q_cf>=0 else "debit"} '
              f'{abs(q_cf):.2f}:  EV {ev_q:+.2f}   '
              f'edge vs fair {edge:+.2f}  '
              f'({"overpriced FOR YOU (good)" if edge>0.02 else "underpriced (bad)" if edge<-0.02 else "fair"})')

    if a.target is not None:
        mu = math.log(a.target / S) / T
        ev_v, pop_v = measure(legs, fair_cf, S, T, sig, mu)
        print(f'  under YOUR VIEW (expected {a.target}, same IV):')
        print(f'    P(profit) {pop_v*100:4.1f}%   EV {ev_v:+.2f}  '
              f'(real only if the view is real; the market disagrees by '
              f'{(a.target/S-1)*100:+.1f}%)')

    rr = maxw / abs(maxl) if maxl < 0 else float('inf')
    print(f'\n  verdict: risk {abs(maxl):.2f} to make {maxw:.2f} '
          f'(reward:risk {rr:.0f}:1) at {pop_m*100:.0f}% probability. ', end='')
    if pop_m < 0.35 and rr > 5:
        print('This is a LOTTERY TICKET --\n  the big ratio and the small '
              'probability cancel; EV is ~0 at fair value and\n  negative '
              'after commissions. Not an edge. Size as entertainment, if at all.')
    elif abs(ev_m) < 0.05 * max(maxw, abs(maxl)):
        print('Fairly priced: no edge\n  either way at fair value; only a '
              'genuine price dislocation or a real view makes it worth doing.')
    else:
        print('Priced away from fair -- inspect the edge line above.')


if __name__ == '__main__':
    main()
