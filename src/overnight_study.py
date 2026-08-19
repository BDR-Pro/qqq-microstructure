# Part of qqq-microstructure.
#
# Signal #2: the overnight/intraday return split (Lou, Polk & Skouras; Asness).
# Documented for decades: US equity returns accrue almost entirely overnight while
# intraday averages ~zero. Tested here on the validated HF minute-bar spines.
#
# Unadjusted-price caveats, both conservative for this result: split days are dropped
# (|overnight| > 15%), and dividends land as negative overnight PRICE moves while the
# holder actually receives them -- so measured overnight UNDERSTATES the true return.
#
# The stack logic: the overnight leg holds 16:00 -> 09:30 and the momentum leg trades
# 10:30 -> 16:00. They never overlap, so ONE pot of capital runs both.

import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = 0.34


def stats(r, lab):
    n = len(r)
    sh = r.mean() / r.std() * np.sqrt(252)
    t = r.mean() / (r.std() / np.sqrt(n))
    eq = (1 + pd.Series(r) / 1e4).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = (eq.iloc[-1] ** (252 / n) - 1) * 100
    print(f"  {lab:<16} {r.mean():>+6.2f} bps/d  t={t:>5.2f}  Sharpe {sh:.2f}  "
          f"CAGR {cagr:5.2f}%  maxDD {dd*100:6.1f}%  = {((1+cagr/100)**(1/12)-1)*100:.2f}%/mo")


def main():
    print(f"{'ticker':>7}{'days':>7} | {'overnight bps/d':>16}{'t':>7} | "
          f"{'intraday bps/d':>15}{'t':>7}")
    for tk in ('QQQ', 'SPY', 'IWM', 'DIA'):
        d = pd.read_parquet(os.path.join(ROOT, 'data', f'daily_hf_{tk}.parquet')) \
              .dropna(subset=['oc_bps', 'prev_close'])
        on = np.log(d['open'].values / d['prev_close'].values) * 1e4
        keep = np.abs(on) < 1500
        on, ic = on[keep], d['oc_bps'].values[keep]
        n = len(on)
        print(f"{tk:>7}{n:>7} | {on.mean():>16.2f}"
              f"{on.mean()/(on.std()/np.sqrt(n)):>7.2f} | {ic.mean():>15.2f}"
              f"{ic.mean()/(ic.std()/np.sqrt(n)):>7.2f}")

    d = pd.read_parquet(os.path.join(ROOT, 'data', 'daily_hf_QQQ.parquet')) \
          .dropna(subset=['oc_bps', 'prev_close', 'p60'])
    d['on_r'] = np.log(d['open'] / d['prev_close']) * 1e4
    d = d[d.on_r.abs() < 1500].reset_index(drop=True)
    on = d.on_r.values - COST
    mom = (np.sign(d['ret60_bps'].values)
           * np.log(d['close'].values / d['p60'].values) * 1e4 - COST)

    print(f"\ncorrelation(overnight, momentum) = {np.corrcoef(on, mom)[0,1]:+.3f}")
    print("\nQQQ, full history, net of costs:")
    stats(on, 'overnight only')
    stats(mom, 'momentum only')
    stats(on + mom, 'STACK')

    print("\nstack by 4-year era:")
    d2 = d.assign(c=on + mom)
    d2['era'] = (d2.day.str[:4].astype(int) // 4) * 4
    for e, g in d2.groupby('era'):
        print(f"  {e}-{e+3}: {g.c.mean():+6.2f} bps/day")

    h = d.day > '20251031'
    print(f"\nholdout (Nov 2025 - Mar 2026, {h.sum()} days):")
    print(f"  overnight {on[h].sum()/100:+.2f}%   momentum {mom[h].sum()/100:+.2f}%   "
          f"stack {(on+mom)[h].sum()/100:+.2f}%")
    print("  Zero long-run correlation did not prevent a jointly bad Feb-Mar 2026.")


if __name__ == '__main__':
    main()
