# Part of qqq-microstructure.
#
# Two sizing/hedging layers on the measured book, pre-declared here before any
# real-data run. Neither touches the frozen models or the registered forward
# metrics -- both are overlays on outputs, so the evidence clock (RESULTS 19)
# is unaffected. Both are honest tests with the sign at risk, not knobs:
#
#   1. BETA-SCALED HEDGE. RESULTS 16 flagged that NEU's 1:1 QQQ hedge
#      under-hedges (Q5 names carry overnight beta > 1; corr(NEU, QQQ_ON)
#      printed +0.55 where a hedge should print ~0). Replace the 1 with a
#      trailing regression beta. Spec, fixed: 252-day rolling cov/var of the
#      basket's overnight on QQQ's, at least 126 pairs, lagged one day (day t
#      is hedged with the beta known at t-1), clamped to [0.5, 2.0]; the QQQ
#      leg's 0.34 bps round trip scales with beta.
#        PASS iff |corr(NEUb, QQQ_ON)| < 0.20 on the beta-defined window.
#
#   2. VOL-TARGET OVERLAY -- de-lever only. RESULTS 19's Monte Carlo priced
#      v2's tail (1-in-4 chance of a -20% month) and vetoed margin; this is
#      the mirror image: exposure = min(1, target / trailing vol), never above
#      1. Spec, fixed: 21-day trailing std of the book's own dailies; target =
#      the expanding median of that vol series (causal, parameter-free), first
#      defined after 252 observations; both lagged one day. Applied to v2,
#      v2n, and v2nb (= NEUb + MOM). Raw and scaled are compared on the SAME
#      days, then both run through RESULTS 19's block bootstrap (mc_risk
#      machinery, 21-day blocks, seed 7) at leverage 1.
#        PASS iff bootstrap CAGR p50 / |maxDD p95| improves AND P(-20% month)
#        does not rise.
#      Stated risk, on the record before the run: v2 earned most in storms
#      (2008-11 was its best era, +15.4 bps/day), so de-levering storms may
#      cut the mean more than the tail -- FAIL is a live outcome.
#
# Validated on planted truth before real data (see RESULTS): on a two-regime
# series (calm sigma 50 / storm sigma 250, same mean everywhere) the overlay
# holds storm vol near target and PASSes; on a series whose entire mean lives
# in the storms -- the stated v2 risk, enacted -- it correctly FAILs; a
# planted beta of 1.4 is recovered to 0.02 with residual correlation 0.00,
# and forcing beta = 1 reproduces stack_v2's NEU column to the float. No
# lookahead: beta and exposure at hand-checked positions equal the values
# computable from t-1 information to 1e-9.
#
# Needs data/stack_daily.csv (run stack_v2.py first).
#
#   python src/overlay.py [--paths 10000]

import os, argparse
import numpy as np, pandas as pd
from xsec_backtest import stats
from stack_v2 import QQQ_RT
from mc_risk import paths, BLOCK, SEED, HORIZON_A

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOL_WIN = 21            # trailing window for realised vol
VOL_BURN = 252          # observations of trailing vol before the target exists
BETA_WIN = 252          # trailing window for the overnight beta
BETA_MIN = 126          # minimum pairs before a beta is used
BETA_CLAMP = (0.5, 2.0)


def beta_hedged(L):
    """NEUb: the basket hedged with trailing beta instead of 1:1. Cost
    constants cancel out of cov/var, so beta on the net columns IS the gross
    beta; the hedge cost scales with the QQQ notional actually shorted."""
    j = L[['ON', 'QQQ_ON']].dropna()
    cov = j.ON.rolling(BETA_WIN, min_periods=BETA_MIN).cov(j.QQQ_ON)
    var = j.QQQ_ON.rolling(BETA_WIN, min_periods=BETA_MIN).var()
    b = (cov / var).shift(1).clip(*BETA_CLAMP)
    neub = j.ON - b * (j.QQQ_ON + 2 * QQQ_RT)
    return b.dropna(), neub.dropna()


def vol_scaled(r):
    """Exposure min(1, target/vol), vol and target known at t-1, never > 1."""
    rv = r.rolling(VOL_WIN).std()
    rv = rv.where(rv > 0)
    tgt = rv.expanding(VOL_BURN).median()
    w = (tgt / rv).clip(upper=1.0).shift(1)
    return w.dropna()


def mc_row(r, n, rng):
    p = paths(r, n, HORIZON_A, BLOCK, rng)
    eq = np.cumprod(1 + p / 1e4, axis=1)
    cagr = eq[:, -1] ** (252 / HORIZON_A) - 1
    dd = (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1)
    mo = p[:, :HORIZON_A - HORIZON_A % 21].reshape(len(p), -1, 21).sum(2) / 1e4
    return dict(p5=np.percentile(cagr, 5), p50=np.percentile(cagr, 50),
                dd95=np.percentile(dd, 5), wmo95=np.percentile(mo.min(1), 5),
                p20=(mo < -0.20).any(1).mean())


def mc_compare(name, raw, sca, n, rng):
    a, b = mc_row(raw.values, n, rng), mc_row(sca.values, n, rng)
    print(f'  MC ({n:,} x 5y, lev 1):'
          f'{"CAGR p5":>10}{"p50":>7}{"maxDD p95":>11}{"worst-mo p95":>14}'
          f'{"P(mo<-20%)":>12}{"p50/|maxDD|":>13}')
    for lab, m in (('raw', a), ('scaled', b)):
        print(f'  {lab:>8}{m["p5"]*100:>9.1f}%{m["p50"]*100:>6.1f}%'
              f'{m["dd95"]*100:>10.1f}%{m["wmo95"]*100:>13.1f}%'
              f'{m["p20"]*100:>11.1f}%{m["p50"]/abs(m["dd95"]):>13.2f}')
    ok = (b['p50'] / abs(b['dd95']) > a['p50'] / abs(a['dd95'])
          and b['p20'] <= a['p20'])
    print(f'  {name}: pre-declared criterion -> {"PASS" if ok else "FAIL"} '
          f'(return per unit of tail '
          f'{"improves" if ok else "does not improve"})')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paths', type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    d = os.path.join(ROOT, 'data')
    sp = os.path.join(d, 'stack_daily.csv')
    if not os.path.exists(sp):
        raise SystemExit('missing data/stack_daily.csv -- run stack_v2.py first')
    L = pd.read_csv(sp, dtype={'day': str}).set_index('day')
    out = pd.DataFrame(index=L.index)

    print('== 1. beta-scaled hedge '
          f'(win {BETA_WIN}, min {BETA_MIN}, clamp {BETA_CLAMP}, lag 1) ==')
    b, neub = beta_hedged(L)
    w = neub.index
    print(f'  beta: mean {b.mean():.2f}  p5 {b.quantile(.05):.2f}  '
          f'p95 {b.quantile(.95):.2f}  last {b.iloc[-1]:.2f}   '
          f'({len(b):,} days, {b.index[0]}..{b.index[-1]})')
    c1 = L.NEU.loc[w].corr(L.QQQ_ON.loc[w])
    c2 = neub.corr(L.QQQ_ON.loc[w])
    print(f'  corr with QQQ_ON, same window:  NEU (1:1) {c1:+.2f}   '
          f'NEUb {c2:+.2f}')
    stats(L.NEU.loc[w].values, 'NEU  (1:1)')
    stats(neub.values, 'NEUb (beta)')
    print(f'  pre-declared criterion |corr| < 0.20 -> '
          f'{"PASS" if abs(c2) < 0.20 else "FAIL"}')
    out['beta'], out['NEUb'] = b, neub

    print(f'\n== 2. vol-target overlay (win {VOL_WIN}, expanding-median '
          f'target, burn {VOL_BURN}, cap 1.0, lag 1) ==')
    books = [('v2', ['ON', 'MOM']), ('v2n', ['NEU', 'MOM'])]
    if len(neub) > VOL_BURN + VOL_WIN:
        L = L.join(neub.rename('NEUb'))
        books.append(('v2nb', ['NEUb', 'MOM']))
    for name, cols in books:
        if not all(c in L for c in cols):
            print(f'\n {name}: skipped (missing {cols})')
            continue
        r = L[cols].dropna().sum(axis=1)
        wgt = vol_scaled(r)
        if len(wgt) < 500:
            print(f'\n {name}: only {len(wgt)} days after burn-in -- skipped')
            continue
        raw, sca = r.loc[wgt.index], r.loc[wgt.index] * wgt
        print(f'\n {name} ({wgt.index[0]}..{wgt.index[-1]}, {len(wgt):,} days '
              f'-- raw and scaled on identical days):')
        stats(raw.values, 'raw')
        stats(sca.values, 'scaled')
        print(f'  exposure: mean {wgt.mean():.2f}   at cap 1.0 '
              f'{(wgt > 0.999).mean()*100:.0f}% of days   p5 '
              f'{wgt.quantile(.05):.2f}')
        mc_compare(name, raw, sca, a.paths, rng)
        print(f'  by 4-year era (scaled | raw):')
        er = pd.DataFrame({'s': sca, 'r': raw})
        for e0 in range(2000, 2030, 4):
            m = (er.index.str[:4].astype(int) >= e0) \
                & (er.index.str[:4].astype(int) < e0 + 4)
            if m.sum() >= 60:
                print(f'    {e0}-{(e0+3)%100:02d}: {er.s[m].mean():+7.2f} | '
                      f'{er.r[m].mean():+7.2f} bps/day  ({m.sum()} days)')
        out[f'{name}_w'], out[f'{name}_vt'] = wgt, sca

    p = os.path.join(d, 'overlay_daily.csv')
    out.dropna(how='all').to_csv(p, float_format='%.4f', index_label='day')
    print(f'\ndaily series -> {p}')
    print('nothing here alters the frozen models or the registered metrics; '
          'RESULTS 19\'s clock runs untouched.')


if __name__ == '__main__':
    main()
