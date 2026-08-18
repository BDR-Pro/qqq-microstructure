# Part of qqq-microstructure.
#
# Can a learned model beat the sign rule at the 10:30 entry?
#
# The baseline is sign(first 60 minutes), held to the close. The challenger is LightGBM
# over everything observable at 10:30: the overnight gap, the path of the first hour
# (returns and ranges at 1/5/10/15/30/60 minutes), yesterday's move, realised vol, and
# day of week. Positions are sized by the model's conviction, capped at 1x.
#
# Model class is chosen for the data, not for fashion. ~6,700 daily rows with ~15
# features is squarely gradient-boosting territory; sequence models (CNN/TFT) start to
# pay at ~1e5+ samples and are the wrong tool at daily frequency.
#
# Validation is WALK-FORWARD: for each test year, the model trains only on years that
# came before it. No test year ever influences its own predictions. This is the only
# scheme that respects how the model would actually be deployed.

import os, sys, argparse
import numpy as np, pandas as pd
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_BPS = 0.34
# rv_1m_bps is EXCLUDED: build_daily computes it over the WHOLE session, so at the
# 10:30 entry it leaks the afternoon's volatility -- and through the vol-return
# asymmetry (high vol days skew down) that leaks the sign of the remaining move.
# Including it produced Sharpe 2.4 out of sample, which was the tell, not a result.
# Trailing-vol regime features were tried and REJECTED: they cut OOS bps/day from
# 3.85 to 2.08. Vol tells you how big the move will be, not which way -- and the model
# spent its splits on it. Sizing, not signing, is where vol belongs (see vol targeting
# below). Recorded here so the next reader does not re-add them.
FEATS = ['gap_bps', 'ret1_bps', 'ret5_bps', 'ret10_bps', 'ret15_bps', 'ret30_bps',
         'ret60_bps', 'range15_bps', 'range60_bps', 'prev_oc_bps', 'prev_rv', 'dow']
PARAMS = dict(objective='regression', learning_rate=0.03, num_leaves=15,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=10.0, verbosity=-1, seed=7)


def load(tk):
    p = os.path.join(ROOT, 'data', f'daily_hf_{tk}.parquet')
    df = pd.read_parquet(p).dropna(subset=['oc_bps']).reset_index(drop=True)
    df['y'] = np.log(df['close'] / df['p60']) * 1e4          # entry -> close, bps
    df['year'] = df.day.str[:4].astype(int)
    # regime features: everything shifted so only days strictly before entry are seen
    df['trail5_vol'] = df.oc_bps.rolling(5).std().shift(1)
    df['trail20_vol'] = df.oc_bps.rolling(20).std().shift(1)
    df['trail20_mom'] = df.oc_bps.rolling(20).mean().shift(1)
    df['vol_ratio'] = df.trail5_vol / df.trail20_vol
    return df.dropna(subset=[f for f in FEATS if f != 'dow'] + ['y']).reset_index(drop=True)


def stats(r, label):
    r = r[np.isfinite(r)]
    n = len(r)
    sh = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    t = r.mean() / (r.std() / np.sqrt(n)) if r.std() > 0 else np.nan
    eq = (1 + r / 1e4).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(label=label, n=n, bps=r.mean(), t=t, sharpe=sh,
                cagr=(eq.iloc[-1] ** (252 / n) - 1) * 100, maxdd=dd * 100,
                hit=(r > 0).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', default='QQQ')
    ap.add_argument('--min-train-years', type=int, default=5)
    a = ap.parse_args()
    df = load(a.ticker)
    print(f'{a.ticker}: {len(df)} days  {df.day.iloc[0]} .. {df.day.iloc[-1]}')
    print(f'walk-forward: first test year = {df.year.min() + a.min_train_years}\n')

    years = sorted(df.year.unique())
    preds = pd.Series(np.nan, index=df.index)
    for ty in years:
        tr = df[df.year < ty]
        te = df[df.year == ty]
        if len(tr) < a.min_train_years * 240 or not len(te):
            continue
        ds = lgb.Dataset(tr[FEATS], tr['y'],
                         categorical_feature=['dow'], free_raw_data=True)
        m = lgb.train(PARAMS, ds, num_boost_round=300)
        preds.loc[te.index] = m.predict(te[FEATS])
    ok = preds.notna()
    d, p = df[ok].reset_index(drop=True), preds[ok].reset_index(drop=True)
    print(f'out-of-sample days: {len(d)}  ({d.year.min()}-{d.year.max()})\n')

    rule = np.sign(d.ret60_bps.values) * d.y.values - COST_BPS
    mlsgn = np.sign(p.values) * d.y.values - COST_BPS
    # conviction sizing: scale by |pred| relative to its trailing typical size, cap 1x
    scale = p.abs().rolling(250, min_periods=50).median().shift(1).bfill()
    size = np.minimum(p.abs() / (2 * scale), 1.0)
    mlsized = np.sign(p.values) * size.values * d.y.values - COST_BPS * (size.values > 0.05)
    agree = np.sign(p.values) == np.sign(d.ret60_bps.values)
    both = np.where(agree, np.sign(p.values), 0.0) * d.y.values - COST_BPS * agree
    # vol targeting: constant expected risk. Size = target / trailing vol, capped at 2x.
    # Uses only past days; this is where volatility belongs -- in the denominator.
    tv = d.y.abs().rolling(20, min_periods=10).mean().shift(1).bfill()
    lev = np.minimum(tv.median() / tv, 2.0)
    voltgt = np.sign(p.values) * lev.values * d.y.values - COST_BPS * lev.values

    print(f'{"strategy":<26}{"bps/day":>9}{"t":>7}{"Sharpe":>8}{"CAGR%":>8}'
          f'{"maxDD%":>9}{"hit%":>7}')
    for r, lab in ((rule, 'sign rule (baseline)'), (mlsgn, 'ML sign'),
                   (mlsized, 'ML conviction-sized'), (both, 'ML+rule agree only'),
                   (voltgt, 'ML sign, vol-targeted')):
        s = stats(pd.Series(r), lab)
        print(f'{s["label"]:<26}{s["bps"]:>9.2f}{s["t"]:>7.2f}{s["sharpe"]:>8.2f}'
              f'{s["cagr"]:>8.2f}{s["maxdd"]:>9.2f}{s["hit"]:>7.1f}')

    ic = np.corrcoef(p.values, d.y.values)[0, 1]
    print(f'\nmodel IC (pred vs realised): {ic:+.4f}')
    print(f'direction agreement with rule: {agree.mean() * 100:.1f}%')

    # per-year comparison -- consistency matters more than the average
    print(f'\n{"year":>6}{"rule":>9}{"ML sign":>9}{"sized":>9}')
    d2 = d.assign(rule=rule, ml=mlsgn, sz=mlsized)
    for y, g in d2.groupby('year'):
        print(f'{y:>6}{g.rule.mean():>9.2f}{g.ml.mean():>9.2f}{g.sz.mean():>9.2f}')

    imp = m.feature_importance(importance_type='gain')
    top = sorted(zip(FEATS, imp), key=lambda x: -x[1])[:6]
    print('\ntop features (last fold, gain):')
    for f, v in top:
        print(f'  {f:<14}{v / imp.sum() * 100:5.1f}%')


if __name__ == '__main__':
    main()
