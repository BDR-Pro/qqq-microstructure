# Part of qqq-microstructure.
#
# The forward test, exactly as RESULTS 12 demanded it for the QQQ stack: the
# frozen cross-sectional models (xsec_ml.py --save-model) evaluated on months
# that postdate their training, as each new month's file lands. 515 backtested
# days could not settle a Sharpe-0.9 effect and 26 backtested years still cannot
# settle next year; only forward months add information. The strategy trades at
# two fixed timestamps (the close and the open), so monthly HF files replay it
# exactly -- a free, zero-infrastructure forward test.
#
# The monthly ritual:
#   python src/xsec_extract.py --end <YYYY-MM>     # fetch the new month(s)
#   python src/xsec_replay.py                      # evaluate past the freeze
#
# Reads models/xsec_on_lgbm.txt (the ceiling: overnight at MOO/MOC prices) and
# models/xsec_on15_lgbm.txt (the floor: 09:45 exit), reports each post-freeze
# month's realised Q5-Q1, the running total, and monthly rank IC, against the
# walk-forward references from RESULTS 15b (+12.2 and +7.3 bps/day). Nothing is
# retrained here -- if the numbers rot, the honest move is recorded in RESULTS
# 12: diagnose, do not filter.

import os, json
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import load_panel, stats
from xsec_ml import build_table, portfolio, monthly_ic, FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {'on': 12.2, 'on15': 7.3}          # RESULTS 15b walk-forward bps/day


def main():
    md = os.path.join(ROOT, 'models')
    mp = os.path.join(md, 'xsec_lgbm.json')
    if not os.path.exists(mp):
        raise SystemExit('no frozen models -- run: python src/xsec_ml.py --save-model')
    meta = json.load(open(mp))
    cut = meta['last_tmonth']
    df = load_panel()
    bym = {m: g[['ticker', 'day', 'cc_bps', 'on_bps', 'on15_bps']]
           for m, g in df.groupby('month')}
    t = build_table(df)
    new = t[t.tmonth > cut]
    print(f'\nfrozen through {cut}; {new.tmonth.nunique()} month(s) after the freeze')
    if new.empty:
        print('nothing to replay -- fetch newer months with xsec_extract.py --end')
        return
    if meta['features'] != FEATS:
        raise SystemExit('feature list changed since the freeze -- re-freeze '
                         'deliberately or check out the freezing commit')

    for tag, ycol, col in (('on', 'yon_r', 'on_bps'),
                           ('on15', 'yon15_r', 'on15_bps')):
        m = lgb.Booster(model_file=os.path.join(md, f'xsec_{tag}_lgbm.txt'))
        pred = pd.Series(m.predict(new[FEATS]), index=new.index)
        ls = portfolio(new, pred, bym, col)
        ic = monthly_ic(new, pred, ycol)
        print(f'\n{tag}: frozen model, {len(ls)} days '
              f'(walk-forward reference {REF[tag]:+.1f} bps/day)')
        mo = ls.ls.groupby(ls.index.str[:6]).agg(['mean', 'sum'])
        for mm, r in mo.iterrows():
            key = f'{mm[:4]}-{mm[4:]}'
            icv = ic.get(key, np.nan)
            print(f'  {key}  {r["mean"]:+7.2f} bps/day  {r["sum"]/100:+6.2f}%   '
                  f'IC {icv:+.3f}')
        print(f'  since freeze: {ls.ls.mean():+.2f} bps/day, '
              f'total {ls.ls.sum()/100:+.2f}%')
        if len(ls) >= 50:
            stats(ls.ls.values, f'{tag} since freeze')


if __name__ == '__main__':
    main()
