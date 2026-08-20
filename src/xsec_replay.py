# Part of qqq-microstructure.
#
# The forward test, exactly as RESULTS 12 demanded it for the QQQ stack -- as ONE
# command, for the same reason daemon mode replaced three scheduled runs: rituals
# with three steps do not get followed.
#
#   python src/xsec_replay.py
#
# What one run does, in order:
#
#   1. FREEZE (first run only). If models/xsec_lgbm.json does not exist, train the
#      production models on every month currently on disk and save them, stamped
#      with the freeze date. The freeze is the whole point: a model that can be
#      quietly retrained proves nothing, because backtests are run by people who
#      have seen the answers. Only months that arrive after this commitment count
#      as evidence. Commit models/ so the git hash makes the freeze tamper-evident.
#   2. FETCH. Work out the last complete calendar month and download any missing
#      month files (resumable, same mechanics as xsec_extract). A month the source
#      has not published yet fails gracefully and is retried next run. No --end
#      bookkeeping.
#   3. VERDICT. Score the frozen models on every month after their training
#      cutoff: per-month Q5-Q1, running total, monthly rank IC, against the
#      walk-forward references from RESULTS 15b (+12.2 ceiling / +7.3 floor
#      bps/day). Months that predate the freeze DATE but postdate the training
#      cutoff are quasi-holdout (they existed, but no decision in this repo saw
#      them); months after the freeze date are true forward evidence.
#
# Nothing is ever retrained here. If the numbers rot, the honest move is the one
# RESULTS 12 records: diagnose, do not filter -- and any deliberate re-freeze is
# its own dated commit via xsec_ml.py --save-model.

import os, json, datetime as dt
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import load_panel, stats, XSEC
from xsec_ml import build_table, portfolio, monthly_ic, save_production, FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = {'on': 12.2, 'on15': 7.3}          # RESULTS 15b walk-forward bps/day


def fetch_new():
    from xsec_extract import one_month, months as monthrange, TMP
    os.makedirs(TMP, exist_ok=True)
    have = sorted(f[:-8] for f in os.listdir(XSEC) if f.endswith('.parquet'))
    today = dt.date.today()
    last_complete = (today.replace(day=1) - dt.timedelta(days=1)).strftime('%Y-%m')
    todo = [m for m in monthrange(have[-1], last_complete) if m > have[-1]]
    if not todo:
        return 0
    got = 0
    for m in todo:
        try:
            n = one_month(m)
            print(f'  fetched {m}: {n:,} ticker-days')
            got += 1
        except Exception as ex:
            print(f'  {m}: not published yet ({type(ex).__name__}) -- '
                  f'will retry next run')
            break
    return got


def main():
    md = os.path.join(ROOT, 'models')
    mp = os.path.join(md, 'xsec_lgbm.json')
    if not os.path.exists(mp):
        print('no frozen models -- freezing NOW on every month on disk')
        save_production(build_table(load_panel()))
        print('commit models/ to put the freeze on the record\n')
    meta = json.load(open(mp))
    cut = meta['last_tmonth']
    print(f'frozen {meta.get("frozen_on", "?")}, trained through {cut}; '
          f'checking for new months...')
    fetch_new()

    df = load_panel()
    bym = {m: g[['ticker', 'day', 'cc_bps', 'on_bps', 'on15_bps']]
           for m, g in df.groupby('month')}
    t = build_table(df)
    new = t[t.tmonth > cut]
    print(f'\n{new.tmonth.nunique()} month(s) after the training cutoff {cut}')
    if new.empty:
        print('nothing to replay yet -- run again after the next month publishes')
        return
    if meta['features'] != FEATS:
        raise SystemExit('feature list changed since the freeze -- re-freeze '
                         'deliberately (xsec_ml.py --save-model) or check out '
                         'the freezing commit')

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
            print(f'  {key}  {r["mean"]:+7.2f} bps/day  {r["sum"]/100:+6.2f}%   '
                  f'IC {ic.get(key, np.nan):+.3f}')
        print(f'  since cutoff: {ls.ls.mean():+.2f} bps/day, '
              f'total {ls.ls.sum()/100:+.2f}%')
        if len(ls) >= 50:
            stats(ls.ls.values, f'{tag} since cutoff')


if __name__ == '__main__':
    main()
