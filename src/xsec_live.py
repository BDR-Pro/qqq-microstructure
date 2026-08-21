# Part of qqq-microstructure.
#
# The zero-capital live record. RESULTS 12's verdict for every strategy in this
# repo is the same: backtests cannot settle what only forward months can. This
# script starts that clock at daily granularity, with no broker and no capital:
# run it before the close, it scores the FROZEN models on everything knowable
# today, logs tonight's Q5/Q1 overnight basket to reports/xsec_paper.csv, and on
# later runs grades every past entry against the official next-day open. The
# log is the evidence; commit it.
#
# Honesty rails, in code rather than in promises:
#   - Feature parity check every run: the live feature builder is exercised
#     against build_table's own rows for the last panel month and must agree to
#     the decimal, so the live signal can never silently drift from the
#     backtested one.
#   - The live universe is the last complete month's top-150 (the lagged rule).
#     The backtest's additional still-in-top-150-next-month conditioning is
#     unknowable in real time; RESULTS 15 measured it as mild optimism, and
#     live simply does not get it.
#   - Nothing is retrained. Models come from models/xsec_*_lgbm.txt as frozen.
#   - If the panel's last month is stale (xsec_extend.py not run), the script
#     says so loudly -- features silently aging is how live records rot.
#
#   python src/xsec_live.py            # grade past entries, then log tonight's basket

import os, argparse, datetime as dt, json
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import lightgbm as lgb
from xsec_backtest import load_panel, ETF, MIN_CC_DAYS
from xsec_ml import build_table, rank_pm, FEATS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'reports', 'xsec_paper.csv')
ET = ZoneInfo('America/New_York')


def month_aggs(df):
    d = df.assign(fh=np.log(df.p60 / df.open) * 1e4,
                  rng=(df.high - df.low) / df.close * 1e4)
    return d.groupby(['ticker', 'month']).agg(
        cc_sum=('cc_bps', 'sum'), cc_n=('cc_bps', 'count'),
        on_mean=('on_bps', 'mean'), on_n=('on_bps', 'count'),
        cc_std=('cc_bps', 'std'), fh_mean=('fh', 'mean'),
        rng_mean=('rng', 'mean'), dv=('dollar_vol', 'mean'),
        close_last=('close', 'last')).to_dict('index')


def features(agg, look, universe):
    """The build_table feature block for one target month: `look` is its 12
    lookback months oldest-first, `universe` the candidate tickers."""
    rows = []
    for tk in sorted(universe):
        hist = [agg.get((tk, mm)) for mm in look]
        if any(h is None or h['cc_n'] < MIN_CC_DAYS for h in hist):
            continue
        if hist[-1]['on_n'] < 8:
            continue
        on12 = [h['on_mean'] for h in hist if h['on_n'] >= 8]
        if len(on12) < 8:
            continue
        rows.append(dict(
            ticker=tk,
            mom_12_2=sum(h['cc_sum'] for h in hist[:-1]),
            ret_1m=hist[-1]['cc_sum'], on_1m=hist[-1]['on_mean'],
            on_12m=float(np.mean(on12)),
            vol_3m=float(np.nanmean([h['cc_std'] for h in hist[-3:]])),
            rng_3m=float(np.mean([h['rng_mean'] for h in hist[-3:]])),
            fh_3m=float(np.mean([h['fh_mean'] for h in hist[-3:]])),
            dvol=float(np.log10(hist[-1]['dv'])),
            dvol_chg=float(np.log10(hist[-1]['dv'])
                           - np.mean([np.log10(h['dv']) for h in hist[:-1]])),
            lprice=float(np.log10(hist[-1]['close_last']))))
    out = pd.DataFrame(rows)
    return out.dropna() if len(out) else out


def parity_check(df, t, agg, months):
    T = t.tmonth.max()
    i = months.index(T)
    uni_prev = set(df[df.month == months[i - 1]].ticker)
    uni_T = set(df[df.month == T].ticker)
    mine = features(agg, months[i - 12:i], (uni_prev & uni_T) - ETF)
    ref = t[t.tmonth == T][['ticker'] + FEATS]
    # ref features are rank-transformed; recompute ranks on mine to compare
    for f in FEATS:
        mine[f] = rank_pm(mine[f], pd.Series(0, index=mine.index))
    j = ref.merge(mine, on='ticker', suffixes=('_bt', '_lv'))
    d = max(float((j[f'{f}_bt'] - j[f'{f}_lv']).abs().max()) for f in FEATS)
    print(f'feature parity vs build_table ({T}, {len(j)} names): '
          f'max |rank diff| = {d:.6f}' + ('' if d < 1e-9 else '   <-- DRIFT'))
    return d < 1e-9


def grade(log):
    import yfinance as yf
    ungraded = log[log.q5_on_bps.isna() & (log.date < dt.date.today()
                                           .strftime('%Y%m%d'))]
    for i, r in ungraded.iterrows():
        names = r.q5.split(';') + ['QQQ']
        d0 = dt.datetime.strptime(r.date, '%Y%m%d').date()
        v = yf.download([n.replace('.', '-') for n in names],
                        start=d0, end=d0 + dt.timedelta(days=8), interval='1d',
                        auto_adjust=False, actions=False, group_by='ticker',
                        progress=False)
        ons, qqq = [], np.nan
        for n in names:
            try:
                s = v[n.replace('.', '-')][['Open', 'Close']].dropna()
            except KeyError:
                continue
            if len(s) < 2 or s.index[0].strftime('%Y%m%d') != r.date:
                continue
            on = float(np.log(s.Open.iloc[1] / s.Close.iloc[0]) * 1e4)
            if n == 'QQQ':
                qqq = on
            else:
                ons.append(on)
        if len(ons) >= 5 and np.isfinite(qqq):
            log.loc[i, 'q5_on_bps'] = float(np.mean(ons))
            log.loc[i, 'qqq_on_bps'] = qqq
            log.loc[i, 'tilt_bps'] = float(np.mean(ons)) - qqq
            print(f'graded {r.date}: basket {np.mean(ons):+.1f} bps  '
                  f'QQQ {qqq:+.1f}  tilt {np.mean(ons) - qqq:+.1f}')
    return log


def main():
    argparse.ArgumentParser().parse_args()
    df = load_panel()
    months = sorted(df.month.unique())
    agg = month_aggs(df)
    t = build_table(df)
    if not parity_check(df, t, agg, months):
        raise SystemExit('live features drifted from build_table -- fix before '
                         'trusting any output')

    now = dt.datetime.now(ET)
    cur = now.strftime('%Y-%m')
    if months[-1] < (pd.Period(cur) - 1).strftime('%Y-%m'):
        print(f'WARNING: panel ends {months[-1]} but current month is {cur} -- '
              f'features are stale; run xsec_extend.py')

    live = features(agg, months[-12:], set(df[df.month == months[-1]].ticker)
                    - ETF)
    raw = live.copy()
    for f in FEATS:
        live[f] = rank_pm(live[f], pd.Series(0, index=live.index))
    mp = os.path.join(ROOT, 'models', 'xsec_lgbm.json')
    if not os.path.exists(mp):
        raise SystemExit('no frozen models -- run xsec_replay.py or '
                         'xsec_ml.py --save-model first')
    meta = json.load(open(mp))
    m = lgb.Booster(model_file=os.path.join(ROOT, 'models', 'xsec_on_lgbm.txt'))
    live['score'] = m.predict(live[FEATS])
    k = len(live) // 5
    q5 = live.nlargest(k, 'score').ticker.tolist()
    q1 = live.nsmallest(k, 'score').ticker.tolist()

    cols = ['date', 'universe_month', 'model_frozen', 'n_scored', 'q5', 'q1',
            'q5_on_bps', 'qqq_on_bps', 'tilt_bps']
    log = pd.read_csv(LOG, dtype={'date': str}) if os.path.exists(LOG) \
        else pd.DataFrame(columns=cols)
    log = grade(log)

    today = now.strftime('%Y%m%d')
    if today in set(log.date):
        print(f'{today} already logged')
    elif now.weekday() >= 5:
        print(f'{today} is a weekend -- nothing to log')
    else:
        log = pd.concat([log, pd.DataFrame([dict(
            date=today, universe_month=months[-1],
            model_frozen=meta.get('frozen_on', '?'), n_scored=len(live),
            q5=';'.join(q5), q1=';'.join(q1))])], ignore_index=True)
        print(f'\nlogged {today}: hold overnight (buy MOC, sell MOO), '
              f'{len(q5)} names from {months[-1]} universe:')
        print('  ' + '  '.join(q5))
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log[cols].to_csv(LOG, index=False)

    g = log.dropna(subset=['tilt_bps'])
    if len(g):
        print(f'\nscorecard: {len(g)} graded nights   basket '
              f'{g.q5_on_bps.astype(float).mean():+.1f} bps/n   QQQ '
              f'{g.qqq_on_bps.astype(float).mean():+.1f}   tilt '
              f'{g.tilt_bps.astype(float).mean():+.1f}   hit '
              f'{(g.tilt_bps.astype(float) > 0).mean() * 100:.0f}%   '
              f'cum tilt {g.tilt_bps.astype(float).sum() / 100:+.2f}%')
    print(f'log -> {LOG}  (commit it -- the log is the evidence)')


if __name__ == '__main__':
    main()
