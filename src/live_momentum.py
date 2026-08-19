# Part of qqq-microstructure.
#
# Forward test of the intraday-momentum signal, in QQQ shares, against an IBKR PAPER
# account. This exists because the backtest cannot settle the question: 515 days cannot
# distinguish a Sharpe-0.9 effect from noise (RESULTS §8, §8b), and every extra variant
# tried on those same days makes the estimate worse, not better. Only new days add
# information.
#
#   python src/live_momentum.py --dry-run     # no broker, prints what it would do
#   python src/live_momentum.py --paper       # IBKR paper account, port 7497
#
# The rules, fixed in advance so there is nothing to tune while it runs:
#   09:30 ET  record the opening price
#   10:30 ET  entry. long if the first 60 minutes are up, short if down
#   15:58 ET  flatten, unconditionally
#
# It will NOT run against a live account. `--paper` is the only broker mode, the port is
# checked, and the account code must start with 'DU' (IBKR's paper prefix). Funding and
# opening the account are yours to do; this connects to one that already exists.
#
# Position sizing is deliberately fixed and small. The backtest says the honest edge is
# ~5-6 bps/day at Sharpe ~0.9. Sizing up an unproven edge is how a marginal strategy
# becomes an expensive one.

import os, sys, json, argparse, datetime as dt
from zoneinfo import ZoneInfo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'reports', 'live')
ET = ZoneInfo('America/New_York')

SYMBOL = 'QQQ'
ENTRY_MIN = 60                 # signal window, minutes after the open
OPEN_T = dt.time(9, 30)
ENTRY_T = dt.time(10, 30)
FLAT_T = dt.time(15, 58)
PAPER_PORTS = (7497, 4002)     # TWS paper, IB Gateway paper
DEFAULT_QTY = 10               # shares. Small on purpose.


def log_path(day):
    os.makedirs(LOG_DIR, exist_ok=True)
    return os.path.join(LOG_DIR, f'{day}.json')


def record(day, **kw):
    p = log_path(day)
    doc = json.load(open(p)) if os.path.exists(p) else {'day': day, 'events': []}
    doc.update({k: v for k, v in kw.items() if k != 'event'})
    if 'event' in kw:
        doc['events'].append({'at': dt.datetime.now(ET).isoformat(timespec='seconds'),
                              **kw['event']})
    json.dump(doc, open(p, 'w'), indent=2)
    return p


def decide(open_px, entry_px):
    """The sign rule. Positive first hour -> long, negative -> short."""
    ret_bps = (entry_px / open_px - 1) * 1e4
    return (1 if ret_bps > 0 else -1 if ret_bps < 0 else 0), ret_bps


def load_model():
    """The production LightGBM model (see momentum_ml.py --save-model)."""
    import json, lightgbm as lgb
    mp = os.path.join(ROOT, 'models', 'momentum_lgbm.txt')
    man = json.load(open(os.path.join(ROOT, 'models', 'momentum_lgbm.json')))
    return lgb.Booster(model_file=mp), man['features']


def features_from_minutes(bars, day_et):
    """Build the model's features from RTH 1-minute bars.

    `bars`: DataFrame with columns [et, open, high, low, close] covering at least
    yesterday's full session and today up to 10:30, ET-localized. Formulas mirror
    hf_history.build_daily exactly -- same names, same definitions.
    """
    import numpy as np
    b = bars.copy()
    b['d'] = b.et.dt.strftime('%Y%m%d')
    days = sorted(b.d.unique())
    if len(days) < 2:
        return None
    prev, today = b[b.d == days[-2]], b[b.d == days[-1]]
    prev_close = float(prev['close'].iloc[-1])
    prev_oc = float(np.log(prev['close'].iloc[-1] / prev['open'].iloc[0]) * 1e4)
    pmr = np.diff(np.log(prev['close'].values)) * 1e4
    prev_rv = float(np.std(pmr)) if len(pmr) > 10 else np.nan
    t = today.reset_index(drop=True)
    o = float(t['open'].iloc[0])
    f = {'gap_bps': float(np.log(o / prev_close) * 1e4),
         'prev_oc_bps': prev_oc, 'prev_rv': prev_rv,
         'dow': float(t.et.iloc[0].dayofweek)}
    mfo = ((t.et.dt.hour - 9) * 60 + t.et.dt.minute - 30).astype(int)
    for k in (1, 5, 10, 15, 30, 60):
        idx = mfo[mfo <= k].index
        px = float(t['open'].iloc[idx[-1]]) if len(idx) else o
        f[f'ret{k}_bps'] = float(np.log(px / o) * 1e4)
        w = t.loc[idx]
        f[f'range{k}_bps'] = float(np.log(w['high'].max() / w['low'].min()) * 1e4) if len(w) else 0.0
    return f


def decide_ml(feats, model, feat_names):
    """Model decision: sign of the predicted entry->close move, plus the value."""
    import numpy as np, pandas as pd
    X = pd.DataFrame([[feats[k] for k in feat_names]], columns=feat_names)
    pred = float(model.predict(X)[0])
    return (1 if pred > 0 else -1 if pred < 0 else 0), pred


# ---------------------------------------------------------------- dry run

def dry_run(args):
    """Replay the most recent day through the live decision path."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pandas as pd
    daily = os.path.join(ROOT, 'data', 'daily_hf_QQQ.parquet')
    if not os.path.exists(daily):
        daily = os.path.join(ROOT, 'data', 'daily.parquet')
    if not os.path.exists(daily):
        raise SystemExit('run: python src/hf_history.py --build')
    df = pd.read_parquet(daily).dropna(subset=['oc_bps'])
    row = df.iloc[-1]
    if args.rule:
        side, ret_bps = decide(row['open'], row[f'p{ENTRY_MIN}'])
    else:
        model, fn = load_model()
        feats = {k: float(row[k]) for k in fn}
        side, pred = decide_ml(feats, model, fn)
        ret_bps = (row[f'p{ENTRY_MIN}'] / row['open'] - 1) * 1e4
        print(f'  model prediction      {pred:+.2f} bps for entry->close')
    qty = args.qty * side
    gross = (row['close'] / row[f'p{ENTRY_MIN}'] - 1) * 1e4 * side
    print(f'DRY RUN on {row["day"]} (no broker connection)\n')
    print(f'  09:30 open            ${row["open"]:.2f}')
    print(f'  10:30 entry           ${row[f"p{ENTRY_MIN}"]:.2f}   '
          f'first {ENTRY_MIN}m {ret_bps:+.1f} bps')
    print(f'  decision              {"LONG" if side>0 else "SHORT" if side<0 else "FLAT"}'
          f'  {qty:+d} shares  (${abs(qty)*row[f"p{ENTRY_MIN}"]:,.2f} notional)')
    print(f'  15:58 flatten at      ${row["close"]:.2f}')
    print(f'  gross                 {gross:+.1f} bps')
    print(f'  net of 0.34 bps cost  {gross-0.34:+.1f} bps '
          f'= ${(gross-0.34)/1e4*abs(qty)*row[f"p{ENTRY_MIN}"]:+.2f}')
    print('\n  This is the same decision function the paper path calls. It replays one')
    print('  archive day; it is a wiring check, not evidence.')


# ---------------------------------------------------------------- paper

def paper(args):
    from ib_async import IB, Stock, MarketOrder
    if args.port not in PAPER_PORTS:
        raise SystemExit(f'port {args.port} is not a paper port {PAPER_PORTS}. '
                         'This script does not connect to live trading accounts.')
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=False)
    accts = ib.managedAccounts()
    if not accts or not all(a.startswith('DU') for a in accts):
        ib.disconnect()
        raise SystemExit(f'accounts {accts} are not IBKR paper accounts (expected DU*). '
                         'Refusing to trade.')
    print(f'connected to paper account(s): {accts}')

    contract = Stock(SYMBOL, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    now = dt.datetime.now(ET)
    day = now.strftime('%Y%m%d')

    def px():
        t = ib.reqMktData(contract, '', True, False)
        ib.sleep(2)
        v = t.marketPrice()
        if v != v:                     # NaN -> fall back to the last close
            v = t.close
        return float(v)

    state = json.load(open(log_path(day))) if os.path.exists(log_path(day)) else {}

    if now.time() < OPEN_T:
        print(f'before the open; nothing to do until {OPEN_T}')
    elif now.time() < ENTRY_T:
        if 'open_px' not in state:
            p = px()
            record(day, open_px=p, symbol=SYMBOL,
                   event={'what': 'recorded open', 'px': p})
            print(f'recorded 09:30 reference ${p:.2f}; entry at {ENTRY_T}')
        else:
            print(f'open ${state["open_px"]:.2f} already recorded; waiting for {ENTRY_T}')
    elif now.time() < FLAT_T:
        if 'entry_px' in state:
            print(f'already in the trade: {state.get("qty")} @ ${state["entry_px"]:.2f}')
        elif 'open_px' not in state:
            print('no 09:30 reference recorded today; skipping (never guess the signal)')
        else:
            p = px()
            if args.rule:
                side, ret_bps = decide(state['open_px'], p)
            else:
                bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='2 D',
                                            barSizeSetting='1 min', whatToShow='TRADES',
                                            useRTH=True, formatDate=2)
                import pandas as pd
                bd = pd.DataFrame([dict(et=b.date, open=b.open, high=b.high,
                                        low=b.low, close=b.close) for b in bars])
                bd['et'] = pd.to_datetime(bd.et, utc=True).dt.tz_convert('America/New_York')
                feats = features_from_minutes(bd, now)
                if feats is None:
                    print('not enough bar history; skipping today')
                    ib.disconnect(); return
                model, fn = load_model()
                side, pred = decide_ml(feats, model, fn)
                ret_bps = feats['ret60_bps']
                record(day, model_pred_bps=pred,
                       event={'what': 'model', 'pred_bps': pred, **feats})
            qty = args.qty * side
            if qty == 0:
                print('signal flat; no trade')
            else:
                o = ib.placeOrder(contract, MarketOrder('BUY' if qty > 0 else 'SELL',
                                                        abs(qty)))
                ib.sleep(3)
                record(day, entry_px=p, qty=qty, signal_bps=ret_bps,
                       event={'what': 'entry', 'side': 'BUY' if qty > 0 else 'SELL',
                              'qty': abs(qty), 'px': p, 'signal_bps': ret_bps,
                              'status': o.orderStatus.status})
                print(f'ENTRY {"LONG" if qty>0 else "SHORT"} {abs(qty)} {SYMBOL} '
                      f'@ ~${p:.2f}  (first {ENTRY_MIN}m {ret_bps:+.1f} bps)')
    else:
        pos = next((p for p in ib.positions()
                    if p.contract.symbol == SYMBOL and p.position != 0), None)
        if pos is None:
            print('no position to flatten')
        else:
            q = int(pos.position)
            o = ib.placeOrder(contract, MarketOrder('SELL' if q > 0 else 'BUY', abs(q)))
            ib.sleep(3)
            p = px()
            ep = state.get('entry_px')
            gross = ((p / ep - 1) * 1e4 * (1 if q > 0 else -1)) if ep else None
            record(day, exit_px=p, gross_bps=gross,
                   event={'what': 'flatten', 'qty': q, 'px': p, 'gross_bps': gross,
                          'status': o.orderStatus.status})
            print(f'FLAT {q:+d} @ ~${p:.2f}' +
                  (f'   gross {gross:+.1f} bps' if gross is not None else ''))
    ib.disconnect()
    print(f'log -> {os.path.relpath(log_path(day), ROOT)}')


def daemon(args):
    """One long-running process instead of three scheduled runs.

    Computes the ET event times itself (so the host machine's timezone and US
    daylight-saving shifts are irrelevant), sleeps until each one, and invokes the same
    idempotent step logic the scheduled path uses. Trade-off vs a scheduler: this dies
    with a reboot or laptop sleep -- restart it and, because each step re-reads the
    day's log, it resumes exactly where the day left off."""
    import time as _t
    EVENTS = [dt.time(9, 31), ENTRY_T.replace(minute=ENTRY_T.minute + 1), FLAT_T]
    print(f'daemon mode: acting at {", ".join(str(e) for e in EVENTS)} ET; Ctrl+C to stop')
    while True:
        now = dt.datetime.now(ET)
        todays = [dt.datetime.combine(now.date(), e, ET) for e in EVENTS]
        nxt = next((t for t in todays if t > now), None)
        if nxt is None or now.weekday() >= 5:
            nxt = dt.datetime.combine(now.date() + dt.timedelta(days=1), EVENTS[0], ET)
            while nxt.weekday() >= 5:
                nxt += dt.timedelta(days=1)
        wait = (nxt - dt.datetime.now(ET)).total_seconds()
        print(f'  next action {nxt:%a %H:%M} ET ({wait/3600:.1f}h from now)', flush=True)
        _t.sleep(max(wait, 1))
        try:
            paper(args)
        except SystemExit as ex:
            print(f'  step refused: {ex}', flush=True)
        except Exception as ex:
            print(f'  step failed: {type(ex).__name__}: {ex} -- will retry at the '
                  f'next event', flush=True)
        _t.sleep(90)          # step past the minute so the same event does not refire


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run', action='store_true', help='no broker; replay one archive day')
    g.add_argument('--paper', action='store_true', help='IBKR paper account only')
    g.add_argument('--daemon', action='store_true',
                   help='stay running and do all three daily steps at the right ET times')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7497)
    ap.add_argument('--client-id', type=int, default=17)
    ap.add_argument('--qty', type=int, default=DEFAULT_QTY)
    ap.add_argument('--rule', action='store_true',
                    help='use the plain sign rule instead of the ML model')
    a = ap.parse_args()
    (dry_run if a.dry_run else daemon if a.daemon else paper)(a)


if __name__ == '__main__':
    main()
