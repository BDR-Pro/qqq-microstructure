# Part of qqq-microstructure.
#
# RESULTS 17's verdict earned exactly one purchase: the QQQ option chain at the
# 10:30 decision time, every day 2023-04 -> present, so RESULTS 8's structures
# can be valued against MEASURED IV instead of an assumed one. This pulls it at
# ~1/40th of full-day cost: a ten-minute window around the entry (10:25-10:35
# ET). The window is computed per day through America/New_York -- the RESULTS
# 10 class of bug (a fixed UTC constant that is right in summer and one hour
# wrong in winter, which opra_load.py's snapshot constants inherit) cannot
# recur here, because every minute inside a day's file IS the entry window.
#
# Money discipline: the default run spends NOTHING. It samples the billing API
# across the range, prints the extrapolated total, and stops. Only --yes pulls
# -- resumably, one parquet per day in data/opra/ with existing days skipped,
# an exact per-day cost tally from the billing API as it goes, and holidays
# saved as empty markers so they are not re-queried. Terminal payoffs need no
# options data at all: the underlying close is in the equity panel, and
# put-call parity recovers spot from the chain itself (opra_load.py).
#
#   set DATABENTO_API_KEY=db-...            (PowerShell: $env:DATABENTO_API_KEY="db-...")
#   python src/opra_pull.py                 # estimate the cost, spend nothing
#   python src/opra_pull.py --yes           # pull
#
import os, glob, argparse, datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XSEC = os.path.join(ROOT, 'data', 'xsec')
OUT = os.path.join(ROOT, 'data', 'opra')
DATASET, SCHEMA, SYM = 'OPRA.PILLAR', 'cbbo-1m', 'QQQ.OPT'
ET = ZoneInfo('America/New_York')
W0, W1 = dt.time(10, 25), dt.time(10, 35)


def window_utc(day):
    d = dt.date(int(day[:4]), int(day[4:6]), int(day[6:8]))
    a = dt.datetime.combine(d, W0, ET).astimezone(dt.timezone.utc)
    b = dt.datetime.combine(d, W1, ET).astimezone(dt.timezone.utc)
    return a, b


def trading_days(start, end):
    """Panel calendar inside the range; plain weekdays beyond the panel's end
    (holidays there come back empty and cost ~nothing)."""
    days = set()
    for f in sorted(glob.glob(os.path.join(XSEC, '*.parquet'))):
        days.update(pd.read_parquet(f, columns=['day']).day.unique())
    days = sorted(days)
    out = [d for d in days if start <= d <= end]
    if days and end > days[-1]:
        extra = pd.bdate_range(pd.to_datetime(days[-1]) + pd.Timedelta(days=1),
                               pd.to_datetime(end)).strftime('%Y%m%d')
        out += [d for d in extra if d >= start]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2023-04-01')
    ap.add_argument('--end', default=dt.date.today().isoformat())
    ap.add_argument('--yes', action='store_true', help='actually pull (spends money)')
    ap.add_argument('--ipv4', action='store_true',
                    help='force IPv4 -- workaround when Python times out '
                         'connecting while Test-NetConnection succeeds '
                         '(broken IPv6 path, v6 address resolved first)')
    a = ap.parse_args()
    if a.ipv4:
        import socket
        from urllib3.util import connection as _uc
        _uc.allowed_gai_family = lambda: socket.AF_INET
    import databento as db
    if not os.environ.get('DATABENTO_API_KEY'):
        raise SystemExit('set DATABENTO_API_KEY first')
    client = db.Historical()

    s, e = a.start.replace('-', ''), a.end.replace('-', '')
    days = trading_days(s, e)
    os.makedirs(OUT, exist_ok=True)
    todo = [d for d in days if not os.path.exists(os.path.join(OUT, f'{d}.parquet'))]
    print(f'{len(days)} trading days {a.start} .. {a.end}; '
          f'{len(todo)} not yet pulled -> data/opra/')
    if not todo:
        print('nothing to do')
        return

    def _retry(fn, tries=2):
        import time
        for k in range(tries):
            try:
                return fn()
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(5 * (k + 1))

    def cost(day):
        t0, t1 = window_utc(day)
        return float(_retry(lambda: client.metadata.get_cost(
            dataset=DATASET, symbols=[SYM], stype_in='parent', schema=SCHEMA,
            start=t0, end=t1)))

    UNREACHABLE = (
        'cannot reach the Databento API ({}).\n'
        'This is a network-path problem, not a key problem -- it happens before\n'
        'authentication. Check, in order:\n'
        '  1. Test-NetConnection hist.databento.com -Port 443\n'
        '     TcpTestSucceeded False means the host is blocked from this network;\n'
        '     databento.com loading in a browser does NOT imply the API host does.\n'
        '  2. https://status.databento.com in a browser.\n'
        '  3. Retry over a VPN or from another network -- the pull is small and\n'
        '     resumable, so one session anywhere finishes it.\n'
        'Fallback without API access: a portal batch download of FULL days\n'
        '(QQQ.OPT parent, cbbo-1m, 2023-04 -> now) costs ~40x more (~$110) and\n'
        'opra_load.py can read the zip it produces.')

    if not a.yes:
        stride = max(1, len(todo) // 60)
        sample = todo[::stride]
        try:
            cs = [cost(d) for d in sample]
        except Exception as ex:
            raise SystemExit(UNREACHABLE.format(type(ex).__name__))
        est = float(np.mean(cs)) * len(todo)
        print(f'sampled {len(sample)} of {len(todo)} days: per-day '
              f'${min(cs):.4f} .. ${max(cs):.4f}, mean ${sum(cs)/len(cs):.4f}')
        print(f'ESTIMATED TOTAL: ${est:.2f}   (ten ET minutes/day, {SYM} parent, '
              f'{SCHEMA})')
        print('this run spent nothing. re-run with --yes to pull.')
        return

    spent, nbytes, fails = 0.0, 0, 0
    for i, d in enumerate(todo, 1):
        t0, t1 = window_utc(d)
        try:
            c = cost(d)
            data = _retry(lambda: client.timeseries.get_range(
                dataset=DATASET, schema=SCHEMA, symbols=[SYM],
                stype_in='parent', start=t0, end=t1))
            df = data.to_df(map_symbols=True)
        except Exception as ex:
            # A client-side rejection (BentoClientError) on a day the API has no
            # session for -- a market holiday the plain-weekday calendar beyond
            # the panel naively included. Save an empty marker so sweeps stop
            # retrying it, but only once this store holds real pulls, so an
            # auth failure can never masquerade as a holiday.
            if 'Client' in type(ex).__name__ and any(
                    os.path.getsize(os.path.join(OUT, f)) > 2000
                    for f in os.listdir(OUT)):
                pd.DataFrame().to_parquet(os.path.join(OUT, f'{d}.parquet'))
                print(f'  {d}: no session (holiday?) -- marker saved', flush=True)
                continue
            print(f'  {d}: {type(ex).__name__}', flush=True)
            fails += 1
            if fails == i and fails >= 3:
                raise SystemExit(UNREACHABLE.format(type(ex).__name__))
            if fails >= 8:
                raise SystemExit('8 consecutive failures -- network likely '
                                 'dropped mid-run; re-run to resume')
            continue
        fails = 0
        p = os.path.join(OUT, f'{d}.parquet')
        df.reset_index().to_parquet(p, index=False)
        spent += c
        nbytes += os.path.getsize(p)
        if df.empty:
            print(f'  {d}: empty (holiday?) -- marker saved', flush=True)
        if i % 25 == 0 or i == len(todo):
            print(f'  [{i}/{len(todo)}] {d}  spent ${spent:.2f}  '
                  f'{nbytes/1e6:.0f} MB on disk', flush=True)
    print(f'done: {len(todo)} day(s), ~${spent:.2f} billed, '
          f'{nbytes/1e6:.0f} MB in data/opra/')


if __name__ == '__main__':
    main()
