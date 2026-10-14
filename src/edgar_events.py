# Part of qqq-microstructure.
#
# The EDGAR event layer: every filing of every mapped wide-panel name that an
# event hypothesis (event_test.py) can use, each placed in time by WHEN THE SEC
# ACCEPTED IT, in US/Eastern -- the one thing that makes an event backtest
# leak-free by construction. Free, public, timestamped data only; no API keys.
#
# Where the data comes from (all www.sec.gov, a declared User-Agent, <= 8
# requests/second, retries with backoff, everything cached under data/edgar/
# and every fetch resumable):
#   - The filing lists: the bulk submissions archive
#     (Archives/edgar/daily-index/bulkdata/submissions.zip, ONE request,
#     rebuilt nightly by the SEC) read in place -- or, with --source api, the
#     per-company data.sec.gov/submissions JSON (one request per page).
#   - The names: the research's validated (ticker, year) -> CIK map
#     (data/edgar/research/assign_v3.parquet). It is keyed by year because
#     tickers are reused (T was AT&T Corp until 2005-11; CPS, MI, JCI, SUNW...).
#   - Exact acceptance times: the filing index page (".../{acc}-index.htm",
#     "Accepted"), which also names each party's role (Subject / Filed by).
#   - Equity vs debt for prospectuses: the first 20,000 characters of the
#     primary document (read with a byte cap; Range headers are ignored).
#   - Insider purchases: the SEC Insider Transactions Data Sets (quarterly
#     zips, 2006q1 on), processed one at a time and deleted.
#
# THE TIMESTAMP RULES (the research's leak traps; event_test.py relies on them)
#   1. The submissions JSON's acceptanceDateTime ends in 'Z' but is NOT UTC:
#      JSON minus the true ET wall clock is 0 h, +4/5 h or +8/10 h depending on
#      the filing (never negative, never more than 10 h). Converting 'Z' to ET
#      places ~20% of filings hours too EARLY -- a lookahead leak. So the JSON
#      wall clock, read AS ET, is used only as an UPPER BOUND U on the true
#      acceptance A, with A in [U - 10 h, U]. When every A in that interval
#      gives the same placement (night or overnight segment, below), U is
#      exact enough and no request is made; otherwise the index page is
#      fetched and its "Accepted" (true ET) is used. A filing that is still
#      unresolved stays at U: late, never early.
#   2. EFFECT notices are accepted ~00:15 ET the day AFTER their filingDate;
#      placing by filingDate is a leak. Acceptance time only.
#   3. Early-close sessions (13:00 ET): the decision time is 12:30 there, not
#      15:30 (the Calendar below; half days = the NYSE rule set UNION days the
#      panel's SPY/QQQ printed < 300 minute bars -- an extra half day only
#      makes the cutoff earlier, which is the safe direction).
#   4. Before 2002-05 the JSON carries date-only placeholders (T00/04/05/08/
#      10:00:00Z) and the index page shows 00:00:00. A filing submitted after
#      17:30 ET gets the next business day's date (22:00 for Forms 3/4/5/144),
#      so a date-only filing is known by filingDate 17:30 ET at the latest:
#      U = filingDate 17:30 ET (EFFECT: filingDate + 1 day 17:30), never
#      fetched (nothing better exists).
#   Decision time of night D: tau_D = 15:30 ET (12:30 on half days). An event
#   may be used for night D only if A < tau_D - 10 min (a dissemination
#   margin; the SEC's posting lag is about a second). First eligible night
#   n(e) = the first trading day D with A < tau_D - 10 min.
#
# OUTPUT  data/edgar/events.parquet, one row per (event class, filing, mapped
# ticker): cik, ticker, form, items, accepted_et (tz-aware US/Eastern: exact if
# the index page was read, else the upper bound U), filing_date, accession,
# event (earn / bad / bad_old / vote / nt / supply / sc13d / buy / effect;
# supply_x = the post-hoc H4a extra, see classify),
# ts_kind (exact / upper / date_only), accepted_lo_et (lower bound), json_raw
# (the JSON string, for the canaries), role (13D: subject / filer / unknown),
# equity (supply: True / False / NaN = not classified), year. Tickers are the
# ones the map gives the CIK in the acceptance year; event_test.py maps names
# to CIKs per (ticker, year of night D) itself.
#
#   python src/edgar_events.py fetch  [--minutes 240] [--rps 8]  # resumable
#   python src/edgar_events.py build                              # events.parquet
#   python src/edgar_events.py coverage                           # map coverage
#   python src/edgar_events.py all    [--minutes 240]             # all three
#   python src/edgar_events.py evidence-map    # POST-HOC canary map (no requests)

import os, sys, re, json, time, zipfile, threading, argparse, html as htmlmod
import urllib.request, urllib.error, urllib.parse, gzip, glob
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGAR = os.path.join(ROOT, 'data', 'edgar')
ZIP = os.path.join(EDGAR, 'submissions.zip')
MAP = os.path.join(EDGAR, 'research', 'assign_v3.parquet')
CACHE = os.path.join(EDGAR, 'cache')
SUBM = os.path.join(EDGAR, 'submissions')           # --source api cache
INSIDER = os.path.join(EDGAR, 'insider')
FILINGS = os.path.join(EDGAR, 'filings_mapped.parquet')
EVENTS = os.path.join(EDGAR, 'events.parquet')
WIDE = os.path.join(ROOT, 'data', 'xsec1000')

UA = 'qqq-microstructure research baderalotaibi3@gmail.com'
MAX_RPS = 8.0
ET = ZoneInfo('America/New_York')
SEC = 'https://www.sec.gov'
ZIP_URL = SEC + '/Archives/edgar/daily-index/bulkdata/submissions.zip'
INSIDER_URL = (SEC + '/files/structureddata/data/insider-transactions-data-'
               'sets/{q}_form345.zip')

DECISION_MIN = 15 * 60 + 30         # tau_D, minutes after midnight ET
HALF_DECISION_MIN = 12 * 60 + 30    # tau_D on a 13:00-close day
LAG_MIN = 10                        # dissemination margin before tau_D
MAX_SHIFT_NS = 10 * 3600 * 10**9    # JSON wall clock is at most 10 h late
DATE_ONLY_BEFORE = '2002-07-01'
PLACEHOLDER_HMS = {'00:00:00', '04:00:00', '05:00:00', '08:00:00', '10:00:00'}
NEW_ITEMS_FROM = '2004-08-23'       # SEC Rel. 33-8400
OLD_12_FROM = '2003-03-28'          # old item 12 (results of operations)
VOTE_FROM = '2010-02-28'            # item 5.07
STRUCTURED_MAX = 12                 # issuer-years with > 12 424Bs are note shelves

F_8K = {'8-K'}
F_NT = {'NT 10-K', 'NT 10-Q'}
F_REG = {'S-1', 'S-3', 'S-3ASR', 'F-1', 'F-3', 'F-3ASR'}
F_424 = {'424B1', '424B3', '424B4', '424B5', '424B7'}
F_424_ALL = {'424B1', '424B2', '424B3', '424B4', '424B5', '424B6', '424B7'}
F_13D = {'SC 13D', 'SCHEDULE 13D'}
F_EFFECT = {'EFFECT'}
F_4 = {'4'}
F_KEEP = F_8K | F_NT | F_REG | F_424_ALL | F_13D | F_EFFECT | F_4
NEW_BAD = {'1.03', '2.05', '2.06', '3.01', '4.01', '4.02'}
OLD_BAD = {'3', '4'}
WINDOW_CLASSES = ('bad', 'bad_old', 'nt', 'supply', 'sc13d', 'buy', 'vote',
                  'effect', 'supply_x')
SUPPLY_CLASSES = ('supply', 'supply_x')    # both need the document test
# fetch order inside a tier: cheapest, most-decisive classes first
CLASS_ORDER = ['bad', 'nt', 'sc13d', 'vote', 'effect', 'bad_old', 'earn',
               'buy', 'supply', 'supply_x']

EQUITY_RE = re.compile(r'shares\s+of\s+(our\s+)?(class\s+[a-z]\s+)?common\s+'
                       r'stock|ordinary\s+shares|american\s+depositary\s+'
                       r'shares', re.I)
DEBT_RE = re.compile(r'%\s*(senior\s+)?notes\s+due|debentures', re.I)
DOC_CHARS = 20000          # text examined for the equity pattern
COVER_CHARS = 3000         # "cover title" = the first 3,000 text characters
DOC_BYTES = 600_000        # byte cap per document read


# --------------------------------------------------------------------------
# time: the calendar, the decision cutoffs, the bounds
# --------------------------------------------------------------------------

def _nth_weekday(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    d += pd.Timedelta(days=(wd - d.weekday()) % 7)
    return d + pd.Timedelta(weeks=n - 1)


def rule_half_days(years):
    """NYSE 13:00 closes by rule (a superset is safe -- it only moves the
    cutoff earlier): the day after Thanksgiving, Dec 24 and Jul 3 when they
    fall Mon-Thu, Jul 5 when it is a Friday, and 2003-12-26."""
    out = {'20031226'}
    for y in years:
        tg = _nth_weekday(y, 11, 3, 4)
        out.add((tg + pd.Timedelta(days=1)).strftime('%Y%m%d'))
        for m, d in ((12, 24), (7, 3)):
            x = pd.Timestamp(y, m, d)
            if x.weekday() <= 3:
                out.add(x.strftime('%Y%m%d'))
        x = pd.Timestamp(y, 7, 5)
        if x.weekday() == 4:
            out.add(x.strftime('%Y%m%d'))
    return out


class Calendar:
    """Trading days (YYYYMMDD) with their decision cutoffs tau_D - LAG,
    opens and closes, all as int64 UTC nanoseconds."""

    def __init__(self, days, half=()):
        self.days = np.array(sorted(set(days)))
        self.pos = {d: i for i, d in enumerate(self.days)}
        half = set(half)
        self.half = np.array([d in half for d in self.days], bool)
        base = pd.DatetimeIndex(pd.to_datetime(self.days, format='%Y%m%d'))

        def at(minutes):
            return as_ns((base + pd.to_timedelta(minutes, unit='min'))
                         .tz_localize(ET))

        dec = np.where(self.half, HALF_DECISION_MIN, DECISION_MIN)
        self.tau = at(dec)
        self.cut = at(dec - LAG_MIN)
        self.open = at(np.full(len(base), 9 * 60 + 30))
        self.close = at(np.where(self.half, 13 * 60, 16 * 60))
        self._bounds = np.empty(2 * len(base), np.int64)
        self._bounds[0::2] = self.open
        self._bounds[1::2] = self.close

    def first_eligible(self, ns):
        """Position of the first night D with A < tau_D - LAG (strict).
        len(days) when after the last cutoff."""
        return np.searchsorted(self.cut, np.asarray(ns, np.int64), side='right')

    def segment(self, ns):
        """Even 2p: inside session p [open, close); odd 2p+1: the overnight
        after day p [close_p, open_p+1); -1 before the first open."""
        return np.searchsorted(self._bounds, np.asarray(ns, np.int64),
                               side='right') - 1

    def on_or_after(self, yyyymmdd):
        return np.searchsorted(self.days, np.asarray(yyyymmdd), side='left')


def panel_calendar(path=WIDE):
    """The wide panel's own trading days (load_panel's calendar), with half
    days = rule set UNION days SPY/QQQ/QQQQ printed < 300 minute bars."""
    fs = sorted(glob.glob(os.path.join(path, '*.parquet')))
    if not fs:
        raise SystemExit(f'no panel files in {path}')
    d = pd.concat([pd.read_parquet(f, columns=['ticker', 'day', 'bars'])
                   for f in fs], ignore_index=True)
    days = d.day.unique()
    ix = d[d.ticker.isin(['SPY', 'QQQ', 'QQQQ'])].groupby('day').bars.max()
    years = range(int(min(days)[:4]), int(max(days)[:4]) + 1)
    half = rule_half_days(years) | set(ix[ix < 300].index)
    return Calendar(days, half)


def as_ns(x):
    """Any datetime-like -> int64 nanoseconds since the epoch (UTC). pandas
    3 keeps whatever unit it parsed (often 'us'); every comparison in this
    layer is done in ns, so every conversion goes through here."""
    return pd.DatetimeIndex(x).as_unit('ns').asi8


def from_ns(a):
    """int64 ns (UTC) -> tz-aware US/Eastern DatetimeIndex."""
    return pd.to_datetime(np.asarray(a, np.int64), unit='ns',
                          utc=True).tz_convert(ET)


def localize_et(naive, later=True):
    """Naive wall-clock datetimes -> tz-aware ET (ns). In the fall-back hour
    take the later instant when `later` (an upper bound must not move
    earlier); a time in the spring-forward gap moves forward."""
    idx = pd.DatetimeIndex(naive).as_unit('ns')
    amb = np.zeros(len(idx), bool) if later else np.ones(len(idx), bool)
    return idx.tz_localize(ET, ambiguous=amb, nonexistent='shift_forward')


def json_wallclock(s):
    """'2026-07-31T00:30:28.000Z' -> naive Timestamp of the wall clock
    (the 'Z' is IGNORED on purpose -- LEAK TRAP 1)."""
    return pd.to_datetime(pd.Series(s, dtype=object).str.slice(0, 19),
                          format='%Y-%m-%dT%H:%M:%S', errors='coerce')


def is_date_only(filing_date, json_raw):
    fd = pd.Series(filing_date, dtype=object).astype(str)
    hms = pd.Series(json_raw, dtype=object).astype(str).str.slice(11, 19)
    return ((fd < DATE_ONLY_BEFORE) & hms.isin(PLACEHOLDER_HMS)).values


def bounds(forms, filing_date, json_raw):
    """(U, L, kind) per filing from the JSON alone: U the leak-free upper
    bound on acceptance (tz-aware ET), L the lower bound, kind 'upper' or
    'date_only'."""
    forms = pd.Series(forms, dtype=object).astype(str).values
    fd = pd.Series(filing_date, dtype=object).astype(str).values
    wall = json_wallclock(json_raw)
    donly = is_date_only(fd, json_raw) | wall.isna().values
    fdt = pd.to_datetime(pd.Series(fd), format='%Y-%m-%d', errors='coerce')
    late = np.isin(forms, ['3', '4', '5', '144', '3/A', '4/A', '5/A'])
    eff = np.isin(forms, list(F_EFFECT))
    dmin = np.where(late, 22 * 60, 17 * 60 + 30)
    du = fdt + pd.to_timedelta(np.where(eff, 1, 0), unit='D') \
        + pd.to_timedelta(dmin, unit='min')
    naive_u = pd.Series(np.where(donly, du.values, wall.values))
    U = localize_et(naive_u.values, later=True)
    L_timed = U - pd.Timedelta(MAX_SHIFT_NS, unit='ns')
    dl = localize_et((fdt - pd.Timedelta(days=4) +
                      pd.Timedelta(minutes=17 * 60 + 30)).values, later=False)
    L = from_ns(np.where(donly, as_ns(dl), as_ns(L_timed)))
    kind = np.where(donly, 'date_only', 'upper')
    return U, L, kind


def needs_index(cls, U_ns, L_ns, kind, cal):
    """Would the index page change this filing's placement? 13Ds always
    (the role lives there); date-only never (nothing better exists);
    otherwise only when the bound interval [L, U] straddles a placement
    boundary -- a decision cutoff for window classes, a session edge for
    earnings sources."""
    cls = np.asarray(cls)
    timed = np.asarray(kind) == 'upper'
    U_ns = np.asarray(U_ns, np.int64)
    L_ns = np.asarray(L_ns, np.int64)
    win = cal.first_eligible(L_ns) != cal.first_eligible(U_ns)
    seg = cal.segment(L_ns) != cal.segment(U_ns)
    out = np.where(cls == 'earn', seg | win, win) & timed
    return out | (cls == 'sc13d')


# --------------------------------------------------------------------------
# the map, the filings, the classes
# --------------------------------------------------------------------------

def load_map(path=MAP):
    a = pd.read_parquet(path)
    a = a[['ticker', 'yr', 'cik']].drop_duplicates(['ticker', 'yr'])
    return a.astype({'yr': int, 'cik': int})


def cik_years(amap):
    """The years each CIK's filings matter: its mapped years, the year
    before (H1's 364-day lookback) and the year after (windows that cross
    a year end)."""
    out = {}
    for c, g in amap.groupby('cik').yr:
        ys = set(g)
        out[int(c)] = ys | {y - 1 for y in ys} | {y + 1 for y in ys}
    return out


COLS = ['accessionNumber', 'filingDate', 'acceptanceDateTime', 'form',
        'items', 'primaryDocument']


def _rows_from_pages(cik, pages, keep_years):
    out = []
    for r in pages:
        n = len(r.get('form', []))
        cols = [r.get(c, [''] * n) for c in COLS]
        for i in range(n):
            f = cols[3][i]
            if f not in F_KEEP:
                continue
            fd = cols[1][i] or ''
            if keep_years is not None and int(fd[:4] or 0) not in keep_years:
                continue
            out.append((cik,) + tuple(c[i] for c in cols))
    return out


def _zip_worker(args):
    zp, ciks, years = args
    z = zipfile.ZipFile(zp)
    names = set(z.namelist())
    out = []
    for cik in ciks:
        nm = f'CIK{cik:010d}.json'
        if nm not in names:
            continue
        d = json.loads(z.read(nm))
        pages = [d['filings']['recent']]
        for f in d['filings'].get('files', []):
            if f['name'] in names:
                pages.append(json.loads(z.read(f['name'])))
        out += _rows_from_pages(cik, pages, years.get(cik))
    return out


def read_zip(ciks, years, zp=ZIP, procs=4):
    """Every kept-form filing of `ciks` straight from the bulk archive, read
    in place (it is 1.5 GB zipped, 5.8 GB unzipped -- never extracted)."""
    from multiprocessing import Pool
    ciks = sorted(ciks)
    chunks = [(zp, ciks[i:i + 100], {c: years.get(c) for c in ciks[i:i + 100]})
              for i in range(0, len(ciks), 100)]
    rows = []
    with Pool(procs) as p:
        for o in p.imap_unordered(_zip_worker, chunks):
            rows += o
    return pd.DataFrame(rows, columns=['cik'] + COLS)


def read_api(ciks, years, fetcher):
    """The same rows from data.sec.gov, one request per page, each page
    cached gzipped under data/edgar/submissions/ (resumable)."""
    os.makedirs(SUBM, exist_ok=True)
    rows = []
    for cik in sorted(ciks):
        def page(name):
            p = os.path.join(SUBM, name + '.gz')
            if os.path.exists(p):
                return json.loads(gzip.decompress(open(p, 'rb').read()))
            b = fetcher.get(f'https://data.sec.gov/submissions/{name}')
            if b is None:
                return None
            open(p, 'wb').write(gzip.compress(b))
            return json.loads(b)
        d = page(f'CIK{cik:010d}.json')
        if d is None:
            continue
        pages = [d['filings']['recent']]
        for f in d['filings'].get('files', []):
            x = page(f['name'])
            if x is not None:
                pages.append(x)
        rows += _rows_from_pages(cik, pages, years.get(cik))
    return pd.DataFrame(rows, columns=['cik'] + COLS)


def parse_items(s):
    if not isinstance(s, str) or not s.strip():
        return frozenset()
    return frozenset(x.strip() for x in s.split(',') if x.strip())


def classify(f):
    """Filing rows -> (row index, event class) pairs, per the hypotheses'
    definitions (amendments are separate form types and never match).
    'supply' still needs the document's equity test; 'sc13d' its role;
    Form 4 rows become 'buy' only via the insider data sets (build())."""
    f = f.reset_index(drop=True)
    fd = f.filingDate.astype(str)
    form = f.form.astype(str)
    its = f['items'].map(parse_items)
    out = []

    def add(mask, cls):
        out.extend((i, cls) for i in np.flatnonzero(np.asarray(mask)))

    is8 = form.isin(F_8K)
    earn = its.map(lambda s: '2.02' in s) & (fd >= NEW_ITEMS_FROM) | \
        its.map(lambda s: '12' in s) & (fd >= OLD_12_FROM) & (fd < NEW_ITEMS_FROM)
    add(is8 & earn, 'earn')
    add(is8 & (fd >= NEW_ITEMS_FROM) & its.map(lambda s: bool(s & NEW_BAD)),
        'bad')
    add(is8 & (fd < NEW_ITEMS_FROM) & its.map(lambda s: bool(s & OLD_BAD)),
        'bad_old')
    add(is8 & (fd >= VOTE_FROM) & its.map(
        lambda s: s in (frozenset({'5.07'}), frozenset({'5.07', '9.01'}))),
        'vote')
    add(form.isin(F_NT), 'nt')
    yr = fd.str[:4]
    n424 = form.isin(F_424_ALL).groupby([f.cik, yr]).transform('sum')
    add(form.isin(F_REG | F_424) & (n424 <= STRUCTURED_MAX), 'supply')
    # POST-HOC (review finding 3, written after the real run; not evidence):
    # the declared count above includes 424Bs filed LATER in the year, which
    # were unknown at the decision. 'supply_x' = the filings that count
    # excluded but whose count KNOWN AT ACCEPTANCE is <= 12. The amended H4
    # uses supply + supply_x; the declared H4 uses supply alone.
    ytd = n424_known_at_acceptance(f.cik, fd, form)
    add(form.isin(F_REG | F_424) & (n424 > STRUCTURED_MAX) &
        (ytd <= STRUCTURED_MAX), 'supply_x')
    add(form.isin(F_13D), 'sc13d')
    add(form.isin(F_EFFECT), 'effect')
    return pd.DataFrame(out, columns=['row', 'event'])


def n424_known_at_acceptance(cik, filing_date, form):
    """Per filing: the CIK's 424B1-7 filings in the same calendar year with
    an EARLIER filingDate, plus 1 if the filing is itself one. A filing
    dated before day t was accepted before any filing dated t (EDGAR dates
    anything after 17:30 ET as the next business day), so every filing
    counted was public when this one was accepted. Same-day 424Bs are not
    counted (their order is unknown) -- an undercount, never a lookahead."""
    x = pd.DataFrame({'cik': np.asarray(cik),
                      'fd': pd.Series(filing_date, dtype=object).astype(str)
                      .values,
                      'is424': pd.Series(form, dtype=object).astype(str)
                      .isin(F_424_ALL).astype(int).values})
    x['yr'] = x.fd.str[:4]
    daily = x.groupby(['cik', 'yr', 'fd'], sort=True).is424.sum() \
        .rename('d').reset_index()
    daily['before'] = daily.groupby(['cik', 'yr']).d.cumsum() - daily.d
    out = x.merge(daily[['cik', 'yr', 'fd', 'before']], on=['cik', 'yr', 'fd'],
                  how='left')
    return pd.Series((out.before + out.is424).values, index=getattr(
        cik, 'index', None))


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

class Fetcher:
    """Rate-limited (<= rps across all threads), retrying GET with the
    declared User-Agent. 404 -> None. `opener` is injectable for tests."""

    def __init__(self, rps=MAX_RPS, opener=None, tries=6, sleep=time.sleep):
        self.gap = 1.0 / min(rps, MAX_RPS)
        self.lock = threading.Lock()
        self.next_t = 0.0
        self.n = 0
        self.errors = 0
        self.tries = tries
        self.sleep = sleep
        self.opener = opener or self._urlopen

    @staticmethod
    def _urlopen(url, max_bytes):
        h = {'User-Agent': UA}
        if max_bytes is None:
            h['Accept-Encoding'] = 'gzip'
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=60) as r:
            b = r.read() if max_bytes is None else r.read(max_bytes)
            if r.headers.get('Content-Encoding') == 'gzip':
                b = gzip.decompress(b)
            return 200, b

    def _wait(self):
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.gap
        if t > now:
            self.sleep(t - now)

    def get(self, url, max_bytes=None):
        for k in range(self.tries):
            self._wait()
            self.n += 1
            try:
                status, b = self.opener(url, max_bytes)
                if status == 200:
                    return b
                if status == 404:
                    return None
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                status = e.code
            except Exception:
                status = -1
            self.errors += 1
            # 403/429 are the SEC's rate-limit answers: back off hard
            self.sleep(min(120, (30 if status in (403, 429) else 2) * 2 ** k))
        raise IOError(f'gave up on {url}')


class JsonlCache:
    """Append-only, thread-safe, resumable: one JSON object per line, keyed
    by 'acc'. A torn last line (killed mid-write) is skipped on load."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data = {}
        if os.path.exists(path):
            for line in open(path):
                try:
                    o = json.loads(line)
                    self.data[o['acc']] = o
                except Exception:
                    continue
        self.lock = threading.Lock()

    def __contains__(self, acc):
        return acc in self.data

    def put(self, o):
        self.put_many([o])

    def put_many(self, objs):
        with self.lock:
            with open(self.path, 'a') as fh:
                for o in objs:
                    self.data[o['acc']] = o
                    fh.write(json.dumps(o) + '\n')


_ACC_RE = re.compile(r'Accepted</div>\s*<div class="info">([^<]+)<')
_FD_RE = re.compile(r'Filing Date</div>\s*<div class="info">([^<]+)<')
_CO_RE = re.compile(r'<span class="companyName">(.*?)</span>', re.S)


def parse_index(html):
    """Filing index page -> {'accepted', 'filing_date', 'roles'}; roles is
    a list of [role, cik] (e.g. ['Subject', 910073], ['Filed by', ...])."""
    if isinstance(html, bytes):
        html = html.decode('utf-8', 'replace')
    m = _ACC_RE.search(html)
    f = _FD_RE.search(html)
    roles = []
    for c in _CO_RE.findall(html):
        k = re.search(r'CIK=(\d+)', c)
        head = re.sub(r'<[^>]+>', '', c.split('<acronym')[0])
        r = re.findall(r'\(([A-Za-z ]+)\)', head)
        if k:
            roles.append([r[-1].strip() if r else '', int(k.group(1))])
    return {'accepted': m.group(1).strip() if m else None,
            'filing_date': f.group(1).strip() if f else None, 'roles': roles}


def html_to_text(raw):
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', 'replace')
    s = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', raw)
    s = re.sub(r'(?s)<[^>]+>', ' ', s)
    s = htmlmod.unescape(s).replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def classify_doc(raw):
    """The declared equity test: the first 20,000 text characters name
    common stock / ordinary shares / ADSs, and the cover (first 3,000) is
    not a note or debenture offering."""
    text = html_to_text(raw)[:DOC_CHARS]
    eq = bool(EQUITY_RE.search(text))
    debt = bool(DEBT_RE.search(text[:COVER_CHARS]))
    return {'equity': eq and not debt, 'eq_match': eq, 'debt_cover': debt,
            'n_text': len(text)}


def acc_base(cik, acc):
    return f'{SEC}/Archives/edgar/data/{int(cik)}/{acc.replace("-", "")}/'


def fetch_index(fetcher, cik, acc, idx_cache):
    b = fetcher.get(acc_base(cik, acc) + f'{acc}-index.htm')
    o = {'acc': acc, 'status': 404} if b is None else \
        {'acc': acc, 'status': 200, 'src': 'index', **parse_index(b)}
    idx_cache.put(o)
    return 1


def fetch_doc(fetcher, cik, acc, doc, doc_cache):
    name = doc if isinstance(doc, str) and doc.strip() else f'{acc}.txt'
    b = fetcher.get(acc_base(cik, acc) + name, max_bytes=DOC_BYTES)
    if b is None and name != f'{acc}.txt':
        b = fetcher.get(acc_base(cik, acc) + f'{acc}.txt', max_bytes=DOC_BYTES)
    doc_cache.put({'acc': acc, 'status': 404} if b is None else
                  {'acc': acc, 'status': 200, **classify_doc(b)})
    return 1


# The company filing feed (cgi-bin/browse-edgar ... output=atom) lists up to
# 100 filings per request, newest first, each with <updated> = the SEC
# acceptance datetime WITH its true UTC offset (e.g. 2026-07-30T16:30:28-04:00
# for Apple's 8-K whose index page says Accepted 16:30:28). Checked against 64
# index pages across 16 forms, 2003-2026: 63 found, 63 identical to the
# second. One feed page replaces up to 100 index-page requests -- the cheap
# compute that makes exact times for every 8-K affordable.
_ENTRY_RE = re.compile(r'<entry>(.*?)</entry>', re.S)


def parse_feed(xml):
    """Atom page -> list of {'acc', 'form', 'filing_date', 'accepted'} with
    'accepted' the ET wall clock 'YYYY-MM-DD HH:MM:SS'."""
    if isinstance(xml, bytes):
        xml = xml.decode('latin-1')
    out = []
    for e in _ENTRY_RE.findall(xml):
        acc = re.search(r'<accession-number>([^<]+)<', e)
        upd = re.search(r'<updated>([^<]+)</updated>', e)
        if not acc or not upd:
            continue
        fd = re.search(r'<filing-date>([^<]+)<', e)
        ft = re.search(r'<filing-type>([^<]+)<', e)
        ts = pd.Timestamp(upd.group(1).strip())
        if ts.tzinfo is None:
            continue
        out.append({'acc': acc.group(1).strip(),
                    'form': ft.group(1).strip() if ft else '',
                    'filing_date': fd.group(1).strip() if fd else '',
                    'accepted': ts.tz_convert(ET).strftime('%Y-%m-%d %H:%M:%S')})
    return out


def feed_url(cik, ftype, owner, dateb, start):
    return (f'{SEC}/cgi-bin/browse-edgar?action=getcompany&CIK={int(cik):010d}'
            f'&type={urllib.parse.quote(ftype)}&dateb={dateb}&owner={owner}'
            f'&start={start}&count=100&output=atom')


def fetch_feed(fetcher, cik, ftype, owner, lo, hi, idx_cache, feed_cache,
               max_pages=400):
    """Every filing of `ftype` (a prefix, as EDGAR matches it) for `cik` with
    filing date in [lo, hi], newest first, re-anchoring `dateb` if paging
    runs long; each entry lands in the index cache with src='feed'."""
    key = f'{int(cik)}|{ftype}|{owner}|{lo}|{hi}'
    dateb, start, n, pages = hi.replace('-', ''), 0, 0, 0
    seen = set()
    while pages < max_pages:
        ents = parse_feed(fetcher.get(feed_url(cik, ftype, owner, dateb,
                                               start)) or b'')
        pages += 1
        new = [dict(e, status=200, src='feed') for e in ents
               if e['acc'] not in seen]
        seen.update(e['acc'] for e in ents)
        idx_cache.put_many(new)
        n += len(new)
        if len(ents) < 100 or min(e['filing_date'] for e in ents) < lo:
            break
        if start >= 2000 and new:          # re-anchor long histories
            dateb = min(e['filing_date'] for e in ents).replace('-', '')
            start = 0
        elif not new:                      # stuck on one crowded date
            d = pd.Timestamp(min(e['filing_date'] for e in ents))
            dateb = (d - pd.Timedelta(days=1)).strftime('%Y%m%d')
            start = 0
        else:
            start += 100
    feed_cache.put({'acc': key, 'n': n, 'pages': pages})
    return pages


def run_tasks(tasks, fetcher, deadline, threads=4, label='', every=500):
    """tasks: list of (fn, args); each fn writes its own cache. Runs in
    order on `threads` threads until done or `deadline` (time.time())."""
    done = [0]
    t0, n0 = time.time(), fetcher.n

    def one(t):
        fn, args = t
        if time.time() > deadline:
            return
        try:
            fn(fetcher, *args)
        except Exception as e:                      # gave up: retry next run
            print(f'  {label} skip {fn.__name__}{args[:3]}: {e}', flush=True)
            return
        done[0] += 1
        if done[0] % every == 0:
            el = time.time() - t0
            print(f'  {label} {done[0]:,}/{len(tasks):,} tasks  {el / 60:5.1f} '
                  f'min  {(fetcher.n - n0) / max(el, 1e-9):.2f} req/s  '
                  f'errors {fetcher.errors}', flush=True)

    with ThreadPoolExecutor(threads) as ex:
        for i in range(0, len(tasks), 200):
            if time.time() > deadline:
                break
            list(ex.map(one, tasks[i:i + 200]))
    return done[0]


# --------------------------------------------------------------------------
# insider purchases (H6)
# --------------------------------------------------------------------------

def insider_quarters(last='2026q1'):
    out = []
    for y in range(2006, int(last[:4]) + 1):
        for q in range(1, 5):
            s = f'{y}q{q}'
            if s <= last:
                out.append(s)
    return out


def purchases_from_zip(zp):
    """Form 4 (not 4/A) filings with >= 1 non-derivative 'P' transaction and
    >= 1 reporting owner who is a director or officer."""
    import csv
    z = zipfile.ZipFile(zp)

    def tsv(name, cols):
        with z.open(name) as fh:
            return pd.read_csv(fh, sep='\t', dtype=str, usecols=cols,
                               quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
    s = tsv('SUBMISSION.tsv', ['ACCESSION_NUMBER', 'FILING_DATE',
                               'DOCUMENT_TYPE', 'ISSUERCIK'])
    s = s[s.DOCUMENT_TYPE == '4']
    o = tsv('REPORTINGOWNER.tsv', ['ACCESSION_NUMBER', 'RPTOWNER_RELATIONSHIP'])
    rel = o.RPTOWNER_RELATIONSHIP.fillna('')
    do = set(o.ACCESSION_NUMBER[rel.str.contains('Director') |
                                rel.str.contains('Officer')])
    t = tsv('NONDERIV_TRANS.tsv', ['ACCESSION_NUMBER', 'TRANS_CODE',
                                   'TRANS_SHARES', 'TRANS_PRICEPERSHARE'])
    t = t[t.TRANS_CODE == 'P']
    v = pd.to_numeric(t.TRANS_SHARES, errors='coerce') * \
        pd.to_numeric(t.TRANS_PRICEPERSHARE, errors='coerce')
    g = t.assign(v=v).groupby('ACCESSION_NUMBER').agg(n_p=('v', 'size'),
                                                       p_value=('v', 'sum'))
    s = s[s.ACCESSION_NUMBER.isin(g.index) & s.ACCESSION_NUMBER.isin(do)]
    s = s.join(g, on='ACCESSION_NUMBER')
    return pd.DataFrame({'accession': s.ACCESSION_NUMBER.values,
                         'issuer_cik': pd.to_numeric(s.ISSUERCIK,
                                                     errors='coerce').values,
                         'filing_date_ds': s.FILING_DATE.values,
                         'n_p': s.n_p.values, 'p_value': s.p_value.values})


def fetch_insider(fetcher, quarters):
    os.makedirs(INSIDER, exist_ok=True)
    for q in quarters:
        out = os.path.join(INSIDER, f'{q}_purchases.parquet')
        if os.path.exists(out):
            continue
        b = fetcher.get(INSIDER_URL.format(q=q))
        if b is None:
            print(f'  insider {q}: not published (404)')
            continue
        tmp = os.path.join(INSIDER, f'{q}.zip.part')
        open(tmp, 'wb').write(b)
        try:
            p = purchases_from_zip(tmp)
        finally:
            os.remove(tmp)
        p.to_parquet(out)
        print(f'  insider {q}: {len(b) / 1e6:.1f} MB -> {len(p):,} '
              f'director/officer purchase filings', flush=True)


def load_insider():
    fs = sorted(glob.glob(os.path.join(INSIDER, '*_purchases.parquet')))
    if not fs:
        return pd.DataFrame(columns=['accession', 'issuer_cik', 'n_p',
                                     'p_value'])
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) \
        .drop_duplicates('accession')


# --------------------------------------------------------------------------
# tiers, candidates, the build
# --------------------------------------------------------------------------

def wide_name_months(path=WIDE):
    fs = sorted(glob.glob(os.path.join(path, '*.parquet')))
    d = pd.concat([pd.read_parquet(f, columns=['ticker', 'month',
                                               'dollar_vol'])
                   for f in fs], ignore_index=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from xsec_backtest import TEST_RE, TEST_SYM
    d = d[~(d.ticker.str.match(TEST_RE) | d.ticker.isin(TEST_SYM))]
    return d


def tier_of_tickers(nm):
    """xsec_wide's tiers from month-local dollar-volume rank: CORE = ever
    ranked <= 150; never-core otherwise (MID and DEEP together)."""
    from xsec_wide import monthly_rank, CORE_N
    rk = monthly_rank(nm)
    best = rk.groupby('ticker').min()
    return (best <= CORE_N).map({True: 'CORE', False: 'never-core'})


def load_filings(amap, source='zip', fetcher=None, refresh=False):
    """Kept-form filings of every mapped CIK, cached in filings_mapped.parquet
    (rebuilt when the map gains a CIK, or with --refresh-filings)."""
    want = sorted(int(c) for c in amap.cik.unique())
    tag = FILINGS + '.ciks'
    if os.path.exists(FILINGS) and os.path.exists(tag) and not refresh:
        if set(want) <= set(json.load(open(tag))):
            f = pd.read_parquet(FILINGS)
            return f[f.cik.isin(want)]
    years = cik_years(amap)
    t0 = time.time()
    if source == 'zip':
        if not os.path.exists(ZIP):
            print(f'downloading the bulk archive (one request) -> {ZIP}')
            b = (fetcher or Fetcher()).get(ZIP_URL)
            open(ZIP + '.part', 'wb').write(b)
            os.replace(ZIP + '.part', ZIP)
        f = read_zip(set(want), years)
    else:
        f = read_api(set(want), years, fetcher or Fetcher())
    f = f.drop_duplicates(['cik', 'accessionNumber', 'form'])
    f.to_parquet(FILINGS)
    json.dump(want, open(tag, 'w'))
    print(f'filings of {len(want):,} mapped CIKs: {len(f):,} kept-form '
          f'rows in {time.time() - t0:.0f} s (source {source})')
    return f


def candidates(f, amap, cal, tiers, insider):
    """One row per (filing, event class) with bounds, tier and fetch needs."""
    c = classify(f)
    base = f.reset_index(drop=True)
    ev = base.loc[c.row].reset_index(drop=True)
    ev['event'] = c.event.values
    # Form 4 purchases from the insider data sets (issuer = this CIK)
    if len(insider):
        ins = insider.dropna(subset=['issuer_cik'])
        key = set(zip(ins.accession, ins.issuer_cik.astype(int)))
        b4 = base[base.form.isin(F_4)]
        hit = [(a, int(k)) in key for a, k in zip(b4.accessionNumber, b4.cik)]
        b4 = b4[np.array(hit, bool)].copy()
        b4['event'] = 'buy'
        ev = pd.concat([ev, b4], ignore_index=True)
    ev = ev.drop_duplicates(['cik', 'accessionNumber', 'event'])
    U, L, kind = bounds(ev.form, ev.filingDate, ev.acceptanceDateTime)
    ev['U'], ev['L'], ev['kind'] = U, L, kind
    ev['need_index'] = needs_index(ev.event.values, as_ns(U), as_ns(L), kind,
                                   cal)
    tk = amap.merge(tiers.rename('tier'), left_on='ticker', right_index=True,
                    how='left')
    core = set(tk[tk.tier == 'CORE'].cik)
    ev['tier'] = np.where(ev.cik.isin(core), 'CORE', 'never-core')
    return ev


FAMILIES = [   # (feed type prefix as EDGAR matches it, owner=, forms it times)
    ('8-K', 'exclude', F_8K), ('NT 10', 'exclude', F_NT),
    ('EFFECT', 'exclude', F_EFFECT), ('S-1', 'exclude', {'S-1'}),
    ('S-3', 'exclude', {'S-3', 'S-3ASR'}), ('F-1', 'exclude', {'F-1'}),
    ('F-3', 'exclude', {'F-3', 'F-3ASR'}), ('424B', 'exclude', F_424),
    ('4', 'only', F_4)]


def plan(ev, f, idx_cache, doc_cache, feed_cache,
         tiers=('CORE', 'never-core')):
    """The fetch list, in three phases, each CORE before never-core:
      1. feeds: per (CIK, form family), one walk of the company feed over the
         candidates' date range -- when its estimated pages are fewer than
         the index pages the same filings would need one by one;
      2. index pages: every SC 13D (the role is only there), and every timed
         candidate no feed covered whose placement the bound leaves open;
      3. documents: the equity test for every supply filing.
    Anything already cached is skipped, so a re-run resumes."""
    fam_dates = {}
    for ftype, _, _ in FAMILIES:
        x = f[f.form.astype(str).str.startswith(ftype)]
        fam_dates[ftype] = {c: np.sort(g.values) for c, g in
                            x.groupby('cik').filingDate}
    have = set(idx_cache.data)
    feeds = {t: [] for t in tiers}
    idx = {t: [] for t in tiers}
    docs = {t: [] for t in tiers}
    ev = ev[ev.tier.isin(tiers)].sort_values('filingDate', ascending=False)
    done_feeds = {}
    for k in feed_cache.data:
        c, ft, ow, l, h = k.split('|')
        done_feeds.setdefault((int(c), ft, ow), []).append((l, h))
    ev = ev.assign(_have=ev.accessionNumber.isin(have).values,
                   _doc=ev.accessionNumber.isin(set(doc_cache.data)).values)
    timed = ev[(ev.kind == 'upper') & (ev.event != 'sc13d') & ~ev._have]
    for ftype, owner, forms in FAMILIES:
        sub = timed[timed.form.isin(forms)]
        for (tier, cik), g in sub.groupby(['tier', 'cik']):
            g = g.drop_duplicates('accessionNumber')
            lo, hi = g.filingDate.min(), g.filingDate.max()
            ind = g[g.need_index]
            d = fam_dates[ftype].get(cik, np.array([], object))
            pages = (np.searchsorted(d, hi, 'right') -
                     np.searchsorted(d, lo, 'left')) // 100 + 1
            covered = any(l <= lo and h >= hi for l, h in
                          done_feeds.get((int(cik), ftype, owner), ()))
            if not covered and pages < len(ind):
                feeds[tier].append((fetch_feed, (int(cik), ftype, owner, lo,
                                                 hi, idx_cache, feed_cache)))
            else:
                idx[tier] += [(fetch_index, (int(r.cik), r.accessionNumber,
                                             idx_cache))
                              for r in ind.itertuples()]
    for r in ev[(ev.event == 'sc13d') & ~ev._have] \
            .drop_duplicates('accessionNumber').itertuples():
        idx[r.tier].append((fetch_index, (int(r.cik), r.accessionNumber,
                                          idx_cache)))
    for r in ev[ev.event.isin(SUPPLY_CLASSES) & ~ev._doc] \
            .drop_duplicates('accessionNumber').itertuples():
        docs[r.tier].append((fetch_doc, (int(r.cik), r.accessionNumber,
                                         r.primaryDocument, doc_cache)))
    out = []
    for phase in (feeds, idx, docs):
        for t in tiers:
            seen = set()
            for fn, args in phase[t]:
                k = (fn.__name__,) + (tuple(args[:5]) if fn is fetch_feed
                                      else (args[1],))
                if k not in seen:
                    seen.add(k)
                    out.append((fn, args))
    return out, {p: {t: len(v[t]) for t in tiers} for p, v in
                 (('feeds', feeds), ('index', idx), ('docs', docs))}


def resolve(ev, idx_cache, doc_cache):
    """Apply cached index pages and documents: exact times, roles, equity."""
    ev = ev.copy()
    acc = ev.accessionNumber.values
    got = [idx_cache.data.get(a) for a in acc]
    exact = pd.Series([g.get('accepted') if g and g.get('status') == 200
                       else None for g in got], dtype=object)
    fd = ev.filingDate.astype(str).values
    ok = exact.notna().to_numpy().copy()
    hms = exact.fillna('').str.slice(11, 19).values
    ok &= ~((hms == '00:00:00') & (fd < DATE_ONLY_BEFORE))
    A = pd.to_datetime(exact.where(ok), format='%Y-%m-%d %H:%M:%S',
                       errors='coerce')
    ok &= A.notna().values
    Aet = localize_et(A.fillna(pd.Timestamp('2000-01-01')).values, later=True)
    acc_ns = np.where(ok, as_ns(Aet), as_ns(ev.U))
    ev['accepted_et'] = from_ns(acc_ns)
    ev['ts_kind'] = np.where(ok, 'exact', ev.kind.values)
    # validation of LEAK TRAP 1 on every resolved filing: U - A in [0, 10 h]
    shift = (as_ns(ev.U) - acc_ns) / 3.6e12
    ev['json_minus_exact_h'] = np.where(ok, shift, np.nan)
    roles = []
    for g, k in zip(got, ev.cik.values):
        if not g or g.get('status') != 200:
            roles.append('unknown')
            continue
        rs = [r for r, c in g.get('roles', []) if c == int(k)]
        roles.append('subject' if 'Subject' in rs else
                     'filer' if rs else 'unknown')
    ev['role'] = np.where(ev.event == 'sc13d', roles, '')
    dg = [doc_cache.data.get(a) for a in acc]
    ev['equity'] = [(g.get('equity') if g and g.get('status') == 200
                     else np.nan) if e in SUPPLY_CLASSES else np.nan
                    for g, e in zip(dg, ev.event)]
    return ev


def explode_tickers(ev, amap):
    """One row per mapped ticker of the CIK in the acceptance year."""
    ev = ev.assign(year=ev.accepted_et.dt.year.astype(int))
    m = amap.rename(columns={'yr': 'year'})
    out = ev.merge(m, on=['cik', 'year'], how='left')
    out['ticker'] = out.ticker.fillna('')
    return out


def build(ev, amap, idx_cache, doc_cache, path=EVENTS):
    r = resolve(ev, idx_cache, doc_cache)
    r = explode_tickers(r, amap)
    out = pd.DataFrame({
        'cik': r.cik.astype(int), 'ticker': r.ticker, 'form': r.form,
        'items': r['items'].fillna(''), 'accepted_et': r.accepted_et,
        'filing_date': r.filingDate, 'accession': r.accessionNumber,
        'event': r.event, 'ts_kind': r.ts_kind,
        'accepted_lo_et': pd.DatetimeIndex(r.L).tz_convert(ET),
        'upper_et': pd.DatetimeIndex(r.U).tz_convert(ET),
        'json_raw': r.acceptanceDateTime, 'role': r.role,
        'equity': r.equity.astype(float), 'tier': r.tier,
        'need_index': r.need_index, 'json_minus_exact_h':
            r.json_minus_exact_h, 'primary_doc': r.primaryDocument,
        'year': r.year})
    out.to_parquet(path)
    return out


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------

def coverage_table(nm, amap, tiers):
    """Share of the panel's name-months with a mapped CIK, by era x tier,
    count- and dollar-weighted. ETFs (the frozen list) are shown apart:
    they have no issuer events by construction."""
    from xsec_backtest import ETF
    d = nm.groupby(['ticker', 'month'], as_index=False).dollar_vol.sum()
    d['yr'] = d.month.str[:4].astype(int)
    d = d.merge(amap, on=['ticker', 'yr'], how='left')
    d['mapped'] = d.cik.notna()
    d['tier'] = d.ticker.map(tiers).fillna('never-core')
    d['era'] = pd.cut(d.yr, [0, 2000, 2009, 2017, 2100],
                      labels=['1999-2000', '2001-2009', '2010-2017',
                              '2018-2026']).astype(str)
    d['etf'] = d.ticker.isin(ETF)
    rows = []
    for (tier, era), g in list(d[~d.etf].groupby(['tier', 'era'])) + \
            [(('ALL', 'all'), d[~d.etf])] + \
            [((t, 'all'), g) for t, g in d[~d.etf].groupby('tier')]:
        rows.append(dict(tier=tier, era=era, name_months=len(g),
                         mapped=g.mapped.mean(),
                         mapped_dollar=(g.mapped * g.dollar_vol).sum()
                         / g.dollar_vol.sum(),
                         names=g.ticker.nunique(),
                         mapped_names=g[g.mapped].ticker.nunique()))
    return pd.DataFrame(rows), int(d.etf.sum())


def coverage_by_survival(nm, amap, horizon=24):
    """POST-HOC (review finding 2): the mapped share of non-ETF name-months
    split by whether the ticker is STILL IN THE PANEL `horizon` months later.
    The research map's candidate CIKs come partly from today's ticker owners
    and from 2009+ XBRL file names, so names that later died are under-
    mapped; this shows by how much. Months closer than `horizon` to the
    panel's end cannot be classed and are left out."""
    from xsec_backtest import ETF
    d = nm[~nm.ticker.isin(ETF)].groupby(['ticker', 'month'],
                                         as_index=False).dollar_vol.sum()
    d['yr'] = d.month.str[:4].astype(int)
    d = d.merge(amap, on=['ticker', 'yr'], how='left')
    d['mapped'] = d.cik.notna()
    d['o'] = pd.PeriodIndex(d.month, freq='M').asi8
    d['last'] = d.groupby('ticker').o.transform('max')
    d = d[d.o + horizon <= d.o.max()].copy()
    d['later'] = np.where(d['last'] >= d.o + horizon, 'still there', 'gone')
    d['era'] = pd.cut(d.yr, [0, 2000, 2009, 2017, 2100],
                      labels=['1999-2000', '2001-2009', '2010-2017',
                              '2018-2026']).astype(str)
    rows = []
    for (era, later), g in d.groupby(['era', 'later']):
        rows.append(dict(era=era, later=later, name_months=len(g),
                         mapped=g.mapped.mean(),
                         mapped_dollar=(g.mapped * g.dollar_vol).sum()
                         / g.dollar_vol.sum()))
    return pd.DataFrame(rows)


def evidence_map(ticker_years, tickers_norm, xbrl, efts, entities):
    """POST-HOC canary map (review finding 2): (ticker, yr) -> CIK from
    TICKER EVIDENCE ALONE -- no overnight gap is looked at, unlike the
    research map, whose pairs were accepted and assigned by counting large
    |gaps| on earnings dates (the nights the test later scores). Priority:
      1. 'efts': the CIK is the dominant press-release entity ("NYSE: X")
         of an EDGAR full-text search window covering yr (>= 5 hits and
         >= 50% of the window's); a ticker-year with two dominant CIKs is
         dropped. Contemporaneous; run for top-150 tickers only, 2001 on.
      2. 'xbrl': exactly one CIK filed an iXBRL document whose file-name
         prefix is the ticker in yr. Contemporaneous; ~2019 on.
      3. 'current': the ticker's single current owner, if that CIK's filing
         history spans yr (first <= yr <= last).
    Still survivorship-conditioned (2 and 3 come from companies alive in the
    XBRL era or today) and still wrong for some reused tickers in old spans:
    it removes the fingerprint, not the hindsight in the candidates.
    ticker_years: ticker, yr. tickers_norm: ticker, sec, cur_cik, cur_n.
    xbrl: prefix, cik, date. efts: ticker, y0, y1, kind, cik, n.
    entities: cik, first, last ('YYYY-MM-DD')."""
    ty = ticker_years[['ticker', 'yr']].drop_duplicates()
    e = efts[efts.cik.notna() & (efts.kind == 'pr')].copy()
    e['cik'] = e.cik.astype(int)
    tot = e.groupby(['ticker', 'y0', 'y1']).n.transform('sum')
    dom = e[(e.n >= 5) & (e.n / tot >= 0.5)]
    if len(dom):
        de = pd.concat([pd.DataFrame({'ticker': r.ticker, 'cik': r.cik,
                                      'yr': np.arange(r.y0, r.y1 + 1)})
                        for r in dom.itertuples()], ignore_index=True)
        de = de.drop_duplicates()
        de = de[~de.duplicated(['ticker', 'yr'], keep=False)]
    else:
        de = pd.DataFrame(columns=['ticker', 'cik', 'yr'])
    x = xbrl.assign(yr=xbrl.date.astype(str).str[:4].astype(int)).merge(
        tickers_norm[['ticker', 'sec']], left_on='prefix', right_on='sec')
    xy = x.groupby(['ticker', 'yr']).cik.nunique()
    one = xy[xy == 1].index
    xb = x.set_index(['ticker', 'yr']).loc[one].reset_index()[
        ['ticker', 'yr', 'cik']].drop_duplicates()
    cur = tickers_norm[tickers_norm.cur_n == 1][['ticker', 'cur_cik']] \
        .rename(columns={'cur_cik': 'cik'}).astype({'cik': int}) \
        .merge(entities[['cik', 'first', 'last']], on='cik', how='inner')
    m = ty.merge(de.rename(columns={'cik': 'c1'}), on=['ticker', 'yr'],
                 how='left') \
        .merge(xb.rename(columns={'cik': 'c2'}), on=['ticker', 'yr'],
               how='left') \
        .merge(cur.rename(columns={'cik': 'c3'}), on='ticker', how='left')
    def year_of(s, missing):
        y = pd.to_numeric(pd.Series(s, dtype=object).str[:4], errors='coerce')
        return y.fillna(missing).values
    span = (year_of(m['first'], 9999) <= m.yr.values) & \
        (year_of(m['last'], 0) >= m.yr.values)
    m['c3'] = m.c3.where(span)
    m['cik'] = m.c1.fillna(m.c2).fillna(m.c3)
    m['src'] = np.select([m.c1.notna(), m.c2.notna(), m.c3.notna()],
                         ['efts', 'xbrl', 'current'], '')
    m = m[m.cik.notna()]
    return m[['ticker', 'yr', 'cik', 'src']].astype({'cik': int}) \
        .reset_index(drop=True)


EVIDENCE_MAP = os.path.join(EDGAR, 'map_evidence.parquet')


def build_evidence_map(nm, path=EVIDENCE_MAP):
    """evidence_map() over the wide panel's non-ETF ticker-years from the
    research's on-disk ticker evidence (no requests), cached at `path`."""
    from xsec_backtest import ETF
    R = os.path.join(EDGAR, 'research')
    ty = nm[~nm.ticker.isin(ETF)][['ticker', 'month']].drop_duplicates()
    ty = ty.assign(yr=ty.month.str[:4].astype(int))[['ticker', 'yr']]
    m = evidence_map(ty, pd.read_parquet(os.path.join(R, 'tickers_norm.parquet')),
                     pd.read_parquet(os.path.join(EDGAR, 'xbrl_prefix.parquet')),
                     pd.read_parquet(os.path.join(R, 'efts_top150.parquet')),
                     pd.read_parquet(os.path.join(EDGAR, 'entities.parquet'),
                                     columns=['cik', 'first', 'last']))
    m.to_parquet(path)
    return m


def print_coverage(nm, amap, tiers):
    t, n_etf = coverage_table(nm, amap, tiers)
    print(f'\nCIK coverage of the wide panel (research map, keyed (ticker, '
          f'year)); {n_etf:,} ETF name-months set apart')
    print(f'  {"tier":<11}{"era":<11}{"name-months":>12}{"mapped":>9}'
          f'{"$-wtd":>8}{"names":>8}{"mapped":>8}')
    for r in t.itertuples():
        print(f'  {r.tier:<11}{r.era:<11}{r.name_months:>12,}{r.mapped:>9.1%}'
              f'{r.mapped_dollar:>8.1%}{r.names:>8,}{r.mapped_names:>8,}')
    s = coverage_by_survival(nm, amap)
    print('\n  by LATER SURVIVAL (ticker still in the panel 24 months on?) -- '
          'the map is built partly from today\'s owners and 2009+ XBRL, so '
          'names that died are under-mapped:')
    print(f'  {"era":<11}{"later":<13}{"name-months":>12}{"mapped":>9}'
          f'{"$-wtd":>8}')
    for r in s.itertuples():
        print(f'  {r.era:<11}{r.later:<13}{r.name_months:>12,}'
              f'{r.mapped:>9.1%}{r.mapped_dollar:>8.1%}')
    return t


def print_fetch_status(ev, idx_cache, doc_cache):
    print('\nfetch status by tier x class (index = exact-time/role pages '
          'needed; doc = equity test):')
    print(f'  {"tier":<11}{"class":<9}{"filings":>9}{"need idx":>10}'
          f'{"have":>8}{"docs":>8}{"have":>8}')
    for (t, c), g in ev.groupby(['tier', 'event']):
        g = g.drop_duplicates('accessionNumber')
        ni = g[g.need_index]
        hi = sum(a in idx_cache for a in ni.accessionNumber)
        nd = len(g) if c in SUPPLY_CLASSES else 0
        hd = sum(a in doc_cache for a in g.accessionNumber) if nd else 0
        print(f'  {t:<11}{c:<9}{len(g):>9,}{len(ni):>10,}{hi:>8,}{nd:>8,}'
              f'{hd:>8,}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fetch', 'build', 'coverage', 'all',
                                    'evidence-map'])
    ap.add_argument('--minutes', type=float, default=240,
                    help='wall-clock budget for page fetching (resumable)')
    ap.add_argument('--rps', type=float, default=MAX_RPS)
    ap.add_argument('--source', choices=['zip', 'api'], default='zip')
    ap.add_argument('--tiers', default='CORE,never-core')
    ap.add_argument('--refresh-filings', action='store_true')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    t0 = time.time()
    amap = load_map()
    nm = wide_name_months()
    tiers = tier_of_tickers(nm)
    if a.cmd == 'evidence-map':
        m = build_evidence_map(nm)
        print(f'evidence map: {len(m):,} (ticker, year) rows, '
              f'{m.cik.nunique():,} CIKs -> {EVIDENCE_MAP}')
        print(m.src.value_counts().to_string())
        return
    if a.cmd == 'coverage':
        print_coverage(nm, amap, tiers)
        return
    fetcher = Fetcher(a.rps)
    cal = panel_calendar()
    print(f'map: {len(amap):,} (ticker, year) rows, {amap.ticker.nunique():,} '
          f'tickers, {amap.cik.nunique():,} CIKs; calendar {cal.days[0]}..'
          f'{cal.days[-1]}, {len(cal.days):,} days, {int(cal.half.sum())} '
          f'half days')
    f = load_filings(amap, a.source, fetcher, a.refresh_filings)
    idx_cache = JsonlCache(os.path.join(CACHE, 'index.jsonl'))
    doc_cache = JsonlCache(os.path.join(CACHE, 'docs.jsonl'))
    feed_cache = JsonlCache(os.path.join(CACHE, 'feeds.jsonl'))
    if a.cmd in ('fetch', 'all'):
        t1 = time.time()
        n0 = fetcher.n
        fetch_insider(fetcher, insider_quarters())
        print(f'insider data sets: {fetcher.n - n0} requests, '
              f'{time.time() - t1:.0f} s')
    insider = load_insider()
    ev = candidates(f, amap, cal, tiers, insider)
    if a.cmd in ('fetch', 'all'):
        deadline = time.time() + a.minutes * 60
        t1, n0 = time.time(), fetcher.n
        # two passes: a feed that misses a filing leaves it to pass 2's
        # index page
        for rnd in (1, 2):
            tasks, counts = plan(ev, f, idx_cache, doc_cache, feed_cache,
                                 tuple(a.tiers.split(',')))
            print(f'\npass {rnd}: {len(tasks):,} tasks planned {counts} '
                  f'({(deadline - time.time()) / 60:.0f} min left, <= '
                  f'{a.rps:.0f} req/s)', flush=True)
            if not tasks:
                break
            k = run_tasks(tasks, fetcher, deadline, label=f'pass {rnd}')
            print(f'pass {rnd}: {k:,} of {len(tasks):,} tasks done', flush=True)
        el = time.time() - t1
        print(f'fetch: {fetcher.n - n0:,} requests in {el / 60:.1f} min '
              f'({(fetcher.n - n0) / max(el, 1):.2f} req/s), '
              f'{fetcher.errors} retried errors')
    print_fetch_status(ev, idx_cache, doc_cache)
    if a.cmd in ('build', 'all', 'fetch'):
        out = build(ev, amap, idx_cache, doc_cache)
        print(f'\nevents.parquet: {len(out):,} rows ({out.accession.nunique():,}'
              f' filings, {out.cik.nunique():,} CIKs)')
        print(out.groupby(['event', 'ts_kind']).size().unstack(fill_value=0)
              .to_string())
        s = out.drop_duplicates('accession').json_minus_exact_h.dropna()
        if len(s):
            bad = ((s < -1e-9) | (s > 10 + 1e-9)).sum()
            print(f'\nLEAK TRAP 1 check on {len(s):,} exact filings: JSON '
                  f'wall clock minus true ET (h):')
            print('  ' + s.round(2).value_counts().head(8).to_string()
                  .replace('\n', '\n  '))
            print(f'  outside [0, 10] h: {bad}  (must be 0 for the upper '
                  f'bound to be leak-free)')
    print_coverage(nm, amap, tiers)
    print(f'\ntotal {time.time() - t0:.0f} s, {fetcher.n:,} requests')


if __name__ == '__main__':
    main()
