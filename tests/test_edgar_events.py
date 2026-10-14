# Part of qqq-microstructure.
#
# edgar_events.py on synthetic filings, pages and calendars -- never the
# network (an autouse fixture makes any socket connection fail the test).
# What is pinned: the strict decision-time rule and its 10-minute margin,
# DST, half days, the JSON 'Z' trap (wall clock read as ET, an upper bound
# 0-10 h late), date-only filings, when an index page is needed, the
# (ticker, year) CIK map, the event classes, the page parsers, the fetcher's
# rate limit / retries, the resumable caches and the insider-purchase filter.

import os, sys, io, json, socket, zipfile
import numpy as np, pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import edgar_events as ee

ET = ee.ET


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('a test tried to use the network')
    monkeypatch.setattr(socket.socket, 'connect', boom)
    monkeypatch.setattr(ee.urllib.request, 'urlopen', boom)


def ns(s):
    return pd.Timestamp(s, tz=ET).value


def weekdays(a, b):
    return [d.strftime('%Y%m%d') for d in pd.bdate_range(a, b)]


# ---------------------------------------------------------------- the rule

def test_strict_cutoff_with_the_declared_margin():
    cal = ee.Calendar(['20240103', '20240104', '20240105'])
    fe = lambda t: int(cal.first_eligible([ns(t)])[0])
    # declared: usable for night D only if A < 15:30 - 10 min, strictly
    assert fe('2024-01-03 15:19:59') == 0
    assert fe('2024-01-03 15:20:00') == 1          # strict: not D
    assert fe('2024-01-03 15:29:00') == 1          # inside the margin
    assert fe('2024-01-03 15:31:00') == 1
    assert fe('2024-01-03 23:00:00') == 1
    assert fe('2024-01-05 15:21:00') == 3          # after the last cutoff


def test_strict_cutoff_at_the_bare_decision_time(monkeypatch):
    monkeypatch.setattr(ee, 'LAG_MIN', 0)
    cal = ee.Calendar(['20240103', '20240104'])
    fe = lambda t: int(cal.first_eligible([ns(t)])[0])
    assert fe('2024-01-03 15:29:00') == 0          # 15:29 flags night D
    assert fe('2024-01-03 15:30:00') == 1          # strictly before 15:30
    assert fe('2024-01-03 15:31:00') == 1          # 15:31 does not


def test_cutoff_follows_dst():
    cal = ee.Calendar(['20240103', '20240701', '20240311'])
    utc = pd.DatetimeIndex(cal.cut).tz_localize('UTC')
    by = dict(zip(cal.days, utc))
    assert by['20240103'].strftime('%H:%M') == '20:20'    # EST, UTC-5
    assert by['20240701'].strftime('%H:%M') == '19:20'    # EDT, UTC-4
    assert by['20240311'].strftime('%H:%M') == '19:20'    # day after switch
    # the same ET wall clock is eligible on both sides of the switch
    for d in ('2024-01-03', '2024-07-01', '2024-03-11'):
        p = cal.pos[d.replace('-', '')]
        assert cal.first_eligible([ns(d + ' 15:19')])[0] == p
        assert cal.first_eligible([ns(d + ' 15:21')])[0] == p + 1


def test_half_days_move_the_cutoff_to_1220():
    half = ee.rule_half_days([2024])
    assert {'20241129', '20241224', '20240703'} <= half
    assert '20241127' not in half
    assert '20031226' in ee.rule_half_days([2003])
    assert '20020705' in ee.rule_half_days([2002])     # Jul 4 on a Thursday
    cal = ee.Calendar(['20241127', '20241129', '20241202'], half)
    p = cal.pos['20241129']
    assert cal.first_eligible([ns('2024-11-29 12:19')])[0] == p
    assert cal.first_eligible([ns('2024-11-29 12:25')])[0] == p + 1
    assert cal.first_eligible([ns('2024-11-29 14:00')])[0] == p + 1   # post-close


def test_segments_overnight_and_session():
    cal = ee.Calendar(weekdays('2024-01-01', '2024-01-12'))
    p = cal.pos['20240109']                                # a Tuesday
    seg = lambda t: int(cal.segment([ns(t)])[0])
    assert seg('2024-01-09 16:05') == 2 * p + 1            # night of the 9th
    assert seg('2024-01-09 16:00') == 2 * p + 1            # at the close
    assert seg('2024-01-09 07:00') == 2 * (p - 1) + 1      # night before
    assert seg('2024-01-09 11:00') == 2 * p                # in session
    fri = cal.pos['20240105']
    assert seg('2024-01-06 10:00') == 2 * fri + 1          # Saturday -> Fri


# ------------------------------------------------- bounds from the JSON alone

def test_json_z_is_read_as_et_wall_clock_an_upper_bound():
    # Apple 8-K 0000320193-26-000018: JSON ...T00:30:28Z, true 16:30:28 ET
    U, L, kind = ee.bounds(['8-K'], ['2026-07-30'],
                           ['2026-07-31T00:30:28.000Z'])
    assert kind[0] == 'upper'
    assert U[0] == pd.Timestamp('2026-07-31 00:30:28', tz=ET)  # NOT 20:30
    assert L[0] == U[0] - pd.Timedelta(hours=10)
    true = pd.Timestamp('2026-07-30 16:30:28', tz=ET)
    assert L[0] <= true <= U[0]
    # all three observed classes stay inside [L, U]
    for shift in (0, 4, 8):
        U, L, _ = ee.bounds(['8-K'], ['2026-07-30'],
                            [(true + pd.Timedelta(hours=shift))
                             .strftime('%Y-%m-%dT%H:%M:%S.000Z')])
        assert L[0] <= true <= U[0]


def test_date_only_filings():
    U, L, kind = ee.bounds(['8-K', '4', 'EFFECT', '8-K'],
                           ['2001-05-10'] * 3 + ['2003-05-10'],
                           ['2001-05-10T04:00:00.000Z'] * 3 +
                           ['2003-05-10T04:00:00.000Z'])
    assert list(kind) == ['date_only'] * 3 + ['upper']    # placeholders end
    assert U[0] == pd.Timestamp('2001-05-10 17:30', tz=ET)
    assert U[1] == pd.Timestamp('2001-05-10 22:00', tz=ET)  # Forms 3/4/5/144
    assert U[2] == pd.Timestamp('2001-05-11 17:30', tz=ET)  # EFFECT: a day on
    cal = ee.Calendar(['20010510', '20010511'])
    assert cal.first_eligible([U[0].value])[0] == 1          # next day


def test_needs_index_only_when_the_bound_straddles():
    cal = ee.Calendar(weekdays('2024-01-01', '2024-01-12'))
    U, L, kind = ee.bounds(['8-K'] * 3, ['2024-01-09'] * 3,
                           ['2024-01-09T20:05:00.000Z',   # 10:05..20:05 straddles
                            '2024-01-09T11:00:00.000Z',   # 01:00..11:00 does not
                            '2024-01-09T11:00:00.000Z'])
    need = ee.needs_index(['bad', 'bad', 'sc13d'], U.asi8, L.asi8, kind, cal)
    assert list(need) == [True, False, True]               # 13D: role
    # earnings sources also need the session edge resolved: 08:00..18:00
    U, L, kind = ee.bounds(['8-K'], ['2024-01-09'],
                           ['2024-01-09T18:00:00.000Z'])
    assert ee.needs_index(['earn'], U.asi8, L.asi8, kind, cal)[0]
    U, L, kind = ee.bounds(['8-K'], ['2001-01-09'],
                           ['2001-01-09T05:00:00.000Z'])
    cal2 = ee.Calendar(weekdays('2001-01-01', '2001-01-12'))
    assert not ee.needs_index(['bad'], U.asi8, L.asi8, kind, cal2)[0]


# ------------------------------------------------------------ the CIK map

def test_map_is_keyed_by_ticker_and_year(tmp_path):
    p = tmp_path / 'm.parquet'
    pd.DataFrame({'ticker': ['T', 'T', 'T', 'GOOG', 'GOOGL', 'T'],
                  'yr': [2004, 2005, 2006, 2015, 2015, 2004],
                  'cik': [5907, 5907, 732717, 1652044, 1652044, 999]}) \
        .to_parquet(p)
    m = ee.load_map(str(p))
    assert len(m) == 5                                    # one CIK per key
    assert m[(m.ticker == 'T') & (m.yr == 2004)].cik.item() == 5907
    yrs = ee.cik_years(m)
    assert yrs[5907] == {2003, 2004, 2005, 2006}
    ev = pd.DataFrame({'cik': [5907, 732717, 5907, 1652044],
                       'accepted_et': pd.to_datetime(
                           ['2005-06-01 16:00', '2006-06-01 16:00',
                            '2006-06-01 16:00', '2015-03-01 16:00'])
                       .tz_localize(ET)})
    out = ee.explode_tickers(ev, m)
    got = out.groupby('cik').ticker.apply(lambda s: sorted(s)).to_dict()
    assert got[732717] == ['T']                 # AT&T Inc holds T in 2006
    assert sorted(out[out.cik == 5907].ticker) == ['', 'T']   # 2006: unmapped
    assert got[1652044] == ['GOOG', 'GOOGL']    # dual class, one CIK


# ------------------------------------------------------------- the classes

def _f(rows):
    return pd.DataFrame(rows, columns=['cik', 'accessionNumber', 'filingDate',
                                       'acceptanceDateTime', 'form', 'items',
                                       'primaryDocument'])


def test_event_classes():
    j = '2010-01-01T16:00:00.000Z'
    f = _f([
        (1, 'a1', '2010-05-01', j, '8-K', '2.02,9.01', ''),     # earn
        (1, 'a2', '2003-06-01', j, '8-K', '12', ''),            # old earn
        (1, 'a3', '2002-06-01', j, '8-K', '12', ''),            # too early
        (1, 'a4', '2010-05-01', j, '8-K/A', '2.02', ''),        # amendment
        (1, 'a5', '2010-05-01', j, '8-K', '2.06,7.01', ''),     # bad
        (1, 'a6', '2003-01-01', j, '8-K', '3', ''),             # bad_old
        (1, 'a7', '2011-05-01', j, '8-K', '5.07', ''),          # vote
        (1, 'a8', '2011-05-01', j, '8-K', '5.07,9.01', ''),     # vote
        (1, 'a9', '2011-05-01', j, '8-K', '5.07,8.01', ''),     # not vote
        (1, 'b1', '2009-05-01', j, '8-K', '5.07', ''),          # before 5.07
        (1, 'b2', '2010-05-01', j, 'NT 10-K', '', ''),
        (1, 'b3', '2010-05-01', j, 'SC 13D', '', ''),
        (1, 'b4', '2025-05-01', j, 'SCHEDULE 13D', '', ''),
        (1, 'b5', '2010-05-01', j, 'SC 13D/A', '', ''),
        (1, 'b6', '2010-05-01', j, '424B5', '', 'd.htm'),
        (1, 'b7', '2010-05-01', j, 'EFFECT', '', ''),
    ] + [(2, f'n{i}', '2012-01-05', j, '424B2', '', '') for i in range(12)]
      + [(2, 'n99', '2012-02-01', j, '424B5', '', ''),        # 13th 424B
         (2, 'n98', '2013-02-01', j, 'S-3', '', '')])          # other year
    c = ee.classify(f)
    acc = f.accessionNumber.values
    got = {}
    for r in c.itertuples():
        got.setdefault(r.event, set()).add(acc[r.row])
    assert got['earn'] == {'a1', 'a2'}
    assert got['bad'] == {'a5'}
    assert got['bad_old'] == {'a6'}
    assert got['vote'] == {'a7', 'a8'}
    assert got['nt'] == {'b2'}
    assert got['sc13d'] == {'b3', 'b4'}
    assert got['effect'] == {'b7'}
    assert got['supply'] == {'b6', 'n98'}        # 2012 is a note shelf


# ------------------------------------------------------------- the parsers

INDEX_13D = '''<div class="infoHead">Filing Date</div>
 <div class="info">2024-06-14</div>
 <div class="infoHead">Accepted</div>
 <div class="info">2024-06-14 15:44:56</div>
 <span class="companyName">NEW YORK COMMUNITY BANCORP, INC. (Subject)
 <acronym title="Central Index Key">CIK</acronym>: <a href="/cgi-bin/browse-edgar?CIK=0000910073&amp;action=getcompany">0000910073 (see all company filings)</a></span>
 <span class="companyName">RCP GenPar HoldCo LLC (Filed by)
 <acronym title="Central Index Key">CIK</acronym>: <a href="/cgi-bin/browse-edgar?CIK=0001731503&amp;action=getcompany">0001731503 (see all company filings)</a></span>'''

INDEX_4 = '''<div class="infoHead">Accepted</div><div class="info">2026-03-25 16:35:08</div>
 <span class="companyName">VERISIGN INC/CA (<a href="/cgi-bin/own-disp?CIK=0001014473&amp;action=getissuer">Issuer</a>)
 <acronym title="Central Index Key">CIK</acronym>: <a href="/cgi-bin/browse-edgar?CIK=0001014473&amp;action=getcompany">0001014473 (see all company filings)</a></span>'''


def test_parse_index_roles_and_time():
    p = ee.parse_index(INDEX_13D.encode())
    assert p['accepted'] == '2024-06-14 15:44:56'
    assert p['filing_date'] == '2024-06-14'
    assert p['roles'] == [['Subject', 910073], ['Filed by', 1731503]]
    assert ee.parse_index(INDEX_4)['roles'] == [['Issuer', 1014473]]


def feed_xml(entries):
    body = ''.join(
        f'<entry><content type="text/xml"><accession-number>{a}'
        f'</accession-number><filing-date>{fd}</filing-date><filing-type>{ft}'
        f'</filing-type></content><updated>{u}</updated></entry>'
        for a, fd, ft, u in entries)
    return ('<?xml version="1.0"?><feed><updated>2026-10-14T06:55:42-04:00'
            f'</updated>{body}</feed>').encode('latin-1')


def test_parse_feed_keeps_the_true_offset():
    x = feed_xml([('0000320193-26-000018', '2026-07-30', '8-K',
                   '2026-07-30T16:30:28-04:00'),
                  ('0001689548-24-003763', '2024-01-12', '424B5',
                   '2024-01-12T17:20:31-05:00')])
    p = ee.parse_feed(x)
    assert [e['accepted'] for e in p] == ['2026-07-30 16:30:28',
                                          '2024-01-12 17:20:31']
    assert p[0]['acc'] == '0000320193-26-000018' and p[1]['form'] == '424B5'


def test_classify_doc():
    eq = '<html><b>PROSPECTUS SUPPLEMENT</b> 5,000,000 Shares of Common ' \
         'Stock</html>'
    assert ee.classify_doc(eq)['equity']
    assert ee.classify_doc('10,000,000 American Depositary Shares')['equity']
    debt = '$500,000,000 4.250% Senior Notes due 2030 ... ' \
           'convertible into shares of our common stock'
    d = ee.classify_doc(debt)
    assert d['eq_match'] and d['debt_cover'] and not d['equity']
    late = 'x ' * 2000 + '5.0% notes due 2030 ' + 'shares of common stock'
    assert ee.classify_doc(late)['equity']           # debt not on the cover
    assert not ee.classify_doc('units of limited partnership')['equity']


# ------------------------------------------------- fetcher, caches, paging

def test_fetcher_rate_limit_retries_and_404():
    calls, slept = [], []

    def opener(url, max_bytes):
        calls.append(url)
        if url.endswith('flaky') and len(calls) < 3:
            return 503, b''
        if url.endswith('gone'):
            return 404, b''
        return 200, b'ok'

    f = ee.Fetcher(rps=100, opener=opener, sleep=slept.append)
    assert f.gap == 1 / ee.MAX_RPS                      # capped at 8 req/s
    assert f.get('https://x/flaky') == b'ok'
    assert f.errors == 2 and len(calls) == 3
    assert f.get('https://x/gone') is None
    bad = ee.Fetcher(opener=lambda u, m: (500, b''), tries=3,
                     sleep=lambda s: None)
    with pytest.raises(IOError):
        bad.get('https://x/dead')


def test_fetcher_spaces_requests():
    import time as _t
    stamps = []
    f = ee.Fetcher(rps=8, opener=lambda u, m: (stamps.append(_t.monotonic())
                                               or (200, b'')))
    for _ in range(5):
        f.get('https://x/')
    gaps = np.diff(stamps)
    assert (gaps >= 0.125 - 0.01).all()


def test_jsonl_cache_resumes_and_skips_a_torn_line(tmp_path):
    p = str(tmp_path / 'c' / 'x.jsonl')
    c = ee.JsonlCache(p)
    c.put({'acc': 'a', 'v': 1})
    c.put_many([{'acc': 'b', 'v': 2}, {'acc': 'c', 'v': 3}])
    with open(p, 'a') as fh:
        fh.write('{"acc": "d", "v"')                     # killed mid-write
    c2 = ee.JsonlCache(p)
    assert set(c2.data) == {'a', 'b', 'c'} and 'd' not in c2


def test_fetch_feed_pages_until_older_than_lo(tmp_path):
    pages = {
        0: [(f'0000000001-24-{i:06d}', '2024-06-01', '8-K',
             '2024-06-01T16:00:00-04:00') for i in range(100)],
        100: [(f'0000000001-23-{i:06d}', '2023-06-01', '8-K',
               '2023-06-01T08:00:00-04:00') for i in range(100)],
        200: [('0000000001-22-000001', '2022-06-01', '8-K',
               '2022-06-01T16:00:00-04:00')],
    }
    seen = []

    class F:
        n = 0

        def get(self, url):
            seen.append(url)
            start = int(url.split('start=')[1].split('&')[0])
            return feed_xml(pages[start])

    idx = ee.JsonlCache(str(tmp_path / 'i.jsonl'))
    fc = ee.JsonlCache(str(tmp_path / 'f.jsonl'))
    n = ee.fetch_feed(F(), 1, '8-K', 'exclude', '2023-01-01', '2024-12-31',
                      idx, fc)
    # page 2 is full and still >= lo, so page 3 is read; it reaches 2022
    # (older than lo) and the walk stops there
    assert n == 3 and len(seen) == 3
    assert 'dateb=20241231' in seen[0] and 'type=8-K' in seen[0]
    assert 'start=200' in seen[2]
    assert len(idx.data) == 201
    assert idx.data['0000000001-23-000005']['accepted'] == '2023-06-01 08:00:00'
    assert '1|8-K|exclude|2023-01-01|2024-12-31' in fc


def test_resolve_prefers_exact_and_keeps_date_only(tmp_path):
    idx = ee.JsonlCache(str(tmp_path / 'i.jsonl'))
    doc = ee.JsonlCache(str(tmp_path / 'd.jsonl'))
    idx.put({'acc': 'x', 'status': 200, 'accepted': '2026-07-30 16:30:28',
             'roles': [['Subject', 7]]})
    idx.put({'acc': 'old', 'status': 200, 'accepted': '2001-05-10 00:00:00',
             'roles': []})
    doc.put({'acc': 's', 'status': 200, 'equity': True})
    ev = pd.DataFrame({
        'accessionNumber': ['x', 'old', 's', 'u'],
        'filingDate': ['2026-07-30', '2001-05-10', '2010-01-01', '2010-01-01'],
        'form': ['SC 13D', '8-K', '424B5', '8-K'],
        'acceptanceDateTime': ['2026-07-31T00:30:28.000Z',
                               '2001-05-10T04:00:00.000Z',
                               '2010-01-01T21:00:00.000Z',
                               '2010-01-01T21:00:00.000Z'],
        'event': ['sc13d', 'bad_old', 'supply', 'bad'], 'cik': [7, 1, 2, 3]})
    U, L, kind = ee.bounds(ev.form, ev.filingDate, ev.acceptanceDateTime)
    ev['U'], ev['L'], ev['kind'] = U, L, kind
    r = ee.resolve(ev, idx, doc)
    assert list(r.ts_kind) == ['exact', 'date_only', 'upper', 'upper']
    assert r.accepted_et[0] == pd.Timestamp('2026-07-30 16:30:28', tz=ET)
    assert r.json_minus_exact_h[0] == pytest.approx(8.0)
    assert r.accepted_et[1] == pd.Timestamp('2001-05-10 17:30', tz=ET)
    assert r.role[0] == 'subject' and r.equity[2] == 1.0
    assert r.accepted_et[3] == pd.Timestamp('2010-01-01 21:00', tz=ET)


def test_insider_purchase_filter(tmp_path):
    zp = str(tmp_path / 'q.zip')
    sub = ('ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERCIK\tREMARKS\n'
           'A1\t01-JAN-2020\t4\t0000000011\t"a ""quoted"" remark"\n'
           'A2\t01-JAN-2020\t4\t0000000012\t\n'
           'A3\t01-JAN-2020\t4/A\t0000000013\t\n'
           'A4\t01-JAN-2020\t4\t0000000014\t\n')
    own = ('ACCESSION_NUMBER\tRPTOWNER_RELATIONSHIP\n'
           'A1\tDirector,TenPercentOwner\nA2\tTenPercentOwner\n'
           'A3\tOfficer\nA4\tOfficer\n')
    tr = ('ACCESSION_NUMBER\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n'
          'A1\tP\t100\t10\nA1\tP\t50\t10\nA2\tP\t100\t10\nA3\tP\t1\t1\n'
          'A4\tS\t100\t10\n')
    with zipfile.ZipFile(zp, 'w') as z:
        z.writestr('SUBMISSION.tsv', sub)
        z.writestr('REPORTINGOWNER.tsv', own)
        z.writestr('NONDERIV_TRANS.tsv', tr)
    p = ee.purchases_from_zip(zp)
    assert list(p.accession) == ['A1']            # director P, an original 4
    assert p.issuer_cik.item() == 11 and p.p_value.item() == 1500


def test_coverage_table():
    from xsec_backtest import ETF
    nm = pd.DataFrame({'ticker': ['A', 'A', 'B', 'SPY'],
                       'month': ['2005-01', '2019-01', '2005-01', '2005-01'],
                       'dollar_vol': [3.0, 1.0, 1.0, 9.0]})
    amap = pd.DataFrame({'ticker': ['A'], 'yr': [2005], 'cik': [1]})
    tiers = pd.Series({'A': 'CORE', 'B': 'never-core'})
    t, n_etf = ee.coverage_table(nm, amap, tiers)
    assert 'SPY' in ETF and n_etf == 1
    allrow = t[(t.tier == 'ALL')].iloc[0]
    assert allrow.name_months == 3 and allrow.mapped == pytest.approx(1 / 3)
    assert allrow.mapped_dollar == pytest.approx(3 / 5)
    core05 = t[(t.tier == 'CORE') & (t.era == '2001-2009')].iloc[0]
    assert core05.mapped == 1.0


# ------------------------------------------- review fixes (post-hoc, 2026-10-14)

def test_shelf_count_known_at_acceptance_finding_3():
    """Finding 3: the declared H4 shelf rule counts a CIK's 424Bs over the
    WHOLE calendar year, so an early filing is dropped because of 424Bs
    filed later. The post-hoc class 'supply_x' keeps exactly the filings
    whose count known at acceptance (earlier filing dates + itself) is
    <= 12; the declared 'supply' class is unchanged."""
    j = '2012-01-01T16:00:00.000Z'
    rows = [(5, 's3', '2012-01-02', j, 'S-3', '', ''),           # before any
            (5, 'p1', '2012-01-03', j, '424B5', '', ''),         # 1st 424B
            (5, 'd1', '2012-01-04', j, '424B3', '', ''),         # same day:
            (5, 'd2', '2012-01-04', j, '424B3', '', '')]         # 2 + 1 each
    rows += [(5, f'n{i}', f'2012-03-{i + 1:02d}', j, '424B2', '', '')
             for i in range(10)]                                 # 3 + 10 = 13
    rows += [(5, 'late', '2012-06-01', j, '424B5', '', ''),      # 14th
             (6, 'q1', '2012-01-05', j, '424B5', '', '')]        # other CIK
    f = _f(rows)
    ytd = ee.n424_known_at_acceptance(f.cik, f.filingDate, f.form)
    got = dict(zip(f.accessionNumber, ytd))
    assert got['s3'] == 0 and got['p1'] == 1
    assert got['d1'] == got['d2'] == 2          # same-day 424Bs not counted
    assert got['n9'] == 13 and got['late'] == 14 and got['q1'] == 1
    c = ee.classify(f)
    acc = f.accessionNumber.values
    by = {}
    for r in c.itertuples():
        by.setdefault(r.event, set()).add(acc[r.row])
    assert by['supply'] == {'q1'}               # declared: CIK 5's year > 12
    assert by['supply_x'] == {'s3', 'p1', 'd1', 'd2'}
    assert 'late' not in by['supply_x']         # 14 known at acceptance
    assert not (by['supply'] & by['supply_x'])


def test_supply_x_gets_its_document_and_equity(tmp_path):
    idx = ee.JsonlCache(str(tmp_path / 'i.jsonl'))
    doc = ee.JsonlCache(str(tmp_path / 'd.jsonl'))
    feed = ee.JsonlCache(str(tmp_path / 'f.jsonl'))
    ev = pd.DataFrame({
        'cik': [5, 5], 'accessionNumber': ['x1', 'x2'],
        'filingDate': ['2012-01-03', '2012-01-04'],
        'form': ['424B5', '424B5'], 'event': ['supply_x', 'supply'],
        'kind': ['date_only', 'date_only'], 'need_index': [False, False],
        'tier': ['CORE', 'CORE'], 'primaryDocument': ['a.htm', 'b.htm']})
    f = ev[['cik', 'form', 'filingDate']]
    tasks, counts = ee.plan(ev, f, idx, doc, feed)
    docs = sorted(args[1] for fn, args in tasks if fn is ee.fetch_doc)
    assert docs == ['x1', 'x2']
    doc.put({'acc': 'x1', 'status': 200, 'equity': True})
    ev['acceptanceDateTime'] = '2012-01-03T00:00:00.000Z'
    U, L, kind = ee.bounds(ev.form, ev.filingDate, ev.acceptanceDateTime)
    ev['U'], ev['L'] = U, L
    r = ee.resolve(ev, idx, doc)
    assert r.equity.tolist()[0] == 1.0 and np.isnan(r.equity.tolist()[1])


def test_coverage_by_later_survival_finding_2():
    """Finding 2: coverage is reported split by whether the ticker is still
    in the panel 24 months later -- the map under-covers names that died."""
    months = [f'{y}-{m:02d}' for y in (2001, 2002, 2003) for m in range(1, 13)]
    nm = pd.DataFrame(
        [('LIVE', m, 2.0) for m in months] +
        [('DEAD', m, 1.0) for m in months[:6]] +
        [('SPY', m, 9.0) for m in months],
        columns=['ticker', 'month', 'dollar_vol'])
    amap = pd.DataFrame({'ticker': ['LIVE', 'DEAD'], 'yr': [2001, 2001],
                         'cik': [1, 2]})
    amap = amap[amap.ticker == 'LIVE']           # the dead name is unmapped
    s = ee.coverage_by_survival(nm, amap, horizon=24)
    s = s.set_index('later')
    # only months with 24 months of panel after them are classed (2001-01..
    # 2001-12); ETFs are left out
    assert s.loc['still there'].name_months == 12
    assert s.loc['gone'].name_months == 6
    assert s.loc['still there'].mapped == 1.0 and s.loc['gone'].mapped == 0.0


def test_evidence_map_uses_ticker_evidence_only_finding_2():
    """Finding 2's canary map: EFTS dominance first, then a unique dated
    XBRL prefix, then the single current owner while its filing history
    spans the year -- no return or gap enters it."""
    ty = pd.DataFrame({'ticker': ['T'] * 5 + ['DUP', 'OLD'],
                       'yr': [2002, 2006, 2021, 1999, 2030, 2021, 1999]})
    tn = pd.DataFrame({'ticker': ['T', 'DUP', 'OLD'],
                       'sec': ['T', 'DUP', 'OLD'],
                       'cur_cik': [732717.0, np.nan, 77.0],
                       'cur_n': [1, 0, 1]})
    xbrl = pd.DataFrame({'prefix': ['T', 'DUP', 'DUP'],
                         'cik': [732717, 1, 2],
                         'date': ['2021-02-01', '2021-03-01', '2021-04-01']})
    efts = pd.DataFrame({'ticker': ['T', 'T'], 'y0': [2001, 2001],
                         'y1': [2004, 2004], 'kind': ['pr', 'pr'],
                         'cik': [5907.0, 732717.0], 'n': [9, 3]})
    ent = pd.DataFrame({'cik': [732717, 77], 'first': ['1983-01-01',
                                                        '2005-01-01'],
                        'last': ['2026-09-30', '2026-09-30']})
    m = ee.evidence_map(ty, tn, xbrl, efts, ent)
    got = {(r.ticker, r.yr): (r.cik, r.src) for r in m.itertuples()}
    assert got[('T', 2002)] == (5907, 'efts')        # AT&T Corp, dominant
    assert got[('T', 2006)] == (732717, 'current')
    assert got[('T', 2021)] == (732717, 'xbrl')       # dated file names
    assert got[('T', 1999)] == (732717, 'current')    # history spans 1999
    assert ('T', 2030) not in got                     # after its last filing
    assert ('DUP', 2021) not in got                   # two XBRL CIKs: no pick
    assert ('OLD', 1999) not in got                   # owner not yet filing
    assert set(m.columns) == {'ticker', 'yr', 'cik', 'src'}
