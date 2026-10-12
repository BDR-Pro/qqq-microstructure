# Part of qqq-microstructure.
#
# xsec_live.grade() writes the forward record RESULTS 19 is judged on. On
# 2026-08-27 and -28 it graded the basket on 9 of 10 names: it ran during the
# next morning's session, Yahoo's in-progress bar for GEV had an open and no
# close, the whole row was dropped, and the night was never revisited -- the
# 28th was logged at -24.3 instead of the basket's -52.0. These tests drive
# the grader with a fake downloader and a fixed clock.
#
# Run: python -m pytest tests -q

import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import xsec_live as xl

NAMES = ['A', 'B', 'C', 'D', 'E', 'GEV']
LOG = pd.DataFrame([dict(emitted_on='20260831', month='2026-09',
                         q5=';'.join(NAMES))])
EMPTY = pd.DataFrame(columns=xl.DCOLS)


def fake(prices):
    """prices: {ticker: {YYYYMMDD: (open, close)}} -> a yf.download stand-in
    returning the group_by='ticker' frame shape."""
    def download(tickers, **kw):
        frames = {}
        for t in tickers:
            p = prices.get(t, {})
            idx = pd.to_datetime(sorted(p))
            frames[t] = pd.DataFrame(
                {'Open': [p[d][0] for d in sorted(p)],
                 'Close': [p[d][1] for d in sorted(p)]}, index=idx)
        return pd.concat(frames, axis=1)
    return download


def market(gev_open_0902=101.0, gev_close_0902=np.nan):
    days = ['20260901', '20260902']
    px = {t: {'20260901': (100.0, 100.0), '20260902': (101.0, 101.0)}
          for t in NAMES + ['QQQ']}
    # 0902 is in progress at grading time: GEV has an open and no close yet
    px['GEV']['20260902'] = (gev_open_0902, gev_close_0902)
    return px


def test_an_open_without_a_close_still_grades_the_night():
    d = xl.grade(LOG, EMPTY.copy(), today='20260902', download=fake(market()))
    r = d[d.date == '20260901'].iloc[0]
    assert int(r.n) == 6, 'GEV dropped because its in-progress close was NaN'


def test_an_incomplete_night_is_regraded_when_the_name_arrives():
    first = xl.grade(LOG, EMPTY.copy(), today='20260902',
                     download=fake(market(gev_open_0902=np.nan)))
    assert int(first[first.date == '20260901'].n.iloc[0]) == 5
    second = xl.grade(LOG, first.copy(), today='20260903',
                      download=fake(market()))
    r = second[second.date == '20260901']
    assert len(r) == 1, 'regraded night was appended, not replaced'
    assert int(r.n.iloc[0]) == 6
    assert np.isclose(r.q5_on_bps.iloc[0], np.log(101 / 100) * 1e4)


def test_an_old_incomplete_night_is_final():
    first = xl.grade(LOG, EMPTY.copy(), today='20260902',
                     download=fake(market(gev_open_0902=np.nan)))
    later = xl.grade(LOG, first.copy(), today='20260915',
                     download=fake(market()))
    assert int(later[later.date == '20260901'].n.iloc[0]) == 5


def test_a_complete_night_is_never_rewritten():
    first = xl.grade(LOG, EMPTY.copy(), today='20260902',
                     download=fake(market()))
    again = xl.grade(LOG, first.copy(), today='20260903',
                     download=fake(market(gev_open_0902=99.0)))
    pd.testing.assert_frame_equal(first.reset_index(drop=True),
                                  again.reset_index(drop=True))


def test_a_forced_regrade_reaches_an_old_night_and_only_adds_names():
    first = xl.grade(LOG, EMPTY.copy(), today='20260902',
                     download=fake(market(gev_open_0902=np.nan)))
    # too old for the automatic retry, but named explicitly
    later = xl.grade(LOG, first.copy(), today='20260915',
                     download=fake(market()), force={'20260901'})
    assert int(later[later.date == '20260901'].n.iloc[0]) == 6
    # a forced regrade never thins a night: GEV gone again -> unchanged
    thin = xl.grade(LOG, later.copy(), today='20260916',
                    download=fake(market(gev_open_0902=np.nan)),
                    force={'20260901'})
    pd.testing.assert_frame_equal(later.reset_index(drop=True),
                                  thin.reset_index(drop=True))
