# Part of qqq-microstructure.
#
# xsec_wide_model.py: the declared verdict rule, and the tiering it shares
# with xsec_wide.py (no never-core name is ever core).

import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import xsec_wide_model as wm


def r(mean, t):
    return dict(mean=mean, t=t)


def test_the_verdict_is_the_declared_rule():
    assert wm.verdict({'MID': r(3, 2.5), 'DEEP': r(1, 2.1)}).startswith(
        'GENERALISES')
    assert wm.verdict({'MID': r(3, 2.5), 'DEEP': r(1, 1.2)}).startswith(
        'PARTIAL')
    assert wm.verdict({'MID': r(3, 2.5), 'DEEP': r(-1, -3)}).startswith(
        'DOES NOT')
    assert wm.verdict({'MID': r(3, 2.5), 'DEEP': None}).startswith('too thin')


def test_never_core_names_are_never_core():
    rows = []
    for m in ('2020-01', '2020-02'):
        for i in range(400):
            # name i's dollar volume falls with i; X0 swaps rank in month 2
            dv = 1e9 / (i + 1) if not (i == 0 and m == '2020-02') else 1.0
            rows.append(dict(month=m, ticker=f'X{i}', dollar_vol=dv))
    df = pd.DataFrame(rows)
    tr = wm.tiers(df)
    assert 'X0' in tr['CORE']                  # core once is core forever
    assert not (tr['MID'] | tr['DEEP']) & tr['CORE']
    # X0 falling to the bottom lifts X150 into the top 150 in month 2
    assert 'X150' in tr['CORE'] and len(tr['CORE']) == 151
    assert 'X200' in tr['MID'] and 'X151' in tr['MID']
