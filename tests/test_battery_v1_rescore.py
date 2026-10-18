# Part of qqq-microstructure.
#
# The decider battery's v1 result is a committed record (RESULTS 27): the
# live run's calls, scored and selected at the declaration's commit. Any
# later change to the scorer must still turn those calls into the same
# summary and the same selection -- otherwise the selection could drift after
# the fact without anyone re-running anything.
#
# "The same" is exact for everything but floats: every key, count, gate,
# ranking, flag and string. Floats must agree to 1e-9 relative. The record was
# scored under numpy 2.5; numpy 2.4 (the newest for Python 3.11, which CI
# runs) moves 121 OLS and correlation floats in their last digits (at most
# 6.6e-14 relative), which a byte comparison failed on -- and that failure
# kept the daily ritual from running from 2026-10-13 (RESULTS 19). A real
# scorer change moves a number by far more than 1e-9, or a gate, or the
# selection. The selection's summary_sha256 must be the hash of the summary
# it was selected from.

import os, sys, shutil, json, hashlib, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
import battery_score as bs

RUN = os.path.join(ROOT, 'bench', 'battery', 'runs', 'v1')
REL = 1e-9


def same(a, b, path='', skip=()):
    """Paths where a and b differ: exact for non-floats, REL for floats."""
    if path in skip:
        return []
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return []
        return [] if abs(a - b) <= REL * max(abs(a), abs(b)) else [path]
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        if list(a) != list(b):
            return [path]
        return [p for k in a for p in same(a[k], b[k], f'{path}/{k}', skip)]
    if isinstance(a, list):
        if len(a) != len(b):
            return [path]
        return [p for i, (x, y) in enumerate(zip(a, b))
                for p in same(x, y, f'{path}[{i}]', skip)]
    return [] if a == b else [path]


def test_the_committed_v1_run_rescores_to_the_same_record(tmp_path):
    d = tmp_path / 'v1'
    shutil.copytree(RUN, d)
    want = {f: json.loads((d / f).read_bytes()) for f in ('summary.json',
                                                          'selection_v1.json')}
    for f in want:
        (d / f).unlink()
    bs.score(str(d), plan_path=os.path.join(ROOT, 'bench', 'battery',
                                            'plan_v1.json'))
    bs.select(str(d))
    got = {f: json.loads((d / f).read_bytes()) for f in want}
    assert same(want['summary.json'], got['summary.json']) == []
    assert same(want['selection_v1.json'], got['selection_v1.json'],
                skip=('/summary_sha256',)) == []
    assert got['selection_v1.json']['summary_sha256'] == \
        hashlib.sha256((d / 'summary.json').read_bytes()).hexdigest()


def test_the_comparison_catches_what_a_drift_would_change():
    rec = {'gates': {'G1': True}, 'q': 0.905, 'rank': ['a', 'b'], 'n': 3}
    assert same(rec, json.loads(json.dumps(rec))) == []
    assert same(rec, {**rec, 'q': 0.905 * (1 + 1e-13)}) == []
    assert same(rec, {**rec, 'q': 0.905 * (1 + 1e-8)}) == ['/q']
    assert same(rec, {**rec, 'gates': {'G1': False}}) == ['/gates/G1']
    assert same(rec, {**rec, 'rank': ['b', 'a']}) == ['/rank[0]', '/rank[1]']
    assert same(rec, {**rec, 'n': 3.0}) == ['/n']
    assert same(rec, {'q': 0.905, 'gates': {'G1': True}, 'rank': ['a', 'b'],
                      'n': 3}) == ['']
