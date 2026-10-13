# Part of qqq-microstructure.
#
# monthly.py keeps the forward test's binding metric: xsec_replay.py's
# printed verdict, verbatim and as a per-month table, under reports/. OUT
# is in xsec_replay's exact print format (a missing IC prints '+nan').

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))
import monthly

OUT = '''frozen 2026-08-20, trained through 2026-03; checking for new months...

6 month(s) after the training cutoff 2026-03

on: frozen model, 120 days (walk-forward reference +12.2 bps/day)
  2026-04   +10.12 bps/day   +2.02%   IC +0.081
  2026-05    -3.40 bps/day   -0.71%   IC +nan
  since cutoff: +3.36 bps/day, total +1.31%

on15: frozen model, 120 days (walk-forward reference +7.3 bps/day)
  2026-04    +5.00 bps/day   +1.00%   IC +0.050
  since cutoff: +5.00 bps/day, total +1.00%
'''


def test_the_replay_verdict_is_kept_verbatim_and_as_a_table(tmp_path):
    txt, csv = tmp_path / 'r.txt', tmp_path / 'r.csv'
    rows = monthly.keep_replay(OUT, '2026-11-02', str(txt), str(csv))
    assert txt.read_text().endswith(OUT)
    lines = csv.read_text().splitlines()
    assert lines[0] == 'run_date,tag,month,bps_per_day,total_pct,ic'
    assert lines[1:] == [
        '2026-11-02,on,2026-04,+10.12,+2.02,+0.081',
        '2026-11-02,on,2026-05,-3.40,-0.71,',
        '2026-11-02,on,since_cutoff,+3.36,+1.31,',
        '2026-11-02,on15,2026-04,+5.00,+1.00,+0.050',
        '2026-11-02,on15,since_cutoff,+5.00,+1.00,']
    assert len(rows) == 5


def test_a_run_with_nothing_to_score_keeps_the_last_table(tmp_path):
    txt, csv = tmp_path / 'r.txt', tmp_path / 'r.csv'
    monthly.keep_replay(OUT, '2026-11-02', str(txt), str(csv))
    before = csv.read_text()
    monthly.keep_replay('nothing to replay yet\n', '2026-11-09', str(txt),
                        str(csv))
    assert csv.read_text() == before
    assert 'nothing to replay yet' in txt.read_text()
