# Part of qqq-microstructure.
#
# Resolve a merge conflict in an APPEND-ONLY report log.
#
# reports/*.csv are written from two places that never see each other: the VM
# next to IB Gateway (xsec_live, orders, fills -- see roi.yml's crontab) and
# GitHub Actions (monthly.yml). Both commit and push. When they append on the
# same day, or when a schema change adds a column on one side, git reports a
# content conflict in a file where BOTH sides are right -- the rows are
# observations, not edits, and the resolution is always the union.
#
# Doing that by hand invites the one mistake that cannot be undone: dropping a
# graded night. This script does it mechanically:
#
#   - reads the two conflicted sides straight out of the index (:2 ours, :3
#     theirs), so no conflict markers are ever parsed;
#   - unions the rows on the log's key column (date / run_at_et / month);
#   - where both sides carry the same key, keeps the row with more filled
#     fields (a re-grade that added columns beats the older thinner row);
#   - takes the union of the columns, ours first, so a schema addition on
#     either side survives;
#   - sorts by the key and stages the result.
#
#   python src/merge_daily.py                 # every conflicted csv under reports/
#   python src/merge_daily.py reports/xsec_paper_daily.csv
#   python src/merge_daily.py --dry-run       # print the plan, touch nothing

import os, io, sys, argparse, subprocess
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ['date', 'run_at_et', 'month']          # first one present wins


def _git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True,
                       text=True)
    return r.returncode, r.stdout, r.stderr


def conflicted():
    _, out, _ = _git('diff', '--name-only', '--diff-filter=U')
    return [p for p in out.split() if p.endswith('.csv')]


def stage(path, n):
    rc, out, err = _git('show', f':{n}:{path}')
    if rc != 0:
        return None                            # side deleted the file
    return pd.read_csv(io.StringIO(out), dtype=str) if out.strip() else None


def merge_one(path, dry=False):
    ours, theirs = stage(path, 2), stage(path, 3)
    if ours is None and theirs is None:
        print(f'{path}: neither side has content -- resolve by hand')
        return False
    if ours is None or theirs is None:
        keep = ours if theirs is None else theirs
        side = 'ours' if theirs is None else 'theirs'
        print(f'{path}: only {side} has content, keeping it ({len(keep)} rows)')
        out = keep
    else:
        cols = list(ours.columns) + [c for c in theirs.columns
                                     if c not in ours.columns]
        key = next((k for k in KEYS if k in cols), cols[0])
        both = pd.concat([ours.reindex(columns=cols),
                          theirs.reindex(columns=cols)], ignore_index=True)
        # a row that fills more fields is the later/richer observation
        both['_filled'] = both.notna().sum(axis=1)
        both = (both.sort_values('_filled', kind='stable')
                    .drop_duplicates(subset=[key], keep='last')
                    .drop(columns='_filled')
                    .sort_values(key, kind='stable'))
        out = both.reindex(columns=cols)
        added = [c for c in theirs.columns if c not in ours.columns]
        print(f'{path}: ours {len(ours)} + theirs {len(theirs)} rows on "{key}"'
              f' -> {len(out)} kept'
              + (f'; columns adopted from theirs: {", ".join(added)}'
                 if added else ''))
    if dry:
        return True
    out.to_csv(os.path.join(ROOT, path), index=False)
    _git('add', path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='*',
                    help='conflicted csv paths (default: all under reports/)')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    paths = a.paths or conflicted()
    if not paths:
        print('no conflicted csv files -- nothing to do')
        return
    ok = all([merge_one(p, a.dry_run) for p in paths])
    if a.dry_run:
        print('\ndry run -- nothing written')
    elif ok:
        print('\nstaged. finish the merge with:  git commit')
    else:
        raise SystemExit('some files need a human')


if __name__ == '__main__':
    main()
