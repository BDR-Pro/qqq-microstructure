# Part of qqq-microstructure.
#
# Resolve a merge conflict in an APPEND-ONLY report log.
#
# reports/*.csv are written from places that never see each other: the VM next
# to IB Gateway (xsec_live, orders, fills -- see roi.yml's crontab), GitHub
# Actions (monthly.yml), and anyone's own checkout. All of them commit. When two
# of them touch the same log, git reports a content conflict in a file where
# BOTH sides are right -- the rows are observations, not edits -- so the
# resolution is the union. Doing that by hand invites the one mistake that
# cannot be undone: dropping or doubling a graded night.
#
#   python src/merge_daily.py --dry-run       # show the plan, touch nothing
#   python src/merge_daily.py                 # every conflicted csv under reports/
#   python src/merge_daily.py reports/xsec_paper_daily.csv
#
# Each rule below exists because the simpler version lost or corrupted rows
# (audit 2026-10, reproduced in tests/test_merge_daily.py):
#
#   - Only paths git actually holds in conflict are accepted, by name. A pasted
#     sentence once became eight "paths" and eight false "resolve by hand"s.
#   - Every side is read from the index (:1 base, :2 ours, :3 theirs) as raw
#     text and parsed with the csv module, never through pandas: no float is
#     re-formatted, and no stray first line can be mistaken for the header
#     (pandas once read a side whose line 1 was "<<<<<<< HEAD" as a 7-level
#     index and silently discarded every field).
#   - Conflict markers are stripped from every side first, with the diff3 base
#     section between "|||||||" and "=======". A side can carry markers BAKED
#     IN by an earlier `git add . && git commit` made mid-merge; both versions
#     of the rows between them are kept and the union settles them.
#   - A log may switch header mid-file (a side whose header was in the
#     conflict). Headers must form a strict append chain -- each one the
#     previous plus new columns at the end -- so every row maps to exactly one
#     header by its width. A RENAMED column is ambiguous and is refused.
#   - Rows are unioned on each log's natural key -- date for the paper log,
#     (run_at_et, leg, mode) for the order log, exec_id for fills, month for
#     the basket registry -- and on the whole row for any other file, so two
#     legitimate rows can never collapse into one.
#   - A row whose key is not in the log's key format is junk (a marker line
#     with trailing commas, a duplicated header) only if its other fields are
#     empty or it repeats a header; it is dropped and named. Anything else with
#     a malformed key is refused, not guessed at.
#   - Where the sides carry the same key with different text, the row with
#     more filled fields wins, then theirs (the shared record), and every such
#     key is printed.
#   - Nothing is written until EVERY conflicted file has merged and passed a
#     safety check run on the text that would be written: every key from every
#     side present, none twice. Output is LF and byte-exact per field.

import os, io, re, csv, sys, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = {'xsec_paper_daily.csv': ['date'],
        'xsec_paper.csv': ['month'],
        'orders_log.csv': ['run_at_et', 'leg', 'mode'],
        'fills.csv': ['exec_id']}
KEYFMT = {'date': r'\d{8}', 'month': r'\d{4}-\d{2}',
          'run_at_et': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', 'exec_id': r'\S+',
          'leg': r'(open|close)', 'mode': r'(dry|paper|live)'}
ORDER = {'fills.csv': ['time_et', 'exec_id']}      # default: sort by the key
MARKER = re.compile(r'^(<{7}|={7}|>{7}|\|{7})( |$)')


def _git(*args):
    r = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True)
    return (r.returncode, r.stdout.decode('utf-8', 'replace'),
            r.stderr.decode('utf-8', 'replace'))


def unmerged():
    """{path: set of stages present} for every path git holds in conflict."""
    _, out, _ = _git('ls-files', '-u', '-z')
    paths = {}
    for rec in filter(None, out.split('\0')):
        meta, path = rec.split('\t', 1)
        paths.setdefault(path, set()).add(int(meta.split()[2]))
    return paths


class Refuse(Exception):
    """This file needs a human; nothing will be written."""


def strip_markers(text):
    """Drop conflict-marker lines and diff3 base sections. Returns the kept
    lines and how many marker lines were removed."""
    kept, n, in_base = [], 0, False
    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            n += 1
            in_base = m.group(1) == '|' * 7       # diff3: base runs to =======
            continue
        if not in_base:
            kept.append(line)
    return kept, n


def parse(text, label):
    """Raw csv text -> (columns, rows as dicts, markers stripped). Values stay
    exactly as written; nothing is converted."""
    lines, n = strip_markers(text)
    recs = [r for r in csv.reader(l for l in lines if l.strip())]
    if not recs:
        return [], [], n
    first = recs[0][0]
    headers, rows = [], []
    for rec in recs:
        if rec[0] == first:                      # a (possibly alternate) header
            if rec not in headers:
                headers.append(rec)
            continue
        rows.append(rec)
    chain = sorted(headers, key=len)
    for a, b in zip(chain, chain[1:]):
        if len(a) == len(b) or b[:len(a)] != a:
            raise Refuse(f'{label} has headers that are not a column-append '
                         f'chain ({a} vs {b}) -- a renamed or reordered '
                         f'column is ambiguous')
    width = {len(h): h for h in chain}
    out = []
    for rec in rows:
        h = width.get(len(rec))
        if h is None:
            raise Refuse(f'{label} has a {len(rec)}-field line that fits no '
                         f'header ({sorted(width)}): {rec[:4]}')
        out.append(dict(zip(h, rec)))
    return chain[-1], out, n


def plan(path, stages):
    """Merge one file in memory. Returns (text to write, report lines)."""
    name = os.path.basename(path)
    if 2 not in stages and 3 not in stages:
        raise Refuse('both sides deleted it')
    notes, sides = [], {}
    for n, label in ((1, 'base'), (2, 'ours'), (3, 'theirs')):
        if n not in stages:
            continue
        rc, text, err = _git('show', f':{n}:{path}')
        if rc:
            raise Refuse(f'cannot read {label} (git: {err.strip()})')
        cols, rows, stripped = parse(text, label)
        if stripped:
            notes.append(f'{label} carried {stripped} conflict-marker line(s) '
                         f'from an earlier unresolved commit -- stripped; the '
                         f'rows on both sides of them are kept')
        sides[label] = (cols, rows)
    if not any(sides.get(s, ([], []))[1] for s in ('ours', 'theirs')):
        raise Refuse('no data rows on either side')

    cols = []
    for label in ('ours', 'theirs', 'base'):
        for c in sides.get(label, ([], []))[0]:
            if c not in cols:
                cols.append(c)
    key = KEYS.get(name)
    if key and not all(k in cols for k in key):
        raise Refuse(f'expected key {key} not in header {cols}')
    kf = (lambda r: tuple(r.get(k, '') for k in key)) if key else \
        (lambda r: tuple(r.get(c, '') for c in cols))

    def valid(r):
        return not key or all(re.fullmatch(KEYFMT.get(k, r'.+'), r.get(k, ''))
                              for k in key)

    rank = {'base': 0, 'ours': 1, 'theirs': 2}
    best, seen, first, junk = {}, {}, {}, []
    for label in ('base', 'ours', 'theirs'):
        for r in sides.get(label, ([], []))[1]:
            if not valid(r):
                vals = [v for c, v in r.items() if c not in key]
                if not any(vals) or list(r.values()) == list(r.keys()) or \
                        MARKER.match(next(iter(r.values()), '') + ' '):
                    junk.append((label, list(r.values())[:3]))
                    continue
                raise Refuse(f'{label} has a row whose key is not '
                             f'{"/".join(KEYFMT.get(k, k) for k in key)}: '
                             f'{list(r.values())[:4]}')
            k = kf(r)
            full = tuple(r.get(c, '') for c in cols)
            seen.setdefault(label, {})[k] = full
            first.setdefault(k, len(first))
            cand = (sum(bool(v) for v in full), rank[label], full)
            if k not in best or cand[:2] >= best[k][:2]:
                best[k] = cand
    for label, v in junk:
        notes.append(f'dropped a junk row from {label} (left by an earlier '
                     f'bad merge): {v}')

    order = ORDER.get(name) or key
    sk = (lambda k: tuple(dict(zip(cols, best[k][2])).get(c, '')
                          for c in order)) if order else (lambda k: first[k])
    out = [best[k][2] for k in sorted(best, key=sk)]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    w.writerows(out)
    text = buf.getvalue()

    # the safety check runs on what would be WRITTEN, parsed back from text
    _, back, _ = parse(text, 'result')
    got = [kf(r) for r in back]
    want = {k for s in seen.values() for k in s}
    lost, dup = want - set(got), len(got) - len(set(got))
    if lost or dup or len(back) != len(out):
        raise Refuse(f'SAFETY CHECK FAILED: {len(lost)} key(s) would be lost, '
                     f'{dup} duplicated')

    o, t = seen.get('ours', {}), seen.get('theirs', {})
    notes.insert(0, f'ours {len(o)} + theirs {len(t)} rows on '
                 f'{"+".join(key) if key else "the whole row (no natural key known)"}'
                 f' -> {len(out)} kept')
    for k in sorted(x for x in o if x in t and o[x] != t[x]):
        notes.append(f'{"/".join(k) if key else "a row"}: the sides differ -- '
                     f'kept the {"theirs" if best[k][1] == 2 else "ours"} version')
    return text, notes


def main():
    ap = argparse.ArgumentParser(
        description='Resolve a merge conflict in an append-only report log '
                    'by union. Touches nothing unless every file merges.')
    ap.add_argument('paths', nargs='*',
                    help='conflicted csv paths (default: every one under reports/)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and write nothing')
    a = ap.parse_args()
    u = unmerged()
    if a.paths:
        want = [os.path.relpath(os.path.abspath(p), ROOT).replace(os.sep, '/')
                for p in a.paths]
        bad = [p for p in want if p not in u]
        if bad:
            raise SystemExit(
                'not in conflict: ' + ', '.join(bad) + '\n' +
                ('in conflict right now: ' + ', '.join(sorted(u)) if u else
                 'nothing is in conflict right now -- there is no merge to '
                 'resolve') + '\n(nothing was written)')
    else:
        want = [p for p in sorted(u)
                if p.startswith('reports/') and p.endswith('.csv')]
        if not want:
            print('no conflicted csv under reports/ -- nothing to do'
                  + (f' (other conflicts: {", ".join(sorted(u))})' if u else ''))
            return

    plans, failed = {}, []
    for p in want:
        try:
            plans[p] = plan(p, u[p])
        except Refuse as ex:
            failed.append(f'{p}: {ex}')
    for p, (_, notes) in plans.items():
        print(f'{p}: ' + notes[0])
        for n in notes[1:]:
            print(f'    {n}')
    if failed:
        raise SystemExit('\n'.join(failed) + '\nnothing was written -- '
                         'these need a human, no file was guessed at')
    if a.dry_run:
        print('\ndry run -- nothing written')
        return
    for p, (text, _) in plans.items():
        with open(os.path.join(ROOT, p), 'w', newline='', encoding='utf-8') as f:
            f.write(text)
        rc, _, err = _git('add', '--', p)
        if rc:
            raise SystemExit(f'{p}: written, but `git add` failed: {err.strip()}')
    rest = sorted(set(unmerged()) - set(want))
    print('\nstaged.' + (f' still in conflict: {", ".join(rest)}' if rest
                         else ' finish the merge with:  git commit --no-edit'))


if __name__ == '__main__':
    main()
