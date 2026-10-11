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
#   - Rows are unioned on each log's identity -- date for the paper log,
#     exec_id for fills, month for the basket registry. The order log has no
#     identity column (orders.py only appends runs), so it and any unknown
#     log are unioned as a MULTISET: a row twice on either side stays twice.
#     A side that lacks the identity column is refused, never collapsed.
#   - Junk an earlier bad merge committed (a marker line with trailing
#     commas, a duplicated header, an empty row) is dropped and named. Any
#     other row with a malformed date/month/time/leg/mode is refused.
#   - Two versions of one observation that agree on every shared column --
#     up to a float's last printed digit, which is formatting, not data --
#     are the same row, and the one with more filled fields wins. Versions
#     that differ MATERIALLY: the remote's (theirs) wins in the graded log and
#     is printed; in the basket registry a month pre-registered twice is an
#     integrity event and is refused for a human to decide.
#   - Nothing is written until EVERY conflicted file has merged and passed a
#     safety check run on the text that would be written: every key from every
#     side present, none twice. Output is LF and byte-exact per field.

import os, io, re, csv, sys, math, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The column(s) that identify one observation. A log not listed here has no
# identity column (orders.py only ever appends a run, never rewrites one), so
# its rows are unioned as a MULTISET: a row present twice on either side is
# present twice in the result.
KEYS = {'xsec_paper_daily.csv': ['date'],
        'xsec_paper.csv': ['month'],
        'fills.csv': ['exec_id']}
# Logs where two materially different rows for one key are an integrity
# event, not a merge: a month pre-registered twice must be decided by a
# human (the earlier committed basket is the pre-registration).
STRICT = {'xsec_paper.csv'}
# Column formats every real row satisfies; a row that fails one is either
# junk from an earlier bad merge (dropped and named) or refused.
FORMAT = {'date': r'\d{8}', 'month': r'\d{4}-\d{2}', 'emitted_on': r'\d{8}',
          'run_at_et': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', 'exec_id': r'\S+',
          'leg': r'open|close', 'mode': r'dry|paper|live'}
ORDER = {'fills.csv': ['time_et', 'exec_id'],
         'orders_log.csv': ['run_at_et']}          # default: sort by the key
MARKER = re.compile(r'^(<{7}|={7}|>{7}|\|{7})( |$)')
LOOSE = re.compile(r'^(<{7}|={7}|>{7}|\|{7})')      # a marker that became a row


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

    cols, hdrs = [], [h for h in [sides.get(x, ([], []))[0] for x in
                                  ('ours', 'theirs', 'base')] if h]
    for h in hdrs:
        cols += [c for c in h if c not in cols]
    key = KEYS.get(name)
    if key:
        for label, (h, rows) in sides.items():
            if rows and not all(k in h for k in key):
                raise Refuse(f'{label} has no {"+".join(key)} column, so its '
                             f'rows cannot be matched: header {h}')
    shared = [c for c in cols if all(c in h for h in hdrs)]

    def junk(r):
        v = [r.get(c, '') for c in cols]
        return (not any(v) or LOOSE.match(v[0] or '') is not None
                or any(v[:len(h)] == h for h in hdrs))

    def bad_format(r):
        return [c for c in cols if c in FORMAT and r.get(c) and
                not re.fullmatch(FORMAT[c], r[c])] + \
               [k for k in (key or []) if not r.get(k)]

    def compatible(a, b):
        """Same observation? Compared on shared columns only; an empty cell
        is a schema difference, and a float that differs only in its last
        printed digit is the same number."""
        for c in shared:
            x, y = a.get(c, ''), b.get(c, '')
            if x == y or not x or not y:
                continue
            try:
                if math.isclose(float(x), float(y), rel_tol=1e-9, abs_tol=1e-9):
                    continue
            except ValueError:
                pass
            return False
        return True

    versions, dropped = {}, []
    ident = (lambda r: tuple(r[k] for k in key)) if key else \
        (lambda r: tuple(r.get(c, '') for c in shared))
    for label in ('base', 'ours', 'theirs'):
        for r in sides.get(label, ([], []))[1]:
            if junk(r):
                dropped.append((label, [r.get(c, '') for c in cols][:3]))
                continue
            bad = bad_format(r)
            if bad:
                raise Refuse(f'{label} has a row with a malformed '
                             f'{"/".join(bad)}: {[r.get(c, "") for c in cols][:4]}')
            versions.setdefault(ident(r), {}).setdefault(label, []).append(r)
    for label, v in dropped:
        notes.append(f'dropped a junk row from {label} (left by an earlier '
                     f'bad merge): {v}')

    filled = lambda r: sum(bool(r.get(c)) for c in cols)
    out, material = [], []
    for k, by in versions.items():
        o, t = by.get('ours', []), by.get('theirs', [])
        if not key:                       # multiset: keep the larger count
            src = max((o, t, by.get('base', [])), key=len)
            for i in range(len(src)):
                cand = [x[i] for x in (o, t) if i < len(x)] or [src[i]]
                out.append(max(cand, key=filled))
            continue
        if t:
            # theirs is the version already on the remote; a side that
            # carries committed markers may hold BOTH versions of a night,
            # and the remote's settles which one stands
            t0 = t[-1]
            if not all(compatible(t0, x) for x in t[:-1]):
                raise Refuse(f'theirs holds {"/".join(k)} twice with '
                             f'different values')
            if all(compatible(t0, x) for x in o):
                out.append(max([t0] + o, key=filled))     # ties: theirs
            elif name in STRICT:
                raise Refuse(f'{"/".join(k)} differs between the sides '
                             f'(ours {list(o[-1].values())[:5]} vs theirs '
                             f'{list(t0.values())[:5]}) -- this log is a '
                             f'pre-registration; a human decides which stands')
            else:
                out.append(t0)
                material.append(k)
        elif o:
            if not all(compatible(o[-1], x) for x in o[:-1]):
                raise Refuse(f'ours holds {"/".join(k)} twice with different '
                             f'values and the remote has no version to '
                             f'settle it')
            out.append(max(o, key=filled))
        else:
            out.append(by['base'][-1])

    order = ORDER.get(name) or key or []
    out.sort(key=lambda r: tuple(r.get(c, '') for c in order))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    w.writerows([r.get(c, '') for c in cols] for r in out)
    text = buf.getvalue()

    # the safety check runs on what would be WRITTEN, parsed back from text
    _, back, _ = parse(text, 'result')
    got = {}
    for r in back:
        got[ident(r)] = got.get(ident(r), 0) + 1
    short = [k for k, by in versions.items()
             if got.get(k, 0) < max(len(v) for v in by.values())
             and (not key or got.get(k, 0) < 1)]
    dup = [k for k, n in got.items() if key and n > 1]
    if short or dup or len(back) != len(out):
        raise Refuse(f'SAFETY CHECK FAILED: {len(short)} observation(s) would '
                     f'be lost, {len(dup)} duplicated')

    count = (lambda side: sum(bool(by.get(side)) for by in versions.values())) \
        if key else \
        (lambda side: sum(len(by.get(side, [])) for by in versions.values()))
    n_o, n_t = count('ours'), count('theirs')      # nights, or runs
    notes.insert(0, f'ours {n_o} + theirs {n_t} rows on '
                 f'{"+".join(key) if key else "the whole row (multiset)"}'
                 f' -> {len(out)} kept')
    for k in material:
        notes.append(f'{"/".join(k)}: the sides differ materially -- kept the '
                     f'version already on the remote (theirs)')
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
