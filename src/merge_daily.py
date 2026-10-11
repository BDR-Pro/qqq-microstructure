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
#   python src/merge_daily.py --repair        # no conflict in progress, but a
#                                             # log has markers committed in it
#
# Each rule below exists because a simpler version lost or corrupted rows
# (audits 2026-10, every case reproduced in tests/test_merge_daily.py):
#
#   - The git operation is detected first. In a MERGE, stage :3 is the
#     incoming (remote) side; in a REBASE (`git pull --rebase`) git swaps
#     them and stage :2 is upstream. The remote's version is the shared
#     record, so which stage is remote decides who wins a disagreement, and
#     the continue command printed at the end is the operation's own. An
#     operation that cannot be identified (a `git stash pop`) gets no
#     winner: every material disagreement is refused.
#   - Only report logs (reports/*.csv) git actually holds in conflict are
#     accepted. A pasted sentence once became eight "paths".
#   - Each side is read from the index as raw text and parsed with the csv
#     module, never pandas: no float is re-formatted, and no stray first line
#     can become the header (pandas read a side whose line 1 was "<<<<<<<"
#     as a 7-level index and silently discarded every field).
#   - Conflict markers are stripped from every side first, with any diff3
#     base section. A side can carry markers BAKED IN by an earlier
#     `git add . && git commit` made mid-merge; both versions of the rows
#     between them are kept and the union settles them.
#   - Headers must form a strict append chain (each is the previous plus new
#     columns at the end), so every row maps to one header by its width. A
#     renamed column is ambiguous and is refused.
#   - The merge base (stage :1) is never unioned in: a night both sides
#     deliberately removed must not come back. In a modify/delete conflict
#     the surviving side is kept and the deletion is named.
#   - A log is recognised by its HEADER, not its filename, and unioned on its
#     identity: date for the graded log, exec_id for fills, month for the
#     basket registry. The order log has no identity column (orders.py only
#     appends runs), so it -- and any unrecognised log -- is a MULTISET: a
#     row twice on either side stays twice.
#   - Junk an earlier bad merge committed (a marker line with trailing
#     commas, a duplicated header, an empty row) is dropped and named. Any
#     other malformed date/month/time/leg/mode is refused.
#   - Versions of one observation that agree on every shared column -- up to
#     a float's last printed digit, which is formatting -- are one row; the
#     one with more filled fields wins. Versions that differ MATERIALLY: the
#     remote's wins in the graded log and is named; in the basket registry a
#     month pre-registered twice is an integrity event and is refused.
#   - Nothing is written until every conflicted file has merged and passed a
#     safety check on the text that would be written (every observation from
#     every side present, none twice). Writes go to temp files first and are
#     rolled back together if any one fails (a log open in Excel).
#   - With nothing in conflict, the logs are still checked for committed
#     markers and duplicated identities -- "nothing to do" is never said about
#     a log that is broken.

import os, io, re, csv, sys, math, argparse, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Known logs: the header columns that identify each one, its identity, its
# sort order. identity None = multiset (no identity column).
LOGS = {
    'xsec_paper_daily.csv': dict(sig={'date', 'q5_on_bps'}, key=['date'],
                                 order=['date']),
    'fills.csv': dict(sig={'exec_id', 'symbol', 'side'}, key=['exec_id'],
                      order=['time_et', 'exec_id']),
    'xsec_paper.csv': dict(sig={'emitted_on', 'q5', 'q1'}, key=['month'],
                           order=['month'], strict=True),
    'orders_log.csv': dict(sig={'run_at_et', 'leg', 'mode'}, key=None,
                           order=['run_at_et']),
}
# Column formats every real row satisfies; a row that fails one is junk from
# an earlier bad merge (dropped and named) or it is refused.
FORMAT = {'date': r'\d{8}', 'month': r'\d{4}-\d{2}', 'emitted_on': r'\d{8}',
          'run_at_et': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}', 'exec_id': r'\S+',
          'leg': r'open|close', 'mode': r'dry|paper|live'}
MARKER = re.compile(r'^(<{7}|={7}|>{7}|\|{7})( |$)')
LOOSE = re.compile(r'^(<{7}|={7}|>{7}|\|{7})')      # a marker that became a row


class Refuse(Exception):
    """This file needs a human; nothing will be written."""


def _git(*args):
    try:
        r = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True)
    except FileNotFoundError:
        raise SystemExit('git is not on PATH in this shell -- open one where '
                         '`git --version` works and re-run (nothing was written)')
    return (r.returncode, r.stdout.decode('utf-8', 'replace'),
            r.stderr.decode('utf-8', 'replace'))


def unmerged():
    """{path: set of stages present} for every path git holds in conflict."""
    rc, out, err = _git('ls-files', '-u', '-z')
    if rc:
        # e.g. Windows' safe.directory ownership guard: every git command
        # fails, and an empty answer would read as "nothing in conflict"
        raise SystemExit(f'git ls-files -u failed (rc {rc}): {err.strip()}\n'
                         f'nothing was written')
    paths = {}
    for rec in filter(None, out.split('\0')):
        meta, path = rec.split('\t', 1)
        paths.setdefault(path, set()).add(int(meta.split()[2]))
    return paths


def operation():
    """(name, the stage that is the remote/shared record or None, the command
    that finishes it)."""
    def exists(name):
        rc, out, _ = _git('rev-parse', '--git-path', name)
        p = out.strip()
        return rc == 0 and os.path.exists(p if os.path.isabs(p)
                                          else os.path.join(ROOT, p))
    if exists('rebase-merge') or exists('rebase-apply'):
        return 'rebase', 2, 'git rebase --continue'
    if exists('MERGE_HEAD'):
        return 'merge', 3, 'git commit --no-edit'
    if exists('CHERRY_PICK_HEAD'):
        return 'cherry-pick', 3, 'git cherry-pick --continue'
    if exists('REVERT_HEAD'):
        return 'revert', 3, 'git revert --continue'
    return 'unknown', None, 'git commit (or finish whatever left this conflict)'


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


def recognise(path, cols):
    """The log's spec, from its header. A known filename whose header does
    not match its own log is refused rather than merged under the wrong
    identity."""
    name = os.path.basename(path)
    hits = [n for n, s in LOGS.items() if s['sig'] <= set(cols)]
    if name in LOGS and name not in hits:
        raise Refuse(f'header {cols} does not look like {name} '
                     f'(expected columns {sorted(LOGS[name]["sig"])})')
    if len(hits) > 1:
        hits = [name] if name in hits else hits
    if len(hits) != 1:
        return None, dict(key=None, order=[])     # unknown: multiset
    return hits[0], LOGS[hits[0]]


def merge_texts(path, texts, remote):
    """The merge proper. texts: {label: csv text} for the sides present;
    remote: the label that is the shared record, or None when that cannot
    be known. Returns (text to write, report lines)."""
    notes, sides = [], {}
    for label, text in texts.items():
        cols, rows, stripped = parse(text, label)
        if stripped:
            notes.append(f'{label} carried {stripped} conflict-marker line(s) '
                         f'from an earlier unresolved commit -- stripped; the '
                         f'rows on both sides of them are kept')
        sides[label] = (cols, rows)
    if not any(rows for _, rows in sides.values()):
        raise Refuse('no data rows on any side')
    cols, hdrs = [], [h for h, _ in sides.values() if h]
    for h in hdrs:
        cols += [c for c in h if c not in cols]
    kind, spec = recognise(path, cols)
    key = spec['key']
    if kind is None:
        notes.append('not a log this script knows -- unioned as a multiset '
                     'of whole rows')
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
    for label, (_, rows) in sides.items():
        for r in rows:
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

    labels = list(sides)
    filled = lambda r: sum(bool(r.get(c)) for c in cols)
    out, material = [], []
    for k, by in versions.items():
        if not key:                       # multiset: keep the larger count
            src = max((by.get(l, []) for l in labels), key=len)
            for i in range(len(src)):
                out.append(max([by[l][i] for l in labels
                                if i < len(by.get(l, []))], key=filled))
            continue
        allv = [r for l in labels for r in by.get(l, [])]
        if all(compatible(allv[0], x) for x in allv[1:]):
            # one observation, however many copies: the widest copy, the
            # remote's on ties
            pref = by.get(remote, []) + [r for l in labels if l != remote
                                         for r in by.get(l, [])]
            out.append(max(pref, key=filled))
            continue
        rv = by.get(remote, []) if remote else []
        if spec.get('strict') or not rv:
            who = (', '.join(f'{l} {[list(r.values())[:5] for r in by[l]]}'
                             for l in labels if l in by))
            raise Refuse(f'{"/".join(k)} has materially different versions '
                         f'({who}) -- ' +
                         ('this log is a pre-registration; a human decides '
                          'which stands' if spec.get('strict') else
                          'and no remote version settles which one stands'))
        if not all(compatible(rv[-1], x) for x in rv[:-1]):
            raise Refuse(f'{remote} holds {"/".join(k)} twice with '
                         f'different values')
        out.append(rv[-1])
        material.append(k)

    order = spec['order'] or (key or [])
    if order:
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
    need = {k: (max(len(v) for v in by.values()) if not key else 1)
            for k, by in versions.items()}
    short = [k for k, n in need.items() if got.get(k, 0) < n]
    dup = [k for k, n in got.items() if key and n > 1]
    if short or dup or len(back) != len(out):
        raise Refuse(f'SAFETY CHECK FAILED: {len(short)} observation(s) would '
                     f'be lost, {len(dup)} duplicated')

    unit = 'nights' if kind == 'xsec_paper_daily.csv' else 'rows'
    counts = ' + '.join(
        f'{l} {sum(bool(by.get(l)) for by in versions.values()) if key else sum(len(by.get(l, [])) for by in versions.values())}'
        for l in labels)
    notes.insert(0, f'{counts} {unit} on '
                 f'{"+".join(key) if key else "the whole row (multiset)"}'
                 f' -> {len(out)} kept')
    for k in material:
        notes.append(f'{"/".join(k)}: the sides differ materially -- kept '
                     f'the remote\'s version ({remote})')
    return text, notes


def plan(path, stages, remote_stage):
    """Merge one conflicted file in memory from its index stages."""
    names = {2: 'ours', 3: 'theirs'}
    texts = {}
    for n in (2, 3):
        if n in stages:
            rc, text, err = _git('show', f':{n}:{path}')
            if rc:
                raise Refuse(f'cannot read {names[n]} (git: {err.strip()})')
            texts[names[n]] = text
    if not texts:
        raise Refuse('both sides deleted it')
    remote = names.get(remote_stage) if remote_stage else None
    text, notes = merge_texts(path, texts, remote if remote in texts else None)
    gone = [names[n] for n in (2, 3) if n not in stages]
    if gone:
        notes.append(f'{gone[0]} DELETED this file; it is kept with the other '
                     f'side\'s rows -- if the deletion was meant, `git rm` it '
                     f'after finishing')
    if remote:
        notes[0] += f'   [{remote} = the remote]'
    return text, notes


def check_logs():
    """No conflict in progress: are the committed logs themselves sound?"""
    problems = {}
    rdir = os.path.join(ROOT, 'reports')
    for fn in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        if not fn.endswith('.csv'):
            continue
        p = f'reports/{fn}'
        text = open(os.path.join(ROOT, p), encoding='utf-8',
                    errors='replace').read()
        lines = [i + 1 for i, l in enumerate(text.splitlines())
                 if LOOSE.match(l)]
        msg = []
        if lines:
            msg.append(f'conflict-marker lines committed at line(s) '
                       f'{lines[:6]}')
        try:
            merge_texts(p, {'file': text}, None)
        except Refuse as ex:
            msg.append(str(ex))
        if msg:
            problems[p] = msg
    return problems


def write_all(plans):
    """Every file, or none: temp files first, then replace, and on any
    failure restore what was already replaced."""
    done, tmps = [], {}
    try:
        for p, (text, _) in plans.items():
            full = os.path.join(ROOT, p)
            tmp = full + '.merge_daily.tmp'
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                f.write(text)
            tmps[p] = tmp
        for p, tmp in tmps.items():
            full = os.path.join(ROOT, p)
            old = open(full, 'rb').read() if os.path.exists(full) else None
            os.replace(tmp, full)
            done.append((full, old))
    except OSError as ex:
        for full, old in done:
            if old is not None:
                with open(full, 'wb') as f:
                    f.write(old)
        for tmp in tmps.values():
            if os.path.exists(tmp):
                os.remove(tmp)
        raise SystemExit(f'could not write {getattr(ex, "filename", "")} '
                         f'({ex.strerror}) -- is it open in Excel or being '
                         f'synced? Nothing was changed.')
    for p in plans:
        rc, _, err = _git('add', '--', p)
        if rc:
            raise SystemExit(f'{p}: written, but `git add` failed: {err.strip()}')


def main():
    ap = argparse.ArgumentParser(
        description='Resolve a merge conflict in an append-only report log '
                    'by union. Touches nothing unless every file merges.')
    ap.add_argument('paths', nargs='*',
                    help='conflicted reports/*.csv paths (default: all of them)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and write nothing')
    ap.add_argument('--repair', action='store_true',
                    help='no conflict in progress: rewrite a log that has '
                         'committed markers in it, then commit it yourself')
    a = ap.parse_args()
    u = unmerged()
    if a.paths:
        want = []
        for p in a.paths:
            try:
                want.append(os.path.relpath(os.path.abspath(p), ROOT)
                            .replace(os.sep, '/'))
            except ValueError:                    # another drive / a share
                want.append(p)
        bad = [p for p in want if p not in u or not
               (p.startswith('reports/') and p.endswith('.csv'))]
        if bad:
            raise SystemExit(
                'not a conflicted report log: ' + ', '.join(bad) + '\n' +
                ('in conflict right now: ' + ', '.join(sorted(u)) if u else
                 'nothing is in conflict right now -- there is no merge to '
                 'resolve') + '\n(nothing was written)')
    else:
        want = [p for p in sorted(u)
                if p.startswith('reports/') and p.endswith('.csv')]

    if not want:
        broken = check_logs()
        if not broken:
            print('no conflicted csv under reports/, and every log parses '
                  'clean -- nothing to do'
                  + (f' (other conflicts: {", ".join(sorted(u))})' if u else ''))
            return
        for p, msgs in broken.items():
            print(f'{p}: not in conflict, but BROKEN:')
            for m in msgs:
                print(f'    {m}')
        if not a.repair:
            raise SystemExit('\nrun  python src/merge_daily.py --repair  to '
                             'rewrite them, then commit (nothing was written)')
        plans = {}
        for p in broken:
            text = open(os.path.join(ROOT, p), encoding='utf-8').read()
            try:
                plans[p] = merge_texts(p, {'file': text}, None)
            except Refuse as ex:
                raise SystemExit(f'{p}: {ex}\nnothing was written')
        for p, (_, notes) in plans.items():
            print(f'{p}: ' + '\n    '.join(notes))
        if not a.dry_run:
            write_all(plans)
            print('\nrepaired and staged -- review, then:  git commit -m '
                  '"repair report logs"')
        return

    op, remote_stage, cont = operation()
    if op == 'unknown':
        print('could not tell which git operation left this conflict (a '
              '`git stash pop`?) -- no side is treated as the remote, so any '
              'material disagreement will be refused')
    plans, failed = {}, []
    for p in want:
        try:
            plans[p] = plan(p, u[p], remote_stage)
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
        print(f'\ndry run ({op}) -- nothing written')
        return
    write_all(plans)
    rest = sorted(set(unmerged()) - set(want))
    print(f'\nstaged ({op}).' + (f' still in conflict: {", ".join(rest)}'
                                 if rest else f' finish with:  {cont}'))


if __name__ == '__main__':
    main()
