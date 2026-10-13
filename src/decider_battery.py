# Part of qqq-microstructure.
#
# The decider battery's command line (RESULTS 27; spec v1, section 7). Each
# step from the committed declaration to the selection is one subcommand and
# a thin call into the module that owns it:
#
#   make-items   battery_items   the stimuli, generated once and locked
#   make-iter    battery_items   an iteration round's stimuli (section 4)
#   verify       battery_items   the lock, the bytes, the counts, a clean
#                + battery_run   tree, what the plan could cost
#   run          battery_run     send every item to every model, resumably
#   score        battery_score   every metric and gate -> summary.json
#   select       battery_score   section 3's rule -> selection_v1.json, or
#                                section 4's -> iter_r{r}_outcome.json
#   scrub-check  battery_run     no key-shaped string in any output
#
# Why one script with fixed exit codes: the cloud session runs exactly these
# commands, and the code alone says what to do next (resume, fix, stop)
# without anyone reading output that could hold a secret. Every line printed
# passes battery_run.scrub.
#
# What the cloud session runs -- this list is the runbook (it supersedes the
# draft in the design spec, whose test file and --max-usd 15 are gone):
#
#   git fetch origin && git checkout <the declaration's merge sha> \
#       && git switch -c claude/decider-battery-v1
#   pip install numpy pandas pytest
#   python -m pytest tests/test_battery_items.py tests/test_battery_run.py \
#       tests/test_battery_score.py tests/test_battery_e2e.py \
#       tests/test_guard_randomization.py tests/test_openrouter.py \
#       tests/test_guard.py -q -p no:cacheprovider
#   python src/decider_battery.py verify --plan bench/battery/plan_v1.json
#   python src/decider_battery.py run --plan bench/battery/plan_v1.json \
#       --out bench/battery/runs/v1 --auth proxy --max-usd 8 --deadline-min 90
#   python src/decider_battery.py score --run bench/battery/runs/v1
#   python src/decider_battery.py select --run bench/battery/runs/v1
#   python src/decider_battery.py scrub-check --run bench/battery/runs/v1
#   git add bench/battery/runs/v1 && git commit \
#       && git push -u origin claude/decider-battery-v1
#
# The run takes 10-30 minutes, longer than a foreground command may live
# (a cloud session's shell kills one after 10): start it in the background
# and wait for its exit code -- or pass --deadline-min 9 and run it in the
# foreground again while it exits 4. By exit code: 0, score; 4, the same
# command again; 3, the same command with --max-usd 12, the ceiling (after
# a 402, once the credit is topped up). A killed or crashed run resumes
# with the same command too.
# score and select refuse a run that is not done or capped at the ceiling,
# and run refuses to resume a run that has been scored: whether to extend
# a run is never decided after seeing its result.
#
# Iteration (spec section 4), after the selection is committed: the
# selected decider is re-asked in up to four of the menu's ways (q2, s3,
# b2, and k5 only where the selection measured a reason; sep not until
# decide_local can send it) on fresh states, against its current version.
# Arms are request variants of the one pinned model -- the bytes carry the
# pin and the selection's provider pin, as the runtime will send them, and
# the arm's id only names its record. Each declaration is made, committed
# and pushed BEFORE its run, in the designing session, offline, inside the
# selection's battery directory:
#
#   python src/decider_battery.py make-iter \
#       --selection bench/battery/runs/v1/selection_v1.json --round 1 \
#       --variants q2,s3,b2 --out bench/battery/iter_r1
#   git add bench/battery/iter_r1 && git commit && git push
#
# then the cloud session runs it as above, its run directory under the
# declaration's own runs/ (cap $3 a round, $2 the holdout; no raise):
#
#   python src/decider_battery.py verify --plan \
#       bench/battery/iter_r1/plan_iter_r1.json
#   python src/decider_battery.py run --plan \
#       bench/battery/iter_r1/plan_iter_r1.json \
#       --out bench/battery/iter_r1/runs/r1 --auth proxy
#   python src/decider_battery.py score --run bench/battery/iter_r1/runs/r1
#   python src/decider_battery.py select --run bench/battery/iter_r1/runs/r1
#   python src/decider_battery.py scrub-check --run \
#       bench/battery/iter_r1/runs/r1
#
# select writes iter_r1_outcome.json: the variant adopted (or none), the
# version the decider now stands at, and 'next' -- the stopping rule.
# Round 2 starts from it (--selection .../iter_r1_outcome.json --round 2,
# its control round 1's adopted version, --variants the remaining menu on
# top, exactly), and the holdout from round 2's (--round holdout, no
# variants: the final version against the selection's own). make-iter
# re-derives the selection or outcome it starts from out of the summary
# and plan beside it, and refuses any step the stopping rule does not
# name, a round declared twice, a round 2 other than the remaining menu,
# and k5 or sep the selection did not allow. The outcome that says next:
# stop carries its decider in decide_local.DECLARED's own fields, with the
# suffix the runtime will log (guard.handling_suffix).
#
# Exit codes: 0 done; 2 auth failure; 3 capped, out of credit, an outage or
# incomplete (resumable); 4 deadline (resumable); 5 refused; 6 insufficient
# analysis set (no selection from this run); 7 scrub hit.

import os, sys, glob, json, time, shutil, argparse, tempfile
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import battery_items as bi
import battery_run as br
import battery_score as bs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DONE, REFUSED, INSUFFICIENT, SCRUB_HIT = 0, br.REFUSED, \
    bs.EXIT_INSUFFICIENT, 7
say = br.say


def _refuse(why):
    say(f'refused: {why}')
    return REFUSED


def _read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _rel(path):
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if br._inside(p, ROOT) else p


# ------------------------------------------------------------ make-items --

def make_items(plan_path, force=False):
    """Generate plan, states, items and lock into the plan's directory from
    the committed S0 and anchors there. Built in a scratch directory first:
    when a lock already exists and the new bytes differ from it (another
    numpy, an edited generator), nothing is written -- the committed bytes
    are the declaration, and only --force replaces them."""
    plan_path = os.path.abspath(plan_path)
    d, name = os.path.split(plan_path)
    want = f'plan_{bi.VERSION}.json'
    if name != want:
        return _refuse(f'make-items writes {want}, not {name}')
    old = _read_json(bi.lock_path(plan_path))
    was = ((_read_json(plan_path) or {}).get('generator') or {}).get('numpy')
    with tempfile.TemporaryDirectory() as tmp:
        for f in (bi.S0_FILE, bi.ANCHORS_FILE):
            if not os.path.exists(os.path.join(d, f)):
                return _refuse(f'{f} is not in {_rel(d)}')
            shutil.copyfile(os.path.join(d, f), os.path.join(tmp, f))
        try:
            out = bi.make_items(tmp)
        except (AssertionError, KeyError, ValueError) as ex:
            return _refuse(f'make-items: S0 or the anchors are not what the '
                           f'generator expects ({type(ex).__name__})')
        new = out['lock']
        changed = sorted(k for k in new if old and old.get(k) != new[k])
        if changed and not force:
            return _refuse(
                f'the regenerated {", ".join(changed)} differ from the '
                f'committed lock (numpy {bi.GENERATOR["numpy"]} here, {was} '
                f'recorded); nothing written -- --force replaces the '
                f'declaration')
        names = [os.path.basename(p) for p in out['paths'].values()]
        names.append(os.path.basename(bi.lock_path(out['paths']['plan'])))
        for f in names:                 # the lock last
            shutil.copyfile(os.path.join(tmp, f), os.path.join(d, f + '.tmp'))
            os.replace(os.path.join(d, f + '.tmp'), os.path.join(d, f))
    status = 'new' if not old else ('replaced (--force)' if changed
                                    else 'unchanged')
    say(f'{len(out["items"])} items, {len(out["states"])} states -> '
        f'{_rel(d)}; lock {status}')
    for k, v in new.items():
        say(f'  {k:8s} {v}')
    return DONE


# ------------------------------------------------------------- make-iter --

# what make-iter reads from a selection or an outcome: each is re-derived
# from the summary and plan beside it before it is trusted
SELECTION_KEYS = ('eligible', 'top', 'incumbent', 'branch', 'decider',
                  'reference')
OUTCOME_KEYS = ('round', 'holdout', 'pin', 'control', 'variants', 'adopted',
                'adopted_arm', 'kept', 'final', 'next', 'origin', 'prior',
                'decider')


def iter_source(path, rnd):
    """What an iteration round starts from (battery_items.make_iter's src,
    less the variants), read from a selection_v1.json (round 1) or the
    previous round's outcome (round 2, the holdout), checked against the
    summary.json and the plan beside it that it was made from -- and the
    decision itself re-derived from them (battery_score.choose, or
    iterate_outcome) and refused unless the file says the same: a hand edit
    of the decider, the adopted version, the stopping rule's 'next' or the
    origin (k5, sep) unlocks nothing. The outcome's 'next' then decides
    which round may come: a round after one that adopted nothing, or a
    holdout of the selection's own version, is refused."""
    path = os.path.abspath(path)
    src = _read_json(path)
    if not isinstance(src, dict) or src.get('mode') not in ('select',
                                                            'iterate'):
        raise ValueError(f'{_rel(path)} is not a selection or an iteration '
                         f'outcome')
    run = os.path.dirname(path)
    sp = os.path.join(run, 'summary.json')
    if not os.path.exists(sp) or br._sha(sp) != src.get('summary_sha256'):
        raise ValueError(f'the summary.json beside {_rel(path)} is not the '
                         f'one it was made from')
    plan_path = bs.plan_of(run)
    if br._sha(plan_path) != src.get('plan_sha256'):
        raise ValueError(f'{_rel(plan_path)} is not the plan {_rel(path)} '
                         f'was made from')
    plan, summary = _read_json(plan_path), _read_json(sp)
    if src['mode'] != plan.get('mode', 'select'):
        raise ValueError(f'{_rel(path)} was not made from {_rel(plan_path)}')
    try:
        if src['mode'] == 'select':
            fresh, keys = bs.choose(summary, plan), SELECTION_KEYS
        else:
            fresh, keys = bs.iterate_outcome(summary, plan), OUTCOME_KEYS
    except bs.InsufficientAnalysisSet as ex:
        raise ValueError(ex.msg)
    fresh = bs._clean(fresh)
    edited = [k for k in keys if src.get(k) != fresh.get(k)]
    if edited:
        raise ValueError(f'{_rel(path)} is not what its summary and plan '
                         f'decide ({", ".join(edited)} differ)')
    anchors = os.path.join(os.path.dirname(plan_path),
                           plan['files']['anchors'])
    where = {'path': _rel(path).replace(os.sep, '/'), 'sha256': br._sha(path),
             'summary_sha256': src['summary_sha256'],
             'plan_sha256': src['plan_sha256']}
    if src['mode'] == 'select':
        if rnd != 1:
            raise ValueError('a selection starts round 1; round 2 and the '
                             'holdout start from the previous outcome')
        origin = bi.origin_of(src, summary, plan, where)
        dec = origin['decider']
        base, prior = [dec['version'], dec['handling']], None
    else:
        want = 'holdout' if rnd == 'holdout' else f'round {rnd}'
        if src.get('next') != want:
            raise ValueError(f'the stopping rule says next: '
                             f'{src.get("next")!r} after {_rel(path)}, not '
                             f'{want}')
        origin = src['origin']
        base = [src['final']['version'], src['final']['handling']]
        prior = dict(where, round=src['round'], adopted=src['adopted'])
    pin = origin['decider']['pin']
    bs.refuse_aliases([pin])
    return {'round': rnd, 'pin': pin, 'base': base, 'origin': origin,
            'prior': prior, 'anchors': anchors}


def battery_dir(origin):
    """The selection's battery directory (bench/battery: its plan's), where
    every round of its iteration is declared -- found from the selection
    the origin names, which must still be the file it was."""
    p = origin['selection']['path']
    p = p if os.path.isabs(p) else os.path.join(ROOT, p)
    if not os.path.exists(p) or br._sha(p) != origin['selection']['sha256']:
        raise ValueError(f'the selection this iteration started from is not '
                         f'{origin["selection"]["path"]} as it was')
    return os.path.dirname(os.path.abspath(bs.plan_of(os.path.dirname(p))))


def declared_elsewhere(home, version, out):
    """Another declaration of this round (its plan, or an outcome of it)
    anywhere under the battery directory, other than out's own."""
    name = 'iter_holdout_outcome.json' if version == 'iter_holdout' else \
        f'{version}_outcome.json'
    hits = [os.path.abspath(p) for f in (f'plan_{version}.json', name)
            for p in glob.glob(os.path.join(home, '**', f), recursive=True)]
    return sorted(p for p in hits if not br._inside(p, out))


def make_iter(selection, rnd, variants, out, force=False):
    """Write an iteration round's (or the holdout's) declaration into out:
    plan_iter_r{r}.json (plan_iter_holdout.json), its states, items, a copy
    of the anchors and the lock -- deterministic bytes, built in a scratch
    directory first and, like make-items, never written over a different
    lock without --force.

    A round is declared once: out must be inside the selection's battery
    directory, and a declaration of the same round anywhere else in it is
    refused (round 1 again from the same selection is the same 18 states,
    re-rolled), as is --force over a declaration that has a run under its
    runs/."""
    rnd = 'holdout' if rnd == 'holdout' else int(rnd)
    tags = [t.strip() for t in (variants or '').split(',') if t.strip()]
    out = os.path.abspath(out)
    version = 'iter_holdout' if rnd == 'holdout' else f'iter_r{rnd}'
    try:
        src = iter_source(selection, rnd)
        home = battery_dir(src['origin'])
    except (OSError, ValueError, KeyError) as ex:
        return _refuse(f'make-iter: {ex}')
    if not br._inside(out, home) or out == home:
        return _refuse(f'make-iter: {_rel(out)} is not inside {_rel(home)}, '
                       f'the selection\'s battery directory: every round of '
                       f'its iteration is declared there')
    other = declared_elsewhere(home, version, out)
    if other:
        return _refuse(f'make-iter: {version} is already declared '
                       f'({_rel(other[0])}): a round is declared once')
    src['variants'] = tags
    with tempfile.TemporaryDirectory() as tmp:
        try:
            made = bi.make_iter(tmp, src)
        except (ValueError, KeyError) as ex:
            return _refuse(f'make-iter: {ex}')
        lock = os.path.basename(bi.lock_path(made['paths']['plan']))
        old = _read_json(os.path.join(out, lock))
        new = made['lock']
        changed = sorted(k for k in new if old and old.get(k) != new[k])
        if changed and not force:
            return _refuse(f'{_rel(out)} already holds another declaration '
                           f'({", ".join(changed)} differ); nothing written '
                           f'-- --force replaces it')
        runs = os.path.join(out, 'runs')
        if changed and os.path.isdir(runs) and os.listdir(runs):
            return _refuse(f'{_rel(out)} holds a run of its declaration: it '
                           f'is never replaced, not even with --force')
        os.makedirs(out, exist_ok=True)
        for f in [os.path.basename(p) for p in made['paths'].values()] + \
                [lock]:                 # the lock last
            shutil.copyfile(os.path.join(tmp, f),
                            os.path.join(out, f + '.tmp'))
            os.replace(os.path.join(out, f + '.tmp'), os.path.join(out, f))
    plan, pr = made['plan'], made['plan']['projection']
    say(f'{plan["version"]}: {plan["pin"]}, {len(made["states"])} states, '
        f'{pr["calls"]} calls, projected ${pr["usd"]:.2f} (cap '
        f'${pr["cap_usd"]}) -> {_rel(out)}')
    for a in plan['arms']:
        n = sum(plan['calls_per_model']['per_arm'][a['id']]['nominal']
                .values())
        say(f'  {a["role"]:8s} {a["tag"]:12s} {n} calls')
    if pr['dropped_blocks'] or pr['dropped_variants']:
        say(f'  over the cap: dropped blocks {pr["dropped_blocks"]}, '
            f'variants {pr["dropped_variants"]}')
    for k, v in new.items():
        say(f'  {k:8s} {v}')
    return DONE


# ---------------------------------------------------------------- verify --

def _per_model(plan, items, sd):
    """{experiment: calls} for one model, after a scale-down."""
    by = {it['item_id']: it for it in items}
    m = plan['incumbent']
    return dict(sorted(Counter(
        by[k.split('|', 1)[1]]['experiment']
        for k in br.planned(plan, items, sd)
        if k.split('|', 1)[0] == m).items()))


def projection(plan, items):
    """What the plan could cost at catalogue prices: every model at the
    pessimistic per-call figure scaled by its input price (battery_run's
    estimate before a pilot lands), and the scale-down the run would apply
    if the pilot measured exactly that -- block 4, then unscored models,
    dearest first, never I or R (an iteration plan: each call priced by its
    bytes, its last block only above the 2-block floor, variants dropped
    from the end of the menu). The run decides from measured costs; this is
    the bound, before any call."""
    sp, models = plan['spend'], br.models_of(plan)
    cand = {c['id']: c for c in plan['candidates']}

    def c0(m):
        price = cand.get(m, {}).get('price_in')
        return sp['pessimistic_per_call'] * (
            price / sp['price_ref'] if price is not None else 1.0)
    cm = {m: c0(m) for m in models}
    wt = br.call_weights(plan, items)

    def proj(sd):
        return br.projected(plan, items, cm, sd, wt)
    droppable = br.droppable(plan, models, cm)
    lim = 0.8 * sp['max_usd']
    sd = {'dropped_blocks': [], 'dropped_models': []}
    nominal = proj(sd)
    if nominal[0] > lim:
        sd['dropped_blocks'] = br.last_block(plan)
        for m in droppable:
            if proj(sd)[0] <= lim:
                break
            sd['dropped_models'].append(m)
    smallest = proj({'dropped_blocks': br.last_block(plan),
                     'dropped_models': droppable})
    return {'per_call': cm, 'limit_usd': lim, 'nominal': nominal,
            'scale_down': sd, 'after': proj(sd), 'smallest': smallest}


def verify(plan_path, allow_dirty=False, check_tree=True):
    """Everything the run will check before its first call, and more,
    offline: the lock and every body (battery_items.verify), the run's own
    refusal checks (battery_run.problems), the floor counts, the tree
    (battery_run.tree_problems: the plan committed in this repository,
    nothing uncommitted) and the catalogue-price bound. Exit 0 when nothing
    is wrong, else 5. check_tree=False: tests on a scratch plan only."""
    plan_path = os.path.abspath(plan_path)
    probs = [f'[items] {p}' for p in bi.verify(plan_path)]
    try:
        plan, items, d = br.read_plan(plan_path)
    except (OSError, ValueError, KeyError) as ex:
        for p in probs + [f'cannot read {_rel(plan_path)}: '
                          f'{type(ex).__name__}']:
            say(p)
        return REFUSED
    probs += [f'[run] {p}' for p in br.problems(plan_path, plan, items, d)]
    models = br.models_of(plan)
    nominal = _per_model(plan, items, {})
    floor = _per_model(plan, items, {'dropped_blocks': br.last_block(plan)})
    want = (plan.get('calls_per_model') or {}).get('floor')
    if want is not None and floor != want:
        probs.append(f'[run] floor calls per model {floor} != plan {want}')
    pr = projection(plan, items)
    if pr['smallest'][0] > plan['spend']['max_usd']:
        probs.append(f'even the smallest run (blocks '
                     f'{br.last_block(plan)} and every unscored model '
                     f'dropped) projects '
                     f'${pr["smallest"][0]:.2f} > max_usd '
                     f'{plan["spend"]["max_usd"]} at catalogue prices')
    dirty = br.tree_problems(plan_path, plan, d) if check_tree else []
    note = None
    if dirty:
        msg = '; '.join(dirty) + ' -- run will refuse'
        if allow_dirty:
            note = msg
        else:
            probs.append(msg)
    say(f'plan {_rel(plan_path)} sha256 {br._sha(plan_path)[:16]}')
    say(f'{len(models)} models; per model {sum(nominal.values())} calls '
        f'nominal, {sum(floor.values())} at the floor; '
        f'{len(br.planned(plan, items, {}))} calls in all')
    sd = pr['scale_down']
    say(f'catalogue-pessimistic cost ${pr["nominal"][0]:.2f} for '
        f'{pr["nominal"][1]} calls (scale-down above '
        f'${pr["limit_usd"]:.2f}: blocks {sd["dropped_blocks"]}, models '
        f'{sd["dropped_models"]} -> ${pr["after"][0]:.2f} for '
        f'{pr["after"][1]} calls); the pilot decides from measured costs')
    tree = 'not checked' if not check_tree else 'dirty' if dirty else 'clean'
    say(f'tree: {tree}' + (f' (allowed; {note})' if note else ''))
    for p in probs:
        say(p)
    say('ok' if not probs else f'{len(probs)} problem(s)')
    return DONE if not probs else REFUSED


# ------------------------------------------------------------------ score --

def score(run_dir):
    try:
        s = bs.score(run_dir)
    except (OSError, ValueError, KeyError) as ex:
        return _refuse(f'score: {type(ex).__name__}: {ex}')
    a = s['analysis_set']
    say(f'analysis set: blocks {a["blocks"]}, {len(a["states"])} states'
        + ('' if a['sufficient'] else
           f' -- fewer than {a["blocks_min"]} blocks: select will exit 6, '
           f'no selection from this run'))
    say('baselines: ' + ', '.join(f'{k} {v:.4f}' if v is not None else
                                  f'{k} n/a'
                                  for k, v in s['baselines'].items()))
    for m, r in s['models'].items():
        q = r['headline']['Q_h']
        say(f'  {m}: h={r["h"]} Q_h={"n/a" if q is None else f"{q:.4f}"} '
            f'failed={",".join(r["failed"]) or "-"}')
    say(f'-> {_rel(os.path.join(run_dir, "summary.json"))}')
    return DONE


# ----------------------------------------------------------------- select --

def select(run_dir):
    try:
        o = bs.select(run_dir)
    except bs.InsufficientAnalysisSet:
        return INSUFFICIENT             # select has said why, on stderr
    except (OSError, ValueError, KeyError) as ex:
        return _refuse(f'select: {type(ex).__name__}: {ex}')
    if o.get('mode') == 'iterate':
        f = o['final']
        say(f'adopted: {o.get("adopted") or "nothing"}; the decider stands '
            f'at {f["tag"]} (suffix {f["suffix"] or "none"}); next: '
            f'{o["next"]}')
        name = 'iter_holdout_outcome.json' if o['holdout'] else \
            f'iter_r{o["round"]}_outcome.json'
        say(f'-> {_rel(os.path.join(run_dir, name))}')
        return DONE
    dec, ref = o['decider'], o['reference']
    say(f'eligible: {", ".join(o["eligible"]) or "none"}')
    say(f'branch {o["branch"]}: decider {dec["pin"]} ({dec["handling"]}'
        f'{", suffix " + dec["suffix"] if dec["suffix"] else ""}, provider '
        f'pin {"none" if not dec["provider_pin"] else dec["provider"]}); '
        f'reference {ref["id"]} {ref["status"]}')
    out = f'selection_{o.get("version") or "v1"}.json'
    say(f'-> {_rel(os.path.join(run_dir, out))}')
    return DONE


def scrub_check(run_dir):
    if not os.path.isdir(run_dir):
        return _refuse(f'no run directory {_rel(run_dir)}')
    hits = br.scrub_check(run_dir)
    for h in hits:
        say(f'scrub hit: {h}')
    say('clean' if not hits else f'{len(hits)} scrub hit(s)')
    return SCRUB_HIT if hits else DONE


# ------------------------------------------------------------------- main --

def parser():
    ap = argparse.ArgumentParser(
        prog='decider_battery.py',
        description='The decider battery (RESULTS 27): make-items, '
                    'make-iter, verify, run, score, select, scrub-check.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make-items', help='generate and lock the stimuli '
                       '(designing session only; offline)')
    p.add_argument('--plan', required=True)
    p.add_argument('--force', action='store_true',
                   help='replace a committed lock whose bytes differ')
    p = sub.add_parser('make-iter', help='an iteration round or the holdout '
                       '(spec section 4): plan, states, items, lock; offline')
    p.add_argument('--selection', required=True,
                   help='selection_v1.json (round 1), or the previous '
                        'round\'s iter_r{r}_outcome.json')
    p.add_argument('--round', required=True, choices=['1', '2', 'holdout'])
    p.add_argument('--variants', default='',
                   help='menu tags on top of the current version, comma-'
                        'separated: ' + ','.join(bi.MENU))
    p.add_argument('--out', required=True,
                   help='the declaration\'s directory, e.g. '
                        'bench/battery/iter_r1')
    p.add_argument('--force', action='store_true',
                   help='replace a declaration whose bytes differ')
    p = sub.add_parser('verify', help='offline checks before a run')
    p.add_argument('--plan', required=True)
    p.add_argument('--allow-dirty', action='store_true',
                   help='report uncommitted changes without failing '
                        '(the run itself still refuses them)')
    p = sub.add_parser('run', help='send every item to every model')
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--auth', choices=['proxy', 'env'], default='proxy')
    p.add_argument('--max-usd', type=float)
    p.add_argument('--deadline-min', type=float)
    p.add_argument('--concurrency', type=int)
    p.add_argument('--per-model', type=int)
    for name, what in (('score', 'metrics and gates -> summary.json'),
                       ('select', 'section 3 -> selection_v1.json'),
                       ('scrub-check', 'no key-shaped strings in outputs')):
        p = sub.add_parser(name, help=what)
        p.add_argument('--run', required=True)
    return ap


def main(argv=None, opener=urllib.request.urlopen, sleep=time.sleep,
         now=time.monotonic, check_tree=True):
    """Exit code of one subcommand. opener, sleep and now reach the run
    unchanged: tests pass a stand-in API and a fake clock. check_tree=False
    (tests on a scratch plan only; no argument sets it) skips verify's and
    run's checks that the plan and the code are the committed ones."""
    a = parser().parse_args(argv)
    if a.cmd == 'make-items':
        return make_items(a.plan, a.force)
    if a.cmd == 'make-iter':
        return make_iter(a.selection, a.round, a.variants, a.out, a.force)
    if a.cmd == 'verify':
        return verify(a.plan, a.allow_dirty, check_tree)
    if a.cmd == 'run':
        try:
            return br.run(a.plan, a.out, auth=a.auth, max_usd=a.max_usd,
                          deadline_min=a.deadline_min,
                          concurrency=a.concurrency, per_model=a.per_model,
                          opener=opener, sleep=sleep, now=now,
                          check_tree=check_tree)
        except (OSError, ValueError, KeyError) as ex:
            return _refuse(f'run: {type(ex).__name__}: {ex}')
    if a.cmd == 'score':
        return score(a.run)
    if a.cmd == 'select':
        return select(a.run)
    return scrub_check(a.run)


if __name__ == '__main__':
    sys.exit(main())
