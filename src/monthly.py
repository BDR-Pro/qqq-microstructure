# Part of qqq-microstructure.
#
# The whole forward-test ritual as one command. The strategy is monthly by
# construction, so this is the entire operating burden: run it in the first
# days of each month (or let Task Scheduler do it), commit reports/, done.
#
# Steps, each continuing past failure with a summary at the end:
#   1. xsec_extend        build newly completed panel months from Yahoo
#   2. xsec_replay        frozen-model verdict on everything past the cutoff
#   3. xsec_live          pre-register this month's basket, grade elapsed days
#   4. opra_pull --yes    extend the options record (~$0.50/month; skipped
#      + opra_value       unless DATABENTO_API_KEY is set)
#
#   python src/monthly.py
#
# Hands-free (Windows, machine must be on; run once to register):
#   schtasks /Create /TN qqq-monthly /SC MONTHLY /D 1 /ST 18:00 /TR ^
#     "cmd /c cd /d C:\Users\bader\qqq-microstructure && python src\monthly.py >> reports\monthly.log 2>&1"

import os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, 'src')


def run(name, args):
    print(f'\n{"=" * 20} {name} {"=" * 20}', flush=True)
    rc = subprocess.call([sys.executable, os.path.join(S, args[0])] + args[1:],
                         cwd=ROOT)
    return name, rc


def main():
    steps = [('extend', ['xsec_extend.py']),
             ('replay', ['xsec_replay.py']),
             ('live', ['xsec_live.py'])]
    if os.environ.get('DATABENTO_API_KEY'):
        steps += [('opra_pull', ['opra_pull.py', '--yes']),
                  ('opra_value', ['opra_value.py'])]
    else:
        print('DATABENTO_API_KEY not set -- options record skipped this month')
    results = [run(n, a) for n, a in steps]
    print(f'\n{"=" * 20} summary {"=" * 20}')
    for n, rc in results:
        print(f'  {n:<12} {"ok" if rc == 0 else f"FAILED (rc={rc})"}')
    print('\nnow: git add reports data && git commit -m "monthly forward log" '
          '&& git push\n(reports/xsec_paper*.csv are the evidence; '
          'data/opra_daily.parquet stays local)')


if __name__ == '__main__':
    main()
