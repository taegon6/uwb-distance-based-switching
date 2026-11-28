import csv, re
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'
runs = {r['trial']: r for r in csv.DictReader(open(BASE / 'analysis' / 'runs_summary_dedup.csv', encoding='utf-8-sig'))}
out = []
for t, r in runs.items():
    f = DAY / r['run'] / 'tables' / 'runtime_status.csv'
    ev = list(csv.DictReader(open(f, encoding='utf-8')))
    last_commit = None; mode = None; t0 = int(ev[0]['host_monotonic_ns'])
    for e in ev:
        ts = (int(e['host_monotonic_ns']) - t0) / 1e9
        if 'MODE_COMMITTED' in e['event']:
            last_commit = ts; mode = re.search(r'mode=(\w+)', e['detail']).group(1); n_after = 0
        elif 'WATCHDOG' in e['event'] or 'STALL' in e['event']:
            out.append((t, r['path'], r['policy'], e['event'], mode, round(ts, 2), round(ts - last_commit, 2)))
import collections
print(len(out), 'events in', len({o[0] for o in out}), 'runs')
for o in out: print(o)
by = collections.Counter((o[2] == 'STAGED', o[4], 'within 3 s of commit' if o[6] < 3 else 'later') for o in out)
for k, v in sorted(by.items(), key=str): print(k, v)


