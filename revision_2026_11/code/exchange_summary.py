"""Ranging exchanges attempted and failed in each run, from the diagnostic counters of the tags (failure_stage_diagnostics.json).
The counters are cumulative since the last reset of a tag. The count of a run is the difference to the previous record of the same link and mode.
If the counter is lower than in the previous record, the tag was reset in between and the cumulative value is taken (the mode was not used between the reset and the run; checked below against the other tag).
failed = attempts - success. Output: exchange_failures_runs.csv, exchange_failures_by_policy.csv"""
import json, numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; RC = BASE / 'analysis' / 'revision_checks'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run'])
rows = []
for d in sorted(p for p in DAY.iterdir() if p.is_dir()):
    f = d / 'tables' / 'failure_stage_diagnostics.json'
    if not f.exists(): continue
    J = json.loads(f.read_text(encoding='utf-8'))
    for tag, items in J.items():
        for it in items:
            if not isinstance(it, dict) or 'attempts' not in it: continue
            rows.append(dict(run=d.name, tag=it['tag_id'], anchor=it['anchor_id'], link=f"{it['tag_id']}_{it['anchor_id']}", mode=it['mode'], attempts=it['attempts'], success=it['success'],
                             timeouts=it.get('response_timeout', 0) + it.get('report_timeout', 0), rx_err=it.get('response_rx_error', 0) + it.get('report_rx_error', 0)))
D = pd.DataFrame(rows).sort_values('run', kind='stable')
D[['d_att', 'd_suc']] = D.groupby(['link', 'mode'])[['attempts', 'success']].diff()
reset = D.d_att.isna() | (D.d_att < 0)
D['counter_restarted'] = reset
D.loc[reset, 'd_att'] = D.loc[reset, 'attempts']; D.loc[reset, 'd_suc'] = D.loc[reset, 'success']
D = D.merge(R, on='run', how='inner'); D = D[D.d_att > 0].copy(); D['failed'] = D.d_att - D.d_suc
MODES = {'FIXED_SS': {'SS'}, 'FIXED_DS': {'DS'}, 'FIXED_PHASE': {'PHASE'}, 'STAGED': {'SS', 'DS', 'PHASE'}}
bad = D[[m not in MODES[p] for m, p in zip(D['mode'], D.policy)]]
assert bad.empty, bad
# consistency of the restarted counters with the other tag in the same run and mode
for (run, mode), g in D.groupby(['run', 'mode']):
    assert g.d_att.max() - g.d_att.min() <= 4, (run, mode, g[['link', 'd_att', 'counter_restarted']])
    assert len(g) == 4, (run, mode, len(g))
print('restarted counters used in:', D[D.counter_restarted].groupby('trial').size().to_dict())
T = D.groupby(['trial', 'path', 'policy']).agg(attempts=('d_att', 'sum'), failed=('failed', 'sum'), links_modes=('link', 'size')).reset_index()
T['failed_pct'] = 100 * T.failed / T.attempts
worst = D.sort_values('failed', ascending=False).groupby('trial').head(2).groupby('trial').apply(lambda g: ', '.join(f"{l} {m} {int(x)}" for l, m, x in zip(g.link, g['mode'], g.failed) if x > 0))
T['links_with_most_failures'] = T.trial.map(worst)
T[['attempts', 'failed']] = T[['attempts', 'failed']].astype(int)
T.round(4).to_csv(RC / 'exchange_failures_runs.csv', index=False, encoding='utf-8-sig')
G = T.groupby(['path', 'policy']).agg(runs=('trial', 'size'), attempts=('attempts', 'sum'), failed=('failed', 'sum'), runs_with_failures=('failed', lambda s: int((s > 0).sum())), median_run_pct=('failed_pct', 'median'), max_run_pct=('failed_pct', 'max')).reset_index()
G['failed_pct'] = 100 * G.failed / G.attempts
G.round(4).to_csv(RC / 'exchange_failures_by_policy.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 240); pd.set_option('display.max_rows', 100)
print(G.round(3).to_string()); print(T[T.failed > 0].round(3).to_string())
s = T[T.policy == 'STAGED']; print('switching runs', len(s), 'attempts', s.attempts.sum(), 'failed', s.failed.sum(), 'max run pct %.3f' % s.failed_pct.max())
a = D[D.failed > 0].groupby(['trial', 'anchor']).failed.sum().unstack(fill_value=0); print(a.to_string())

