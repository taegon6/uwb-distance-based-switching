import json, numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run'])
rows = []
for d in sorted(p for p in DAY.iterdir() if p.is_dir()):
    f = d / 'tables' / 'failure_stage_diagnostics.json'
    if not f.exists(): continue
    J = json.loads(f.read_text(encoding='utf-8'))
    for tag, items in J.items():
        for it in items:
            if not isinstance(it, dict) or 'attempts' not in it: continue
            rows.append(dict(run=d.name, link=f"{it['tag_id']}_{it['anchor_id']}", mode=it['mode'], attempts=it['attempts'], success=it['success'],
                             timeouts=it.get('response_timeout', 0) + it.get('report_timeout', 0), rx_err=it.get('response_rx_error', 0) + it.get('report_rx_error', 0)))
D = pd.DataFrame(rows).sort_values('run')
print('runs with diagnostics', D.run.nunique(), 'rows', len(D)); print(D.groupby('mode').size().to_dict())
D[['d_att', 'd_to', 'd_err']] = D.groupby(['link', 'mode'])[['attempts', 'timeouts', 'rx_err']].diff()
D = D.merge(R, on='run', how='left')
ok = D[(D.d_att > 50)]            # counters that advanced in this run (negative or zero = reset or mode not used)
ok = ok[ok.trial.notna()]
ok['fail_pct'] = 100 * (ok.d_to + ok.d_err) / ok.d_att
pd.set_option('display.width', 220)
print('== exchange failures per run from the tag diagnostic counters (% of attempts, median over runs [max])')
print(ok.groupby(['path', 'mode', 'link']).fail_pct.agg(['median', 'max', 'size']).round(2).unstack('link').to_string())
print(ok[ok.fail_pct > 1].sort_values('fail_pct', ascending=False)[['trial', 'path', 'policy', 'mode', 'link', 'd_att', 'd_to', 'd_err', 'fail_pct']].head(20).round(1).to_string())
ok.to_csv(BASE / 'analysis' / 'revision_checks' / 'link_exchange_failures_by_run.csv', index=False)

