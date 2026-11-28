"""Frame-time budget from DW3000 device timestamps (Response-RX time of each link, tag clock).
Intervals: same-tag (A1 -> A2 link of one tag), frame period (same link in consecutive frames),
cross-tag (derived: frame period minus both same-tag intervals, shared by the two hand-overs)."""
import io
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run'])
TICK = 1.0 / (499.2e6 * 128); WRAP = 2 ** 40
LINK_MS = {'SS': 1.19, 'DS': 4.05, 'PHASE': 8.61}           # calculated link times of manuscript Table 2
MODE = {'FIXED_SS': 'SS', 'FIXED_DS': 'DS', 'FIXED_PHASE': 'PHASE'}
rows = []
for r in R[R.policy.isin(MODE)].itertuples():
    L = pd.read_csv(DAY / r.run / 'tables' / 'links.csv', usecols=['frame_id', 'link_id', 'device_timestamp_ticks', 'success'], on_bad_lines='skip')
    L = L[L.success.astype(str).isin(['1', 'True', 'true'])].drop_duplicates(['frame_id', 'link_id'], keep='last')
    W = L.pivot(index='frame_id', columns='link_id', values='device_timestamp_ticks').dropna().sort_index()
    same = {}
    for tag in ('T1', 'T2'):
        same[tag] = ((W[f'{tag}_A2'] - W[f'{tag}_A1']) % WRAP) * TICK * 1e3
    consecutive = np.diff(W.index.to_numpy()) == 1
    period = (np.diff(W['T1_A1'].to_numpy()) % WRAP) * TICK * 1e3
    period = period[consecutive]
    ok = (period < 200)
    m = MODE[r.policy]
    P = np.median(period[ok]); s1 = same['T1'].median(); s2 = same['T2'].median()
    cross = (P - s1 - s2) / 2
    rows.append(dict(trial=r.trial, path=r.path, mode=m, frames=int(ok.sum()), period_ms=P, same_T1_ms=s1, same_T2_ms=s2, cross_mean_ms=cross,
                     gap_same_ms=(s1 + s2) / 2 - LINK_MS[m], gap_cross_ms=cross - LINK_MS[m], overhead_ms=P - 4 * LINK_MS[m],
                     period_p05=np.percentile(period[ok], 5), period_p95=np.percentile(period[ok], 95)))
D = pd.DataFrame(rows)
D.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'frame_time_budget.csv', index=False)
pd.set_option('display.width', 220)
print(D.groupby('mode')[['period_ms', 'same_T1_ms', 'same_T2_ms', 'cross_mean_ms', 'gap_same_ms', 'gap_cross_ms', 'overhead_ms', 'period_p05', 'period_p95']].median().round(3).to_string())
print(D.groupby('mode')[['gap_same_ms', 'gap_cross_ms', 'overhead_ms']].agg(['min', 'max']).round(3).to_string())
print(D.groupby('mode').trial.count().to_dict())

