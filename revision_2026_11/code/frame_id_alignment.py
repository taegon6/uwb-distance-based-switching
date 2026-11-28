
import sys
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r'D:/sandbox/uwb/output/exp_ready_20261124/repo/tools')
from measurement_time import tag_measure_times
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
rows = []
for t in R.index[R.policy.str.startswith('FIXED')]:
    L = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['frame_id', 'tag_id', 'link_id', 'host_monotonic_ns', 'device_timestamp_ticks', 'success'], on_bad_lines='skip')
    L = L[L.device_timestamp_ticks.notna() & L.success.astype(str).isin(['1', 'True', 'true'])].copy()
    L['t_meas'] = pd.concat([tag_measure_times(g) for _, g in L.groupby('tag_id')])
    W = L.drop_duplicates(['frame_id', 'link_id'], keep='last').pivot(index='frame_id', columns='link_id', values='t_meas').dropna()
    d = (W['T2_A1'] - W['T1_A1']) / 1e6; s = (W['T1_A2'] - W['T1_A1']) / 1e6
    rows.append(dict(trial=t, policy=R.loc[t, 'policy'], frames=len(W), T2A1_minus_T1A1_median=d.median(), p05=d.quantile(.05), p95=d.quantile(.95), T1A2_minus_T1A1=s.median()))
D = pd.DataFrame(rows); pd.set_option('display.width', 200)
print(D.round(2).to_string(index=False))
print(D.groupby('policy')[['T2A1_minus_T1A1_median', 'T1A2_minus_T1A1']].agg(['median', 'min', 'max']).round(2).to_string())
D.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'frame_id_alignment.csv', index=False)

