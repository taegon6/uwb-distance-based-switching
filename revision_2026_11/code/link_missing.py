import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
tot = []
for t in R.index:
    pf = EV / f'pairs_{t}.csv'
    if not pf.exists(): continue
    Fr = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'frames.csv', usecols=['frame_id', 'mode', 'L1_present', 'L2_present', 'L3_present', 'L4_present', 'complete', 'frame_start_host_monotonic_ns'])
    L = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['frame_id', 'link_id', 'success'], on_bad_lines='skip')
    P = pd.read_csv(pf, usecols=['frame_id', 't', 'seg']); C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    P['psi'] = np.interp(P.t, C.t, C.psi); P = P.sort_values('frame_id')
    # heading and segment of every frame id by interpolation over the paired frames (frames without a pose have no pair)
    Fr = Fr[(Fr.frame_id >= P.frame_id.min()) & (Fr.frame_id <= P.frame_id.max())].copy()
    Fr['psi'] = np.interp(Fr.frame_id, P.frame_id, P.psi)
    idx = np.clip(np.searchsorted(P.frame_id.to_numpy(), Fr.frame_id.to_numpy()), 0, len(P) - 1); Fr['seg'] = P.seg.to_numpy()[idx]
    Fr['trial'] = t; Fr['path'] = R.loc[t, 'path']; Fr['policy'] = R.loc[t, 'policy']
    Fr['succ_rows_not1'] = (~L.success.astype(str).isin(['1', 'True', 'true'])).sum()
    tot.append(Fr)
F = pd.concat(tot)
print('link rows with success != 1 per run (max over runs):', F.groupby('trial').succ_rows_not1.first().max())
F = F[F.seg.isin(['far', 'mid'])]
F['hbin'] = pd.cut(F.psi.abs(), [0, 15, 30, 45, 100])
cols = ['L1_present', 'L2_present', 'L3_present', 'L4_present']
for path in ('S', 'C1', 'C2'):
    g = F[F.path == path]
    tab = g.groupby('hbin', observed=True)[cols].agg(lambda x: 100 * (1 - x.mean())); tab['frames'] = g.groupby('hbin', observed=True).size(); tab['incomplete_%'] = g.groupby('hbin', observed=True).complete.agg(lambda x: 100 * (1 - x.mean()))
    print('== missing links (% of frames), L1..L4 = T1_A1, T1_A2, T2_A1, T2_A2,', path); print(tab.round(2).to_string())
print(F[(F.path != 'S')].groupby(['policy', 'trial'])[cols + ['complete']].agg(lambda x: 100 * (1 - x.mean())).round(2).query('complete > 0.5').to_string())

