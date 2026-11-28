import pandas as pd, numpy as np
from pathlib import Path
B = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = B / 'results_20261124'; EV = B / 'cam' / 'eval'; RC = B / 'analysis' / 'revision_checks'
R = pd.read_csv(B / 'analysis' / 'runs_summary_dedup.csv').set_index('trial'); T = pd.read_csv(EV / 'segment_times.csv').set_index('trial')
rows = []
for t, r in R.iterrows():
    P = pd.read_csv(DAY / r.run / 'tables' / 'poses.csv', usecols=['host_pose_wall_ns', 'valid', 'status', 'failure_reason'])
    P = P[~P.status.astype(str).str.contains('INITIALIZING')]
    v = P.valid.astype(str).str.lower().isin(['true', '1']).to_numpy(); w = P.host_pose_wall_ns.to_numpy() / 1e9
    d = dict(trial=t, path=r.path, policy=r.policy, all_pct=100 * v.mean(), n=len(P))
    if t in T.index:
        s = T.loc[t]; ap = (w >= s.t_motion) & (w < s.t_mid_end); st = w >= s.t_mid_end
        d.update(approach_pct=100 * v[ap].mean() if ap.any() else np.nan, approach_invalid=int((~v[ap]).sum()), stop_pct=100 * v[st].mean() if st.any() else np.nan,
                 top_reason=P[~v].failure_reason.astype(str).value_counts().index[0] if (~v).any() else '')
    rows.append(d)
D = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(D[(D.all_pct < 98) | (D.approach_pct < 98)].round(1).to_string(index=False))
print('approach share: median %.1f, min %.1f; runs below 99%%: %d' % (D.approach_pct.median(), D.approach_pct.min(), (D.approach_pct < 99).sum()))
print(D.groupby(['path', 'policy'])[['all_pct', 'approach_pct', 'stop_pct']].median().round(1).to_string())
D.round(2).to_csv(RC / 'valid_share_by_phase.csv', index=False)

