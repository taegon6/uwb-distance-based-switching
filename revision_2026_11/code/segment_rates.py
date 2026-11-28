
import pandas as pd, numpy as np
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124')
R = pd.read_csv(BASE/'analysis'/'runs_summary_dedup.csv', usecols=['trial','path','policy','run'])
rows = []
for _, r in R.iterrows():
    p = BASE/'results_20261124'/r.run/'tables'/'poses.csv'
    P = pd.read_csv(p, usecols=['host_pose_monotonic_ns','coil_to_tx_distance_m','valid'])
    P = P[P.valid.astype(str).str.lower().isin(['true','1'])]
    t = P.host_pose_monotonic_ns.to_numpy(float)/1e9; d = P.coil_to_tx_distance_m.to_numpy(float)
    for name, lo, hi in (('far', 4.0, 10.5), ('mid', 0.5, 4.0)):
        m = (d >= lo) & (d < hi)
        if m.sum() < 10: continue
        span = t[m].max() - t[m].min()
        rows.append(dict(trial=r.trial, path=r.path, policy=r.policy, seg=name, n_valid=int(m.sum()), span_s=round(span,3), rate_hz=round(m.sum()/span,2)))
S = pd.DataFrame(rows)
S.to_csv(BASE/'analysis'/'revision_checks'/'segment_update_rates.csv', index=False)
print(S.groupby(['path','seg','policy']).rate_hz.median().unstack().round(1).to_string())
print(S.groupby(['path','seg','policy']).span_s.median().unstack().round(1).to_string())

