import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = Path(r'D:/sandbox/uwb/output/field_20261124/cam/eval')
fig, ax = plt.subplots(1, 3, figsize=(15, 5.5))
for k, (t, title) in enumerate([('T10', 'S T10 STAGED'), ('T28', 'C1 T28 STAGED'), ('T50', 'C2 T50 STAGED')]):
    p = pd.read_csv(E / f'pairs_{t}.csv')
    a = ax[k]
    for m, col in (('SS', 'tab:orange'), ('DS', 'tab:blue'), ('PHASE', 'tab:green')):
        g = p[p['mode'] == m]
        a.plot(g.tag_midpoint_x_m, g.tag_midpoint_y_m, '.', ms=2, color=col, label=f'UWB {m}')
    a.plot(p.cam_X, p.cam_Y, 'k-', lw=1.2, label='camera')
    a.set_title(title); a.set_xlabel('x (m, code frame)'); a.set_ylabel('y (m)'); a.axis('equal'); a.grid(alpha=.3)
    a.axhline(-2.2, color='gray', lw=.6)
ax[0].legend(markerscale=4, fontsize=8)
fig.tight_layout(); fig.savefig(E / 'example_trajectories.png', dpi=110)
S = pd.read_csv(E / 'segment_errors.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k, path in enumerate(('S', 'C1')):
    g = S[S.path == path]; pols = [p for p in ['FIXED_SS', 'FIXED_DS', 'FIXED_PHASE', 'STAGED'] if p in set(g.policy)]
    segs = ['far', 'mid', 'near']; w = 0.8 / len(pols)
    for i, p in enumerate(pols):
        v = [g[(g.policy == p) & (g.seg == s)].log_lat_rmse.median() for s in segs]
        ax[k].bar(np.arange(3) + i * w, v, w, label=p)
    ax[k].set_xticks(np.arange(3) + 0.4 - w / 2); ax[k].set_xticklabels(['far 4-10.5 m', 'mid 0.5-4 m', 'near <0.5 m'])
    ax[k].set_ylabel('lateral RMSE vs camera (cm, median of runs)'); ax[k].set_title(path); ax[k].grid(axis='y', alpha=.3); ax[k].legend(fontsize=8)
fig.tight_layout(); fig.savefig(E / 'lateral_rmse_by_policy.png', dpi=110)
print('ok')

