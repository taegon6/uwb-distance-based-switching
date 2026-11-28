"""Figure for cam/speed_replay.py: far-segment lateral error at the RX coil against the replay speed."""
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE = r'D:/sandbox/uwb/output/field_20261124'
R = pd.read_csv(BASE + '/analysis/revision_checks/speed_replay_runs.csv'); R = R[R.speed > 0]
S = pd.read_csv(BASE + '/analysis/revision_checks/speed_replay_summary.csv')
print(S[S.speed.isin([-1.0, 1.0, 1.4, 2.0, 2.8, 4.2])][['speed', 'rx_SS', 'rx_DS', 'rx_HP', 'rx_SW', 'rx_SW-DS', 'rx_SW-DS_ci', 'rx_SW-DS_neg', 'rxq_SW-DS', 'rxq_SW-DS_ci', 'rxq_SW-DS_neg', 'tag_SW-DS', 'tag_SW-DS_ci', 'tagq_SW-DS', 'tagq_SW-DS_ci']].round(1).to_string(index=False))
LAB = {'FIXED_SS': 'Fixed SS-TWR', 'FIXED_DS': 'Fixed DS-TWR', 'FIXED_PHASE': 'Fixed high-precision DS-TWR', 'STAGED': 'Distance-based switching'}
COL = {'FIXED_SS': '#1f5fbf', 'FIXED_DS': '#e06c1b', 'FIXED_PHASE': '#2e8b3a', 'STAGED': 'k'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9.5, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for p in LAB:
    g = R[R.policy == p].groupby('speed').rx_lat_rmse.agg(['mean', 'min', 'max'])
    ax[0].plot(3.6 * g.index, g['mean'], color=COL[p], label=LAB[p], ls='--' if p == 'STAGED' else '-'); ax[0].fill_between(3.6 * g.index, g['min'], g['max'], color=COL[p], alpha=0.12, lw=0)
g = S[S.speed > 0]
for key, c, lab in (('rx_SW-DS', 'k', 'Recorded error plus calculated shift'), ('rxq_SW-DS', '0.45', 'Shift added in quadrature')):
    ax[1].plot(3.6 * g.speed, g[key], color=c, label=lab); ax[1].fill_between(3.6 * g.speed, g[key] - g[key + '_ci'], g[key] + g[key + '_ci'], color=c, alpha=0.15, lw=0)
ax[1].axhline(0, color='0.3', lw=0.8)
for a in ax: a.axvspan(2.5, 5.0, color='0.6', alpha=0.25, lw=0); a.set_xlabel('Replay speed [km/h]'); a.grid(alpha=.3)
ax[0].text(2.6, 0.97, 'speeds driven\nin this study', transform=ax[0].get_xaxis_transform(), fontsize=7.5, color='0.25', va='top')
ax[0].set_ylabel('Lateral RMSE at the RX coil, 4–10.5 m [cm]'); ax[0].set_title('(a) Recorded runs replayed at a constant speed', fontsize=9.5); ax[0].legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(0.30, 1.0))
ax[1].set_ylabel('Switching minus fixed DS-TWR [cm]'); ax[1].set_title('(b) Paired difference over five blocks (95% interval)', fontsize=9.5); ax[1].legend(fontsize=7.5, loc='lower left')
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(BASE + f'/fig/speed_replay.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)
print('saved')

