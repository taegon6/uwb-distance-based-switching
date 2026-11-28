"""Sequential-link model for 2A2T: bias caused by vehicle motion inside one frame, against speed.
Straight approach on the centerline, measured link timing of each mode, no ranging noise."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read().split('# ---------------- main ----------------')[0])
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUTR = BASE / 'analysis' / 'revision_checks'
B = pd.read_csv(OUTR / 'frame_time_budget.csv').groupby('mode')[['period_ms', 'same_T1_ms', 'cross_mean_ms']].median()
LAB = {'SS': 'SS-TWR', 'DS': 'DS-TWR', 'PHASE': 'High-precision DS-TWR'}; COL = {'SS': '#1f5fbf', 'DS': '#e06c1b', 'PHASE': '#2e8b3a'}
rows = []
for m in LAB:
    same, cross = B.loc[m, 'same_T1_ms'] / 1e3, B.loc[m, 'cross_mean_ms'] / 1e3
    link_time = np.array([0.0, same, same + cross, 2 * same + cross]); link_time -= link_time.mean()
    for v in np.arange(0.0, 4.21, 0.1):
        for dist in (4.0, 7.0, 10.0):                       # coil distance; tag midpoint is 2.2 m closer to the anchors
            y0 = dist - 2.2
            ranges = np.zeros((1, 4))
            for k, l in enumerate(LINKS):
                tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]
                y = y0 - v * link_time[k]                    # the vehicle drives toward -y
                ranges[0, k] = math.sqrt((b[0] - A[0]) ** 2 + (y - A[1]) ** 2 + (A[2] - TZ) ** 2)
            sol, _ = resolve(ranges, np.array([[0.0, y0, 0.0]]))
            yaw = sol[0, 2]; rx_x = sol[0, 0] - 2.2 * math.sin(yaw)
            rows.append(dict(mode=m, speed_mps=round(v, 2), speed_kmh=round(3.6 * v, 2), coil_distance_m=dist, tag_lat_cm=100 * sol[0, 0], heading_deg=math.degrees(yaw), rx_lat_cm=100 * rx_x,
                             travel_per_update_cm=100 * v * B.loc[m, 'period_ms'] / 1e3))
D = pd.DataFrame(rows); D.round(3).to_csv(OUTR / 'sequential_link_model.csv', index=False)
pd.set_option('display.width', 200)
show = D[D.speed_mps.isin([1.0, 1.4, 2.8, 4.2]) & (D.coil_distance_m == 7.0)]
print(show.round(2).to_string(index=False))
print(D[(D.speed_mps == 2.8)].round(2).to_string(index=False))
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9.5, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for m in LAB:
    g = D[(D['mode'] == m) & (D.coil_distance_m == 7.0)]
    ax[0].plot(g.speed_kmh, g.heading_deg.abs(), color=COL[m], label=LAB[m])
    ax[1].plot(g.speed_kmh, g.rx_lat_cm.abs(), color=COL[m], label=LAB[m])
    lo = D[(D['mode'] == m) & (D.coil_distance_m == 4.0)]; hi = D[(D['mode'] == m) & (D.coil_distance_m == 10.0)]
    ax[1].fill_between(g.speed_kmh, lo.rx_lat_cm.abs().to_numpy(), hi.rx_lat_cm.abs().to_numpy(), color=COL[m], alpha=0.15, lw=0)
for a in ax: a.axvspan(2.5, 5.0, color='0.6', alpha=0.25, lw=0)   # 95th percentile of the speed in the runs: 0.7-1.4 m/s
ax[0].text(2.6, 0.93, 'speeds driven\nin this study', transform=ax[0].get_xaxis_transform(), fontsize=7.5, color='0.25', va='top')
ax[1].axhline(10, color='crimson', ls=':', lw=1); ax[1].text(0.15, 10.6, 'SAE J2954\nlateral tolerance', fontsize=7.5, color='crimson', va='bottom', linespacing=1.1)
ax[0].set_xlabel('Vehicle speed [km/h]'); ax[0].set_ylabel('Heading bias [°]'); ax[0].set_title('(a) Heading bias from sequential links', fontsize=9.5); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[1].set_xlabel('Vehicle speed [km/h]'); ax[1].set_ylabel('Lateral bias at the RX coil [cm]'); ax[1].set_title('(b) RX coil, coil distance 7 m (band: 4 to 10 m)', fontsize=9.5); ax[1].grid(alpha=.3)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(BASE / 'fig' / f'sequential_link_model.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)

