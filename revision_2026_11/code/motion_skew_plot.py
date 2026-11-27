"""Heading and lateral bias against speed on the fixed straight runs: observed (UWB - camera) and predicted
from the sequential link timing. Writes motion_skew_by_speed.csv and fig/motion_skew.png."""
import math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'; OUTR = BASE / 'analysis' / 'revision_checks'
D = pd.read_csv(OUTR / 'motion_skew.csv')
rows = []
for t, g in D.groupby('trial'):
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['t', 'yaw_deg', 'rx_coil_x_m'])
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    g = g.merge(P, on='t', how='left')
    g['cam_psi'] = np.interp(g.t, C.t, C.psi); g['cam_rx_x'] = np.interp(g.t, C.t, C.RX_X)
    g['obs_yaw'] = (g.yaw_deg - g.cam_psi + 180) % 360 - 180
    g['obs_rx_lat'] = 100 * (g.rx_coil_x_m - g.cam_rx_x)
    rows.append(g)
D = pd.concat(rows)
D['pred_rx_lat'] = D.pred_lat - 100 * 2.2 * np.sin(np.radians(D.pred_yaw))       # RX = midpoint + R(psi)(0, 2.2): x shifts by -2.2 sin(dpsi)
bins = [0.2, 0.5, 0.8, 1.1, 1.6]; D['vbin'] = pd.cut(D.speed, bins)
run = D.groupby(['mode', 'trial', 'vbin'], observed=True).agg(speed=('speed', 'mean'), obs_yaw=('obs_yaw', 'mean'), pred_yaw=('pred_yaw', 'mean'),
        obs_lat=('obs_lat', 'mean'), pred_lat=('pred_lat', 'mean'), obs_rx=('obs_rx_lat', 'mean'), pred_rx=('pred_rx_lat', 'mean'), n=('speed', 'size')).reset_index()
run = run[run.n >= 20]
T = run.groupby(['mode', 'vbin'], observed=True).agg(speed=('speed', 'mean'), obs_yaw=('obs_yaw', 'mean'), obs_yaw_sd=('obs_yaw', 'std'), pred_yaw=('pred_yaw', 'mean'),
        obs_lat=('obs_lat', 'mean'), obs_lat_sd=('obs_lat', 'std'), pred_lat=('pred_lat', 'mean'), obs_rx=('obs_rx', 'mean'), obs_rx_sd=('obs_rx', 'std'), pred_rx=('pred_rx', 'mean'), runs=('trial', 'nunique')).reset_index()
pd.set_option('display.width', 220); print(T.round(2).to_string(index=False))
T.round(3).to_csv(OUTR / 'motion_skew_by_speed.csv', index=False)
# slope of the bias against speed per run (deg per m/s, cm per m/s), then mean and 95 % CI over the five runs of a mode
TQ = {5: 2.776, 4: 3.182}
B = pd.read_csv(OUTR / 'frame_id_alignment.csv'); B['mode'] = B.policy.map({'FIXED_SS': 'SS', 'FIXED_DS': 'DS', 'FIXED_PHASE': 'PHASE'})
tag_gap_ms = B.groupby('mode').T2A1_minus_T1A1_median.median()
print('slopes against speed (per run, moving samples faster than 0.2 m/s):')
for m in ('SS', 'DS', 'PHASE'):
    s_yaw, s_lat, s_rx = [], [], []
    for t, g in D[(D['mode'] == m) & (D.speed > 0.2)].groupby('trial'):
        A = np.c_[np.ones(len(g)), g.speed]
        ok = np.isfinite(g.obs_yaw) & np.isfinite(g.obs_rx_lat)
        s_yaw.append(np.linalg.lstsq(A[ok], g.obs_yaw[ok], rcond=None)[0][1]); s_lat.append(np.linalg.lstsq(A[ok], g.obs_lat[ok], rcond=None)[0][1]); s_rx.append(np.linalg.lstsq(A[ok], g.obs_rx_lat[ok], rcond=None)[0][1])
    def ci(v): v = np.array(v); return f'{v.mean():+.2f} +- {TQ[len(v)] * v.std(ddof=1) / math.sqrt(len(v)):.2f}'
    formula = -math.degrees(tag_gap_ms[m] / 1e3 / 0.8)
    print(f'  {m:5s} heading {ci(s_yaw)} deg per m/s (formula {formula:+.2f}) | tag-midpoint lateral {ci(s_lat)} cm per m/s | RX lateral {ci(s_rx)} cm per m/s (formula {-220 * math.radians(formula):+.2f} from heading alone)')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9.5, 'mathtext.fontset': 'stix'})
COL = {'SS': '#1f5fbf', 'DS': '#e06c1b', 'PHASE': '#2e8b3a'}; LAB = {'SS': 'SS-TWR', 'DS': 'DS-TWR', 'PHASE': 'High-precision DS-TWR'}
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
v = np.linspace(0, 1.7, 20)
for m in COL:
    g = T[T['mode'] == m]
    ax[0].errorbar(g.speed, g.obs_yaw, yerr=g.obs_yaw_sd, fmt='o', color=COL[m], capsize=3, label=LAB[m] + ', measured')
    ax[0].plot(v, -np.degrees(tag_gap_ms[m] / 1e3 / 0.8) * v, color=COL[m], lw=1.2, ls='--', label=LAB[m] + ', sequential-link model')
    ax[1].errorbar(g.speed, g.obs_lat, yerr=g.obs_lat_sd, fmt='o', color=COL[m], capsize=3)
    ax[1].plot(g.speed, g.pred_lat, color=COL[m], lw=1.2, ls='--')
ax[0].set_xlabel('Vehicle speed [m/s]'); ax[0].set_ylabel('Heading bias, UWB − camera [°]'); ax[0].set_title('(a) Heading', fontsize=9.5); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7, ncol=1)
ax[1].set_xlabel('Vehicle speed [m/s]'); ax[1].set_ylabel('Lateral bias of the tag midpoint [cm]'); ax[1].set_title('(b) Lateral position', fontsize=9.5); ax[1].grid(alpha=.3)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(BASE / 'fig' / f'motion_skew.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)

