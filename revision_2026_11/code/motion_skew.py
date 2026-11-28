"""Motion skew inside a frame: the four links are measured one after another while the vehicle moves.
(1) Prediction: ranges are generated from the camera trajectory at the actual link times of each mode and
    solved with the 2A2T solver, without noise. (2) Observation: signed lateral error of the fixed straight
    runs against speed."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read().split('# ---------------- main ----------------')[0])
OUTR = BASE / 'analysis' / 'revision_checks'
B = pd.read_csv(OUTR / 'frame_time_budget.csv').groupby('mode')[['same_T1_ms', 'cross_mean_ms']].median()
E = pd.read_csv(OUT / 'segment_errors.csv'); sched = E.drop_duplicates('trial').set_index('trial')[['path', 'policy', 'block']]
MODE = {'FIXED_SS': 'SS', 'FIXED_DS': 'DS', 'FIXED_PHASE': 'PHASE'}
rows = []
for t in sched.index[(sched.path == 'S') & sched.policy.isin(list(MODE))]:
    m = MODE[sched.loc[t, 'policy']]
    same, cross = B.loc[m, 'same_T1_ms'] / 1e3, B.loc[m, 'cross_mean_ms'] / 1e3
    link_time = np.array([0.0, same, same + cross, 2 * same + cross])        # T1_A1, T1_A2, T2_A1, T2_A2
    link_time -= link_time.mean()                                             # pose time = mean of the link times
    c = pd.read_csv(OUT / f'camtrack_{t}.csv').sort_values('t')
    P = pd.read_csv(OUT / f'pairs_{t}.csv', usecols=['t', 'seg', 'cam_dc', 'log_lat', 'cam_X', 'cam_Y'])
    P = P[P.seg.isin(['far', 'mid'])].copy()
    Xs = c.X.rolling(15, center=True, min_periods=3).mean(); Ys = c.Y.rolling(15, center=True, min_periods=3).mean()
    psi = np.radians(c.psi.rolling(15, center=True, min_periods=3).mean())
    speed = np.hypot(np.gradient(Xs, c.t), np.gradient(Ys, c.t))
    P['speed'] = np.interp(P.t, c.t, speed)
    ranges = np.zeros((len(P), 4))
    for k, l in enumerate(LINKS):
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]; tk = P.t.to_numpy() + link_time[k]
        x = np.interp(tk, c.t, Xs); y = np.interp(tk, c.t, Ys); a = np.interp(tk, c.t, psi)
        px = x + np.cos(a) * b[0] - np.sin(a) * b[1]; py = y + np.sin(a) * b[0] + np.cos(a) * b[1]
        ranges[:, k] = np.sqrt((px - A[0]) ** 2 + (py - A[1]) ** 2 + (A[2] - TZ) ** 2)
    x0 = np.c_[np.interp(P.t, c.t, Xs), np.interp(P.t, c.t, Ys), np.interp(P.t, c.t, psi)]
    sol, _ = resolve(ranges, x0.copy())
    P['pred_lat'] = 100 * (sol[:, 0] - x0[:, 0]); P['pred_yaw'] = np.degrees(sol[:, 2] - x0[:, 2])
    P['obs_lat'] = 100 * P.log_lat; P['trial'] = t; P['mode'] = m; P['block'] = sched.loc[t, 'block']
    rows.append(P)
D = pd.concat(rows)
D.to_csv(OUTR / 'motion_skew.csv', index=False)
pd.set_option('display.width', 200)
D['vbin'] = pd.cut(D.speed, [0, 0.2, 0.5, 0.8, 1.1, 1.6])
print('== predicted lateral error from sequential link timing [cm], straight, far+mid, by speed')
print(D.groupby(['mode', 'vbin'], observed=True).pred_lat.agg(['mean', 'std', 'count']).round(2).unstack(0).to_string())
print('== observed signed lateral error [cm] by speed')
print(D.groupby(['mode', 'vbin'], observed=True).obs_lat.agg(['mean', 'count']).round(1).unstack(0).to_string())
far = D[D.seg == 'far']
print('== far segment per mode: predicted mean / RMS lateral [cm], predicted yaw mean [deg], observed mean lateral [cm]')
print(far.groupby('mode').agg(pred_mean=('pred_lat', 'mean'), pred_rms=('pred_lat', lambda v: math.sqrt((v ** 2).mean())), pred_yaw=('pred_yaw', 'mean'), obs_mean=('obs_lat', 'mean'), speed=('speed', 'median'), dist=('cam_dc', 'median')).round(2).to_string())
# observed bias: standing at the start (speed < 0.2 m/s, d_c > 9.5 m) versus moving in the same far segment
start = far[(far.speed < 0.2) & (far.cam_dc > 9.5)]; moving = far[far.speed > 0.6]
print('== observed lateral bias, far segment: standing at the start vs moving faster than 0.6 m/s [cm], mean of run means')
print(pd.DataFrame({'standing': start.groupby(['mode', 'trial']).obs_lat.mean().groupby('mode').mean(),
                    'moving': moving.groupby(['mode', 'trial']).obs_lat.mean().groupby('mode').mean(),
                    'runs_standing': start.groupby('mode').trial.nunique()}).round(1).to_string())

