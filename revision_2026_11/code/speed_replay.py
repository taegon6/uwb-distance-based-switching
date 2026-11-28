"""Speed replay (26 Nov): what the far-segment errors of the recorded straight runs would be at a higher, constant speed.
Each recorded pose keeps its measured error. The pose shift that the sequential link measurement causes is calculated for
the frame (true pose from the camera, measured link times of the mode, 2A2T solver, no noise) and scaled from the speed
that was driven to the replay speed. Nothing else is changed, so range noise and range bias are those of the recording.
This is a calculation on measured errors, not a measurement at these speeds."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read().split('# ---------------- main ----------------')[0])
OUTR = BASE / 'analysis' / 'revision_checks'
B = pd.read_csv(OUTR / 'frame_time_budget.csv').groupby('mode')[['same_T1_ms', 'cross_mean_ms']].median()
E = pd.read_csv(OUT / 'segment_errors.csv'); sched = E.drop_duplicates('trial').set_index('trial')[['path', 'policy', 'block']]
LT = {}
for m in ('SS', 'DS', 'PHASE'):
    same, cross = B.loc[m, 'same_T1_ms'] / 1e3, B.loc[m, 'cross_mean_ms'] / 1e3
    lt = np.array([0.0, same, same + cross, 2 * same + cross]); LT[m] = lt - lt.mean()
SPEEDS = [round(v, 1) for v in np.arange(0.5, 4.21, 0.1)]
rows = []; frames = []
for t in sched.index[sched.path == 'S']:
    c = pd.read_csv(OUT / f'camtrack_{t}.csv').sort_values('t'); P = pd.read_csv(OUT / f'pairs_{t}.csv')
    P = P[P.seg == 'far'].copy()
    Xs = c.X.rolling(15, center=True, min_periods=3).mean(); Ys = c.Y.rolling(15, center=True, min_periods=3).mean(); psi = np.radians(c.psi.rolling(15, center=True, min_periods=3).mean())
    speed = np.hypot(np.gradient(Xs, c.t), np.gradient(Ys, c.t)); P['speed'] = np.interp(P.t, c.t, speed)
    for k in ('RX_X', 'RX_Y', 'psi'): P['c' + k] = np.interp(P.t, c.t, c[k], left=np.nan, right=np.nan)
    x0 = np.c_[np.interp(P.t, c.t, Xs), np.interp(P.t, c.t, Ys), np.interp(P.t, c.t, psi)]
    S1 = np.zeros((len(P), 3))
    for m in P['mode'].unique():
        idx = np.flatnonzero((P['mode'] == m).to_numpy()); sol = {}
        for v in (0.0, 1.0):
            ranges = np.zeros((len(idx), 4))
            for k, l in enumerate(LINKS):
                tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]; a = x0[idx, 2]
                x = x0[idx, 0]; y = x0[idx, 1] - v * LT[m][k]                      # the vehicle drives toward -y
                px = x + np.cos(a) * b[0] - np.sin(a) * b[1]; py = y + np.sin(a) * b[0] + np.cos(a) * b[1]
                ranges[:, k] = np.sqrt((px - A[0]) ** 2 + (py - A[1]) ** 2 + (A[2] - TZ) ** 2)
            sol[v], _ = resolve(ranges, x0[idx].copy())
        S1[idx] = sol[1.0] - sol[0.0]
    P['s1_x_cm'] = 100 * S1[:, 0]; P['s1_yaw_deg'] = np.degrees(S1[:, 2])
    ok = P.cRX_X.notna().to_numpy()
    def errors(v):
        dv = (P.speed.to_numpy() if v is None else v) - P.speed.to_numpy()
        x = P.x_m.to_numpy() + dv * S1[:, 0]; y = P.y_m.to_numpy() + dv * S1[:, 1]; yaw = np.radians(P.yaw_deg.to_numpy()) + dv * S1[:, 2]
        tag_lat = x - P.cam_X.to_numpy()
        ex = (x - 2.2 * np.sin(yaw)) - P.cRX_X.to_numpy(); ey = (y + 2.2 * np.cos(yaw)) - P.cRX_Y.to_numpy(); ps = np.radians(P.cpsi.to_numpy())
        rx_lat = ex * np.cos(ps) + ey * np.sin(ps)
        return 100 * math.sqrt(np.nanmean(tag_lat ** 2)), 100 * math.sqrt(np.nanmean(rx_lat[ok] ** 2)), 100 * np.nanmean(tag_lat), 100 * np.nanmean(rx_lat[ok])
    base = errors(None)
    one = errors_frames = None
    # second variant: the calculated shift is treated as uncorrelated with the recorded error (added in quadrature)
    dv1 = np.ones(len(P)); x1 = P.x_m.to_numpy() + S1[:, 0]; y1 = P.y_m.to_numpy() + S1[:, 1]; yaw0 = np.radians(P.yaw_deg.to_numpy()); yaw1 = yaw0 + S1[:, 2]; ps = np.radians(P.cpsi.to_numpy())
    rx0 = (P.x_m.to_numpy() - 2.2 * np.sin(yaw0) - P.cRX_X.to_numpy()) * np.cos(ps) + (P.y_m.to_numpy() + 2.2 * np.cos(yaw0) - P.cRX_Y.to_numpy()) * np.sin(ps)
    rx1 = (x1 - 2.2 * np.sin(yaw1) - P.cRX_X.to_numpy()) * np.cos(ps) + (y1 + 2.2 * np.cos(yaw1) - P.cRX_Y.to_numpy()) * np.sin(ps)
    s_rx = rx1 - rx0; s_tag = S1[:, 0]; e_tag = P.x_m.to_numpy() - P.cam_X.to_numpy(); va = P.speed.to_numpy()
    def quad(v):
        if v is None: v = va
        return (100 * math.sqrt(max(np.nanmean(e_tag ** 2 + (v ** 2 - va ** 2) * s_tag ** 2), 0)), 100 * math.sqrt(max(np.nanmean((rx0 ** 2 + (v ** 2 - va ** 2) * s_rx ** 2)[ok]), 0)))
    for v in [None] + SPEEDS:
        e = errors(v); q = quad(v)
        rows.append(dict(trial=t, policy=sched.loc[t, 'policy'], block=sched.loc[t, 'block'], speed=-1.0 if v is None else v, tag_lat_rmse=e[0], rx_lat_rmse=e[1], tag_lat_bias=e[2], rx_lat_bias=e[3], tagq_lat_rmse=q[0], rxq_lat_rmse=q[1]))
    frames.append(dict(trial=t, policy=sched.loc[t, 'policy'], n=len(P), speed_mean=P.speed.mean(), speed_p95=P.speed.quantile(0.95), s1_x_cm=P.s1_x_cm.mean(), s1_yaw_deg=P.s1_yaw_deg.mean(), tag_lat_rmse=base[0], logged_rmse=100 * math.sqrt((P.log_lat ** 2).mean()), rx_lat_rmse=base[1]))
R = pd.DataFrame(rows); R.round(3).to_csv(OUTR / 'speed_replay_runs.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
F = pd.DataFrame(frames); print(F.round(2).to_string(index=False))
print(F.groupby('policy')[['speed_mean', 's1_x_cm', 's1_yaw_deg', 'tag_lat_rmse', 'rx_lat_rmse']].mean().round(2).to_string())
def paired(v, a, b, col):
    g = R[R.speed == v]; A = g[g.policy == a].set_index('block')[col]; Bb = g[g.policy == b].set_index('block')[col]; d = (A - Bb).dropna()
    return d.mean(), 2.776 * d.std(ddof=1) / math.sqrt(len(d)), int((d < 0).sum()), len(d)
out = []
for v in [-1.0] + SPEEDS:
    g = R[R.speed == v].groupby('policy')[['tag_lat_rmse', 'rx_lat_rmse']].mean(); r = dict(speed=v, kmh=round(3.6 * v, 1) if v > 0 else np.nan)
    for p_, s in (('FIXED_SS', 'SS'), ('FIXED_DS', 'DS'), ('FIXED_PHASE', 'HP'), ('STAGED', 'SW')):
        r['tag_' + s] = g.loc[p_, 'tag_lat_rmse']; r['rx_' + s] = g.loc[p_, 'rx_lat_rmse']
    for a, b, s in (('FIXED_SS', 'FIXED_DS', 'SS-DS'), ('STAGED', 'FIXED_DS', 'SW-DS'), ('FIXED_SS', 'FIXED_PHASE', 'SS-HP')):
        for col, c2 in (('tag_lat_rmse', 'tag'), ('rx_lat_rmse', 'rx'), ('tagq_lat_rmse', 'tagq'), ('rxq_lat_rmse', 'rxq')):
            mu, ci, neg, n = paired(v, a, b, col); r[f'{c2}_{s}'] = mu; r[f'{c2}_{s}_ci'] = ci; r[f'{c2}_{s}_neg'] = neg
    out.append(r)
S = pd.DataFrame(out); S.round(3).to_csv(OUTR / 'speed_replay_summary.csv', index=False)
show = S[S.speed.isin([-1.0, 0.5, 1.0, 1.4, 2.0, 2.8, 3.5, 4.2])]
print(show[['speed', 'kmh', 'tag_SS', 'tag_DS', 'tag_HP', 'tag_SW', 'rx_SS', 'rx_DS', 'rx_HP', 'rx_SW']].round(1).to_string(index=False))
print(R[R.speed == -1.0].groupby('policy')[['tag_lat_bias', 'rx_lat_bias']].mean().round(2).to_string())
for c2 in ('tag', 'rx', 'tagq', 'rxq'):
    for s in ('SS-DS', 'SW-DS', 'SS-HP'):
        print('==', c2, s)
        print(show[['speed', 'kmh', f'{c2}_{s}', f'{c2}_{s}_ci', f'{c2}_{s}_neg']].round(2).to_string(index=False))
        g = S[S.speed > 0]; lower = g[g[f'{c2}_{s}'] < 0]; res = g[(g[f'{c2}_{s}'] + g[f'{c2}_{s}_ci']) < 0]
        print('   first speed with a lower mean: %s m/s; first speed with the 95%% interval below zero: %s m/s' % (lower.speed.min() if len(lower) else 'none', res.speed.min() if len(res) else 'none'))

