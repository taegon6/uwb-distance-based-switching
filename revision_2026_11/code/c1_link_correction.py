"""R1 Major 5 (26 Nov): can the pair-specific excess of link T2-A1 on the curved paths be corrected with the recorded data?
The excess is modelled as a function of the bearing of anchor A1 seen from tag T2 (10 degree bins, median over the other C1 runs,
camera pose used only in those training runs). In the evaluated run the bearing comes from the UWB pose itself (first pass),
the T2-A1 range is corrected, and the pose is solved again. Exploratory, defined after the experiment."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read().split('# ---------------- main ----------------')[0])
OUTR = BASE / 'analysis' / 'revision_checks'; LAB = GEOM['link_offsets_m']
E = pd.read_csv(OUT / 'segment_errors.csv'); sched = E.drop_duplicates('trial').set_index('trial')[['path', 'policy', 'block']]
RUN = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'run']).set_index('trial').run
BINS = np.arange(-90, 91, 10)
def bearing(x, y, psi):
    """bearing of anchor A1 seen from tag T2 in the body frame [deg]; psi in rad"""
    c, s = np.cos(psi), np.sin(psi); b = BODY['T2']; A = ANC['A1']
    px = x + c * b[0] - s * b[1]; py = y + s * b[0] + c * b[1]; v = np.c_[A[0] - px, A[1] - py]; u = v / np.linalg.norm(v, axis=1)[:, None]
    fwd = np.c_[np.sin(psi), -np.cos(psi)]; right = np.c_[np.cos(psi), np.sin(psi)]
    return np.degrees(np.arctan2((u * right).sum(1), (u * fwd).sum(1)))
data = {}
for t in sched.index[sched.path.isin(['C1', 'C2'])]:
    P = pd.read_csv(OUT / f'pairs_{t}.csv'); P = P[P.seg.isin(['far', 'mid'])].copy(); c = pd.read_csv(OUT / f'camtrack_{t}.csv').sort_values('t')
    L = rd(DAY / RUN[t] / 'tables' / 'links.csv'); L = L[L['success'].astype(str).isin(['1', 'True', 'true'])]
    ds = L.pivot_table(index='frame_id', columns='link_id', values='ds_range_m', aggfunc='last')
    mode = P['mode'].to_numpy(); r = np.c_[[ds.reindex(P.frame_id)[l].to_numpy(float) - np.array([LAB[m][l] for m in mode]) for l in LINKS]].T
    ok = np.isfinite(r).all(1); P = P[ok].copy(); r = r[ok]; mode = mode[ok]
    psi_c = np.radians(np.interp(P.t, c.t, c.psi)); xc, yc = P.cam_X.to_numpy(), P.cam_Y.to_numpy()
    pred = np.zeros_like(r)
    for k, l in enumerate(LINKS):
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]; px = xc + np.cos(psi_c) * b[0] - np.sin(psi_c) * b[1]; py = yc + np.sin(psi_c) * b[0] + np.cos(psi_c) * b[1]
        pred[:, k] = np.sqrt((px - A[0]) ** 2 + (py - A[1]) ** 2 + (A[2] - TZ) ** 2)
    res = r - pred; excess = res[:, 2] - res[:, [0, 1, 3]].mean(1)
    X = c.X.rolling(31, center=True, min_periods=5).mean(); Y = c.Y.rolling(31, center=True, min_periods=5).mean(); vx = np.gradient(X, c.t); vy = np.gradient(Y, c.t)
    tx = np.interp(P.t, c.t, vx); ty = np.interp(P.t, c.t, vy); nrm = np.hypot(tx, ty); tang = np.c_[tx, ty] / np.where(nrm > 0.02, nrm, np.nan)[:, None]; tang[np.isnan(tang[:, 0])] = [0.0, -1.0]
    data[t] = dict(P=P, r=r, lat=np.c_[-tang[:, 1], tang[:, 0]], excess=excess, beta_cam=bearing(xc, yc, psi_c), x0=np.c_[P.x_m.astype(float), P.y_m.astype(float), np.radians(P.yaw_deg.astype(float))], seg=P.seg.to_numpy(), xc=xc, yc=yc)
def table(trials):
    b = np.concatenate([data[t]['beta_cam'] for t in trials]); e = np.concatenate([data[t]['excess'] for t in trials]); idx = np.digitize(b, BINS) - 1; out = np.zeros(len(BINS) - 1)
    for k in range(len(out)):
        m = idx == k
        if m.sum() >= 50: out[k] = np.median(e[m])
    return out
def lat_rmse(t, sol):
    d = data[t]; ex = sol[:, 0] - d['xc']; ey = sol[:, 1] - d['yc']; v = 100 * (ex * d['lat'][:, 0] + ey * d['lat'][:, 1])
    return {s: math.sqrt(np.mean(v[d['seg'] == s] ** 2)) if (d['seg'] == s).sum() > 10 else np.nan for s in ('far', 'mid')}
c1 = [t for t in data if sched.loc[t, 'path'] == 'C1']; rows = []
full = table(c1); print('excess by bearing of A1 seen from T2 [cm], all C1 runs:'); print({f'{BINS[k]}..{BINS[k + 1]}': round(100 * full[k], 1) for k in range(len(full)) if full[k] != 0})
for t in data:
    d = data[t]; tab = table([u for u in c1 if u != t])
    base, _ = resolve(d['r'].copy(), d['x0'].copy())
    def corrected(beta):
        r2 = d['r'].copy(); r2[:, 2] -= tab[np.clip(np.digitize(beta, BINS) - 1, 0, len(tab) - 1)]; s, _ = resolve(r2, d['x0'].copy()); return s
    one = corrected(bearing(base[:, 0], base[:, 1], base[:, 2])); two = corrected(bearing(one[:, 0], one[:, 1], one[:, 2])); orc = corrected(d['beta_cam'])
    lg = lat_rmse(t, d['x0']); b0 = lat_rmse(t, base); b1 = lat_rmse(t, one); b2 = lat_rmse(t, two); bo = lat_rmse(t, orc)
    for s in ('far', 'mid'):
        rows.append(dict(trial=t, path=sched.loc[t, 'path'], policy=sched.loc[t, 'policy'], seg=s, logged=lg[s], resolved=b0[s], corrected=b1[s], corrected_2pass=b2[s], corrected_camera_bearing=bo[s]))
R = pd.DataFrame(rows); R.round(2).to_csv(OUTR / 'c1_link_correction.csv', index=False)
pd.concat([pd.DataFrame(dict(trial=t, path=sched.loc[t, 'path'], seg=data[t]['seg'], beta=data[t]['beta_cam'], excess_cm=100 * data[t]['excess'])) for t in data]).round(3).to_csv(OUTR / 'c1_link_bearing_frames.csv', index=False)
pd.set_option('display.width', 220); print(R.round(1).to_string(index=False))
chk = R.merge(E[['trial', 'seg', 'log_lat_rmse']], on=['trial', 'seg']); print('logged lateral RMSE against segment_errors.csv: max abs difference %.2f cm' % (chk.logged - chk.log_lat_rmse).abs().max())
for path in ('C1', 'C2'):
    for s in ('far', 'mid'):
        g = R[(R.path == path) & (R.seg == s)].dropna(); dlt = g.corrected - g.resolved; n = len(g)
        if n < 3: print(path, s, 'n', n, g[['resolved', 'corrected']].round(1).to_dict('records')); continue
        tq = {15: 2.145, 14: 2.160, 13: 2.179, 3: 4.303}.get(n, 2.2)
        print('%s %s n %d: median %.1f -> %.1f cm (2 pass %.1f, camera bearing %.1f); paired change %.1f ± %.1f cm, lower in %d/%d' % (path, s, n, g.resolved.median(), g.corrected.median(), g.corrected_2pass.median(), g.corrected_camera_bearing.median(), dlt.mean(), tq * dlt.std(ddof=1) / math.sqrt(n), (dlt < 0).sum(), n))
        if path == 'C1': print('     by policy:', {p: (round(x.resolved.median(), 1), round(x.corrected.median(), 1)) for p, x in g.groupby('policy')})

