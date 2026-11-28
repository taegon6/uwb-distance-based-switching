"""Camera reference vs UWB for the 24 Nov runs: sync, per-session registration, errors.

    python cam_eval.py
Inputs: cam/tracks/T??.csv (cam_track.py), cam/jobs.csv, results_20261124, analysis/stops_v2_seltos_L439.csv,
        analysis/stop_implied_link_offsets_v2.csv.  No UWB position is used for sync or registration.
"""
from __future__ import annotations
import io, json, math
from pathlib import Path
import numpy as np, pandas as pd, cv2

BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; CAM = BASE / 'cam'; OUT = CAM / 'eval'
OUT.mkdir(exist_ok=True)
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}
LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
SCHED = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv')
BOARD_Z = 0.68

def rd(p):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False)

# ---------------- UWB re-solve (vectorised Gauss-Newton) ----------------
def resolve(r, x0):
    """r: (N,4) offset-corrected slant ranges in LINKS order; x0: (N,3) x, y, yaw(rad)."""
    p = x0.copy()
    for _ in range(8):
        F = np.zeros((len(p), 4)); J = np.zeros((len(p), 4, 3))
        c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
        for k, l in enumerate(LINKS):
            t, a = l.split('_'); b = BODY[t]; A = ANC[a]
            px = p[:, 0] + c * b[0] - s * b[1]; py = p[:, 1] + s * b[0] + c * b[1]
            dx, dy = px - A[0], py - A[1]; d = np.sqrt(dx * dx + dy * dy + (A[2] - TZ) ** 2)
            F[:, k] = d - r[:, k]
            J[:, k, 0] = dx / d; J[:, k, 1] = dy / d
            J[:, k, 2] = (dx * (-s * b[0] - c * b[1]) + dy * (c * b[0] - s * b[1])) / d
        JT = np.transpose(J, (0, 2, 1))
        dp = np.linalg.solve(JT @ J + 1e-9 * np.eye(3), -(JT @ F[:, :, None]))[:, :, 0]
        p += dp
    return p, np.sqrt(np.mean(F ** 2, axis=1))

def uwb_track(run, offsets, variants=None):
    """Logged poses plus a re-solve with the given {mode: {link: m}} offsets."""
    T = DAY / run / 'tables'
    po = rd(T / 'poses.csv'); po = po[po['valid'].astype(str).str.lower().isin(['true', '1'])].copy()
    lk = rd(T / 'links.csv'); lk = lk[lk['success'].astype(str).isin(['1', 'True', 'true'])]
    ds = lk.pivot_table(index='frame_id', columns='link_id', values='ds_range_m', aggfunc='last')
    ph = lk.pivot_table(index='frame_id', columns='link_id', values='phase_corrected_range_m', aggfunc='last')
    po = po[po.frame_id.isin(ds.index)].copy()
    src = po['range_source'].astype(str).to_numpy()
    mode = np.where(src == 'SS_RANGE', 'SS', np.where(src == 'PHASE_CORRECTED_RANGE', 'PHASE', 'DS'))
    r = np.zeros((len(po), 4))
    for k, l in enumerate(LINKS):
        dsv = ds.reindex(po.frame_id)[l].to_numpy(float); phv = ph.reindex(po.frame_id)[l].to_numpy(float)
        v = np.where(mode == 'PHASE', phv, dsv)
        off = np.array([offsets[m][l] for m in mode])
        r[:, k] = v - off
    x0 = np.c_[po.tag_midpoint_x_m.astype(float), po.tag_midpoint_y_m.astype(float), np.radians(po.yaw_deg.astype(float))]
    raw = r + np.array([[offsets[m][l] for l in LINKS] for m in mode])
    for tag, offs in [('f', offsets)] + list((variants or {}).items()):
        rr = raw - np.array([[offs[m][l] for l in LINKS] for m in mode])
        ok = np.isfinite(rr).all(1)
        sol = np.full_like(x0, np.nan)
        if ok.any():
            s, e = resolve(rr[ok], x0[ok]); sol[ok] = s
        po[tag + 'x'], po[tag + 'y'], po[tag + 'yaw'] = sol[:, 0], sol[:, 1], np.degrees(sol[:, 2])
    po['mode'] = mode; po['t'] = po.host_pose_wall_ns.astype('int64') / 1e9
    fr = rd(T / 'frames.csv')
    fr['t'] = fr.frame_end_host_wall_ns.astype('int64') / 1e9
    fr['rmean'] = fr[['r11', 'r12', 'r21', 'r22']].astype(float).mean(1)
    return po, fr[['t', 'rmean']].dropna()

# ---------------- sync: d/dt of camera distance vs mean raw UWB range ----------------
def sync(cam_t, cam_d, uwb_t, uwb_r, lo=-4.0, hi=4.0):
    g0, g1 = max(cam_t.min(), uwb_t.min() - 3), min(cam_t.max(), uwb_t.max() + 3)
    grid = np.arange(cam_t.min(), cam_t.max(), 0.01)
    def smooth_deriv(t, v):
        gi = np.interp(grid, t, v); k = 31; ker = np.ones(k) / k
        sm = np.convolve(gi, ker, 'same'); return np.gradient(sm, 0.01)
    dc = smooth_deriv(cam_t, cam_d)
    best = (-2, 0)
    def score(dt):
        gu = np.interp(grid - dt, uwb_t, uwb_r, left=np.nan, right=np.nan)
        m = np.isfinite(gu)
        if m.sum() < 300: return -2
        k = 31; ker = np.ones(k) / k
        du = np.gradient(np.convolve(np.nan_to_num(gu, nan=np.nanmean(gu)), ker, 'same'), 0.01)
        a, b = dc[m][20:-20], du[m][20:-20]
        mv = (np.abs(a) > 0.05) | (np.abs(b) > 0.05)
        if mv.sum() < 100: return -2
        return np.corrcoef(a[mv], b[mv])[0, 1]
    for dt in np.arange(lo, hi, 0.02):
        s = score(dt)
        if s > best[0]: best = (s, dt)
    for dt in np.arange(best[1] - 0.03, best[1] + 0.03, 0.002):
        s = score(dt)
        if s > best[0]: best = (s, dt)
    return best[1], best[0]

def sync_check(cam_t, cam_d, uwb_t, uwb_r, dt_used):
    """Residual SD of the mean range against the camera distance (quadratic fit, moving samples) as a function of the offset."""
    cd = pd.Series(cam_d).rolling(5, center=True, min_periods=1).median().to_numpy(); ur = pd.Series(uwb_r).rolling(5, center=True, min_periods=1).median().to_numpy()
    sp = np.abs(np.gradient(pd.Series(cd).rolling(15, center=True, min_periods=3).mean().to_numpy(), cam_t)); mv = sp > 0.15
    offs = np.arange(-4.0, 4.0, 0.02); sd = np.full(len(offs), np.nan)
    for i, dt in enumerate(offs):
        u = np.interp(cam_t - dt, uwb_t, ur, left=np.nan, right=np.nan); m = mv & np.isfinite(u)
        if m.sum() < 100: continue
        sd[i] = np.std(u[m] - np.polyval(np.polyfit(cd[m], u[m], 2), cd[m]))
    k = int(np.nanargmin(sd)); at = float(sd[int(np.argmin(np.abs(offs - dt_used)))])
    return dict(dt_resid=float(offs[k]), sd_min_cm=100 * float(sd[k]), sd_at_dt_cm=100 * at, ratio=at / float(sd[k]) if np.isfinite(at) else np.inf)

sync_rows = []
# ---------------- main ----------------
jobs = pd.read_csv(CAM / 'jobs.csv')
stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv')
impl = pd.read_csv(BASE / 'analysis' / 'stop_implied_link_offsets_v2.csv')
lab = GEOM['link_offsets_m']
runs = {}
for j in jobs.itertuples():
    f = CAM / 'tracks' / f'{j.trial}.csv'
    if not f.exists(): continue
    c = pd.read_csv(f); c = c[c.cx.notna() & (c.reproj_px < 2.5)].copy()
    c['d'] = np.sqrt(c.cx ** 2 + c.cy ** 2 + c.cz ** 2)
    st = stops[stops.trial == j.trial].iloc[0]
    oth = impl[(impl.trial != j.trial) & (impl.path == 'S')]
    offs = {m: {l: float(oth[(oth['mode'] == m) & (oth.link == l)].off.median()) for l in LINKS} for m in ('SS', 'DS', 'PHASE')}
    zero = {m: {l: 0.0 for l in LINKS} for m in offs}
    cmn = {m: {l: float(np.mean(list(offs[m].values()))) for l in LINKS} for m in offs}
    po, fr = uwb_track(j.run, offs, {'z': zero, 'c': cmn})
    c['tw'] = j.video_start + c.pts
    dt, corr = sync(c.tw.to_numpy(), c.d.to_numpy(), fr.t.to_numpy(), fr.rmean.to_numpy())
    chk = sync_check(c.tw.to_numpy(), c.d.to_numpy(), fr.t.to_numpy(), fr.rmean.to_numpy(), dt); method = 'correlation'; dt_first, corr_first = dt, corr
    if chk['ratio'] > 2.0:
        rm5 = fr.rmean.rolling(5, center=True, min_periods=1).median().to_numpy()
        dt, corr = sync(c.tw.to_numpy(), c.d.to_numpy(), fr.t.to_numpy(), rm5, chk['dt_resid'] - 0.4, chk['dt_resid'] + 0.4); method = 'correlation within 0.4 s of the residual minimum, five-frame median of the range'
    sync_rows.append(dict(trial=j.trial, dt_s=dt, corr=corr, method=method, dt_first_s=dt_first, corr_first=corr_first, **chk))
    c['t'] = c.tw - dt          # camera at tw matches UWB at tw - dt
    runs[j.trial] = dict(cam=c, po=po, st=st, dt=dt, corr=corr, path=st.path, policy=stops[stops.trial == j.trial].iloc[0].get('stop_mode'))
    print(j.trial, f'sync dt {dt:+.3f} s corr {corr:.3f} cam {len(c)} uwb {len(po)}', flush=True)

def board_dirs(c):
    R = [cv2.Rodrigues(np.array([a, b, d]))[0] for a, b, d in zip(c.rx, c.ry, c.rz)]
    return np.array(R)

def SESSION_OF(t):
    # the phone was touched between S and C1, and again in the 23:07-23:23 break before T37/T50
    n = int(t[1:])
    return 'S' if n <= 20 else ('C' if n <= 36 else 'C2late')

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def register(session, delta_fixed=None):
    """Plane from pooled direct detections; heading from the board normal at the stop (mount offset
    delta = board-normal angle minus direction of motion, estimated in the same session) and the plumb yaw."""
    tr = [t for t in runs if SESSION_OF(t) == session]
    P = np.concatenate([runs[t]['cam'].query("source == 'DIRECT' and reproj_px < 2")[['cx', 'cy', 'cz']].to_numpy() for t in tr])
    keep = np.ones(len(P), bool)
    for _ in range(4):
        m = P[keep].mean(0); _, _, vt = np.linalg.svd(P[keep] - m); n = vt[2]
        r_ = P @ n - n @ m; s = np.std(r_[keep]); keep = np.abs(r_) < 3 * s + 1e-4
    if n[1] > 0: n = -n
    dplane = float(n @ P[keep].mean(0)); plane_rms = float(np.std((P @ n)[keep] - dplane))
    e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    if e1[0] < 0: e1 = -e1
    e2 = np.cross(n, e1)
    if e2[2] < 0: e2 = -e2
    def ang(v): return math.atan2(v @ e2, v @ e1)
    deltas, beta_stop, path_alpha = {}, {}, {}
    for t in tr:
        c = runs[t]['cam'].sort_values('t'); d = c[(c.source == 'DIRECT') & (c.d < 3.0)]
        if len(d) < 20: continue
        Rb = [cv2.Rodrigues(np.array([a, b, e]))[0] for a, b, e in zip(d.rx, d.ry, d.rz)]
        beta = np.array([ang(-R[:, 2]) for R in Rb])          # board normal pointing away from the camera ~ vehicle forward
        Q = d[['cx', 'cy', 'cz']].to_numpy(); h = np.c_[Q @ e1, Q @ e2]
        hs = pd.DataFrame(h).rolling(15, center=True, min_periods=5).mean().to_numpy()
        tt = d.t.to_numpy(); v = np.gradient(hs, tt, axis=0); sp = np.hypot(v[:, 0], v[:, 1])
        # a front-bumper point moves along the heading only when the vehicle is not turning
        bs = pd.Series(np.unwrap(beta)).rolling(15, center=True, min_periods=5).mean().to_numpy()
        yr = np.abs(np.gradient(bs, tt))
        mv = (sp > 0.15) & (yr < math.radians(1.5))
        if mv.sum() > 10:
            deltas[t] = float(np.median(wrap(beta[mv] - np.arctan2(v[mv, 1], v[mv, 0]))))
        beta_stop[t] = float(np.median(beta[tt > tt.max() - 1.5]))
    delta_est = float(np.median(list(deltas.values())))
    delta = delta_est if delta_fixed is None else delta_fixed
    alphas = {}
    for t in beta_stop:
        psi = math.radians(runs[t]['st'].ref_yaw)
        phi = math.atan2(-math.cos(psi), -math.sin(psi))       # physical forward (-sin psi, -cos psi)
        alphas[t] = wrap(phi - (beta_stop[t] - delta))
    al = np.array(list(alphas.values())); alpha = float(np.median(al))
    ca, sa = math.cos(alpha), math.sin(alpha)
    wx = ca * e1 - sa * e2; wy = sa * e1 + ca * e2
    Rwc = np.vstack([wx, wy, n])
    pts = []
    for t in tr:
        c = runs[t]['cam']; st = runs[t]['st']
        last = c[c.t > c.t.max() - 1.5][['cx', 'cy', 'cz']].median().to_numpy()
        pts.append((t, last, np.array([-st.ref_x, st.ref_y, BOARD_Z])))
    tt_ = np.median([w - Rwc @ cc for _, cc, w in pts], axis=0)
    res = {t: float(np.linalg.norm((Rwc @ cc + tt_ - w)[:2])) for t, cc, w in pts}
    loo = {}
    for t, cc, w in pts:
        t2 = np.median([w2 - Rwc @ c2 for t3, c2, w2 in pts if t3 != t], axis=0)
        loo[t] = float(np.linalg.norm((Rwc @ cc + t2 - w)[:2]))
    info = dict(session=session, runs=len(tr), plane_rms_m=plane_rms, plane_points=int(keep.sum()), camera_world_phys=tt_.tolist(),
                camera_height_m=float(tt_[2]), mount_delta_deg=math.degrees(delta), mount_delta_own_estimate_deg=math.degrees(delta_est),
                mount_delta_spread_deg=float(np.degrees(np.std(list(deltas.values())))),
                alpha_deg=math.degrees(alpha), alpha_spread_deg=float(np.degrees(np.std(al))),
                alpha_per_run={t: round(math.degrees(a), 2) for t, a in alphas.items()},
                stop_residual_median_cm=100 * float(np.median(list(res.values()))), stop_loo_median_cm=100 * float(np.median(list(loo.values()))),
                stop_loo_max_cm=100 * float(max(loo.values())))
    return Rwc, tt_, info


regs = {}
delta_S = None
for sess in ('S', 'C', 'C2late'):
    if sum(SESSION_OF(t) == sess for t in runs) < 2: continue
    # curve runs are still turning near the camera, so the board mount offset comes from the straight session
    Rwc, tt, info = register(sess, None if sess == 'S' else delta_S); regs[sess] = (Rwc, tt); print(json.dumps(info, indent=1), flush=True)
    if sess == 'S': delta_S = math.radians(info['mount_delta_deg'])
    (OUT / f'registration_{sess}.json').write_text(json.dumps(info, indent=1), encoding='utf-8')

def seg(d):
    return np.where(d >= 4.0, 'far', np.where(d >= 0.5, 'mid', 'near'))

REAR_AXLE_TO_BOARD_M = 3.5      # Kia Seltos: wheelbase 2.63 m plus front overhang of about 0.9 m (board on the front bumper)
HEADING_GATE_DEG = 25.0

def wrap_deg(a): return (a + 180.0) % 360.0 - 180.0

def repair_heading(c):
    """Camera heading with mirrored board poses replaced.

    A planar target that is small in the image has two pose solutions with mirrored normals, and the optical-flow
    part of the track sometimes returns the wrong one. Walking backward in time from the stop, where the pose is
    reliable, the heading is predicted with the kinematics of a point ahead of the rear axle:
    yaw rate = speed * sin(direction of motion - heading) / distance to the rear axle.
    A measured heading further than HEADING_GATE_DEG from the prediction is replaced by the prediction.
    Returns the repaired heading, the replaced mask and a free-running kinematic heading for validation."""
    t = c.t.to_numpy(); raw = c.psi.to_numpy()
    X = c.X.rolling(31, center=True, min_periods=5).mean().to_numpy(); Y = c.Y.rolling(31, center=True, min_periods=5).mean().to_numpy()
    vx, vy = np.gradient(X, t), np.gradient(Y, t)
    speed = np.hypot(vx, vy); motion = np.degrees(np.arctan2(vx, -vy))
    smooth = c.psi.rolling(9, center=True, min_periods=3).median().to_numpy()
    n = len(c); est = np.empty(n); free = np.empty(n); replaced = np.zeros(n, bool)
    est[-1] = free[-1] = np.median(raw[t > t[-1] - 1.0])
    for i in range(n - 2, -1, -1):
        dt = t[i + 1] - t[i]
        for track in (est, free):
            yaw_rate = speed[i + 1] * math.sin(math.radians(wrap_deg(motion[i + 1] - track[i + 1]))) / REAR_AXLE_TO_BOARD_M
            track[i] = track[i + 1] - math.degrees(yaw_rate) * dt
        replaced[i] = abs(wrap_deg(raw[i] - est[i])) > HEADING_GATE_DEG
        if abs(wrap_deg(smooth[i] - est[i])) <= HEADING_GATE_DEG:
            est[i] = smooth[i]
    return np.where(replaced, est, raw), replaced, free

rows = []
seg_times = []
heading_rows = []
for t, r in runs.items():
    key = SESSION_OF(t)
    if key not in regs: continue
    Rwc, tt = regs[key]
    c = r['cam']; W = (Rwc @ c[['cx', 'cy', 'cz']].to_numpy().T).T + tt
    c['X'], c['Y'] = -W[:, 0], W[:, 1]                         # back to code frame
    # camera heading: board normal (pointing away from the camera) in the world, minus the mount offset
    nb = np.array([-cv2.Rodrigues(np.array([a, b, e]))[0][:, 2] for a, b, e in zip(c.rx, c.ry, c.rz)])
    wn = (Rwc @ nb.T).T
    phi = np.arctan2(wn[:, 1], wn[:, 0]) - delta_S              # physical forward direction angle
    fx, fy = -np.cos(phi), np.sin(phi)                          # code frame (x mirrored)
    c['psi'] = np.degrees(np.arctan2(fx, -fy))
    c = c.sort_values('t').reset_index(drop=True)
    c['psi_raw'] = c.psi
    c['psi'], c['psi_replaced'], free_heading = repair_heading(c)
    moving = (~c.psi_replaced) & (c.t < c.t.max() - 3.0)
    heading_rows.append(dict(trial=t, path=r['path'], samples=len(c), replaced=int(c.psi_replaced.sum()),
                             replaced_pct=100 * float(c.psi_replaced.mean()),
                             free_run_rms_deg=float(np.sqrt(np.mean(wrap_deg(c.psi_raw[moving] - free_heading[moving]) ** 2)))))
    ps = np.radians(c.psi)
    c['RX_X'], c['RX_Y'] = c.X - 2.2 * np.sin(ps), c.Y + 2.2 * np.cos(ps)
    c[['t', 'pts', 'source', 'X', 'Y', 'psi', 'RX_X', 'RX_Y', 'psi_raw', 'psi_replaced']].to_csv(OUT / f'camtrack_{t}.csv', index=False)
    po = r['po'].copy()
    ti = c.t.to_numpy(); order = np.argsort(ti)
    for k in ('X', 'Y'):
        po['cam_' + k] = np.interp(po.t, ti[order], c[k].to_numpy()[order], left=np.nan, right=np.nan)
    # Approach segments are time intervals cut at the first time the camera RX-coil distance falls below
    # 4.0 m and 0.5 m. Assigning segments per sample from the UWB distance put standstill samples whose
    # noisy distance rose above 0.5 m into the 0.5-4 m segment.
    cs = c.sort_values('t')
    cam_t = cs.t.to_numpy()
    cam_dc = np.hypot(cs.RX_X, cs.RX_Y).rolling(15, center=True, min_periods=1).median().to_numpy()
    t_far_end = cam_t[np.argmax(cam_dc < 4.0)] if (cam_dc < 4.0).any() else np.inf
    t_mid_end = cam_t[np.argmax(cam_dc < 0.5)] if (cam_dc < 0.5).any() else np.inf
    all_t = r['po'].t.to_numpy()
    # The approach starts when the vehicle starts to move. On the straight runs the vehicle stood at the start
    # for 1 to 7 s after the recording began, and that standstill is not part of the approach.
    start_x, start_y = cs.X.iloc[:15].median(), cs.Y.iloc[:15].median()
    moved = np.hypot(cs.X - start_x, cs.Y - start_y).rolling(5, center=True, min_periods=1).median().to_numpy()
    t_motion = cam_t[np.argmax(moved > 0.15)]
    windows = {'wait': (all_t.min(), t_motion), 'far': (t_motion, t_far_end), 'mid': (t_far_end, t_mid_end), 'near': (t_mid_end, all_t.max())}
    seg_times.append(dict(trial=t, t_start=all_t.min(), t_motion=t_motion, t_far_end=t_far_end, t_mid_end=t_mid_end, t_end=all_t.max(),
                          cam_dc_first=float(cam_dc[0]), cam_dc_min=float(cam_dc.min())))
    po['cam_dc'] = np.interp(po.t, cam_t, cam_dc, left=np.nan, right=np.nan)
    gap = np.abs(np.searchsorted(ti[order], po.t) - 0)
    near = np.array([np.min(np.abs(ti - x)) if len(ti) else 9 for x in po.t])
    po = po[near < 0.05].copy()
    # path tangent from the camera trajectory (for curves); straight runs use the y axis
    if r['path'] == 'S':
        tang = np.tile([0.0, -1.0], (len(po), 1))
    else:
        cs = c.sort_values('t'); X = cs.X.rolling(31, center=True, min_periods=5).mean(); Y = cs.Y.rolling(31, center=True, min_periods=5).mean()
        vx = np.gradient(X, cs.t); vy = np.gradient(Y, cs.t)
        tx = np.interp(po.t, cs.t, vx); ty = np.interp(po.t, cs.t, vy); nrm = np.hypot(tx, ty)
        tang = np.c_[tx, ty] / np.where(nrm > 0.02, nrm, np.nan)[:, None]
        tang[np.isnan(tang[:, 0])] = [0.0, -1.0]
    lat = np.c_[-tang[:, 1], tang[:, 0]]
    po['d_c'] = po.coil_to_tx_distance_m.astype(float)
    po['seg'] = np.where(po.t < t_motion, 'wait', np.where(po.t < t_far_end, 'far', np.where(po.t < t_mid_end, 'mid', 'near')))
    for name, xs, ys in (('log', po.tag_midpoint_x_m.astype(float), po.tag_midpoint_y_m.astype(float)), ('fld', po.fx, po.fy),
                         ('raw', po.zx, po.zy), ('cmn', po.cx, po.cy)):
        ex, ey = xs - po.cam_X, ys - po.cam_Y
        po[name + '_lat'] = ex * lat[:, 0] + ey * lat[:, 1]; po[name + '_lon'] = ex * tang[:, 0] + ey * tang[:, 1]
    po.to_csv(OUT / f'pairs_{t}.csv', index=False)
    for sg, g in po.groupby('seg'):
        if len(g) < 10: continue
        lo, hi = windows[sg]
        # update rate: all valid UWB poses in the segment window, including those without a camera match
        row = dict(trial=t, path=r['path'], seg=sg, n=len(g), dur_s=hi - lo, rate_hz=((all_t >= lo) & (all_t < hi)).sum() / max(hi - lo, 1e-3),
                   sync_dt=r['dt'], sync_corr=r['corr'])
        for name in ('log', 'fld', 'raw', 'cmn'):
            for ax in ('lat', 'lon'):
                v = g[f'{name}_{ax}'].dropna() * 100
                row[f'{name}_{ax}_bias'] = v.mean(); row[f'{name}_{ax}_sd'] = v.std(); row[f'{name}_{ax}_rmse'] = math.sqrt((v ** 2).mean())
            row[f'{name}_2d_rmse'] = math.sqrt(((g[f'{name}_lat'] ** 2 + g[f'{name}_lon'] ** 2).mean())) * 100
        rows.append(row)
E = pd.DataFrame(rows)
pd.DataFrame(seg_times).to_csv(OUT / 'segment_times.csv', index=False)
pd.DataFrame(sync_rows).round(4).to_csv(OUT / 'sync_check.csv', index=False)
HR = pd.DataFrame(heading_rows); HR.round(2).to_csv(OUT / 'heading_repair.csv', index=False)
print('== camera heading repair'); print(HR.round(2).to_string(index=False))
pol = SCHED.set_index('trial')
E['policy'] = E.trial.map(pol.policy); E['block'] = E.trial.map(pol.block)
E.round(3).to_csv(OUT / 'segment_errors.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
for name in ('log', 'fld', 'raw', 'cmn'):
    cols = ['n', 'rate_hz'] + [f'{name}_{a}_{k}' for a in ('lat', 'lon') for k in ('bias', 'sd', 'rmse')] + [f'{name}_2d_rmse']
    print('==', name); print(E.groupby(['path', 'seg', 'policy'])[cols].median().round(2).to_string())
# paired block differences on far-segment lateral RMSE (field offsets)
pair_rows = []
for var in ('log', 'cmn'):
    for path in ('S', 'C1'):
        for sg in ('far', 'mid'):
            f = E[(E.path == path) & (E.seg == sg)].pivot_table(index='block', columns='policy', values=f'{var}_lat_rmse')
            if 'STAGED' not in f.columns: continue
            for other in [c for c in f.columns if c != 'STAGED']:
                d = (f['STAGED'] - f[other]).dropna()
                if len(d) >= 2:
                    from math import sqrt
                    tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}[len(d)]
                    m, s = d.mean(), d.std(ddof=1); h = tq * s / sqrt(len(d))
                    pair_rows.append(dict(variant=var, path=path, seg=sg, other=other, blocks=len(d), mean_cm=m, ci95_cm=h, values=np.round(d.to_numpy(), 2).tolist()))
                    print(f'[{var}] {path} {sg} lateral RMSE STAGED - {other}: n {len(d)} mean {m:+.2f} cm, 95% CI +-{h:.2f}, {np.round(d.to_numpy(), 2).tolist()}')
pd.DataFrame(pair_rows).to_csv(OUT / 'paired_blocks.csv', index=False)
print('sync corr min', E.sync_corr.min(), 'median dt', E.drop_duplicates('trial').sync_dt.median())

