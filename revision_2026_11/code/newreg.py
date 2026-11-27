
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def register(session):
    """Plane from pooled direct detections; heading from the board normal at the stop (mount offset
    delta = board-normal angle minus direction of motion, estimated in the same session) and the plumb yaw."""
    tr = [t for t, r in runs.items() if (r['path'] == 'S') == (session == 'S')]
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
        mv = sp > 0.15
        if mv.sum() > 10:
            deltas[t] = float(np.median(wrap(beta[mv] - np.arctan2(v[mv, 1], v[mv, 0]))))
        beta_stop[t] = float(np.median(beta[tt > tt.max() - 1.5]))
    delta = float(np.median(list(deltas.values())))
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
                camera_height_m=float(tt_[2]), mount_delta_deg=math.degrees(delta),
                mount_delta_spread_deg=float(np.degrees(np.std(list(deltas.values())))),
                alpha_deg=math.degrees(alpha), alpha_spread_deg=float(np.degrees(np.std(al))),
                alpha_per_run={t: round(math.degrees(a), 2) for t, a in alphas.items()},
                stop_residual_median_cm=100 * float(np.median(list(res.values()))), stop_loo_median_cm=100 * float(np.median(list(loo.values()))),
                stop_loo_max_cm=100 * float(max(loo.values())))
    return Rwc, tt_, info

