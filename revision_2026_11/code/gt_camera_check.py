"""Camera-only check of the two corrected manual entries (no UWB). Session C (T21-T36), registration as in cam_eval.register,
with the references as entered and as corrected. Rotation and translation are medians over the runs, so one entry cannot move them much."""
import math, json, numpy as np, pandas as pd, cv2
from pathlib import Path
B_ = Path(r'D:/sandbox/uwb/output/field_20261124'); CAM = B_ / 'cam'
src = (B_ / 'analyze_stops_v2.py').read_text(encoding='utf-8')
L_REAR = 4.39; STOP = (0.0, -2.2)
def ref_from(lat, lon, rear):
    x, y = STOP[0] + lat / 100, STOP[1] + lon / 100; s = (x - rear / 100) / L_REAR
    return x, y, math.degrees(math.asin(max(-1, min(1, s))))
st = pd.read_csv(B_ / 'results_20261124' / 'stops_20261124.csv').drop_duplicates('trial', keep='last').set_index('trial')
delta = math.radians(json.loads((CAM / 'eval' / 'registration_S.json').read_text(encoding='utf-8'))['mount_delta_deg'])
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi
trials = [f'T{n}' for n in range(21, 37) if (CAM / 'tracks' / f'T{n}.csv').exists()]
cams = {}
for t in trials:
    c = pd.read_csv(CAM / 'tracks' / f'{t}.csv'); c = c[c.cx.notna() & (c.reproj_px < 2.5)].copy(); c['d'] = np.sqrt(c.cx ** 2 + c.cy ** 2 + c.cz ** 2); cams[t] = c.sort_values('pts')
P = np.concatenate([cams[t].query("source == 'DIRECT' and reproj_px < 2")[['cx', 'cy', 'cz']].to_numpy() for t in trials]); keep = np.ones(len(P), bool)
for _ in range(4):
    m = P[keep].mean(0); _, _, vt = np.linalg.svd(P[keep] - m); n = vt[2]; r_ = P @ n - n @ m; s = np.std(r_[keep]); keep = np.abs(r_) < 3 * s + 1e-4
if n[1] > 0: n = -n
e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
if e1[0] < 0: e1 = -e1
e2 = np.cross(n, e1)
if e2[2] < 0: e2 = -e2
beta_stop, last = {}, {}
for t in trials:
    c = cams[t]; d = c[(c.source == 'DIRECT') & (c.d < 3.0)]
    Rb = [cv2.Rodrigues(np.array([a, b, e]))[0] for a, b, e in zip(d.rx, d.ry, d.rz)]
    beta = np.array([math.atan2(-R[:, 2] @ e2, -R[:, 2] @ e1) for R in Rb]); tt = d.pts.to_numpy()
    beta_stop[t] = float(np.median(beta[tt > tt.max() - 1.5])); last[t] = c[c.pts > c.pts.max() - 1.5][['cx', 'cy', 'cz']].median().to_numpy()
def run(corrected):
    ref = {}
    for t in trials:
        s = st.loc[t]; lat, lon, rear = s.lateral_cm, s.long_cm, s.rear_lat_cm
        if corrected and t == 'T21': lon = -lon
        if corrected and t == 'T34': rear = -rear
        ref[t] = ref_from(lat, lon, rear)
    alphas = {}
    for t in trials:
        psi = math.radians(ref[t][2]); phi = math.atan2(-math.cos(psi), -math.sin(psi)); alphas[t] = wrap(phi - (beta_stop[t] - delta))
    alpha = float(np.median(list(alphas.values()))); ca, sa = math.cos(alpha), math.sin(alpha)
    Rwc = np.vstack([ca * e1 - sa * e2, sa * e1 + ca * e2, n])
    pts = [(t, last[t], np.array([-ref[t][0], ref[t][1], 0.0])) for t in trials]
    loo = {}
    for t, cc, w in pts:
        t2 = np.median([w2 - Rwc @ c2 for t3, c2, w2 in pts if t3 != t], axis=0); loo[t] = (Rwc @ cc + t2 - w)[:2]
    return alphas, loo, alpha
out = []
for corrected in (False, True):
    alphas, loo, alpha = run(corrected)
    a = pd.Series(alphas).map(math.degrees); l = pd.Series({t: 100 * float(np.linalg.norm(v)) for t, v in loo.items()})
    oth_a = a.drop('T34'); oth_l = l.drop('T21')
    print('corrected' if corrected else 'as entered', '| session rotation %.2f deg' % math.degrees(alpha))
    print('  T34 rotation implied by its plumb heading: %.2f deg; other 15 runs median %.2f, SD %.2f, range %.2f to %.2f; deviation %.2f deg' % (a['T34'], oth_a.median(), oth_a.std(), oth_a.min(), oth_a.max(), a['T34'] - oth_a.median()))
    print('  T21 leave-one-out stop residual: %.1f cm (dx %.1f, dy %.1f); other 15 runs median %.1f, max %.1f cm' % (l['T21'], 100 * loo['T21'][0], 100 * loo['T21'][1], oth_l.median(), oth_l.max()))
    for t in trials: out.append(dict(reference='corrected' if corrected else 'as entered', trial=t, rotation_deg=a[t], loo_residual_cm=l[t], loo_dx_cm=100 * loo[t][0], loo_dy_cm=100 * loo[t][1]))
pd.DataFrame(out).round(3).to_csv(B_ / 'analysis' / 'revision_checks' / 'reference_sign_camera_check.csv', index=False, encoding='utf-8-sig')

