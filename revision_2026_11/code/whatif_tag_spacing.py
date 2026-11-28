"""What-if (26 Nov): the same range residuals of the 39 stops with a wider tag spacing. Calculation only.
Residual of a link = measured 3 s median minus reference range minus field offset (straight stops, evaluated stop left out)."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); OUT = BASE / 'analysis'
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']; RX_BODY = np.array([0.0, 2.2]); LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
def rot(yaw): c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); return np.array([[c, -s], [s, c]])
def slant(x, y, yaw, half):
    R = rot(yaw); out = {}
    for l in LINKS:
        t, a = l.split('_'); b = np.array([-half if t == 'T1' else half, 0.0]); p = np.array([x, y]) + R @ b; out[l] = float(np.sqrt(np.sum((p - ANC[a][:2]) ** 2) + (ANC[a][2] - TZ) ** 2))
    return out
def solve(r, x0, half):
    p = np.array(x0, float)
    for _ in range(60):
        f = slant(*p, half); fv = np.array([f[l] for l in LINKS]); res = fv - np.array([r[l] for l in LINKS]); J = np.zeros((4, 3)); h = [1e-5, 1e-5, 1e-3]
        for k in range(3):
            q = p.copy(); q[k] += h[k]; g = slant(*q, half); J[:, k] = (np.array([g[l] for l in LINKS]) - fv) / h[k]
        dp = np.linalg.lstsq(J, -res, rcond=None)[0]; p += dp
        if np.abs(dp).max() < 1e-8: break
    return p
def rx(x, y, yaw): return np.array([x, y]) + rot(yaw) @ RX_BODY
I = pd.read_csv(OUT / 'stop_implied_link_offsets_v2.csv'); st = pd.read_csv(OUT / 'stops_v2_seltos_L439.csv'); pol = pd.read_csv(OUT / 'runs_summary_dedup.csv')[['trial', 'policy']]; st = st.merge(pol, on='trial')
rows = []
for _, s in st.iterrows():
    oth = I[(I.trial != s.trial) & (I.path == 'S')].groupby('link')['off'].median(); res = I[I.trial == s.trial].set_index('link')['off'] - oth
    ref = (s.ref_x, s.ref_y, s.ref_yaw); R = rx(*ref); r = dict(trial=s.trial, policy=s.policy, recorded=s.sfit_rx_err_cm)
    for half in (0.4, 0.6, 0.8):
        true = slant(*ref, half); p = solve({l: true[l] + res[l] for l in LINKS}, ref, half); r[f'half_{half}'] = 100 * float(np.linalg.norm(rx(*p) - R)); r[f'yaw_{half}'] = abs(p[2] - ref[2])
    rows.append(r)
D = pd.DataFrame(rows); D.round(2).to_csv(OUT / 'revision_checks' / 'whatif_tag_spacing.csv', index=False)
pd.set_option('display.width', 220)
print('check (0.8 m spacing reproduces the recorded values): max abs diff %.2f cm' % (D['half_0.4'] - D.recorded).abs().max())
for name, g in [('all 39', D), ('switching 12', D[D.policy == 'STAGED']), ('fixed 27', D[D.policy != 'STAGED'])]:
    print(name, ' | '.join('spacing %.1f m: median %.1f cm, within 10 cm %d/%d, heading %.1f deg' % (2 * h, g[f'half_{h}'].median(), (g[f'half_{h}'] <= 10).sum(), len(g), g[f'yaw_{h}'].median()) for h in (0.4, 0.6, 0.8)))
print(D[D.policy == 'STAGED'].round(1).to_string(index=False))

