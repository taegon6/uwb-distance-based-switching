"""Switching stops (26 Nov): what remains in each link range after the field offset, and how a 1 cm range error moves the estimate.
Uses analysis/stop_implied_link_offsets_v2.csv (measured 3 s median minus the range of the reference pose) and the solver of analyze_stops_v2.py."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); OUT = BASE / 'analysis'
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}; RX_BODY = np.array([0.0, 2.2]); LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
def rot(yaw): c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); return np.array([[c, -s], [s, c]])
def slant(x, y, yaw):
    R = rot(yaw); out = {}
    for l in LINKS:
        t, a = l.split('_'); p = np.array([x, y]) + R @ BODY[t]; out[l] = float(np.sqrt(np.sum((p - ANC[a][:2]) ** 2) + (ANC[a][2] - TZ) ** 2))
    return out
def solve(r, x0):
    p = np.array(x0, float)
    for _ in range(40):
        f = slant(*p); fv = np.array([f[l] for l in LINKS]); res = fv - np.array([r[l] for l in LINKS]); J = np.zeros((4, 3)); h = [1e-5, 1e-5, 1e-3]
        for k in range(3):
            q = p.copy(); q[k] += h[k]; g = slant(*q); J[:, k] = (np.array([g[l] for l in LINKS]) - fv) / h[k]
        dp = np.linalg.lstsq(J, -res, rcond=None)[0]; p += dp
        if np.abs(dp).max() < 1e-8: break
    return p
def rx(x, y, yaw): return np.array([x, y]) + rot(yaw) @ RX_BODY
pd.set_option('display.width', 250)
print('anchors', {k: v.round(3).tolist() for k, v in ANC.items()}, 'tag height', TZ)
# sensitivity at the nominal stop
x0 = (0.0, -2.2, 0.0); r0 = slant(*x0); R0 = rx(*x0); rows = []
for l in LINKS:
    r = dict(r0); r[l] += 0.01; p = solve(r, x0); Ru = rx(*p)
    rows.append(dict(link=l, range_m=r0[l], d_tag_x_cm=100 * (p[0] - x0[0]), d_tag_y_cm=100 * (p[1] - x0[1]), d_yaw_deg=p[2] - x0[2], d_rx_lat_cm=100 * (Ru[0] - R0[0]), d_rx_lon_cm=100 * (Ru[1] - R0[1])))
S = pd.DataFrame(rows); print('== effect of +1 cm on one link at the nominal stop'); print(S.round(2).to_string(index=False))
for name, dv in (('T1 links +1 cm, T2 links -1 cm', {'T1_A1': .01, 'T1_A2': .01, 'T2_A1': -.01, 'T2_A2': -.01}), ('A1 links +1 cm, A2 links -1 cm', {'T1_A1': .01, 'T2_A1': .01, 'T1_A2': -.01, 'T2_A2': -.01}), ('all +1 cm', {l: .01 for l in LINKS}), ('diagonal T1_A1,T2_A2 +1, others -1', {'T1_A1': .01, 'T2_A2': .01, 'T1_A2': -.01, 'T2_A1': -.01})):
    p = solve({l: r0[l] + dv[l] for l in LINKS}, x0); Ru = rx(*p); print('%-38s tag (%.2f, %.2f) cm, yaw %+.2f deg, RX lateral %+.2f cm' % (name, 100 * (p[0] - x0[0]), 100 * (p[1] - x0[1]), p[2], 100 * (Ru[0] - R0[0])))
# residuals of the switching stops
I = pd.read_csv(OUT / 'stop_implied_link_offsets_v2.csv'); st = pd.read_csv(OUT / 'stops_v2_seltos_L439.csv'); pol = pd.read_csv(OUT / 'runs_summary_dedup.csv')[['trial', 'policy']]
out = []
for t in pol[pol.policy == 'STAGED'].trial:
    oth = I[(I.trial != t) & (I.path == 'S')].groupby('link')['off'].median(); me = I[I.trial == t].set_index('link')['off']; res = 100 * (me - oth); s = st[st.trial == t].iloc[0]
    out.append(dict(trial=t, path=s.path, **{l: res[l] for l in LINKS}, spread=res.max() - res.min(), yaw_err=s.sfit_dyaw, rx_err=s.sfit_rx_err_cm, tag_err=s.sfit_err_cm))
O = pd.DataFrame(out); O.round(2).to_csv(OUT / 'revision_checks' / 'switching_stop_link_residuals.csv', index=False)
print('== switching stops: range residual after the field offset [cm] (measured minus reference range)'); print(O.round(1).to_string(index=False))
print('field offsets used for a curved stop [cm]:', (100 * I[I.path == 'S'].groupby('link')['off'].median()).round(1).to_dict())
print('abs residual median over 48 links %.1f cm, max %.1f; spread median %.1f' % (O[LINKS].abs().stack().median(), O[LINKS].abs().stack().max(), O.spread.median()))

