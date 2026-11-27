"""Per-link range residual against the camera reference (R1 Major 5): measured raw range minus the slant range
predicted from the camera pose (tag-midpoint position and heading in the TX frame). Binned by camera coil
distance on the straight runs and by vehicle heading on curve C1. Also the pose-solver range residual."""
import json, math
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}
LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
out = []
for t in R.index:
    pf = EV / f'pairs_{t}.csv'
    if not pf.exists(): continue
    P = pd.read_csv(pf, usecols=['frame_id', 't', 'cam_X', 'cam_Y', 'cam_dc', 'seg', 'mode', 'range_residual_rms_m'])
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    P['psi'] = np.interp(P.t, C.t, C.psi)
    L = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['frame_id', 'link_id', 'ds_range_m', 'phase_corrected_range_m', 'success'], on_bad_lines='skip')
    L = L[L.success.astype(str).isin(['1', 'True', 'true'])].drop_duplicates(['frame_id', 'link_id'], keep='last')
    ds = L.pivot(index='frame_id', columns='link_id', values='ds_range_m'); ph = L.pivot(index='frame_id', columns='link_id', values='phase_corrected_range_m')
    psi = np.radians(P.psi.to_numpy()); c, s = np.cos(psi), np.sin(psi)
    for l in LINKS:
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]
        px = P.cam_X.to_numpy() + c * b[0] - s * b[1]; py = P.cam_Y.to_numpy() + s * b[0] + c * b[1]
        pred = np.sqrt((px - A[0]) ** 2 + (py - A[1]) ** 2 + (A[2] - TZ) ** 2)
        meas = np.where(P['mode'].to_numpy() == 'PHASE', ph.reindex(P.frame_id)[l].to_numpy(float), ds.reindex(P.frame_id)[l].to_numpy(float))
        # direction from the tag to the anchor relative to the vehicle forward axis (0 = straight ahead)
        fwd = np.c_[np.sin(psi), -np.cos(psi)]
        to_anchor = np.c_[A[0] - px, A[1] - py]; to_anchor /= np.linalg.norm(to_anchor, axis=1)[:, None]
        off_axis = np.degrees(np.arccos(np.clip((fwd * to_anchor).sum(1), -1, 1)))
        out.append(pd.DataFrame(dict(trial=t, path=R.loc[t, 'path'], policy=R.loc[t, 'policy'], link=l, mode=P['mode'].to_numpy(), seg=P.seg.to_numpy(),
                                     cam_dc=P.cam_dc.to_numpy(), psi=P.psi.to_numpy(), off_axis=off_axis, resid_cm=100 * (meas - pred),
                                     solver_rms_cm=100 * P.range_residual_rms_m.to_numpy(float))))
D = pd.concat(out).dropna(subset=['resid_cm'])
pd.set_option('display.width', 220)
OUT = BASE / 'analysis' / 'revision_checks'
S = D[(D.path == 'S') & D.seg.isin(['far', 'mid'])].copy(); S['bin'] = pd.cut(S.cam_dc, [0.5, 2, 4, 6, 8, 10.5])
tab = S.groupby(['mode', 'bin', 'link'], observed=True).resid_cm.median().unstack().round(1)
print('== straight, approach: median range residual vs camera [cm] by coil-distance bin'); print(tab.to_string())
tab.to_csv(OUT / 'range_residual_straight_by_distance.csv')
near = D[(D.seg == 'near')].groupby(['path', 'mode', 'link']).resid_cm.median().unstack().round(1)
print('== at the stop (near segment): median range residual [cm]'); print(near.to_string())
C1 = D[(D.path == 'C1') & D.seg.isin(['far', 'mid'])].copy(); C1['hbin'] = pd.cut(C1.psi.abs(), [0, 5, 15, 30, 90])
tabc = C1.groupby(['hbin', 'link'], observed=True).resid_cm.agg(['median', 'count']).unstack().round(1)
print('== C1, approach: range residual by |heading| bin (all modes)'); print(tabc.to_string())
C1.groupby(['hbin', 'link'], observed=True).resid_cm.median().unstack().round(1).to_csv(OUT / 'range_residual_c1_by_heading.csv')
C1['obin'] = pd.cut(C1.off_axis, [0, 10, 20, 30, 45, 90])
print('== C1, approach: residual by off-axis angle of the anchor seen from the tag'); print(C1.groupby(['obin', 'link'], observed=True).resid_cm.median().unstack().round(1).to_string())
print('== heading range on C1 by segment'); print(C1.groupby('seg').psi.describe()[['min', '50%', 'max']].round(1).to_string())
# spread between the four links = what the 2A2T solver cannot absorb as a common offset
def spread(g):
    w = g.pivot_table(index=['trial', 'frame'], columns='link', values='resid_cm') if 'frame' in g else None
    return w
for name, g in (('S', S), ('C1', C1)):
    m = g.groupby(['seg', 'link']).resid_cm.median().unstack()
    print(f'== {name}: per-link median by segment'); print(m.round(1).to_string()); print('   max-min across links:', (m.max(1) - m.min(1)).round(1).to_dict())
q = D.drop_duplicates(['trial', 'seg', 'cam_dc', 'psi'])
print('== pose solver range residual RMS [cm], median by path and segment'); print(q.groupby(['path', 'seg']).solver_rms_cm.median().unstack().round(2).to_string())

