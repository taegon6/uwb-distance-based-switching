"""Effective measurement-time offsets of the four links inside a frame, estimated from the data:
range residual (measured - camera-predicted) regressed on the camera range rate. slope = time offset."""
import json, math
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}
LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
MODE = {'FIXED_SS': 'SS', 'FIXED_DS': 'DS', 'FIXED_PHASE': 'PHASE'}
rows = []
for t in R.index[(R.path == 'S') & R.policy.isin(list(MODE))]:
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['frame_id', 't', 'seg', 'cam_dc', 'mode'])
    P = P[P.seg.isin(['far', 'mid'])]
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    Xs = C.X.rolling(15, center=True, min_periods=3).mean(); Ys = C.Y.rolling(15, center=True, min_periods=3).mean(); ps = np.radians(C.psi.rolling(15, center=True, min_periods=3).mean())
    L = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['frame_id', 'link_id', 'ds_range_m', 'phase_corrected_range_m', 'success'], on_bad_lines='skip')
    L = L[L.success.astype(str).isin(['1', 'True', 'true'])].drop_duplicates(['frame_id', 'link_id'], keep='last')
    col = 'phase_corrected_range_m' if MODE[R.loc[t, 'policy']] == 'PHASE' else 'ds_range_m'
    M = L.pivot(index='frame_id', columns='link_id', values=col).reindex(P.frame_id)
    tau = {}
    for l in LINKS:
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]
        def predicted(tt):
            x = np.interp(tt, C.t, Xs); y = np.interp(tt, C.t, Ys); a = np.interp(tt, C.t, ps)
            return np.sqrt((x + np.cos(a) * b[0] - np.sin(a) * b[1] - A[0]) ** 2 + (y + np.sin(a) * b[0] + np.cos(a) * b[1] - A[1]) ** 2 + (A[2] - TZ) ** 2)
        tt = P.t.to_numpy(); pred = predicted(tt); rate = (predicted(tt + 0.05) - predicted(tt - 0.05)) / 0.1
        resid = M[l].to_numpy(float) - pred
        ok = np.isfinite(resid) & np.isfinite(rate)
        X = np.c_[np.ones(ok.sum()), rate[ok], P.cam_dc.to_numpy()[ok]]
        coef = np.linalg.lstsq(X, resid[ok], rcond=None)[0]
        tau[l] = 1000 * coef[1]
    rows.append(dict(trial=t, mode=MODE[R.loc[t, 'policy']], **{f'tau_{l}': tau[l] for l in LINKS},
                     same_T1=tau['T1_A2'] - tau['T1_A1'], same_T2=tau['T2_A2'] - tau['T2_A1'],
                     T2_minus_T1=(tau['T2_A1'] + tau['T2_A2']) / 2 - (tau['T1_A1'] + tau['T1_A2']) / 2))
D = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(D.round(1).to_string(index=False))
print(D.groupby('mode')[['same_T1', 'same_T2', 'T2_minus_T1']].agg(['median', 'min', 'max']).round(1).to_string())
B = pd.read_csv(BASE / 'analysis' / 'revision_checks' / 'frame_time_budget.csv').groupby('mode')[['period_ms', 'same_T1_ms', 'cross_mean_ms']].median()
B['T2_minus_T1_if_same_cycle'] = B.same_T1_ms + B.cross_mean_ms; B['if_T2_one_cycle_later'] = B.T2_minus_T1_if_same_cycle + B.period_ms; B['if_T2_one_cycle_earlier'] = B.T2_minus_T1_if_same_cycle - B.period_ms
print(B.round(1).to_string())
D.round(2).to_csv(BASE / 'analysis' / 'revision_checks' / 'link_time_offsets_from_data.csv', index=False)

