"""R1 Major 5: what the recorded data say about the cause of the long T2-A1 range on the curved path.

No received-power metric was logged. This script tests the candidate causes with the per-link range residual
(raw measured range minus the slant range predicted from the camera pose):
 1. rigid-body reference error (cannot produce a one-link excess; checked by the pattern over the four links)
 2. floor reflection taken as first path (excess would follow 2*h_tag*h_anchor/d)
 3. orientation-dependent bias of a device (additive model: tag term by tag bearing + anchor term by anchor bearing)
 4. weak or blocked link (range noise and packet failures of the link by heading)
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'
OUT = BASE / 'analysis' / 'revision_checks'
GEOM = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in GEOM['anchors_m'].items()}; TZ = GEOM['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}
LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
pd.set_option('display.width', 250)

rows, fails = [], []
for t in R.index:
    pf = EV / f'pairs_{t}.csv'
    if not pf.exists(): continue
    P = pd.read_csv(pf, usecols=['frame_id', 't', 'cam_X', 'cam_Y', 'cam_dc', 'seg', 'mode'])
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    P['psi'] = np.interp(P.t, C.t, C.psi)
    L = pd.read_csv(DAY / R.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['frame_id', 'link_id', 'ds_range_m', 'phase_corrected_range_m', 'success'], on_bad_lines='skip')
    L['ok'] = L.success.astype(str).isin(['1', 'True', 'true'])
    # packet failures: map every link attempt to the heading of the nearest paired frame
    Pm = P.sort_values('frame_id')
    idx = np.clip(np.searchsorted(Pm.frame_id.to_numpy(), L.frame_id.to_numpy()), 0, len(Pm) - 1)
    near = np.abs(Pm.frame_id.to_numpy()[idx] - L.frame_id.to_numpy()) <= 5
    F = pd.DataFrame(dict(trial=t, path=R.loc[t, 'path'], link=L.link_id.to_numpy(), ok=L.ok.to_numpy(), psi=Pm.psi.to_numpy()[idx], seg=Pm.seg.to_numpy()[idx], mode=Pm['mode'].to_numpy()[idx]))[near]
    fails.append(F)
    L = L[L.ok].drop_duplicates(['frame_id', 'link_id'], keep='last')
    ds = L.pivot(index='frame_id', columns='link_id', values='ds_range_m'); ph = L.pivot(index='frame_id', columns='link_id', values='phase_corrected_range_m')
    psi = np.radians(P.psi.to_numpy()); c, s = np.cos(psi), np.sin(psi)
    fwd = np.c_[np.sin(psi), -np.cos(psi)]; right = np.c_[np.cos(psi), np.sin(psi)]   # body +x (T2 side)
    for l in LINKS:
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]
        px = P.cam_X.to_numpy() + c * b[0] - s * b[1]; py = P.cam_Y.to_numpy() + s * b[0] + c * b[1]
        v = np.c_[A[0] - px, A[1] - py]; dh = np.linalg.norm(v, axis=1); u = v / dh[:, None]
        pred = np.sqrt(dh ** 2 + (A[2] - TZ) ** 2)
        meas = np.where(P['mode'].to_numpy() == 'PHASE', ph.reindex(P.frame_id)[l].to_numpy(float), ds.reindex(P.frame_id)[l].to_numpy(float))
        beta_t = np.degrees(np.arctan2((u * right).sum(1), (u * fwd).sum(1)))      # anchor seen from the tag, + toward T2 side
        beta_a = np.degrees(np.arctan2(-u[:, 0], -u[:, 1]))                         # tag seen from the anchor, 0 = +Y, + toward +X
        rows.append(pd.DataFrame(dict(trial=t, path=R.loc[t, 'path'], policy=R.loc[t, 'policy'], link=l, tag=tag, anc=anc, frame=np.arange(len(P)), t=P.t.to_numpy(),
                                      mode=P['mode'].to_numpy(), seg=P.seg.to_numpy(), psi=P.psi.to_numpy(), d=pred, beta_t=beta_t, beta_a=beta_a, resid=100 * (meas - pred))))
D = pd.concat(rows).dropna(subset=['resid']); FL = pd.concat(fails)
D = D[D.seg.isin(['far', 'mid', 'near'])]
D['hbin'] = pd.cut(D.psi.abs(), [0, 15, 30, 45, 100])

def excess_table(frame, col, label):
    w = frame.pivot_table(index=['trial', 'frame'], columns='link', values=col).dropna()
    aux = frame[frame.link == 'T1_A1'].set_index(['trial', 'frame'])[['psi', 'path', 'seg', 'd']]
    w = w.join(aux)
    w['excess'] = w.T2_A1 - w[['T1_A1', 'T1_A2', 'T2_A2']].mean(axis=1)
    w['hbin'] = pd.cut(w.psi.abs(), [0, 15, 30, 45, 100])
    return w

W = excess_table(D, 'resid', 'raw')
print('== 1. pattern over the four links on C1 (median residual relative to the frame mean, cm) by |heading|')
c1 = W[(W.path == 'C1') & W.seg.isin(['far', 'mid'])]
rel = c1[LINKS].sub(c1[LINKS].mean(axis=1), axis=0); rel['hbin'] = c1.hbin
print(rel.groupby('hbin', observed=True).median().round(1).to_string())

print('== 2. floor-reflection test on C1, |heading| 30-45 deg: T2-A1 excess by link distance against 2*h_t*h_a/d')
g = c1[(c1.psi.abs() >= 30) & (c1.psi.abs() < 45)].copy(); g['dbin'] = pd.cut(g.d, [3, 5, 6, 7, 8, 12])
tab = g.groupby('dbin', observed=True).agg(n=('excess', 'size'), d=('d', 'median'), excess=('excess', 'median'))
tab['floor_bounce'] = 100 * 2 * TZ * ANC['A1'][2] / tab.d
print(tab.round(1).to_string())

# 3. additive orientation model
def design(frame, cols=None):
    X = pd.concat([pd.get_dummies(frame['mode'], prefix='m'),
                   pd.get_dummies(frame.tag + '_' + pd.cut(frame.beta_t, np.arange(-100, 101, 10)).astype(str), prefix='t'),
                   pd.get_dummies(frame.anc + '_' + pd.cut(frame.beta_a, np.arange(-100, 101, 10)).astype(str), prefix='a'),
                   pd.get_dummies(pd.cut(frame.d, [0, 2, 3, 4, 5, 6, 8, 12]).astype(str), prefix='d')], axis=1).astype(float)
    return X if cols is None else X.reindex(columns=cols, fill_value=0.0)

M = D[(D.seg != 'near') | (D.frame % 20 == 0)].copy()      # the stop is one geometry repeated, so it is thinned
M = M[M.frame % 3 == 0]
def fit_predict(train, test):
    Xtr = design(train); coef, *_ = np.linalg.lstsq(Xtr.to_numpy(), train.resid.to_numpy(), rcond=None)
    known = (design(test, Xtr.columns).to_numpy() * (np.abs(coef) > 0)).sum(1) == 4       # all four terms seen in training
    return design(test, Xtr.columns).to_numpy() @ coef, known, pd.Series(coef, index=Xtr.columns)

pred, known, coef = fit_predict(M, M); M['res_model'] = M.resid - pred
print('== 3a. additive model fitted on all runs: variance explained', round(1 - M.res_model.var() / M.resid.var(), 3), ' rows', len(M))
is_c1 = M.path == 'C1'
pred2, known2, _ = fit_predict(M[~is_c1], M[is_c1]); M.loc[is_c1, 'res_cv'] = M.resid[is_c1] - pred2; M.loc[is_c1, 'cv_known'] = known2
print('   C1 rows whose bearing and distance bins all occur on the straight path or C2:', round(100 * known2.mean(), 1), '%')
for col, name in (('resid', 'measured'), ('res_model', 'after model fitted on all runs'), ('res_cv', 'after model fitted without C1')):
    sub = M[is_c1 & M.seg.isin(['far', 'mid'])]
    if col == 'res_cv': sub = sub[sub.cv_known == True]
    w = excess_table(sub, col, name)
    print(f'   T2-A1 excess on C1 by |heading| ({name}):', w.groupby('hbin', observed=True).excess.agg(['median', 'size']).round(1).to_dict('index'))
print('== 3b. fitted orientation terms (cm, relative to the bin at 0 deg of the same device)')
for dev, pre in (('T1', 't_T1_'), ('T2', 't_T2_'), ('A1', 'a_A1_'), ('A2', 'a_A2_')):
    cs = coef[[k for k in coef.index if k.startswith(pre)]]
    cs.index = [float(k.split('(')[1].split(',')[0]) + 5 for k in cs.index]; cs = cs.sort_index()
    n = M.groupby(pd.cut(M.beta_t if pre[0] == 't' else M.beta_a, np.arange(-100, 101, 10)), observed=True).size()
    ref = cs.get(-5.0, np.nan) if not np.isnan(cs.get(-5.0, np.nan)) else cs.iloc[len(cs) // 2]
    print(f'   {dev}:', {int(k): round(v - ref, 1) for k, v in cs.items() if abs(v) > 0})
coef.to_csv(OUT / 'link_cause_additive_model_coefficients.csv')

print('== 3c. same tag bearing, two anchors: residual of T2 links by tag bearing (median cm, frame-mean removed)')
relD = D.copy(); relD['rel'] = relD.resid - relD.groupby(['trial', 'frame']).resid.transform('mean'); relD['bt'] = pd.cut(relD.beta_t, [-60, -40, -20, -10, 0, 10, 20, 30, 40, 60])
print(relD[relD.seg.isin(['far', 'mid'])].groupby(['bt', 'link'], observed=True).rel.median().unstack().round(1).to_string())
print('   by path, T2-A1 only:'); print(relD[(relD.link == 'T2_A1') & relD.seg.isin(['far', 'mid'])].groupby(['bt', 'path'], observed=True).rel.agg(['median', 'size']).unstack().round(1).to_string())

print('== 4a. range noise by link and |heading| on C1 (cm): SD of the frame-to-frame difference / sqrt(2), median over runs; DS and SS frames')
N = D[(D.path == 'C1') & D.seg.isin(['far', 'mid']) & D['mode'].isin(['SS', 'DS'])].sort_values(['trial', 'link', 'frame']).copy()
N['diff'] = N.groupby(['trial', 'link']).resid.diff()
noise = N.groupby(['mode', 'hbin', 'link', 'trial'], observed=True)['diff'].agg(lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))) / np.sqrt(2)).groupby(['mode', 'hbin', 'link'], observed=True).median().unstack()
print(noise.round(2).to_string())
# exchange failures are in the tag diagnostic counters, see link_failures_diag.py (links.csv holds successful exchanges only)
W.reset_index().round(3).to_csv(OUT / 'link_cause_frames.csv', index=False)


# ---- 5. decisive test of an additive device-orientation bias: predict T2-A1 on the curves from the other links only
hold = (M.link == 'T2_A1') & M.path.isin(['C1', 'C2'])
pred5, known5, coef5 = fit_predict(M[~hold], M[hold])
H = M[hold].copy(); H['pred'] = pred5; H['known'] = known5
ref = M[~hold & (M.link == 'T2_A1')].resid.median()          # T2-A1 on the straight path
H['hb'] = pd.cut(H.psi.abs(), [0, 15, 30, 45, 100])
print('== 5. T2-A1 on C1/C2 predicted by a model that never saw T2-A1 on a curve (cm). Rows with all terms seen elsewhere:', round(100 * known5.mean(), 1), '%')
print(H[H.known & H.seg.isin(['far', 'mid'])].groupby(['path', 'hb'], observed=True).agg(n=('resid', 'size'), measured=('resid', 'median'), predicted=('pred', 'median')).round(1).to_string())
print('   T2-A1 median on the straight path:', round(ref, 1))
# the other three links, same frames, predicted by the same model (in-sample for them)
O = M[M.path.isin(['C1']) & M.seg.isin(['far', 'mid']) & (M.link != 'T2_A1')].copy(); O['pred'] = design(O, coef5.index).to_numpy() @ coef5.to_numpy(); O['hb'] = pd.cut(O.psi.abs(), [0, 15, 30, 45, 100])
print(O.groupby(['hb', 'link'], observed=True).agg(measured=('resid', 'median'), predicted=('pred', 'median')).round(1).unstack().to_string())

print('== 6. where the vehicle was when T2-A1 read long (C1 and C2, far and mid): position, bearings')
c = W[W.path.isin(['C1', 'C2']) & W.seg.isin(['far', 'mid'])].reset_index()
aux = D[(D.link == 'T2_A1')][['trial', 'frame', 'beta_t', 'beta_a', 'd']].rename(columns={'d': 'd21'})
aux2 = D[(D.link == 'T2_A2')][['trial', 'frame', 'beta_t', 'beta_a']].rename(columns={'beta_t': 'bt22', 'beta_a': 'ba22'})
aux1 = D[(D.link == 'T1_A1')][['trial', 'frame', 'beta_t', 'beta_a']].rename(columns={'beta_t': 'bt11', 'beta_a': 'ba11'})
c = c.merge(aux, on=['trial', 'frame']).merge(aux2, on=['trial', 'frame']).merge(aux1, on=['trial', 'frame'])
c['bt'] = pd.cut(c.beta_t, [-60, 0, 10, 20, 30, 40, 60, 100])
print(c.groupby(['path', 'bt'], observed=True).agg(n=('excess', 'size'), excess=('excess', 'median'), heading=('psi', 'median'), d_T2A1=('d21', 'median'), anchor_bearing_T2A1=('beta_a', 'median'),
      T2_bearing_to_A2=('bt22', 'median'), A2_bearing=('ba22', 'median'), T1_bearing_to_A1=('bt11', 'median'), A1_bearing_T1=('ba11', 'median')).round(1).to_string())
print('== 7. T2-A1 excess by mode on C1, |heading| 30-45 deg')
cm = c[(c.path == 'C1') & (c.psi.abs() >= 30) & (c.psi.abs() < 45)].merge(D[D.link == 'T2_A1'][['trial', 'frame', 'mode', 'policy']], on=['trial', 'frame'])
print(cm.groupby('mode').excess.agg(['median', 'size']).round(1).to_string())
print(cm.groupby('trial').excess.median().round(1).describe()[['min', '50%', 'max']].round(1).to_dict())


# ---- v41: the values quoted in the manuscript, for the audit
_h = '(30, 45]'
_n = noise.copy(); _n.index = [f'{m}|{h}' for m, h in _n.index]
_t5 = H[H.known & H.seg.isin(['far', 'mid'])].groupby(['path', 'hb'], observed=True).agg(measured=('resid', 'median'), predicted=('pred', 'median')); _t5.index = [f'{a}|{b}' for a, b in _t5.index]
_c6 = c.groupby(['path', 'bt'], observed=True).excess.median(); _c6.index = [f'{a}|{b}' for a, b in _c6.index]
_bins = ('(10, 20]', '(20, 30]', '(30, 40]', '(40, 60]')
_c2 = [float(_c6['C2|' + b]) for b in _bins]; _c1 = [float(_c6['C1|' + b]) for b in _bins]
_others = [l for l in noise.columns if l != 'T2_A1']
_no = [float(_n.loc[m + '|' + _h, l]) for m in ('DS', 'SS') for l in _others]
_sm = {'noise_T2A1_DS_cm': float(_n.loc['DS|' + _h, 'T2_A1']), 'noise_T2A1_SS_cm': float(_n.loc['SS|' + _h, 'T2_A1']), 'noise_other_links_min_cm': min(_no), 'noise_other_links_max_cm': max(_no),
       'model_measured_C1_cm': float(_t5.loc['C1|' + _h, 'measured']), 'model_predicted_C1_cm': float(_t5.loc['C1|' + _h, 'predicted']), 'straight_reference_cm': float(ref),
       'floor_bounce_first_bin_cm': float(tab.floor_bounce.iloc[0]), 'floor_bounce_7_8m_cm': float(tab.floor_bounce.iloc[3]), 'excess_first_bin_cm': float(tab.excess.iloc[0]), 'excess_7_8m_cm': float(tab.excess.iloc[3]),
       'excess_C2_by_bearing_min_cm': min(_c2), 'excess_C2_by_bearing_max_cm': max(_c2), 'excess_C1_by_bearing_min_cm': min(_c1), 'excess_C1_by_bearing_max_cm': max(_c1),
       'excess_C1_bearing_0_10_cm': float(_c6['C1|(0, 10]']), 'excess_C1_bearing_below_0_cm': float(_c6['C1|(-60, 0]']),
       'excess_DS_cm': float(cm[cm['mode'] == 'DS'].excess.median()), 'excess_SS_cm': float(cm[cm['mode'] == 'SS'].excess.median())}
pd.Series(_sm, name='value').round(3).to_csv(OUT / 'link_cause_summary.csv', header=True, index_label='quantity')
