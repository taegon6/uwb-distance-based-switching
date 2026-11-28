
import numpy as np, pandas as pd
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/range_residual.py', encoding='utf-8').read().split("pd.set_option('display.width', 220)")[0])
D['frame'] = D.groupby(['trial', 'link']).cumcount()
W = D.pivot_table(index=['trial', 'path', 'policy', 'seg', 'frame'], values='resid_cm', columns='link').dropna().reset_index()
aux = D[D.link == 'T1_A1'][['trial', 'frame', 'psi', 'cam_dc', 'solver_rms_cm']]
W = W.merge(aux, on=['trial', 'frame'])
W['excess'] = W.T2_A1 - W[['T1_A1', 'T1_A2', 'T2_A2']].mean(axis=1)
W['spread'] = W[LINKS].max(axis=1) - W[LINKS].min(axis=1)
pd.set_option('display.width', 220)
g = W[W.seg == 'far'].groupby(['path', 'trial']).agg(excess=('excess', 'median'), spread=('spread', 'median'), rms=('solver_rms_cm', 'median'), psi=('psi', 'median'), n=('excess', 'size')).round(1)
print(g.to_string())
print(g.groupby('path')[['excess', 'spread', 'rms']].describe().loc[:, (slice(None), ['min', '50%', 'max'])].round(1).to_string())
C = W[(W.path == 'C1') & W.seg.isin(['far', 'mid'])].copy(); C['hbin'] = pd.cut(C.psi.abs(), [0, 5, 15, 30, 45, 90])
print(C.groupby('hbin', observed=True).agg(n=('excess', 'size'), excess=('excess', 'median'), spread=('spread', 'median'), rms=('solver_rms_cm', 'median'), dc=('cam_dc', 'median')).round(1).to_string())
S = W[(W.path == 'S') & W.seg.isin(['far', 'mid'])].copy(); S['bin'] = pd.cut(S.cam_dc, [0.5, 2, 4, 6, 8, 10.5])
print(S.groupby('bin', observed=True).agg(n=('excess', 'size'), excess=('excess', 'median'), spread=('spread', 'median'), rms=('solver_rms_cm', 'median'), common=('T1_A1', 'median')).round(1).to_string())
W.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'range_residual_frames.csv', index=False)

