"""R2 Comment 4: is 4.0 m a reasonable SS-TWR to DS-TWR threshold? Composite of the fixed SS-TWR and fixed DS-TWR runs of each
straight block for several thresholds. Lateral error against the camera in 0.5 m bins of the reference alignment distance (0.5-10 m);
every bin has equal weight, SS-TWR above the threshold and DS-TWR below."""
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'
E = pd.read_csv(EV / 'segment_errors.csv').drop_duplicates('trial').set_index('trial')
S = E[(E.path == 'S') & E.policy.isin(['FIXED_SS', 'FIXED_DS'])]
edges = np.arange(0.5, 10.01, 0.5); mids = 0.5 * (edges[1:] + edges[:-1])
RATE = {'FIXED_SS': 116.5, 'FIXED_DS': 50.2}
mse = {}
for t, r in S.iterrows():
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['cam_dc', 'seg', 'log_lat']); P = P[P.seg.isin(['far', 'mid'])]
    b = pd.cut(P.cam_dc, edges)
    mse[(int(r.block), r.policy)] = P.groupby(b, observed=False).log_lat.apply(lambda v: np.mean((100 * v) ** 2) if len(v) > 3 else np.nan).to_numpy()
blocks = sorted({k[0] for k in mse})
rows = []
for th in (0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
    for b in blocks:
        ss, ds = mse[(b, 'FIXED_SS')], mse[(b, 'FIXED_DS')]
        comp = np.where(mids > th, ss, ds); ok = np.isfinite(ss) & np.isfinite(ds)
        rows.append(dict(threshold_m=th, block=b, lat_rmse_cm=float(np.sqrt(np.mean(comp[ok]))), share_ss=float(np.mean(mids[ok] > th)),
                         mean_rate_hz=float(np.mean(np.where(mids[ok] > th, RATE['FIXED_SS'], RATE['FIXED_DS']))), bins=int(ok.sum())))
D = pd.DataFrame(rows); D.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'threshold_composite.csv', index=False)
W = D.pivot(index='block', columns='threshold_m', values='lat_rmse_cm')
T4 = 2.776
out = pd.DataFrame({'lat_rmse_mean': W.mean(), 'rate_hz': D.groupby('threshold_m').mean_rate_hz.mean(), 'share_ss': D.groupby('threshold_m').share_ss.mean()})
for th in W.columns:
    d = W[th] - W[4.0]; out.loc[th, 'diff_vs_4m'] = d.mean(); out.loc[th, 'ci95'] = T4 * d.std(ddof=1) / np.sqrt(len(d)); out.loc[th, 'worse_blocks'] = int((d > 0).sum())
pd.set_option('display.width', 200); print(out.round(2).to_string()); print(W.round(2).to_string())
# per-bin difference SS - DS (mean over blocks, cm RMSE)
ss = np.sqrt(np.array([mse[(b, 'FIXED_SS')] for b in blocks])); ds = np.sqrt(np.array([mse[(b, 'FIXED_DS')] for b in blocks]))
d = ss - ds; n = np.sum(np.isfinite(d), axis=0)
print('bin mid [m]         ', ' '.join(f'{m:5.2f}' for m in mids))
print('SS - DS RMSE [cm]   ', ' '.join(f'{v:5.1f}' for v in np.nanmean(d, axis=0)))
print('blocks with SS worse', ' '.join(f'{int(v):5d}' for v in np.nansum(d > 0, axis=0)))

