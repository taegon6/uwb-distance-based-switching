"""Same composite as threshold_composite.py, with the bins extended to 10.5 m so that the range matches the 4 to 10.5 m range of the paper.
Writes a new CSV only."""
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'
E = pd.read_csv(EV / 'segment_errors.csv').drop_duplicates('trial').set_index('trial')
S = E[(E.path == 'S') & E.policy.isin(['FIXED_SS', 'FIXED_DS'])]
edges = np.arange(0.5, 10.51, 0.5); mids = 0.5 * (edges[1:] + edges[:-1])
mse = {}; nmax = {}
for t, r in S.iterrows():
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['cam_dc', 'seg', 'log_lat']); P = P[P.seg.isin(['far', 'mid'])]
    nmax[t] = float(P.cam_dc.max())
    b = pd.cut(P.cam_dc, edges)
    mse[(int(r.block), r.policy)] = P.groupby(b, observed=False).log_lat.apply(lambda v: np.mean((100 * v) ** 2) if len(v) > 3 else np.nan).to_numpy()
blocks = sorted({k[0] for k in mse})
rows = []
for th in (0.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
    for b in blocks:
        ss, ds = mse[(b, 'FIXED_SS')], mse[(b, 'FIXED_DS')]
        comp = np.where(mids > th, ss, ds); ok = np.isfinite(ss) & np.isfinite(ds)
        rows.append(dict(threshold_m=th, block=b, lat_rmse_cm=float(np.sqrt(np.mean(comp[ok]))), share_ss=float(np.mean(mids[ok] > th)), bins=int(ok.sum())))
D = pd.DataFrame(rows)
out_csv = BASE / 'analysis' / 'revision_checks' / 'threshold_composite_10p5.csv'
assert not out_csv.exists(); D.round(4).to_csv(out_csv, index=False)
W = D.pivot(index='block', columns='threshold_m', values='lat_rmse_cm')
T4 = 2.776
out = pd.DataFrame({'lat_rmse_mean': W.mean(), 'share_ss': D.groupby('threshold_m').share_ss.mean(), 'bins': D.groupby('threshold_m').bins.mean()})
for th in W.columns:
    d = W[th] - W[4.0]; out.loc[th, 'diff_vs_4m'] = d.mean(); out.loc[th, 'ci_lo'] = d.mean() - T4 * d.std(ddof=1) / np.sqrt(len(d)); out.loc[th, 'ci_hi'] = d.mean() + T4 * d.std(ddof=1) / np.sqrt(len(d))
pd.set_option('display.width', 200); print(out.round(3).to_string())
print('max camera alignment distance per straight trial:', {k: round(v, 2) for k, v in nmax.items()})

