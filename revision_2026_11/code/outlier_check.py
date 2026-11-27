"""Outlier (machine-error) removal sensitivity check.
Rule: drop a UWB pose if solver range residual RMS > 0.10 m, or if its lateral/longitudinal error vs the camera
is more than 5 scaled MAD from the run-segment median (Hampel). Scenarios: none / STAGED only (inspection only,
asymmetric, not for the paper) / all policies (same rule for every policy)."""
import glob, os, math
import numpy as np, pandas as pd
E0 = pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/segment_errors.csv')
meta = E0.drop_duplicates('trial').set_index('trial')[['path', 'policy', 'block']]
P = []
for f in glob.glob(r'D:/sandbox/uwb/output/field_20261124/cam/eval/pairs_T*.csv'):
    t = os.path.basename(f)[6:-4]; p = pd.read_csv(f); p['trial'] = t; P.append(p)
P = pd.concat(P, ignore_index=True)
P = P.join(meta, on='trial')
P = P[P.log_lat.notna() & P.log_lon.notna()]
def hampel(g, col):
    med = g[col].median(); mad = 1.4826 * (g[col] - med).abs().median()
    return (g[col] - med).abs() > 5 * max(mad, 0.005)
P['bad_res'] = P.range_residual_rms_m > 0.10
P['bad_lat'] = P.groupby(['trial', 'seg'], group_keys=False).apply(lambda g: hampel(g, 'log_lat'))
P['bad_lon'] = P.groupby(['trial', 'seg'], group_keys=False).apply(lambda g: hampel(g, 'log_lon'))
P['bad'] = P.bad_res | P.bad_lat | P.bad_lon
tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
out = []
for scen in ('none', 'staged_only', 'all'):
    if scen == 'none': Q = P
    elif scen == 'staged_only': Q = P[~(P.bad & (P.policy == 'STAGED'))]
    else: Q = P[~P.bad]
    seg = Q.groupby(['trial', 'path', 'policy', 'block', 'seg']).apply(lambda g: pd.Series({
        'n': len(g), 'lat_rmse': 100 * math.sqrt((g.log_lat ** 2).mean()), 'lon_rmse': 100 * math.sqrt((g.log_lon ** 2).mean())})).reset_index()
    tab = seg.groupby(['path', 'seg', 'policy'])[['lat_rmse', 'lon_rmse']].median().round(1)
    tab.columns = [f'{c}_{scen}' for c in tab.columns]; out.append(tab)
    seg.to_csv(rf'D:/sandbox/uwb/output/field_20261124/cam/eval/outlier_{scen}_segments.csv', index=False)
    print(f'\n### {scen}')
    for path in ('S', 'C1'):
        for sg in ('far', 'mid', 'near'):
            f = seg[(seg.path == path) & (seg.seg == sg)].pivot_table(index='block', columns='policy', values='lat_rmse')
            if 'STAGED' not in f: continue
            for o in [c for c in f.columns if c != 'STAGED']:
                d = (f['STAGED'] - f[o]).dropna()
                if len(d) >= 2:
                    print(f'{path} {sg:4s} STAGED-{o:11s} n{len(d)} mean {d.mean():+6.2f} +-{tq[len(d)] * d.std(ddof=1) / math.sqrt(len(d)):5.2f} cm')
T = pd.concat(out, axis=1)
T.to_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/outlier_compare_table.csv')
pd.set_option('display.width', 250)
print(T[[c for c in T.columns if c.startswith('lat')]].to_string())
rem = P.groupby(['path', 'policy']).agg(n=('bad', 'size'), removed=('bad', 'sum'), by_residual=('bad_res', 'sum'))
rem['pct'] = (100 * rem.removed / rem.n).round(1)
print(rem.to_string())
rem.to_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/outlier_removed_counts.csv')

