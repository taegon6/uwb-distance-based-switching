"""RX coil estimate at the final stop by policy (26 Nov). Same data and definitions as rx_components2.py (field offsets from the straight stops,
3 s median, each stop in its own final mode); only the grouping is new. Saves analysis/revision_checks/rx_stop_by_policy.csv."""
import pandas as pd, numpy as np, math
A = r'D:/sandbox/uwb/output/field_20261124/analysis'
s = pd.read_csv(A + '/stops_v2_seltos_L439.csv')
e = pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/segment_errors.csv')[['trial', 'block']].drop_duplicates()
rs = pd.read_csv(A + '/runs_summary_dedup.csv')[['trial', 'policy']]
d = s.merge(rs, on='trial').merge(e, on='trial', how='left')
lat, lon = d.sfit_rx_dlat_cm, d.sfit_rx_dlon_cm; ax, ay = d.rx_ref_x_cm, d.rx_ref_y_cm
d['yaw_abs'] = d.sfit_dyaw.abs(); d['lat_abs'] = lat.abs(); d['lon_abs'] = lon.abs()
d['in_ref'] = (ax.abs() <= 10) & (ay.abs() <= 7.5); d['in_est'] = ((ax + lat).abs() <= 10) & ((ay + lon).abs() <= 7.5)
rows = []
for name, g in list(d.groupby('policy')) + [('ALL', d)] + [(f'{p}/{q}', g) for (p, q), g in d.groupby(['path', 'policy'])]:
    rows.append(dict(group=name, n=len(g), rx_median_cm=g.sfit_rx_err_cm.median(), rx_within_10cm=int((g.sfit_rx_err_cm <= 10).sum()), lat_median_cm=g.lat_abs.median(), lon_median_cm=g.lon_abs.median(),
                     heading_median_deg=g.yaw_abs.median(), tag_median_cm=g.sfit_err_cm.median(), tag_within_10cm=int((g.sfit_err_cm <= 10).sum()),
                     inside_ref=int(g.in_ref.sum()), est_inside=int(g.in_est.sum()), est_inside_and_ref_inside=int((g.in_est & g.in_ref).sum()), est_inside_ref_outside=int((g.in_est & ~g.in_ref).sum())))
R = pd.DataFrame(rows); R.round(2).to_csv(A + '/revision_checks/rx_stop_by_policy.csv', index=False)
pd.set_option('display.width', 250); print(R.round(2).to_string(index=False))
rng = np.random.default_rng(1)
def perm(col):
    x = d[d.policy == 'STAGED'][col].to_numpy(); y = d[d.policy != 'STAGED'][col].to_numpy(); allv = np.r_[x, y]; r = pd.Series(allv).rank().to_numpy(); obs = r[:len(x)].mean() - r[len(x):].mean(); c = 0; N = 20000
    for _ in range(N):
        p = rng.permutation(len(allv)); c += (r[p[:len(x)]].mean() - r[p[len(x):]].mean()) <= obs
    return np.median(x), np.median(y), c / N
for col in ('sfit_rx_err_cm', 'yaw_abs', 'lat_abs', 'sfit_err_cm'):
    print(col, 'switching %.2f, fixed %.2f, one-sided rank permutation p %.3f' % perm(col))
T = {5: 2.776, 10: 2.262}
for path in (['S'], ['C1'], ['S', 'C1']):
    for b in ('FIXED_DS', 'FIXED_SS', 'FIXED_PHASE'):
        for col in ('sfit_rx_err_cm', 'yaw_abs'):
            g = d[d.path.isin(path)]; A_ = g[g.policy == 'STAGED'].set_index(['path', 'block'])[col]; B_ = g[g.policy == b].set_index(['path', 'block'])[col]; x = (A_ - B_).dropna()
            if len(x) in T: print(path, 'switching -', b, col, '%+.1f ± %.1f (n %d, lower in %d)' % (x.mean(), T[len(x)] * x.std(ddof=1) / math.sqrt(len(x)), len(x), int((x < 0).sum())))

