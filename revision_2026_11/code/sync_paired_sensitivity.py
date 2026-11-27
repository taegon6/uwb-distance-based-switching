"""Sensitivity of the paired policy differences (straight path) to a shift of the camera time. Same error definition as revision_checks.py (section 1)."""
import sys, math
sys.argv = ['x']
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read()
exec(src.split('# ---------------- main ----------------')[0])
sched = SCHED.set_index('trial'); print(list(sched.columns))
rows = []
for t in sched.index[sched.path == 'S']:
    pf = OUT / f'pairs_{t}.csv'; cf = OUT / f'camtrack_{t}.csv'
    if not pf.exists(): continue
    p = pd.read_csv(pf, usecols=['t', 'tag_midpoint_x_m', 'tag_midpoint_y_m', 'seg', 'log_lat']); c = pd.read_csv(cf).sort_values('t')
    for s in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
        cx = np.interp(p.t + s, c.t, c.X, left=np.nan, right=np.nan)
        lat = -(p.tag_midpoint_x_m - cx)          # straight path: tangent (0, -1), lateral = ex * (-ty) + ey * tx = ex
        lat = (p.tag_midpoint_x_m - cx)
        for sg in ('far', 'mid'):
            m = (p.seg == sg) & np.isfinite(lat)
            rows.append(dict(trial=t, policy=sched.loc[t, 'policy'], block=int(sched.loc[t, 'block']), seg=sg, shift_ms=int(round(s * 1000)), lat_rmse=100 * math.sqrt(np.mean(lat[m] ** 2)),
                             check0=100 * math.sqrt(np.mean(p.log_lat[m] ** 2)) if s == 0 else np.nan))
D = pd.DataFrame(rows)
z = D[D.shift_ms == 0]; print('shift 0 against the stored lateral error: max difference %.4f cm' % (z.lat_rmse - z.check0).abs().max())
out = []
for sg in ('far', 'mid'):
    for other in ('FIXED_DS', 'FIXED_SS', 'FIXED_PHASE'):
        for s, g in D[D.seg == sg].groupby('shift_ms'):
            P = g.pivot_table(index='block', columns='policy', values='lat_rmse'); d = (P['STAGED'] - P[other]).dropna()
            out.append(dict(seg=sg, other=other, shift_ms=s, mean_cm=d.mean(), ci95_cm=2.776 * d.std(ddof=1) / math.sqrt(len(d)), blocks=len(d)))
O = pd.DataFrame(out); O.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'sync_sensitivity_paired_differences.csv', index=False)
pd.set_option('display.width', 220)
print(O.pivot_table(index=['seg', 'other'], columns='shift_ms', values='mean_cm').round(2).to_string())
print(O.pivot_table(index=['seg', 'other'], columns='shift_ms', values='ci95_cm').round(2).to_string())

