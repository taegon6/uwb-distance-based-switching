
import math
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'; OUTR = BASE / 'analysis' / 'revision_checks'
D = pd.read_csv(OUTR / 'motion_skew.csv')
rows = []
for t, g in D.groupby('trial'):
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['t', 'yaw_deg']); C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    g = g.merge(P, on='t', how='left'); g['obs_yaw'] = (g.yaw_deg - np.interp(g.t, C.t, C.psi) + 180) % 360 - 180
    rows.append(g)
D = pd.concat(rows)
# same distance band for every mode, so that distance-dependent offsets cancel in the mode differences
D['dbin'] = pd.cut(D.cam_dc, [0.5, 2, 4, 6, 8, 10.5]); D['fast'] = np.where(D.speed >= 0.8, 'fast (>= 0.8 m/s)', 'slow (< 0.8 m/s)')
cell = D.groupby(['block', 'mode', 'dbin', 'fast'], observed=True).agg(yaw=('obs_yaw', 'mean'), lat=('obs_lat', 'mean'), pyaw=('pred_yaw', 'mean'), plat=('pred_lat', 'mean'), v=('speed', 'mean'), n=('speed', 'size')).reset_index()
cell = cell[cell.n >= 15]
W = cell.pivot_table(index=['block', 'dbin', 'fast'], columns='mode', values=['yaw', 'lat', 'pyaw', 'plat', 'v'], observed=True)
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
pd.set_option('display.width', 220)
for other in ('DS', 'PHASE'):
    print('==', other, 'minus SS, same block, same distance band, same speed class')
    out = []
    for (dbin, fast), g in W.groupby(level=['dbin', 'fast'], observed=True):
        r = dict(dbin=str(dbin), speed_class=fast)
        for q, name in (('yaw', 'heading_deg'), ('lat', 'lateral_cm')):
            d = (g[(q, other)] - g[(q, 'SS')]).dropna(); p = (g[('p' + q, other)] - g[('p' + q, 'SS')]).dropna()
            if len(d) >= 3:
                r[name] = f'{d.mean():+.2f} +- {TQ[len(d)] * d.std(ddof=1) / math.sqrt(len(d)):.2f} (n {len(d)})'; r[name + '_model'] = round(p.mean(), 2)
        r['speed'] = round(g[('v', other)].mean(), 2)
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))

