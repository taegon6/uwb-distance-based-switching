
import math
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'
E = pd.read_csv(EV / 'segment_errors.csv'); sched = E.drop_duplicates('trial').set_index('trial')[['path', 'policy', 'block']]
MODE = {'FIXED_SS': 'SS', 'FIXED_DS': 'DS', 'FIXED_PHASE': 'PHASE'}
rows = []
for t in sched.index[(sched.path == 'S') & sched.policy.isin(list(MODE))]:
    P = pd.read_csv(EV / f'pairs_{t}.csv', usecols=['t', 'seg', 'yaw_deg', 'log_lat', 'cam_dc'])
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    P['dyaw'] = (P.yaw_deg - np.interp(P.t, C.t, C.psi) + 180) % 360 - 180
    X = C.X.rolling(15, center=True, min_periods=3).mean(); Y = C.Y.rolling(15, center=True, min_periods=3).mean()
    P['speed'] = np.interp(P.t, C.t, np.hypot(np.gradient(X, C.t), np.gradient(Y, C.t)))
    still_start = P[(P.seg == 'wait')]; still_end = P[(P.seg == 'near') & (P.speed < 0.03)]
    moving = P[P.seg.isin(['far', 'mid']) & (P.speed >= 0.8)]
    for name, g in (('standing at the start (10 m)', still_start), ('moving faster than 0.8 m/s', moving), ('standing at the stop', still_end)):
        if len(g) >= 15: rows.append(dict(block=sched.loc[t, 'block'], mode=MODE[sched.loc[t, 'policy']], state=name, yaw=g.dyaw.mean(), lat=100 * g.log_lat.mean(), n=len(g)))
D = pd.DataFrame(rows)
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
print(D.groupby(['state', 'mode'])[['yaw', 'lat']].mean().round(2).unstack().to_string())
for state, g in D.groupby('state'):
    W = g.pivot(index='block', columns='mode', values=['yaw', 'lat'])
    for other in ('DS', 'PHASE'):
        for q, unit in (('yaw', 'deg'), ('lat', 'cm')):
            d = (W[(q, other)] - W[(q, 'SS')]).dropna()
            print(f'{state:30s} {other:5s} - SS {q}: {d.mean():+.2f} +- {TQ[len(d)] * d.std(ddof=1) / math.sqrt(len(d)):.2f} {unit} (n {len(d)}, negative in {int((d < 0).sum())})')
D.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'mode_bias_standing_vs_moving.csv', index=False)

