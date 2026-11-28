
import pandas as pd, numpy as np
from pathlib import Path
EV = Path(r'D:/sandbox/uwb/output/field_20261124/cam/eval')
E = pd.read_csv(EV/'segment_errors.csv'); sched = E.drop_duplicates('trial').set_index('trial')[['path','policy']]
rows = []
for t in sched.index:
    c = pd.read_csv(EV/f'camtrack_{t}.csv').sort_values('t')
    c = c[np.isfinite(c.X)]
    # 0.5 s moving average on a 30 Hz grid, central difference over +-0.5 s (as in the original Section 4.1)
    g = np.arange(c.t.min(), c.t.max(), 1/30)
    X = pd.Series(np.interp(g, c.t, c.X)).rolling(15, center=True, min_periods=8).mean().to_numpy()
    Y = pd.Series(np.interp(g, c.t, c.Y)).rolling(15, center=True, min_periods=8).mean().to_numpy()
    d = np.hypot(np.interp(g, c.t, c.RX_X), np.interp(g, c.t, c.RX_Y))
    k = 15
    v = np.full(len(g), np.nan); v[k:-k] = np.hypot(X[2*k:] - X[:-2*k], Y[2*k:] - Y[:-2*k]) / (g[2*k:] - g[:-2*k])
    t4 = g[np.argmax(d < 4.0)]; t05 = g[np.argmax(d < 0.5)]
    for seg, m in (('far', g < t4), ('mid', (g >= t4) & (g < t05))):
        vv = v[m & np.isfinite(v)]
        if len(vv) > 10: rows.append(dict(trial=t, path=sched.loc[t,'path'], policy=sched.loc[t,'policy'], seg=seg, p95=np.percentile(vv, 95), mean=vv.mean()))
S = pd.DataFrame(rows)
print(S.groupby(['path','seg']).p95.describe()[['min','50%','max']].round(2))
print(S.groupby(['path','seg','policy']).p95.median().unstack().round(2))
S.to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/approach_speed.csv', index=False)

