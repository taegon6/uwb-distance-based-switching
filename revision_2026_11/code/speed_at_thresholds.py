import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv', usecols=['trial', 'path', 'policy', 'run']).set_index('trial')
pd.set_option('display.width', 220)
rows = []
for t in R.index:
    f = EV / f'camtrack_{t}.csv'
    if not f.exists(): continue
    C = pd.read_csv(f, usecols=['t', 'X', 'Y', 'RX_X', 'RX_Y']).dropna().sort_values('t')
    g = np.arange(C.t.min(), C.t.max(), 0.05); x = np.interp(g, C.t, C.X); y = np.interp(g, C.t, C.Y); dc = np.hypot(np.interp(g, C.t, C.RX_X), np.interp(g, C.t, C.RX_Y))
    k = 6; sp = np.hypot(x[k:] - x[:-k], y[k:] - y[:-k]) / (g[k:] - g[:-k]); dcm = dc[k // 2: k // 2 + len(sp)]
    def at(lo, hi):
        m = (dcm > lo) & (dcm < hi); return float(np.median(sp[m])) if m.sum() > 2 else np.nan
    rows.append(dict(trial=t, path=R.loc[t, 'path'], v_4m=at(3.8, 4.2), v_1m=at(0.9, 1.1), v_0p5m=at(0.45, 0.55), v_0p25m=at(0.2, 0.3)))
V = pd.DataFrame(rows)
print(V[['v_4m', 'v_1m', 'v_0p5m', 'v_0p25m']].describe().loc[['count', '50%', '25%', '75%', 'min', 'max']].round(2).to_string())
print(V.groupby('path')[['v_4m', 'v_0p5m']].median().round(2).to_string())
V.round(3).to_csv(BASE / 'analysis' / 'revision_checks' / 'speed_at_thresholds.csv', index=False)

