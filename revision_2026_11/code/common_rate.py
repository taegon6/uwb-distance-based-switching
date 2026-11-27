"""Common output rate: average each policy's poses into fixed windows (10 Hz, 2 Hz) and compare with the camera.
Straight runs, lateral error (pairs_T*.csv, logged offsets). Block-paired differences between policies."""
import glob, os, math
import numpy as np, pandas as pd
EV = r'D:/sandbox/uwb/output/field_20261124/cam/eval'; OUTR = r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks'
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
TQ = {5: 2.776}
rows = []
for f in glob.glob(EV + '/pairs_T*.csv'):
    t = os.path.basename(f)[6:-4]
    if sched.loc[t, 'path'] != 'S': continue
    p = pd.read_csv(f).dropna(subset=['log_lat'])
    for seg in ('far', 'mid'):
        g = p[p.seg == seg]
        if len(g) < 20: continue
        row = dict(trial=t, policy=sched.loc[t, 'policy'], block=sched.loc[t, 'block'], seg=seg, raw=100 * math.sqrt((g.log_lat ** 2).mean()), n_raw=len(g))
        for hz in (10, 2):
            w = g.groupby(np.floor(g.t * hz)).log_lat.agg(['mean', 'size'])
            row[f'hz{hz}'] = 100 * math.sqrt((w['mean'] ** 2).mean()); row[f'per_window_{hz}'] = w['size'].median()
            sd = g.groupby(np.floor(g.t * hz)).log_lat.std().dropna()
            row[f'within_sd_{hz}'] = 100 * sd.median() if len(sd) else np.nan
        e = g.log_lat.to_numpy() * 100; row['bias'] = e.mean(); row['sd'] = e.std()
        # lag-1 autocorrelation of consecutive lateral errors and the SD of a 0.1 s average around a 1 s moving mean
        row['lag1'] = np.corrcoef(e[:-1], e[1:])[0, 1]
        slow = g.log_lat.rolling(window=max(3, int(len(g) / max(g.t.max() - g.t.min(), 1e-3))), center=True, min_periods=3).mean()
        row['fast_sd'] = 100 * (g.log_lat - slow).std()
        w10 = (g.log_lat - slow).groupby(np.floor(g.t * 10)).mean(); row['fast_sd_10hz'] = 100 * w10.std()
        rows.append(row)
D = pd.DataFrame(rows); D.to_csv(OUTR + '/common_rate_lateral.csv', index=False)
pd.set_option('display.width', 220)
print(D.groupby(['seg', 'policy'])[['raw', 'hz10', 'hz2', 'per_window_10', 'bias', 'sd', 'lag1', 'fast_sd', 'fast_sd_10hz']].median().round(2).to_string())
for seg in ('far', 'mid'):
    for col in ('hz10', 'fast_sd_10hz'):
        P = D[D.seg == seg].pivot(index='block', columns='policy', values=col)
        for a, b in (('FIXED_SS', 'FIXED_DS'), ('FIXED_SS', 'FIXED_PHASE'), ('STAGED', 'FIXED_DS'), ('STAGED', 'FIXED_PHASE')):
            d = (P[a] - P[b]).dropna(); h = TQ[5] * d.std(ddof=1) / math.sqrt(len(d))
            print(f'{seg} {col}: {a} - {b} = {d.mean():+.2f} +- {h:.2f} cm, lower in {int((d < 0).sum())}/{len(d)}')

