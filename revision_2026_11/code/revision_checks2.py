"""R3-M5 pose computation time (replay of logged links with the logger's own estimator) and
R3-M2 alternative staging (2-stage SS->HP, SS->DS only) from fixed-mode runs."""
import sys, time, io, math, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r'D:/sandbox/uwb/output/exp_ready_20261124/repo/pc')
from uwb_pc_logger.pose import Geometry2A2T, PoseEstimator2A2T
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; OUTR = BASE / 'analysis' / 'revision_checks'
geom = Geometry2A2T.from_json(BASE / 'configs' / 'vehicle_geometry_20261124.json')
jobs = pd.read_csv(BASE / 'cam' / 'jobs.csv').set_index('trial')
def rd(p):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False)
B = {'True': True, 'False': False, 'true': True, 'false': False, '1': True, '0': False}
res = []
for t in ('T11', 'T09', 'T12', 'T10', 'T28'):
    lk = rd(DAY / jobs.loc[t, 'run'] / 'tables' / 'links.csv')
    lk = lk[lk.success.astype(str).isin(['1', 'True', 'true'])]
    est = PoseEstimator2A2T(geom)
    times = []
    for fid, g in lk.groupby('frame_id', sort=True):
        links = []
        for r in g.to_dict('records'):
            for k in ('packet_valid', 'phase_valid'):
                r[k] = B.get(str(r.get(k)), bool(r.get(k)))
            links.append(r)
        a = time.perf_counter(); est.estimate(links, int(fid)); times.append(time.perf_counter() - a)
    x = np.array(times) * 1000
    res.append(dict(trial=t, frames=len(x), median_ms=np.median(x), p99_ms=np.percentile(x, 99), max_ms=x.max()))
T = pd.DataFrame(res).round(3); print('== pose estimate() time per frame (logger code, this desktop, single thread)'); print(T.to_string(index=False))
T.to_csv(OUTR / 'pose_compute_time.csv', index=False)

# alternative staging from fixed runs, straight blocks: per distance band use the mode a policy would use
P = []
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
for t in jobs.index:
    f = BASE / 'cam' / 'eval' / f'pairs_{t}.csv'
    if f.exists():
        p = pd.read_csv(f); p['trial'] = t; p['path'] = sched.loc[t, 'path']; p['policy'] = sched.loc[t, 'policy']; p['block'] = sched.loc[t, 'block']; P.append(p)
P = pd.concat(P)
alts = {'3-stage SS>DS>HP (proposed)': [(4.0, 99, 'FIXED_SS'), (0.5, 4.0, 'FIXED_DS'), (0, 0.5, 'FIXED_PHASE')],
        '2-stage SS>HP @0.5 m': [(0.5, 99, 'FIXED_SS'), (0, 0.5, 'FIXED_PHASE')],
        '2-stage SS>HP @4.0 m': [(4.0, 99, 'FIXED_SS'), (0, 4.0, 'FIXED_PHASE')],
        '2-stage SS>DS (no HP)': [(4.0, 99, 'FIXED_SS'), (0, 4.0, 'FIXED_DS')]}
rows = []
for name, bands in alts.items():
    for blk in range(1, 6):
        errs, rate_t, dist = [], 0.0, 0.0
        for lo, hi, pol in bands:
            g = P[(P.path == 'S') & (P.block == blk) & (P.policy == pol) & (P.d_c >= lo) & (P.d_c < hi)]
            if len(g) < 5: continue
            errs.append(g.log_lat.to_numpy())
            dur = g.t.max() - g.t.min()
            rate_t += len(g); dist += dur
        e = np.concatenate(errs)
        rows.append(dict(alternative=name, block=blk, lat_rmse_0_4m=100 * math.sqrt(np.mean(np.concatenate(
            [P[(P.path == 'S') & (P.block == blk) & (P.policy == pol) & (P.d_c >= max(lo, 0.5)) & (P.d_c < min(hi, 4.0))].log_lat.to_numpy() for lo, hi, pol in bands]) ** 2)),
            rate_0_5_4m=np.mean([len(P[(P.path == 'S') & (P.block == blk) & (P.policy == pol) & (P.d_c >= max(lo, 0.5)) & (P.d_c < min(hi, 4.0))]) /
                                  max(1e-3, np.ptp(P[(P.path == 'S') & (P.block == blk) & (P.policy == pol) & (P.d_c >= max(lo, 0.5)) & (P.d_c < min(hi, 4.0))].t) if len(P[(P.path == 'S') & (P.block == blk) & (P.policy == pol) & (P.d_c >= max(lo, 0.5)) & (P.d_c < min(hi, 4.0))]) > 1 else 1e-3)
                                  for lo, hi, pol in bands if min(hi, 4.0) > max(lo, 0.5)])))
A = pd.DataFrame(rows)
print('== alternative staging (straight, stitched from fixed runs, 0.5-4 m band)')
print(A.groupby('alternative')[['lat_rmse_0_4m', 'rate_0_5_4m']].median().round(2).to_string())
A.to_csv(OUTR / 'alternative_staging.csv', index=False)

