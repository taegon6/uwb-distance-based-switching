"""Whole-approach comparison, second version (after the second external review).
Availability is counted on the raw valid poses of poses.csv (independent of the camera). Errors need a camera match and are counted separately.
Window: from the time the camera shows 0.15 m of travel from the starting position (t_motion) to the first time the camera alignment distance falls below 0.50 m.
Grid 10 Hz. A grid time has an output if a valid pose is not older than 0.1 s.
Two error definitions: (A) error of the selected pose at its own time (as in the first version); (B) the selected pose held until the grid time, against the camera at the grid time."""
import io, math, numpy as np, pandas as pd
from pathlib import Path
B = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = B / 'cam' / 'eval'; RC = B / 'analysis' / 'revision_checks'; DAY = B / 'results_20261124'
ST = pd.read_csv(EV / 'segment_times.csv').set_index('trial'); R = pd.read_csv(B / 'analysis' / 'runs_summary_dedup.csv').set_index('trial'); SP = pd.read_csv(B / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
jobs = pd.read_csv(B / 'cam' / 'jobs.csv').set_index('trial')
def rd(p, cols):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False, usecols=cols)
STEP, AGE = 0.1, 0.1; rows = []
def latest(tv, grid):
    idx = np.searchsorted(tv, grid, side='right') - 1; ok = (idx >= 0) & (grid - tv[np.clip(idx, 0, len(tv) - 1)] <= AGE); return np.clip(idx, 0, len(tv) - 1), ok
for t in ST.index:
    f = EV / f'pairs_{t}.csv'
    if not f.exists(): continue
    s = ST.loc[t]; path = SP.loc[t, 'path']
    P = pd.read_csv(f, usecols=['t', 'valid', 'log_lat', 'log_lon', 'tag_midpoint_x_m', 'tag_midpoint_y_m']); P['valid'] = P.valid.astype(str).str.lower().isin(['true', '1'])
    V = P[P.valid & P.log_lat.notna()].sort_values('t'); tv = V.t.to_numpy()
    raw = rd(DAY / jobs.loc[t, 'run'] / 'tables' / 'poses.csv', ['host_pose_wall_ns', 'valid']); raw = raw[raw.valid.astype(str).str.lower().isin(['true', '1'])]
    tr = np.sort(raw.host_pose_wall_ns.astype('int64').to_numpy() / 1e9)
    grid = np.arange(s.t_motion + STEP, s.t_mid_end + 1e-9, STEP)
    _, has_raw = latest(tr, grid); idx, has = latest(tv, grid)
    lat = V.log_lat.to_numpy()[idx]; lon = V.log_lon.to_numpy()[idx]; eA = np.hypot(lat, lon)
    c = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t'); cx = np.interp(grid, c.t, c.X, left=np.nan, right=np.nan); cy = np.interp(grid, c.t, c.Y, left=np.nan, right=np.nan)
    ex = V.tag_midpoint_x_m.to_numpy(float)[idx] - cx; ey = V.tag_midpoint_y_m.to_numpy(float)[idx] - cy; eB = np.hypot(ex, ey)
    if path == 'S': latB = ex
    else:
        Xs = c.X.rolling(31, center=True, min_periods=5).mean(); Ys = c.Y.rolling(31, center=True, min_periods=5).mean(); vx = np.interp(grid, c.t, np.gradient(Xs, c.t)); vy = np.interp(grid, c.t, np.gradient(Ys, c.t)); n = np.hypot(vx, vy)
        tx = np.where(n > 0.02, vx / np.where(n > 0.02, n, 1), 0.0); ty = np.where(n > 0.02, vy / np.where(n > 0.02, n, 1), -1.0); latB = ex * (-ty) + ey * tx
    okB = has & np.isfinite(eB)
    win_raw = tr[(tr >= s.t_motion) & (tr <= s.t_mid_end)]; gaps_in = np.diff(win_raw); gaps_edge = np.diff(np.r_[s.t_motion, win_raw, s.t_mid_end])
    rows.append(dict(trial=t, path=path, policy=R.loc[t, 'policy'], duration_s=s.t_mid_end - s.t_motion, first_distance_m=s.cam_dc_first, grid_points=len(grid), valid_poses_raw=len(win_raw), valid_poses_matched=int(((tv >= s.t_motion) & (tv <= s.t_mid_end)).sum()),
                     available_raw_n=int(has_raw.sum()), available_pct=100 * has_raw.mean(), evaluable_n=int(has.sum()), evaluable_pct=100 * has.mean(),
                     lat_rmse_cm=100 * math.sqrt(np.mean(lat[has] ** 2)), rmse_2d_cm=100 * math.sqrt(np.mean(eA[has] ** 2)),
                     within_50cm_pct=100 * np.mean(has & (eA <= 0.5)), within_30cm_pct=100 * np.mean(has & (eA <= 0.3)), within_20cm_pct=100 * np.mean(has & (eA <= 0.2)),
                     held_lat_rmse_cm=100 * math.sqrt(np.mean(latB[okB] ** 2)), held_within_50cm_pct=100 * np.mean(okB & (eB <= 0.5)), held_within_30cm_pct=100 * np.mean(okB & (eB <= 0.3)), held_within_20cm_pct=100 * np.mean(okB & (eB <= 0.2)),
                     longest_gap_s=gaps_in.max(), longest_gap_with_edges_s=gaps_edge.max(), final_tag_cm=SP.loc[t, 'sfit_err_cm'], final_rx_cm=SP.loc[t, 'sfit_rx_err_cm'], restarts=R.loc[t, 'restarts']))
D = pd.DataFrame(rows).set_index('trial'); D.round(3).to_csv(RC / 'whole_approach_by_run.csv', encoding='utf-8-sig')
cols = ['duration_s', 'valid_poses_raw', 'available_pct', 'evaluable_pct', 'lat_rmse_cm', 'within_50cm_pct', 'within_30cm_pct', 'within_20cm_pct', 'held_lat_rmse_cm', 'held_within_50cm_pct', 'held_within_30cm_pct', 'longest_gap_s', 'longest_gap_with_edges_s', 'final_tag_cm']
G = D.groupby(['path', 'policy'])[cols].median(); G['runs'] = D.groupby(['path', 'policy']).size()
pool = D.groupby(['path', 'policy'])[['available_raw_n', 'evaluable_n', 'grid_points']].sum(); G['pooled_available_pct'] = 100 * pool.available_raw_n / pool.grid_points; G['pooled_n'] = pool.available_raw_n.astype(str) + '/' + pool.grid_points.astype(str)
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 40)
print(G.round(2).to_string()); G.round(3).to_csv(RC / 'whole_approach_by_policy.csv', encoding='utf-8-sig')
print('\nraw minus camera-matched availability per run [points]: max %.2f, runs with a difference: %s' % ((D.available_pct - D.evaluable_pct).abs().max(), D.index[(D.available_pct - D.evaluable_pct).abs() > 1e-9].tolist()))
print('raw poses minus matched poses in the window: max %d' % (D.valid_poses_raw - D.valid_poses_matched).abs().max())
print(D[D.path.isin(['S', 'C1'])].groupby(['path', 'policy'])[['available_pct', 'evaluable_pct', 'within_30cm_pct', 'held_within_30cm_pct', 'longest_gap_with_edges_s']].agg(['min', 'max']).round(2).to_string())
print(D[(D.path == 'S') & (D.policy == 'STAGED')][['duration_s', 'grid_points', 'available_raw_n', 'available_pct', 'evaluable_pct', 'longest_gap_s', 'longest_gap_with_edges_s', 'restarts']].round(2).to_string())

