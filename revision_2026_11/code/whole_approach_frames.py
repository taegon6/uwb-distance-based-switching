"""Frames and lost frames inside the window of Table 6 (start of the motion to an alignment distance of 0.50 m), from the records themselves.
A frame is one cycle of the four ranging exchanges. A frame is lost if it is missing from the record: it is incomplete in frames.csv, one of its link records is marked as failed,
or the device timestamps of two consecutive frames of the same mode epoch are more than 1.5 nominal frame periods apart (then round(dt/period)-1 frames are counted as lost).
Intervals across a mode change, or longer than 0.3 s on the PC clock, are output gaps and are counted separately (they are the transition interruptions of Section 4.2).
Also: share of the sampled outputs within a tolerance (conditional on an output), from whole_approach_by_run.csv, and the baton counters of the tags.
Output: whole_approach_frames_by_run.csv, whole_approach_frames_by_policy.csv"""
import io, json, numpy as np, pandas as pd
from pathlib import Path
B = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = B / 'results_20261124'; EV = B / 'cam' / 'eval'; RC = B / 'analysis' / 'revision_checks'
R = pd.read_csv(B / 'analysis' / 'runs_summary_dedup.csv').set_index('trial'); ST = pd.read_csv(EV / 'segment_times.csv').set_index('trial'); WA = pd.read_csv(RC / 'whole_approach_by_run.csv').set_index('trial')
def rd(p, cols=None):
    tx = p.read_text(encoding='utf-8', errors='replace'); tx = '\n'.join(l for l in tx.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(tx), on_bad_lines='skip', low_memory=False, usecols=cols)
TICK = 1 / (128 * 499.2e6); rows = []
for t in WA.index:
    d = DAY / R.loc[t, 'run'] / 'tables'; s = ST.loc[t]
    L = rd(d / 'links.csv', ['host_wall_ns', 'frame_id', 'link_id', 'mode', 'mode_epoch', 'device_timestamp_ticks', 'success'])
    F = rd(d / 'frames.csv', ['frame_id', 'mode', 'mode_epoch', 'complete', 'frame_end_host_wall_ns'])
    F['t'] = F.frame_end_host_wall_ns / 1e9; Fw = F[(F.t >= s.t_motion) & (F.t <= s.t_mid_end)]
    L['t'] = L.host_wall_ns / 1e9; Lw = L[(L.t >= s.t_motion) & (L.t <= s.t_mid_end)]
    lost = 0; irregular = 0; ratio_max = 0.0; gaps = 0
    for lk, g in Lw.groupby('link_id'):
        g = g.sort_values('host_wall_ns', kind='stable'); ts = g.device_timestamp_ticks.astype('int64').to_numpy(); dt = (np.diff(ts) % (1 << 40)) * TICK * 1e3
        hd = np.diff(g.t.to_numpy()); m = g['mode'].astype(str).to_numpy(); ep = g.mode_epoch.to_numpy(); same = (m[1:] == m[:-1]) & (ep[1:] == ep[:-1]); gap = (~same) | (hd > 0.3)
        nom = np.full(len(dt), np.nan)
        for mode in np.unique(m): nom[m[1:] == mode] = np.median(dt[(m[1:] == mode) & ~gap]) if ((m[1:] == mode) & ~gap).any() else np.nan
        r = dt[~gap] / nom[~gap]; irr = r > 1.5
        lost = max(lost, int(np.round(r[irr] - 1).sum())); irregular = max(irregular, int(irr.sum())); ratio_max = max(ratio_max, float(np.nanmax(r)) if len(r) else 0.0); gaps = max(gaps, int(gap.sum()))
    rows.append(dict(trial=t, path=WA.loc[t, 'path'], policy=WA.loc[t, 'policy'], frames=len(Fw), incomplete_frames=int((Fw.complete != 1).sum()), failed_link_records=int((Lw.success != 1).sum()),
                     irregular_intervals=irregular, lost_frames=lost + int((Fw.complete != 1).sum()), largest_interval_ratio=ratio_max, output_gaps=gaps, valid_poses=int(WA.loc[t, 'valid_poses_raw']),
                     longest_gap_s=WA.loc[t, 'longest_gap_s'], sampled_within_50cm_pct=100 * WA.loc[t, 'within_50cm_pct'] / WA.loc[t, 'available_pct'], sampled_within_30cm_pct=100 * WA.loc[t, 'within_30cm_pct'] / WA.loc[t, 'available_pct'],
                     sampled_within_20cm_pct=100 * WA.loc[t, 'within_20cm_pct'] / WA.loc[t, 'available_pct'], lat_rmse_cm=WA.loc[t, 'lat_rmse_cm'], restarts=int(WA.loc[t, 'restarts'])))
D = pd.DataFrame(rows).set_index('trial'); D['valid_of_frames_pct'] = 100 * D.valid_poses / D.frames
D.round(3).to_csv(RC / 'whole_approach_frames_by_run.csv', encoding='utf-8-sig')
G = D.groupby(['path', 'policy']).agg(runs=('frames', 'size'), frames_median=('frames', 'median'), frames_sum=('frames', 'sum'), lost_frames_sum=('lost_frames', 'sum'), largest_interval_ratio=('largest_interval_ratio', 'max'), valid_poses_median=('valid_poses', 'median'),
                                     valid_of_frames_pct_median=('valid_of_frames_pct', 'median'), valid_of_frames_pct_min=('valid_of_frames_pct', 'min'), within_50_median=('sampled_within_50cm_pct', 'median'), within_50_min=('sampled_within_50cm_pct', 'min'),
                                     within_30_median=('sampled_within_30cm_pct', 'median'), within_30_min=('sampled_within_30cm_pct', 'min'), within_30_max=('sampled_within_30cm_pct', 'max'), within_20_median=('sampled_within_20cm_pct', 'median'),
                                     longest_gap_median=('longest_gap_s', 'median'), longest_gap_min=('longest_gap_s', 'min'), longest_gap_max=('longest_gap_s', 'max'))
G.round(3).to_csv(RC / 'whole_approach_frames_by_policy.csv', encoding='utf-8-sig')
pd.set_option('display.width', 280); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
print(D[['path', 'policy', 'frames', 'valid_poses', 'valid_of_frames_pct', 'incomplete_frames', 'failed_link_records', 'irregular_intervals', 'lost_frames', 'largest_interval_ratio', 'output_gaps', 'longest_gap_s']].round(3).to_string())
print(G.round(2).T.to_string())
print('all runs: lost frames', int(D.lost_frames.sum()), 'of', int(D.frames.sum()), '; largest interval ratio outside the gaps %.3f' % D.largest_interval_ratio.max())
# baton (hand-over) counters of the tags, difference to the previous record
rows = []
for d in sorted(p for p in DAY.iterdir() if p.is_dir()):
    f = d / 'tables' / 'failure_stage_diagnostics.json'
    if not f.exists(): continue
    for tag, items in json.loads(f.read_text(encoding='utf-8')).items():
        for it in items:
            if isinstance(it, dict) and 'baton_timeout' in it and it.get('baton_wait', 0) > 0: rows.append(dict(run=d.name, tag=it['tag_id'], mode=it['mode'], anchor=it['anchor_id'], baton_wait=it['baton_wait'], baton_good=it['baton_rx_good'], baton_timeout=it['baton_timeout']))
Bt = pd.DataFrame(rows).sort_values('run', kind='stable'); Bt[['d_wait', 'd_to']] = Bt.groupby(['tag', 'mode', 'anchor'])[['baton_wait', 'baton_timeout']].diff()
neg = Bt.d_wait.isna() | (Bt.d_wait < 0); Bt.loc[neg, 'd_wait'] = Bt.loc[neg, 'baton_wait']; Bt.loc[neg, 'd_to'] = Bt.loc[neg, 'baton_timeout']
Bt = Bt.merge(R.reset_index()[['trial', 'run', 'policy', 'restarts']], on='run'); Bs = Bt[Bt.d_wait > 0].groupby(['trial', 'policy', 'restarts']).agg(baton_waits=('d_wait', 'sum'), baton_timeouts=('d_to', 'sum')).reset_index()
Bs.to_csv(RC / 'baton_timeouts_by_run.csv', index=False, encoding='utf-8-sig')
print(Bs[(Bs.baton_timeouts > 0) | (Bs.restarts > 0)].to_string())
print('runs with restarts and baton timeouts:', int(((Bs.restarts > 0) & (Bs.baton_timeouts > 0)).sum()), 'of', int((Bs.restarts > 0).sum()), '; runs without restarts but with baton timeouts:', int(((Bs.restarts == 0) & (Bs.baton_timeouts > 0)).sum()), 'of', int((Bs.restarts == 0).sum()))

