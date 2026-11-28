"""Delay of the position estimates to the PC and spacing of the batches (Sections 4.1.2 and 5 of the manuscript).

For each trial with a camera ground truth, the acquisition time of each frame is reconstructed from the device
timestamps of its links as in cam_eval_dejit.py, and the PC time of each position estimate is compared with it.
lag_ms is the median of the PC time minus the acquisition time, and batch_ms the median interval between the batches
in which the position estimates reached the PC (intervals longer than 20 ms). Reads the raw run folders of the
analysis computer (not included), stops/runs_summary_dedup.csv, and camera_eval/segment_times.csv, and writes
derived_data/revision_checks/latency_batches_by_run.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'derived_data'
RES = Path(r'D:/sandbox/uwb/output/field_20261124/results_20261124')
TICK = 1.0 / (128 * 499.2e6)


def frame_times(lk):
    lk = lk[lk['device_timestamp_valid'].astype(str).str.lower().isin(['true', '1']) & lk['success'].astype(str).isin(['1', 'True', 'true'])].copy()
    lk['host'] = lk.host_rx_ns.astype('int64') / 1e9; out = []
    for (tag, ep), g in lk.groupby(['tag_id', 'mode_epoch']):
        g = g.sort_values('host')
        if len(g) < 60: continue
        dev = np.unwrap(g.device_timestamp_ticks.astype('int64').to_numpy() * TICK, period=(2 ** 40) * TICK); host = g.host.to_numpy()
        d = host - dev; a = np.polyfit(dev, d, 1); res = d - np.polyval(a, dev)
        for _ in range(3):
            lo = res <= np.quantile(res, 0.02); a = np.polyfit(dev[lo], d[lo], 1); res = d - np.polyval(a, dev)
        tm = dev + np.polyval(a, dev) + np.quantile(res, 0.005)
        if np.quantile(res, 0.995) - np.quantile(res, 0.005) > 0.25: continue
        out.append(pd.DataFrame(dict(frame_id=g.frame_id.to_numpy(), tm=tm)))
    return pd.concat(out).groupby('frame_id').tm.mean() if out else pd.Series(dtype=float)


R = pd.read_csv(D / 'stops' / 'runs_summary_dedup.csv')
cam = set(pd.read_csv(D / 'camera_eval' / 'segment_times.csv').trial)
rows = []
for _, r in R.iterrows():
    if r.trial not in cam: continue
    T = RES / r.run / 'tables'
    lk = pd.read_csv(T / 'links.csv', low_memory=False); po = pd.read_csv(T / 'poses.csv', low_memory=False)
    ftm = frame_times(lk); th = po.host_pose_wall_ns.astype('int64') / 1e9; tm = ftm.reindex(po.frame_id).to_numpy()
    hp = np.sort(np.unique(po.host_pose_wall_ns.astype('int64'))) / 1e6; dd = np.diff(hp); bb = dd[dd > 20]
    rows.append(dict(trial=r.trial, policy=r.policy, lag_ms=round(1000 * float(np.nanmedian(th - tm)), 2), batch_ms=round(float(np.median(bb)), 2)))
L = pd.DataFrame(rows)
L.to_csv(D / 'revision_checks' / 'latency_batches_by_run.csv', index=False)
print(L.to_string(index=False))
print('lag_ms range', L.lag_ms.min(), L.lag_ms.max(), '| batch_ms median, fixed methods', L[L.policy != 'STAGED'].batch_ms.median())
