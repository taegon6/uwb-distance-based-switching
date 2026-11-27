"""Sensitivity check (v40): the evaluation of cam_eval.py with pose times taken from the DW3000 device timestamps.
The PC receives the records in batches about 78 ms apart, so the PC pose time lags the measurement by 0 to 80 ms.
Here each link record gets the time  a * device_time + b  (per tag and mode epoch, line through the fastest records),
and the pose time of a frame is the mean of its link times. Everything else is cam_eval.py. Output: cam/eval_dejit/."""
from pathlib import Path
src = Path(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py').read_text(encoding='utf-8')
def rep(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:70]); return s.replace(old, new)
src = rep(src, "OUT = CAM / 'eval'", "OUT = CAM / 'eval_dejit'")
src = rep(src, "    po['mode'] = mode; po['t'] = po.host_pose_wall_ns.astype('int64') / 1e9\n",
          "    po['mode'] = mode; po['t_host'] = po.host_pose_wall_ns.astype('int64') / 1e9\n"
          "    ft = frame_times(rd(T / 'links.csv'))\n"
          "    tm = ft.reindex(po.frame_id).to_numpy(); po['t'] = np.where(np.isfinite(tm), tm, po.t_host - 0.040)\n"
          "    DEJIT_STATS.append(dict(run=run, frames=len(po), with_device_time=int(np.isfinite(tm).sum()), lag_median_ms=1000 * float(np.nanmedian(po.t_host - tm)), lag_p95_ms=1000 * float(np.nanquantile(po.t_host - tm, .95))))\n")
src = rep(src, "    fr['t'] = fr.frame_end_host_wall_ns.astype('int64') / 1e9\n",
          "    fr['t_host'] = fr.frame_end_host_wall_ns.astype('int64') / 1e9\n"
          "    tf = ft.reindex(fr.frame_id).to_numpy(); fr['t'] = np.where(np.isfinite(tf), tf, fr.t_host - 0.040)\n")
helper = '''
DEJIT_STATS = []
TICK = 1.0 / (128 * 499.2e6)
def frame_times(lk):
    """Mean measurement time of each frame on the PC wall clock, from the device timestamps of its links."""
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
        if np.quantile(res, 0.995) - np.quantile(res, 0.005) > 0.25: continue       # mapping not credible, keep the PC time
        out.append(pd.DataFrame(dict(frame_id=g.frame_id.to_numpy(), tm=tm)))
    if not out: return pd.Series(dtype=float)
    return pd.concat(out).groupby('frame_id').tm.mean()
'''
src = rep(src, "# ---------------- UWB re-solve (vectorised Gauss-Newton) ----------------", helper + "\n# ---------------- UWB re-solve (vectorised Gauss-Newton) ----------------")
src += "\npd.DataFrame(DEJIT_STATS).round(2).to_csv(OUT / 'dejitter_stats.csv', index=False)\nprint(pd.DataFrame(DEJIT_STATS).describe().round(2).to_string())\n"
src = rep(src, "    return po, fr[['t', 'rmean']].dropna()", "    return po, fr[['t', 't_host', 'rmean']].dropna()")
src = rep(src, "    dt, corr = sync(c.tw.to_numpy(), c.d.to_numpy(), fr.t.to_numpy(), fr.rmean.to_numpy())\n",
          "    _m = MAIN_SYNC.loc[j.trial]; dt = float(_m.dt_s) + float((fr.t_host - fr.t).mean()); corr = float(_m['corr'])\n")
src = rep(src, "    if chk['ratio'] > 2.0:\n", "    if False:\n")
src = rep(src, "# ---------------- main ----------------", "MAIN_SYNC = pd.read_csv(CAM / 'eval' / 'sync_check.csv').set_index('trial')\n# ---------------- main ----------------")
exec(compile(src, 'cam_eval_dejit', 'exec'))

