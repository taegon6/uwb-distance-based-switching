"""Revision checks: sync sensitivity (R2-11/R3-M3), distance-bin accuracy + K replay (R2-4),
stop decomposition offset x aggregation x mode (R3-M4), policy summary (R3-M2)."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read()
exec(src.split('# ---------------- main ----------------')[0])
OUTR = BASE / 'analysis' / 'revision_checks'; OUTR.mkdir(exist_ok=True)
jobs = pd.read_csv(CAM / 'jobs.csv'); stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
impl = pd.read_csv(BASE / 'analysis' / 'stop_implied_link_offsets_v2.csv')
sched = SCHED.set_index('trial')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)

# ---------- 1) sync sensitivity ----------
rows = []
for j in jobs.itertuples():
    pf = OUT / f'pairs_{j.trial}.csv'; cf = OUT / f'camtrack_{j.trial}.csv'
    if not pf.exists(): continue
    p = pd.read_csv(pf); c = pd.read_csv(cf).sort_values('t')
    path = sched.loc[j.trial, 'path']; pol = sched.loc[j.trial, 'policy']
    X = c.X.rolling(31, center=True, min_periods=5).mean(); Y = c.Y.rolling(31, center=True, min_periods=5).mean()
    vx, vy = np.gradient(X, c.t), np.gradient(Y, c.t)
    for s in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
        tq = p.t + s
        cx = np.interp(tq, c.t, c.X, left=np.nan, right=np.nan); cy = np.interp(tq, c.t, c.Y, left=np.nan, right=np.nan)
        if path == 'S': tx, ty = np.zeros(len(p)), -np.ones(len(p))
        else:
            tx = np.interp(tq, c.t, vx); ty = np.interp(tq, c.t, vy); n = np.hypot(tx, ty); ok = n > 0.02
            tx = np.where(ok, tx / np.where(ok, n, 1), 0); ty = np.where(ok, ty / np.where(ok, n, 1), -1)
        ex = p.tag_midpoint_x_m - cx; ey = p.tag_midpoint_y_m - cy
        lat = ex * (-ty) + ey * tx
        for sg in ('far', 'mid'):
            m = (p.seg == sg) & np.isfinite(lat)
            if m.sum() > 10: rows.append(dict(trial=j.trial, path=path, policy=pol, seg=sg, shift_ms=int(s * 1000), lat_rmse=100 * math.sqrt(np.mean(lat[m] ** 2))))
SS_ = pd.DataFrame(rows)
sens = SS_.groupby(['path', 'seg', 'shift_ms']).lat_rmse.median().unstack().round(2)
print('== sync shift sensitivity: median lateral RMSE [cm] vs extra time offset'); print(sens.to_string())
sens.to_csv(OUTR / 'sync_sensitivity.csv')
# half-split sync consistency
hs = []
for j in jobs.itertuples():
    f = CAM / 'tracks' / f'{j.trial}.csv'
    if not f.exists(): continue
    c = pd.read_csv(f); c = c[c.cx.notna() & (c.reproj_px < 2.5)].copy(); c['d'] = np.sqrt(c.cx ** 2 + c.cy ** 2 + c.cz ** 2)
    c['tw'] = j.video_start + c.pts
    fr = rd(DAY / j.run / 'tables' / 'frames.csv'); fr['t'] = fr.frame_end_host_wall_ns.astype('int64') / 1e9
    fr['rm'] = fr[['r11', 'r12', 'r21', 'r22']].astype(float).mean(axis=1)
    fr = fr.dropna(subset=['rm'])
    mid = c.tw.median()
    a = c[c.tw < mid]; b = c[c.tw >= mid - 2]
    try:
        d1, c1 = sync(a.tw.to_numpy(), a.d.to_numpy(), fr.t.to_numpy(), fr.rm.to_numpy())
        d2, c2 = sync(b.tw.to_numpy(), b.d.to_numpy(), fr.t.to_numpy(), fr.rm.to_numpy())
        if c1 > 0.8 and c2 > 0.8: hs.append(dict(trial=j.trial, diff_ms=1000 * (d1 - d2)))
    except Exception: pass
H = pd.DataFrame(hs)
print(f'== half-split sync difference: runs {len(H)}, median |diff| {H.diff_ms.abs().median():.0f} ms, p90 {H.diff_ms.abs().quantile(.9):.0f} ms')
H.to_csv(OUTR / 'sync_halfsplit.csv', index=False)

# ---------- 2) R2-4 distance bins (fixed straight runs) + K replay ----------
P = []
for j in jobs.itertuples():
    pf = OUT / f'pairs_{j.trial}.csv'
    if pf.exists():
        p = pd.read_csv(pf); p['trial'] = j.trial; p['path'] = sched.loc[j.trial, 'path']; p['policy'] = sched.loc[j.trial, 'policy']; P.append(p)
P = pd.concat(P)
# bins by the camera RX-coil distance, approach phase only (standstill samples are excluded)
bins = [0.5, 1, 2, 3, 4, 5, 6, 8, 10.5]
P['bin'] = pd.cut(P.cam_dc, bins)
fx = P[(P.path == 'S') & P.policy.str.startswith('FIXED') & P.seg.isin(['far', 'mid'])]
tab = fx.groupby(['bin', 'policy'], observed=True).apply(lambda g: 100 * math.sqrt(np.mean(g.log_lat ** 2))).unstack().round(1)
print('== straight fixed runs: lateral RMSE [cm] by coil distance bin'); print(tab.to_string())
tab.to_csv(OUTR / 'lateral_rmse_by_distance_fixed_straight.csv')
# K replay: count mode commits a threshold rule would make on fixed-mode d_c series
def replay(d, enter, exit_, K):
    state, cnt_in, cnt_out, n = 0, 0, 0, 0
    for v in d:
        if state == 0:
            cnt_in = cnt_in + 1 if v < enter else 0
            if cnt_in >= K: state, n, cnt_out = 1, n + 1, 0
        else:
            cnt_out = cnt_out + 1 if v > exit_ else 0
            if cnt_out >= K: state, n, cnt_in = 0, n + 1, 0
    return n
kr = []
for (pol, enter, exit_) in (('FIXED_SS', 4.0, 4.5), ('FIXED_DS', 0.5, 0.6)):
    for t in P[(P.policy == pol)].trial.unique():
        po = rd(DAY / jobs.set_index('trial').loc[t, 'run'] / 'tables' / 'poses.csv')
        po = po[po.valid.astype(str).str.lower().isin(['true', '1'])]
        d = po.coil_to_tx_distance_m.astype(float).to_numpy()
        for K in (1, 3, 5, 10):
            for hyst in (True, False):
                kr.append(dict(policy=pol, trial=t, path=sched.loc[t, 'path'], K=K, hysteresis=hyst, commits=replay(d, enter, exit_ if hyst else enter, K)))
KR = pd.DataFrame(kr)
kt = KR.groupby(['policy', 'hysteresis', 'K']).commits.agg(['mean', 'max', lambda x: (x > 1).mean() * 100]).round(2)
kt.columns = ['mean_commits', 'max_commits', 'pct_runs_with_extra_switch']
print('== threshold replay on fixed-mode d_c (ideal = 1 commit)'); print(kt.to_string())
kt.to_csv(OUTR / 'confirmation_frames_replay.csv')

# ---------- 3) R3-M4 stop decomposition ----------
rows = []
for j in jobs.itertuples():
    if j.trial not in stops.index: continue
    st = stops.loc[j.trial]
    oth = impl[(impl.trial != j.trial) & (impl.path == 'S')]
    fo = {m: {l: float(oth[(oth['mode'] == m) & (oth.link == l)].off.median()) for l in LINKS} for m in ('SS', 'DS', 'PHASE')}
    zero = {m: {l: 0.0 for l in LINKS} for m in fo}; lab = {m: {l: float(GEOM['link_offsets_m'][m][l]) for l in LINKS} for m in fo}
    po, _ = uwb_track(j.run, fo, {'z': zero, 'l': lab})
    end = po.t.max(); w = po[po.t > end - 3.0]
    mode = w['mode'].iloc[-1]
    for name, (xc, yc) in (('none', ('zx', 'zy')), ('lab', ('lx', 'ly')), ('field', ('fx', 'fy'))):
        ex = 100 * (w[xc] - st.ref_x); ey = 100 * (w[yc] - st.ref_y); e = np.hypot(ex, ey)
        rows.append(dict(trial=j.trial, path=sched.loc[j.trial, 'path'], stop_mode=mode, offset=name,
                         single_median=float(e.median()), single_p95=float(e.quantile(.95)),
                         agg3s=float(math.hypot(ex.median(), ey.median()))))
D = pd.DataFrame(rows)
dec = D.groupby(['stop_mode', 'offset'])[['single_median', 'single_p95', 'agg3s']].median().round(2)
print('== stop error decomposition [cm] (tag midpoint, median over runs)'); print(dec.to_string())
D.to_csv(OUTR / 'stop_decomposition_runs.csv', index=False); dec.to_csv(OUTR / 'stop_decomposition.csv')

# ---------- 4) R3-M2 policy summary ----------
R = pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv').set_index('trial')
E = pd.read_csv(OUT / 'segment_errors.csv')
rows = []
for t, r in R.iterrows():
    pol = sched.loc[t, 'policy']; path = sched.loc[t, 'path']
    po = rd(DAY / r.run / 'tables' / 'poses.csv'); v = po[po.valid.astype(str).str.lower().isin(['true', '1'])]
    d = v.coil_to_tx_distance_m.astype(float).to_numpy(); ts = v.host_pose_monotonic_ns.astype('int64').to_numpy() / 1e9
    k0 = np.argmax(d < d[:50].min() - 0.2) if len(d) > 50 else 0       # motion start
    k1 = np.argmax(d < 0.05) if (d < 0.05).any() else len(d) - 1
    e = E[E.trial == t].set_index('seg')
    st = stops.loc[t] if t in stops.index else None
    rows.append(dict(trial=t, path=path, policy=pol, approach_s=ts[k1] - ts[k0], n_valid=len(v), valid_pct=r.valid_pct,
                     p95_gap_ms=r.p95_gap_ms, max_gap_ms=r.max_gap_ms, restarts=r.restarts,
                     far_lat=e.log_lat_rmse.get('far', np.nan), mid_lat=e.log_lat_rmse.get('mid', np.nan),
                     stop_err_field=st.sfit_err_cm if st is not None else np.nan, stop_rx_field=st.sfit_rx_err_cm if st is not None else np.nan))
S2 = pd.DataFrame(rows)
summ = S2.groupby(['path', 'policy']).agg(runs=('trial', 'count'), approach_s=('approach_s', 'median'), n_valid=('n_valid', 'median'),
        valid_pct=('valid_pct', 'median'), p95_gap_ms=('p95_gap_ms', 'median'), max_gap_ms=('max_gap_ms', 'max'), restarts=('restarts', 'sum'),
        far_lat=('far_lat', 'median'), mid_lat=('mid_lat', 'median'), stop_err=('stop_err_field', 'median'),
        stop_le10=('stop_err_field', lambda x: f'{(x <= 10).sum()}/{x.notna().sum()}'), rx_err=('stop_rx_field', 'median')).round(1)
print('== R3-M2 policy summary'); print(summ.to_string())
summ.to_csv(OUTR / 'policy_summary.csv'); S2.to_csv(OUTR / 'policy_summary_runs.csv', index=False)

