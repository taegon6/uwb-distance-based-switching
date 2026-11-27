"""Relative displacement from the carrier phase in the last centimetres before the stop.
For every stop recorded in high-precision mode the wrapped phase is unwrapped by continuity (no integer from
DS-TWR is needed) and the change of each link range is compared with the change predicted from the camera.
Window: from the last time the camera speed exceeded V_MAX until the end of the run."""
import sys, io, json, math
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, r'D:/sandbox/uwb/output/exp_ready_20261124/repo/tools')
from measurement_time import tag_measure_times
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'; OUTR = BASE / 'analysis' / 'revision_checks'
G = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in G['anchors_m'].items()}; TZ = G['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}; LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
LAM = 299792458.0 / (2 * 7.9872e9); V_MAX = 0.08; W = 5
jobs = pd.read_csv(BASE / 'cam' / 'jobs.csv').set_index('trial'); st = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
runs = [t for t in st.index if st.loc[t, 'stop_mode'] == 'PHASE' and (EV / f'camtrack_{t}.csv').exists()]
def circular_median(wrapped):
    a = 2 * np.pi * wrapped / LAM; c = np.angle(np.mean(np.exp(1j * a))); d = np.angle(np.exp(1j * (a - c)))
    return (c + np.median(d)) * LAM / (2 * np.pi)
rows, series = [], []
for t in runs:
    C = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    X = C.X.rolling(9, center=True, min_periods=3).median(); Y = C.Y.rolling(9, center=True, min_periods=3).median(); P = np.radians(C.psi.rolling(9, center=True, min_periods=3).median())
    speed = np.hypot(np.gradient(X.rolling(15, center=True, min_periods=5).mean(), C.t), np.gradient(Y.rolling(15, center=True, min_periods=5).mean(), C.t))
    fast = np.flatnonzero(speed > V_MAX)
    t_slow = float(C.t.iloc[fast[-1]]) if len(fast) else float(C.t.iloc[0])
    L = pd.read_csv(DAY / jobs.loc[t, 'run'] / 'tables' / 'links.csv', usecols=['host_wall_ns', 'host_monotonic_ns', 'tag_id', 'link_id', 'success', 'device_timestamp_ticks', 'ds_range_m', 'phase_wrapped_m', 'phase_valid', 'range_source'], on_bad_lines='skip')
    L = L[L.success.astype(str).isin(['1', 'True', 'true']) & L.device_timestamp_ticks.notna()].copy()
    wall_minus_mono = (L.host_wall_ns.astype('int64') - L.host_monotonic_ns.astype('int64')).median()
    L['t_meas'] = (pd.concat([tag_measure_times(g) for _, g in L.groupby('tag_id')]) + wall_minus_mono) / 1e9
    L = L[L.phase_valid.astype(str).isin(['1', 'True', 'true']) & (L.range_source == 'PHASE_CORRECTED_RANGE')]
    for l in LINKS:
        g = L[(L.link_id == l) & (L.t_meas > t_slow - 0.3)].sort_values('t_meas')
        if len(g) < 40: continue
        tm = g.t_meas.to_numpy(); wrapped = np.mod(g.phase_wrapped_m.to_numpy(float), LAM); ds = g.ds_range_m.to_numpy(float)
        track = np.empty(len(g)); previous = None
        for i in range(len(g)):
            frac = circular_median(wrapped[max(0, i - W + 1):i + 1])
            track[i] = frac if previous is None else frac + LAM * round((previous - frac) / LAM)
            previous = track[i]
        tag, anc = l.split('_'); b = BODY[tag]; A = ANC[anc]
        def camera_range(tt):
            x = np.interp(tt, C.t, X); y = np.interp(tt, C.t, Y); a = np.interp(tt, C.t, P)
            return np.sqrt((x + np.cos(a) * b[0] - np.sin(a) * b[1] - A[0]) ** 2 + (y + np.sin(a) * b[0] + np.cos(a) * b[1] - A[1]) ** 2 + (A[2] - TZ) ** 2)
        end = tm > tm.max() - 1.5
        phase_change = track - np.median(track[end])
        ds_smooth = pd.Series(ds).rolling(W, min_periods=1).median().to_numpy(); ds_change = ds_smooth - np.median(ds[end])
        best = None
        for shift in np.arange(-0.15, 0.151, 0.01):       # camera clock was aligned to the PC arrival time, so one shift per link is fitted
            cam = camera_range(tm + shift); cam_change = cam - np.median(cam[end])
            rms = math.sqrt(np.mean((phase_change - cam_change) ** 2))
            if best is None or rms < best[0]: best = (rms, shift, cam_change)
        rms, shift, cam_change = best
        moved = np.abs(cam_change) > 0.002
        rows.append(dict(trial=t, link=l, samples=len(g), window_s=tm.max() - tm.min(), travel_mm=1000 * np.abs(cam_change).max(), shift_ms=1000 * shift,
                         phase_rms_mm=1000 * rms, phase_rms_moving_mm=1000 * math.sqrt(np.mean((phase_change - cam_change)[moved] ** 2)) if moved.sum() > 5 else np.nan,
                         ds_rms_mm=1000 * math.sqrt(np.mean((ds_change - cam_change) ** 2)), start_error_cycles=(phase_change[0] - cam_change[0]) / LAM))
        series.append(pd.DataFrame(dict(trial=t, link=l, t=tm - tm.max(), phase_mm=1000 * phase_change, camera_mm=1000 * cam_change, ds_mm=1000 * ds_change)))
D = pd.DataFrame(rows); D.round(3).to_csv(OUTR / 'phase_relative_displacement.csv', index=False); pd.concat(series).round(3).to_csv(OUTR / 'phase_relative_displacement_series.csv', index=False)
pd.set_option('display.width', 220)
print(D.groupby('trial').agg(links=('link', 'count'), window_s=('window_s', 'median'), travel_mm=('travel_mm', 'median'), phase_rms_mm=('phase_rms_mm', 'median'), phase_moving_mm=('phase_rms_moving_mm', 'median'),
                             ds_rms_mm=('ds_rms_mm', 'median'), worst_start_cycles=('start_error_cycles', lambda v: v.abs().max())).round(2).to_string())
ok = D[D.start_error_cycles.abs() < 0.5]
print('links', len(D), '| no cycle slip (start error below half a cycle):', len(ok), '| travel in window median %.1f mm (max %.1f)' % (D.travel_mm.median(), D.travel_mm.max()))
print('phase vs camera RMS [mm]: all links median %.2f, links without slip median %.2f (p90 %.2f) | DS (5-sample median) vs camera median %.2f' % (D.phase_rms_mm.median(), ok.phase_rms_mm.median(), ok.phase_rms_mm.quantile(.9), D.ds_rms_mm.median()))

