"""Fixes after an internal check: (1) K replay with the field configuration (enter K=3, reverse K=10)
and variants; (2) alternative staging with update rates from all UWB outputs (not camera-matched pairs)
plus paired block tests and the stop heading/RX of each alternative's final mode."""
import io, math
from pathlib import Path
import numpy as np, pandas as pd
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
def t_p_two_sided(t, df):
    """Two-sided p for Student t by numerical integration (no scipy here)."""
    x = np.linspace(abs(t), abs(t) + 200, 400001)
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    pdf = c * (1 + x ** 2 / df) ** (-(df + 1) / 2)
    return 2 * float(np.sum((pdf[1:] + pdf[:-1]) / 2 * np.diff(x)))
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'; OUTR = BASE / 'analysis' / 'revision_checks'
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
jobs = pd.read_csv(BASE / 'cam' / 'jobs.csv').set_index('trial')
def rd(p):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False)
def poses(t):
    po = rd(DAY / jobs.loc[t, 'run'] / 'tables' / 'poses.csv'); po = po[po.valid.astype(str).str.lower().isin(['true', '1'])].copy()
    po['t'] = po.host_pose_monotonic_ns.astype('int64') / 1e9; po['d'] = po.coil_to_tx_distance_m.astype(float); return po

# (1) K replay: state machine like AutoModeManager (consecutive frames, enter/exit thresholds)
def replay(d, enter, exit_, k_in, k_out):
    st, c_in, c_out, n = 0, 0, 0, 0
    for v in d:
        if st == 0:
            c_in = c_in + 1 if v < enter else 0
            if c_in >= k_in: st, n, c_out = 1, n + 1, 0
        else:
            c_out = c_out + 1 if v > exit_ else 0
            if c_out >= k_out: st, n, c_in = 0, n + 1, 0
    return n
rows = []
for pol, enter, ex in (('FIXED_SS', 4.0, 4.5), ('FIXED_DS', 0.5, 0.6)):
    for t in [t for t in sched.index if sched.loc[t, 'policy'] == pol and t in jobs.index]:
        d = poses(t).d.to_numpy()
        for kin, kout in ((1, 1), (3, 3), (3, 10), (5, 10), (10, 10)):
            for hyst in (True, False):
                rows.append(dict(boundary=f'{enter} m', policy=pol, trial=t, k_enter=kin, k_exit=kout, hysteresis=hyst,
                                 commits=replay(d, enter, ex if hyst else enter, kin, kout)))
K = pd.DataFrame(rows)
kt = K.groupby(['boundary', 'hysteresis', 'k_enter', 'k_exit']).commits.agg(
    runs='count', mean='mean', max='max', never=lambda x: int((x == 0).sum()), extra=lambda x: int((x > 1).sum())).round(2)
print('== threshold replay on fixed-mode coil distance (ideal: exactly 1 commit); field config = enter 3 / exit 10 with hysteresis')
print(kt.to_string()); kt.to_csv(OUTR / 'confirmation_frames_replay_v2.csv')
# actual switching runs
st = [t for t in sched.index if sched.loc[t, 'policy'] == 'STAGED' and t in jobs.index]
act = []
for t in st:
    rs = rd(DAY / jobs.loc[t, 'run'] / 'tables' / 'runtime_status.csv')
    act.append(dict(trial=t, commits=int((rs.event == 'AUTO_MODE_COMMITTED').sum())))
A = pd.DataFrame(act); print('== actual switching runs: commits per run', A.commits.value_counts().to_dict(), f'({len(A)} runs)')

# (2) alternative staging (straight), band 0.5-4 m
P = []
for t in sched.index:
    f = EV / f'pairs_{t}.csv'
    if f.exists() and sched.loc[t, 'path'] == 'S':
        p = pd.read_csv(f); p['trial'] = t; p['policy'] = sched.loc[t, 'policy']; p['block'] = sched.loc[t, 'block']; P.append(p)
P = pd.concat(P)
stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
# middle segment = approach interval between the camera crossings of 4.0 m and 0.5 m (cam_eval.py);
# its update rate counts all valid UWB outputs in that interval
SEG = pd.read_csv(EV / 'segment_errors.csv')
MIDRATE = SEG[SEG.seg == 'mid'].set_index('trial').rate_hz
alts = {'3-stage SS>DS>HP (proposed)': ('FIXED_DS', 'FIXED_PHASE'), '2-stage SS>HP @0.5 m': ('FIXED_SS', 'FIXED_PHASE'),
        '2-stage SS>HP @4.0 m': ('FIXED_PHASE', 'FIXED_PHASE'), '2-stage SS>DS (no HP)': ('FIXED_DS', 'FIXED_DS')}
rows = []
for name, (mid_pol, stop_pol) in alts.items():
    for b in range(1, 6):
        tm = sched[(sched.path == 'S') & (sched.block == b) & (sched.policy == mid_pol)].index[0]
        ts = sched[(sched.path == 'S') & (sched.block == b) & (sched.policy == stop_pol)].index[0]
        g = P[(P.trial == tm) & (P.seg == 'mid')]
        s = stops.loc[ts]
        rows.append(dict(alternative=name, block=b, mid_trial=tm, stop_trial=ts, lat_rmse_0_5_4m=100 * math.sqrt(np.mean(g.log_lat ** 2)),
                         rate_0_5_4m_hz=MIDRATE[tm], stop_yaw_err_deg=abs(s.sfit_dyaw), stop_rx_err_cm=s.sfit_rx_err_cm))
AL = pd.DataFrame(rows); AL.to_csv(OUTR / 'alternative_staging_v2.csv', index=False)
print('== alternative staging (straight, block-paired; middle band from the mode used there, stop from the final mode)')
print(AL.groupby('alternative')[['lat_rmse_0_5_4m', 'rate_0_5_4m_hz', 'stop_yaw_err_deg', 'stop_rx_err_cm']].median().round(2).to_string())
base = AL[AL.alternative.str.startswith('3-stage')].set_index('block')
for name in [n for n in alts if not n.startswith('3-stage')]:
    o = AL[AL.alternative == name].set_index('block')
    for col in ('lat_rmse_0_5_4m', 'stop_rx_err_cm'):
        d = (base[col] - o[col]).dropna()
        if len(d) >= 3 and d.std() > 0:
            se = d.std(ddof=1) / math.sqrt(len(d)); tstat = d.mean() / se; h = TQ[len(d)] * se   # TQ is keyed by n (df = n - 1)
            print(f'3-stage - {name:24s} {col:16s}: {d.mean():+6.2f} +-{h:5.2f}  t={tstat:+.2f} p={t_p_two_sided(tstat, len(d) - 1):.3f}  blocks lower {int((d < 0).sum())}/{len(d)}')
        else:
            print(f'3-stage - {name:24s} {col:16s}: identical by construction')

