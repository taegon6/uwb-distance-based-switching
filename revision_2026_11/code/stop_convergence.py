"""Stop convergence: after the vehicle stops (camera), how fast does each method's RX estimate settle?
Methods on the same HP stops: DS, raw HP, HP with stationary integer locking (hp_stationary.lock_link).
Estimate at tau = per-link median of samples in [t_stop, t_stop + tau] -> 2A2T pose -> RX coil error vs
tape+plumb. Also the instantaneous estimate (25-sample running median) at t_stop + tau."""
import io, sys, math, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r'D:/sandbox/uwb/output/exp_ready_20261124/repo/pc')
from uwb_pc_logger.hp_stationary import lock_link, StationaryConfig
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE/'results_20261124'; OUTR = BASE/'analysis'/'revision_checks'; EV = BASE/'cam'/'eval'
G = json.loads((BASE/'configs'/'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in G['anchors_m'].items()}; TZ = G['tag_heights_m']['T1']; LAB = G['link_offsets_m']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}; LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
jobs = pd.read_csv(BASE/'cam'/'jobs.csv').set_index('trial'); st = pd.read_csv(BASE/'analysis'/'stops_v2_seltos_L439.csv').set_index('trial')
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
def rd(p, cols):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False, usecols=cols)
def rot(y): c, s = math.cos(math.radians(y)), math.sin(math.radians(y)); return np.array([[c, -s], [s, c]])
def slant(x, y, yaw):
    R = rot(yaw); return {l: float(np.sqrt(np.sum((np.array([x, y]) + R @ BODY[l[:2]] - ANC[l[3:]][:2]) ** 2) + (ANC[l[3:]][2] - TZ) ** 2)) for l in LINKS}
def solve(r, x0):
    p = np.array(x0, float)
    for _ in range(40):
        f = slant(*p); fv = np.array([f[l] for l in LINKS]); res = fv - np.array([r[l] for l in LINKS]); J = np.zeros((4, 3))
        for k, h in enumerate([1e-5, 1e-5, 1e-3]):
            q = p.copy(); q[k] += h; g = slant(*q); J[:, k] = (np.array([g[l] for l in LINKS]) - fv) / h
        dp = np.linalg.lstsq(J, -res, rcond=None)[0]; p += dp
        if abs(dp).max() < 1e-9: break
    return p
def ref(t): s = st.loc[t]; return s.ref_x, s.ref_y, s.ref_yaw
def rx(x, y, yaw): return np.array([x, y]) + rot(yaw) @ np.array([0, 2.2])

runs = [t for t in st.index if st.loc[t, 'stop_mode'] == 'PHASE' and (EV/f'camtrack_{t}.csv').exists()]
data, tstop = {}, {}
for t in runs:
    c = pd.read_csv(EV/f'camtrack_{t}.csv').sort_values('t')
    X = c.X.rolling(9, center=True, min_periods=3).median(); Y = c.Y.rolling(9, center=True, min_periods=3).median()
    fin = np.array([X.iloc[-30:].median(), Y.iloc[-30:].median()])
    far = np.flatnonzero(np.hypot(X - fin[0], Y - fin[1]).to_numpy() > 0.01)
    tstop[t] = float(c.t.iloc[far[-1] + 1]) if len(far) and far[-1] + 1 < len(c) else float(c.t.iloc[0])
    lk = rd(DAY/jobs.loc[t, 'run']/'tables'/'links.csv', ['host_wall_ns', 'link_id', 'success', 'ds_range_m', 'phase_wrapped_m', 'phase_corrected_range_m', 'phase_valid'])
    lk = lk[lk.success.astype(str).isin(['1', 'True', 'true']) & lk.phase_valid.astype(str).isin(['1', 'True', 'true'])].copy()
    lk['tw'] = lk.host_wall_ns.astype('int64') / 1e9
    d = {}
    for l, g in lk.sort_values('tw').groupby('link_id'):
        g = g[g.tw > tstop[t] - 10]
        o = lock_link(g.ds_range_m.to_numpy(float), g.phase_wrapped_m.to_numpy(float), StationaryConfig())
        d[l] = pd.DataFrame({'tw': g.tw.to_numpy(), 'DS': g.ds_range_m.to_numpy(float), 'HP raw': g.phase_corrected_range_m.to_numpy(float),
                             'HP locked': o['locked_range_m'], 'HP locked filt': o['filtered_range_m']})
    data[t] = d
    print(t, f'stop at {tstop[t] - lk.tw.min():.1f} s into the run, {lk.tw.max() - tstop[t]:.1f} s after the stop', flush=True)
METHODS = ['DS', 'HP raw', 'HP locked']
# offsets per method from the final 3 s, straight runs only, leave-one-out
fin = {t: {m: {l: float(np.nanmedian(data[t][l][m][data[t][l].tw > data[t][l].tw.max() - 3])) for l in LINKS} for m in METHODS} for t in runs}
imp = {t: {m: {l: fin[t][m][l] - slant(*ref(t))[l] for l in LINKS} for m in METHODS} for t in runs}
def offs(t, m): oth = [u for u in runs if u != t and sched.loc[u, 'path'] == 'S']; return {l: float(np.median([imp[u][m][l] for u in oth])) for l in LINKS}
taus = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
rows = []
for t in runs:
    x, y, yaw = ref(t); R0 = rx(x, y, yaw)
    for m in METHODS:
        of = offs(t, m)
        for tau in taus:
            for kind in ('cumulative', 'instant'):
                r = {}
                for l in LINKS:
                    g = data[t][l]
                    if kind == 'cumulative':
                        v = g[m][(g.tw >= tstop[t]) & (g.tw <= tstop[t] + tau)]
                    else:
                        col = 'HP locked filt' if m == 'HP locked' else m
                        w = g[g.tw <= tstop[t] + tau]
                        v = w[m].iloc[-25:] if m != 'HP locked' else w[col].iloc[-1:]
                    v = v.dropna(); r[l] = float(np.median(v)) - of[l] if len(v) else np.nan
                if any(np.isnan(list(r.values()))):
                    rows.append(dict(trial=t, method=m, kind=kind, tau=tau, rx_err=np.nan, yaw_err=np.nan, n=0)); continue
                p = solve(r, (x, y, yaw))
                rows.append(dict(trial=t, method=m, kind=kind, tau=tau, rx_err=100 * float(np.linalg.norm(rx(*p) - R0)), yaw_err=abs(p[2] - yaw), tag_err=100 * math.hypot(p[0] - x, p[1] - y)))
D = pd.DataFrame(rows); D.to_csv(OUTR/'stop_convergence.csv', index=False)
pd.set_option('display.width', 220)
for kind in ('cumulative', 'instant'):
    print(f'\n== {kind}: median RX error [cm] (n = {len(runs)} HP stops) vs time after stop [s]')
    print(D[D.kind == kind].pivot_table(index='method', columns='tau', values='rx_err', aggfunc='median').round(1).to_string())
    print(f'-- {kind}: RX <= 10 cm count')
    print(D[D.kind == kind].pivot_table(index='method', columns='tau', values='rx_err', aggfunc=lambda x: int((x <= 10).sum())).to_string())
    print(f'-- {kind}: median yaw error [deg]')
    print(D[D.kind == kind].pivot_table(index='method', columns='tau', values='yaw_err', aggfunc='median').round(2).to_string())
json.dump({t: tstop[t] for t in runs}, open(OUTR/'stop_times_camera.json', 'w'))

