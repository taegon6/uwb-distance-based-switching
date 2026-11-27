"""Same-stop comparison on the 17 HP stops: DS vs raw HP vs HP with stationary integer locking
(Jang et al. JKIEES 2025 method, pc/uwb_pc_logger/hp_stationary.py). Every value comes from the same
3 s window of the same run; offsets per method from straight stops only (leave-one-out for straight)."""
import io, sys, math, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, r'D:/sandbox/uwb/output/exp_ready_20261124/repo/pc')
from uwb_pc_logger.hp_stationary import lock_link, StationaryConfig
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE/'results_20261124'; OUTR = BASE/'analysis'/'revision_checks'
G = json.loads((BASE/'configs'/'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {k: np.array(v) for k, v in G['anchors_m'].items()}; TZ = G['tag_heights_m']['T1']
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
hp_runs = [t for t in st.index if st.loc[t, 'stop_mode'] == 'PHASE']
vals = {}; sds = []
for t in hp_runs:
    lk = rd(DAY/jobs.loc[t, 'run']/'tables'/'links.csv', ['host_monotonic_ns', 'link_id', 'success', 'ds_range_m', 'phase_wrapped_m', 'phase_corrected_range_m', 'phase_valid'])
    lk = lk[lk.success.astype(str).isin(['1', 'True', 'true']) & lk.phase_valid.astype(str).isin(['1', 'True', 'true'])].sort_values('host_monotonic_ns')
    end = lk.host_monotonic_ns.astype('int64').max(); v = {}
    for l, g in lk.groupby('link_id'):
        g = g[g.host_monotonic_ns.astype('int64') > end - 12e9]                 # last 12 s so the 25-sample window can settle
        out = lock_link(g.ds_range_m.to_numpy(float), g.phase_wrapped_m.to_numpy(float), StationaryConfig())
        g = g.assign(locked=out['locked_range_m'], filt=out['filtered_range_m'], stat=out['stationary'])
        w = g[g.host_monotonic_ns.astype('int64') > end - 3e9]
        v[l] = dict(ds=w.ds_range_m.astype(float).median(), hp=w.phase_corrected_range_m.astype(float).median(), lock=np.nanmedian(w.filt),
                    ds_single=w.ds_range_m.astype(float).iloc[-1], hp_single=w.phase_corrected_range_m.astype(float).iloc[-1], lock_single=w.locked.dropna().iloc[-1] if w.locked.notna().any() else np.nan)
        sds.append(dict(trial=t, link=l, stationary_pct=100 * w.stat.mean(), ds_sd_mm=1000 * w.ds_range_m.astype(float).std(), hp_sd_mm=1000 * w.phase_corrected_range_m.astype(float).std(),
                        locked_sd_mm=1000 * np.nanstd(w.locked), filtered_sd_mm=1000 * np.nanstd(w.filt)))
    vals[t] = v
S = pd.DataFrame(sds); print('== link SD in the last 3 s (median over 17 HP stops x 4 links) [mm]'); print(S[['stationary_pct', 'ds_sd_mm', 'hp_sd_mm', 'locked_sd_mm', 'filtered_sd_mm']].median().round(2).to_string())
def ref(t): s = st.loc[t]; return s.ref_x, s.ref_y, s.ref_yaw
rows = []
for meth in ('ds', 'hp', 'lock', 'ds_single', 'hp_single', 'lock_single'):
    imp = {t: {l: vals[t][l][meth] - slant(*ref(t))[l] for l in LINKS} for t in hp_runs}
    for t in hp_runs:
        oth = [u for u in hp_runs if u != t and sched.loc[u, 'path'] == 'S']
        off = {l: float(np.median([imp[u][l] for u in oth])) for l in LINKS}
        x, y, yaw = ref(t); p = solve({l: vals[t][l][meth] - off[l] for l in LINKS}, (x, y, yaw))
        rx_ref = np.array([x, y]) + rot(yaw) @ np.array([0, 2.2]); rx_u = p[:2] + rot(p[2]) @ np.array([0, 2.2])
        rows.append(dict(method=meth, trial=t, path=sched.loc[t, 'path'], tag_err_cm=100 * math.hypot(p[0] - x, p[1] - y), yaw_err_deg=abs(p[2] - yaw),
                         rx_err_cm=100 * float(np.linalg.norm(rx_u - rx_ref))))
D = pd.DataFrame(rows)
tab = D.groupby('method').agg(tag=('tag_err_cm', 'median'), yaw=('yaw_err_deg', 'median'), rx=('rx_err_cm', 'median'), rx_le10=('rx_err_cm', lambda x: f'{(x <= 10).sum()}/{len(x)}')).round(2)
print('== same 17 HP stops, straight-only offsets (LOO): tag midpoint / yaw / RX coil (median)'); print(tab.to_string())
P = D.pivot(index='trial', columns='method', values='rx_err_cm')
for a, b in (('lock', 'ds'), ('hp', 'ds'), ('lock', 'hp')):
    d = (P[a] - P[b]).dropna(); se = d.std(ddof=1) / math.sqrt(len(d))
    print(f'RX {a} - {b}: mean {d.mean():+.2f} cm, 95% CI +-{2.12 * se:.2f} (t16), lower in {int((d < 0).sum())}/{len(d)} stops')
Y = D.pivot(index='trial', columns='method', values='yaw_err_deg')
for a, b in (('lock', 'ds'), ('hp', 'ds')):
    d = (Y[a] - Y[b]).dropna(); print(f'yaw {a} - {b}: mean {d.mean():+.2f} deg, lower in {int((d < 0).sum())}/{len(d)}')
D.to_csv(OUTR/'hp_locking_same_stop.csv', index=False); S.to_csv(OUTR/'hp_locking_link_sd.csv', index=False); tab.to_csv(OUTR/'hp_locking_summary.csv')

