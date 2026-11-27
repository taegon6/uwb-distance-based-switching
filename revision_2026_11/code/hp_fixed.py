"""HP at the final stop with a fixed integer, and the R3-M4 decomposition on the same HP stops.

Finding that shaped the method: with the integer fixed once and the range following exchange-to-exchange phase
continuity, 8.8 % of the steps exceed a quarter cycle (heavy-tailed single-exchange phase noise), so that
tracker slips. The robust version used here takes the fractional range from the circular median of the last
25 wrapped-phase samples, fixes the integer once from the DS median of the first 2 s after the stop, and then
keeps the cycle by continuity of the 25-sample estimate. Stop time comes from the camera (offline analysis).

Decomposition: offset (none / laboratory / field from straight stops, leave-one-out) x aggregation
(single frame / 25-sample window / last 3 s) x method (DS / raw HP / HP fixed)."""
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/stop_convergence.py', encoding='utf-8').read().split('METHODS = [')[0])
LAM = 299792458.0 / (2 * 7.9872e9); W = 25; ANCHOR_S = 2.0

def circular_median(wrapped):
    angle = 2 * np.pi * wrapped / LAM
    centre = np.angle(np.mean(np.exp(1j * angle)))
    deviation = np.angle(np.exp(1j * (angle - centre)))
    return (centre + np.median(deviation)) * LAM / (2 * np.pi)

step_rows = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; post = np.flatnonzero((g.tw >= tstop[t]).to_numpy())
        wrapped = np.mod(g['HP raw'].to_numpy(float), LAM); ds = g.DS.to_numpy(float); tw = g.tw.to_numpy()
        fixed = np.full(len(g), np.nan); previous = None; anchored = False
        for n, i in enumerate(post):
            frac = circular_median(wrapped[post[max(0, n - W + 1):n + 1]])
            if not anchored:
                reference = np.median(ds[post[:n + 1]])
                fixed[i] = frac + LAM * round((reference - frac) / LAM)
                anchored = tw[i] >= tstop[t] + ANCHOR_S
            else:
                fixed[i] = frac + LAM * round((previous - frac) / LAM)
            previous = fixed[i]
        g['HP fixed'] = fixed
        raw_steps = np.abs(np.angle(np.exp(1j * 2 * np.pi * np.diff(wrapped[post]) / LAM))) * LAM / (2 * np.pi)
        late = (tw > tw.max() - 3) & (tw >= tstop[t] + ANCHOR_S)
        step_rows.append(dict(trial=t, link=l, raw_step_p50_mm=1000 * np.median(raw_steps), raw_step_p95_mm=1000 * np.percentile(raw_steps, 95),
                              raw_steps_gt_quarter_pct=100 * np.mean(raw_steps > LAM / 4),
                              fixed_sd_mm=1000 * np.nanstd(fixed[late]) if late.sum() > 5 else np.nan,
                              fixed_range_mm=1000 * (np.nanmax(fixed[late]) - np.nanmin(fixed[late])) if late.sum() > 5 else np.nan,
                              ds_sd_mm=1000 * np.nanstd(ds[late]) if late.sum() > 5 else np.nan))
S = pd.DataFrame(step_rows); S.to_csv(OUTR / 'hp_fixed_links.csv', index=False)
print('single-exchange phase step [mm]: median %.2f, p95 %.2f, steps > quarter cycle %.1f %%' % (S.raw_step_p50_mm.median(), S.raw_step_p95_mm.median(), S.raw_steps_gt_quarter_pct.mean()))
print('link SD in the last 3 s [mm]: DS %.1f, HP fixed median %.2f (p90 %.2f, max %.2f); links whose HP fixed range exceeds half a cycle: %d / %d' % (
    S.ds_sd_mm.median(), S.fixed_sd_mm.median(), S.fixed_sd_mm.quantile(.9), S.fixed_sd_mm.max(), int((S.fixed_range_mm > 1000 * LAM / 2).sum()), int(S.fixed_range_mm.notna().sum())))

MODE = {'DS': 'DS', 'HP raw': 'PHASE', 'HP fixed': 'PHASE'}
def tail(t, m, l, agg):
    g = data[t][l]; v = g[m][g.tw > g.tw.max() - 3].dropna()
    if agg == '3 s': return float(v.median())
    if agg == '25 samples': return float(v.iloc[-1]) if m == 'HP fixed' else float(v.iloc[-W:].median())
    return float('nan') if m == 'HP fixed' else float(v.iloc[-1])
imp = {m: {t: {l: tail(t, m, l, '3 s') - slant(*ref(t))[l] for l in LINKS} for t in runs} for m in MODE}
rows = []
for m in MODE:
    for t in runs:
        x, y, yaw = ref(t); R0 = rx(x, y, yaw)
        others = [u for u in runs if u != t and sched.loc[u, 'path'] == 'S']
        offsets = {'none': {l: 0.0 for l in LINKS}, 'laboratory': {l: float(LAB[MODE[m]][l]) for l in LINKS},
                   'field': {l: float(np.median([imp[m][u][l] for u in others])) for l in LINKS}}
        for oname, off in offsets.items():
            for agg in ('single frame', '25 samples', '3 s'):
                r = {l: tail(t, m, l, agg) - off[l] for l in LINKS}
                if any(np.isnan(list(r.values()))): continue
                p = solve(r, (x, y, yaw))
                rows.append(dict(method=m, offset=oname, aggregation=agg, trial=t, tag_cm=100 * math.hypot(p[0] - x, p[1] - y),
                                 yaw_deg=abs(p[2] - yaw), rx_cm=100 * float(np.linalg.norm(rx(*p) - R0))))
        off = offsets['field']; tend = min(data[t][l].tw.max() for l in LINKS); t0 = max(tend - 3, tstop[t] + ANCHOR_S + 0.2); est = []
        for tg in np.arange(t0, tend, 0.1):
            r = {}
            for l in LINKS:
                g = data[t][l]; w = g[m][(g.tw <= tg) & (g.tw >= tstop[t])].dropna()
                r[l] = (float(w.iloc[-1]) if m == 'HP fixed' else float(w.iloc[-W:].median())) - off[l] if len(w) else np.nan
            if not any(np.isnan(list(r.values()))): p = solve(r, (x, y, yaw)); est.append(np.r_[rx(*p), p[2]])
        if len(est) >= 5:
            E = np.array(est); rows.append(dict(method=m, offset='field', aggregation='jitter', trial=t,
                                                rx_cm=100 * float(np.sqrt(E[:, 0].var() + E[:, 1].var())), yaw_deg=float(E[:, 2].std())))
D = pd.DataFrame(rows); D.to_csv(OUTR / 'stop_decomposition_hp_stops.csv', index=False)
pd.set_option('display.width', 200)
Q = D[D.aggregation != 'jitter']
T = Q.groupby(['offset', 'aggregation', 'method'])[['tag_cm', 'yaw_deg', 'rx_cm']].median().round(2)
T['rx_le10'] = Q.groupby(['offset', 'aggregation', 'method']).rx_cm.apply(lambda v: f'{int((v <= 10).sum())}/{len(v)}')
print(T.to_string()); T.to_csv(OUTR / 'stop_decomposition_hp_stops_summary.csv')
J = D[D.aggregation == 'jitter']
print('running-estimate jitter over the last 3 s (25-sample window), RX [cm] and yaw [deg], median (min-max) over', J.trial.nunique(), 'stops')
for m in MODE:
    g = J[J.method == m]; print(f'  {m:9s} RX {g.rx_cm.median():.2f} ({g.rx_cm.min():.2f}-{g.rx_cm.max():.2f})  yaw {g.yaw_deg.median():.2f} ({g.yaw_deg.min():.2f}-{g.yaw_deg.max():.2f})')
JP = J.pivot(index='trial', columns='method', values='rx_cm'); print('  HP fixed jitter below DS in', int((JP['HP fixed'] < JP['DS']).sum()), '/', len(JP))
P = Q[Q.offset == 'field'].pivot_table(index='trial', columns=['aggregation', 'method'], values='rx_cm')
for agg in ('25 samples', '3 s'):
    d = (P[(agg, 'HP fixed')] - P[(agg, 'DS')]).dropna(); se = d.std(ddof=1) / math.sqrt(len(d))
    print(f'RX {agg}: HP fixed - DS mean {d.mean():+.2f} cm, 95% CI +-{2.131 * se:.2f} (t15), lower in {int((d < 0).sum())}/{len(d)}')

