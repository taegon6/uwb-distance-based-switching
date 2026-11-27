exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/stop_convergence.py', encoding='utf-8').read().split('taus = [')[0])
# jitter of the running estimate (25-sample running median per link -> pose) over the last 3 s, and N-anchor cycle errors
rows = []
for t in runs:
    x, y, yaw = ref(t)
    for m in METHODS:
        of = offs(t, m); col = 'HP locked filt' if m == 'HP locked' else m
        tend = min(data[t][l].tw.max() for l in LINKS); grid = np.arange(tend - 3, tend, 0.1)
        est = []
        for tg in grid:
            r = {}
            for l in LINKS:
                g = data[t][l]; w = g[g.tw <= tg]
                v = (w[m].iloc[-25:] if m != 'HP locked' else w[col].iloc[-1:]).dropna()
                r[l] = float(np.median(v)) - of[l] if len(v) else np.nan
            if not any(np.isnan(list(r.values()))):
                p = solve(r, (x, y, yaw)); est.append(np.r_[rx(*p), p[2]])
        E = np.array(est)
        if len(E) > 5:
            rows.append(dict(trial=t, method=m, rx_jitter_cm=100 * float(np.sqrt(E[:, 0].var() + E[:, 1].var())), yaw_jitter_deg=float(E[:, 2].std()),
                             rx_err_mean_cm=100 * float(np.linalg.norm(E[:, :2].mean(0) - rx(x, y, yaw)))))
J = pd.DataFrame(rows)
print('== running estimate over the last 3 s (25-sample median per link, sampled every 0.1 s), median over stops')
print(J.groupby('method')[['rx_jitter_cm', 'yaw_jitter_deg', 'rx_err_mean_cm']].median().round(2).to_string())
P = J.pivot(index='trial', columns='method', values='rx_jitter_cm')
print('jitter HP locked < DS in', int((P['HP locked'] < P['DS']).sum()), '/', len(P))
# N-anchor error: locked minus raw-phase-lattice level relative to DS over last 3 s, in cycles
lam = 0.0187670559
cyc = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; w = g[g.tw > g.tw.max() - 3]
        cyc.append(dict(trial=t, link=l, locked_minus_ds_cycles=(np.nanmedian(w['HP locked']) - np.nanmedian(w['DS'])) / lam))
C = pd.DataFrame(cyc)
print('locked - DS median difference in cycles: quantiles', C.locked_minus_ds_cycles.quantile([0.05, .25, .5, .75, .95]).round(2).to_dict())
J.to_csv(OUTR/'stop_estimate_jitter.csv', index=False); C.to_csv(OUTR/'hp_lock_vs_ds_cycles.csv', index=False)

