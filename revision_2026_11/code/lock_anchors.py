exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/stop_convergence.py', encoding='utf-8').read().split('METHODS = [')[0].replace("'HP locked filt': o['filtered_range_m']})", "'HP locked filt': o['filtered_range_m'], 'anch': o['anchored'], 'stat': o['stationary']})"))
rows = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; w = g[g.tw >= tstop[t]]
        lk = w['HP locked'].dropna()
        jumps = int((np.abs(np.diff(w['HP locked filt'].dropna().to_numpy())) > 0.009).sum())
        rows.append(dict(trial=t, link=l, after_stop_s=w.tw.max() - tstop[t], stationary_pct=100 * w.stat.mean(), anchors=int(w.anch.sum()), filt_jumps_gt_half_cycle=jumps))
R = pd.DataFrame(rows)
print(R.groupby('trial')[['after_stop_s', 'stationary_pct', 'anchors', 'filt_jumps_gt_half_cycle']].agg({'after_stop_s': 'first', 'stationary_pct': 'median', 'anchors': 'sum', 'filt_jumps_gt_half_cycle': 'sum'}).round(1).to_string())
print('links with >=2 anchors after stop:', int((R.anchors >= 2).sum()), '/', len(R))

