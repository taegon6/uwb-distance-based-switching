"""How often does the stationary integer lock hold on all four links, and what does it give when it holds?"""
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/stop_convergence.py', encoding='utf-8').read().split('METHODS = [')[0].replace("'HP locked filt': o['filtered_range_m']})", "'HP locked filt': o['filtered_range_m'], 'anch': o['anchored'], 'stat': o['stationary']})"))
import math
rows = []
for t in runs:
    worst_range_mm, anchors_last3 = 0.0, 0
    for l in LINKS:
        g = data[t][l]; w = g[g.tw > g.tw.max() - 3]
        locked = w['HP locked'].dropna()
        worst_range_mm = max(worst_range_mm, 1000 * float(locked.max() - locked.min()) if len(locked) else 999.0)
        anchors_last3 += int(w.anch.sum())
    rows.append(dict(trial=t, worst_link_range_mm=worst_range_mm, anchors_last3s=anchors_last3, all4_stable=worst_range_mm < 9.4))
L = pd.DataFrame(rows).set_index('trial')
J = pd.read_csv(OUTR / 'stop_estimate_jitter.csv').pivot(index='trial', columns='method', values='rx_jitter_cm')
E = pd.read_csv(OUTR / 'stop_estimate_jitter.csv').pivot(index='trial', columns='method', values='rx_err_mean_cm')
L = L.join(J.add_prefix('jit_')).join(E.add_prefix('err_'))
pd.set_option('display.width', 220)
print(L.round(2).to_string())
print('all four links stable in the last 3 s:', int(L.all4_stable.sum()), '/', len(L))
for name, g in L.groupby('all4_stable'):
    print('stable' if name else 'unstable', len(g), 'RX jitter median DS %.2f / HP locked %.2f cm; RX error median DS %.1f / HP locked %.1f cm' % (
        g['jit_DS'].median(), g['jit_HP locked'].median(), g['err_DS'].median(), g['err_HP locked'].median()))
L.to_csv(OUTR / 'hp_lock_stability.csv')
H = pd.read_csv(OUTR / 'hp_locking_same_stop.csv')
for col in ('rx_err_cm', 'yaw_err_deg'):
    P = H.pivot(index='trial', columns='method', values=col)
    d = (P['lock_single'] - P['ds_single']).dropna(); se = d.std(ddof=1) / math.sqrt(len(d))
    print(f'single frame {col}: HP locked - DS mean {d.mean():+.2f}, 95% CI +-{2.12 * se:.2f}, lower in {int((d < 0).sum())}/{len(d)}')

