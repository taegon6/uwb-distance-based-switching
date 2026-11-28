import numpy as np, pandas as pd
R = r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/'
E = pd.read_csv(R + 'stop_detector_fullstream_declarations.csv'); V = pd.read_csv(R + 'stop_detector_fullstream_vehicle.csv'); L = pd.read_csv(R + 'stop_detector_fullstream_links.csv')
def cat(r):
    if r.speed_at_declaration > 0.05: return 'declared while moving'
    if r.declared_rel_s >= 0: return 'after the camera stop'
    if r.dist_at_declaration_m <= 0.1: return 'near the end, before the camera stop'
    return 'standstill at the start'
E['category'] = E.apply(cat, axis=1)
print(E.groupby(['category', 'withdrawn']).agg(n=('trial', 'size'), stops=('trial', 'nunique'), links=('link', lambda v: len(v)), held_median=('held_s', 'median'), vmax=('max_speed_while_declared', 'max')).round(2).to_string())
for c in E.category.unique(): print(c, sorted(E[E.category == c].trial.unique()))
w = E[E.withdrawn]; print('withdrawn after the camera stop: stops', sorted(w[w.category == 'after the camera stop'].trial.unique()), 'n', int((w.category == 'after the camera stop').sum()))
s0 = E[E.category == 'standstill at the start']; print('start-standstill declarations: max speed reached while still declared', s0.groupby('trial').max_speed_while_declared.max().round(2).to_dict())
end = L.groupby('trial').declared_at_end.agg(['sum', 'all']); print('all four links declared at the end of the UWB record: %d of %d; not: %s' % (int(end['all'].sum()), len(end), end[~end['all']].index.tolist()))
print('four links while moving [s]:', V.set_index('trial').time_four_while_moving_s[V.set_index('trial').time_four_while_moving_s > 0].to_dict(), '| max speed with four links', V.set_index('trial').speed_max_while_four.round(2).to_dict())
E.round(4).to_csv(R + 'stop_detector_fullstream_declarations.csv', index=False)
# 13-stop result split by the four-link state at the end
J = pd.read_csv(R + 'stop_decomposition_hp_stops_causal.csv'); J = J[J.aggregation == 'jitter'].pivot(index='trial', columns='method', values='rx_cm')
J['all_four_at_end'] = end['all'].reindex(J.index)
print('\ncausal jitter stops', len(J), '| with all four links at the end', int(J.all_four_at_end.sum()))
for name, g in (('all 13', J), ('four links at end', J[J.all_four_at_end]), ('one link missing', J[~J.all_four_at_end])):
    print(name, len(g), 'DS median %.3f, HP fixed median %.4f, HP lower in %d' % (g['DS'].median(), g['HP fixed'].median(), int((g['HP fixed'] < g['DS']).sum())), sorted(g.index))
print('stops without a jitter value:', sorted(set(V.trial) - set(J.index)))
J.round(4).to_csv(R + 'stop_detector_causal_jitter_by_end_state.csv')

