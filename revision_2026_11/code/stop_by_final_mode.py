import math
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/pose_offset_calibration.py', encoding='utf-8').read()
exec(src.split("loo = evaluate(")[0])
O = OUT / 'revision_checks'
mode = pd.read_csv(O / 'stop_decomposition_runs.csv').drop_duplicates('trial').set_index('trial').stop_mode
straight = [t for t in trials if S.loc[t, 'path'] == 'S']
pooled = evaluate(lambda t: [u for u in straight if u != t], 'pooled')
own = evaluate(lambda t: [u for u in straight if u != t and mode[u] == mode[t]], 'own mode')
pd.set_option('display.width', 220)
for D in (pooled, own):
    C = D[D.model == 'constant'].copy(); C['stop_mode'] = C.trial.map(mode)
    print('== constant offsets from straight stops,', D.split.iloc[0], '(evaluated stop left out)')
    print(C.groupby('stop_mode').agg(n=('trial', 'count'), tag=('tag_cm', 'median'), yaw=('yaw_deg', 'median'), rx=('rx_cm', 'median'), rx_le10=('rx_cm', lambda v: int((v <= 10).sum())), tag_le10=('tag_cm', lambda v: int((v <= 10).sum()))).round(2).to_string())
    print('   all 39: tag %.2f (%d), yaw %.2f, rx %.2f (%d)' % (C.tag_cm.median(), (C.tag_cm <= 10).sum(), C.yaw_deg.median(), C.rx_cm.median(), (C.rx_cm <= 10).sum()))
    print('   training stops per mode:', {m: sum(1 for u in straight if mode[u] == m) for m in ('SS', 'DS', 'PHASE')})
# per-mode offsets (median implied offset over straight stops), cm
print('== implied link offsets at the straight stops by final mode (median, cm)')
print(pd.DataFrame({m: {l: 100 * np.median([implied[u][l] for u in straight if mode[u] == m]) for l in LINKS} for m in ('SS', 'DS', 'PHASE')}).round(1).to_string())
Hd = pd.read_csv(O / 'stop_decomposition_hp_stops.csv'); Hd = Hd[(Hd.offset == 'field') & (Hd.aggregation == '3 s') & (Hd.method == 'HP raw')].set_index('trial')
Cp = pooled[pooled.model == 'constant'].set_index('trial')
cmp = pd.DataFrame({'yaw_pooled': Cp.yaw_deg, 'rx_pooled': Cp.rx_cm, 'yaw_decomp': Hd.yaw_deg, 'rx_decomp': Hd.rx_cm}).dropna()
print(cmp.round(2).to_string()); print(cmp.median().round(2).to_dict())
own.round(3).to_csv(O / 'stop_own_mode_offsets.csv', index=False)


for D in (pooled, own):
    C = D[D.model == 'constant'].copy(); C['stop_mode'] = C.trial.map(mode); C['pol'] = C.trial.map(pd.read_csv(BASE / 'analysis' / 'runs_summary_dedup.csv').set_index('trial').policy)
    print('== by path and final mode,', D.split.iloc[0])
    print(C.groupby(['path', 'stop_mode']).agg(n=('trial', 'count'), tag=('tag_cm', 'median'), yaw=('yaw_deg', 'median'), rx=('rx_cm', 'median'), rx_le10=('rx_cm', lambda v: int((v <= 10).sum()))).round(2).to_string())
    print(C[C.path == 'S'].groupby(['pol']).agg(n=('trial', 'count'), yaw=('yaw_deg', 'median'), rx=('rx_cm', 'median'), rx_mean=('rx_cm', 'mean')).round(2).to_string())

