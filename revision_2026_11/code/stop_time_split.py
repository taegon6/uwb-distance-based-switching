"""Final-stop accuracy with a calibration that is separated in time from the evaluation:
link offsets from the straight stops of blocks 1-2 (T01-T08), evaluation on all later stops."""
import math
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/analyze_stops_v2.py', encoding='utf-8').read()
src = src.replace('BASE = Path(__file__).parent', "BASE = Path(r'D:/sandbox/uwb/output/field_20261124')")
exec(src.split('# 1) original entries')[0])
S = pd.read_csv(OUT / 'stops_v2_seltos_L439.csv').set_index('trial')
I = pd.read_csv(OUT / 'stop_implied_link_offsets_v2.csv')
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
calib = [t for t in S.index if S.loc[t, 'path'] == 'S' and sched.loc[t, 'block'] <= 2]
offsets = I[I.trial.isin(calib)].groupby('link')['off'].median()
print('calibration stops:', calib); print('offsets [cm]:', (100 * offsets).round(1).to_dict())
rows = []
for t in S.index:
    if t in calib: continue
    fm, m, path = meas[t]; x, y, yaw = S.loc[t, 'ref_x'], S.loc[t, 'ref_y'], S.loc[t, 'ref_yaw']
    p = solve({l: m[l] - offsets[l] for l in LINKS}, (x, y, yaw))
    rows.append(dict(trial=t, path=path, stop_mode=fm, block=sched.loc[t, 'block'] if t in sched.index else np.nan, tag_cm=100 * math.hypot(p[0] - x, p[1] - y), yaw_deg=abs(p[2] - yaw),
                     rx_cm=100 * float(np.linalg.norm(rx(*p) - rx(x, y, yaw))), loo_same_day_cm=S.loc[t, 'sfit_err_cm']))
D = pd.DataFrame(rows); D.round(3).to_csv(OUT / 'revision_checks' / 'stop_time_split_calibration.csv', index=False)
print(D.groupby('path').agg(runs=('trial', 'count'), tag_median=('tag_cm', 'median'), tag_max=('tag_cm', 'max'), within10=('tag_cm', lambda v: int((v <= 10).sum())), rx_median=('rx_cm', 'median'),
                            rx_within10=('rx_cm', lambda v: int((v <= 10).sum())), yaw_median=('yaw_deg', 'median'), previous_median=('loo_same_day_cm', 'median')).round(2).to_string())
print('all: tag median %.2f cm, within 10 cm %d of %d; RX median %.2f cm, within 10 cm %d' % (D.tag_cm.median(), int((D.tag_cm <= 10).sum()), len(D), D.rx_cm.median(), int((D.rx_cm <= 10).sum())))

