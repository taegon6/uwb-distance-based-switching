import glob, os, math
import numpy as np, pandas as pd
EV = r'D:/sandbox/uwb/output/field_20261124/cam/eval'
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
rows = []
for f in glob.glob(EV + '/pairs_T*.csv'):
    t = os.path.basename(f)[6:-4]; p = pd.read_csv(f); c = pd.read_csv(EV + f'/camtrack_{t}.csv').sort_values('t')
    for k in ('RX_X', 'RX_Y', 'psi'):
        p['c' + k] = np.interp(p.t, c.t, c[k], left=np.nan, right=np.nan)
    p = p.dropna(subset=['cRX_X'])
    ex = p.rx_coil_x_m - p.cRX_X; ey = p.rx_coil_y_m - p.cRX_Y
    ps = np.radians(p.cpsi); lat = ex * np.cos(ps) + ey * np.sin(ps)        # perpendicular to vehicle heading
    dy = (p.yaw_deg - p.cpsi + 180) % 360 - 180
    for sg in ('far', 'mid', 'near'):
        m = p.seg == sg
        if m.sum() > 10:
            rows.append(dict(trial=t, path=sched.loc[t, 'path'], policy=sched.loc[t, 'policy'], seg=sg,
                             rx_lat_rmse=100 * math.sqrt(np.mean(lat[m] ** 2)), yaw_rmse=math.sqrt(np.mean(dy[m] ** 2))))
R = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(R.groupby(['path', 'seg', 'policy'])[['rx_lat_rmse', 'yaw_rmse']].median().round(1).unstack('seg').to_string())
R.to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/rx_and_yaw_by_segment.csv', index=False)

