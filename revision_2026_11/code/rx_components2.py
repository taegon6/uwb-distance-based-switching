import pandas as pd, numpy as np
from pathlib import Path
B = Path(r'D:/sandbox/uwb/output/field_20261124')
ST = pd.read_csv(B / 'analysis' / 'stops_v2_seltos_L439.csv')
lat, lon = ST.sfit_rx_dlat_cm, ST.sfit_rx_dlon_cm
dx, dy, dyaw = ST.sfit_dx_cm, ST.sfit_dy_cm, ST.sfit_dyaw
print('n', len(ST))
print('RX est err: lat signed mean %.2f SD %.2f | lon signed mean %.2f SD %.2f' % (lat.mean(), lat.std(), lon.mean(), lon.std()))
print('RX est err: |lat| med %.2f, |lon| med %.2f, 2D med %.2f' % (lat.abs().median(), lon.abs().median(), ST.sfit_rx_err_cm.median()))
print('tag mid err: |dx| med %.2f, |dy| med %.2f, 2D med %.2f' % (dx.abs().median(), dy.abs().median(), ST.sfit_err_cm.median()))
print('dyaw unit check: |dyaw| med %.2f max %.2f' % (dyaw.abs().median(), dyaw.abs().max()))
h_lat = lat - dx; h_lon = lon - dy
print('heading term (RX minus tag mid): |lat| med %.2f, |lon| med %.2f' % (h_lat.abs().median(), h_lon.abs().median()))
pred = 220 * np.sin(np.radians(dyaw))
print('220 sin(dyaw) vs heading lat term: corr %.3f, max abs diff of magnitudes %.2f' % (np.corrcoef(pred, h_lat)[0, 1], float((pred.abs() - h_lat.abs()).abs().max())))
print('corr(lat err, heading lat term) %.3f ; var(lat) %.1f var(heading term) %.1f var(dx) %.1f cov %.1f' % (np.corrcoef(lat, h_lat)[0, 1], lat.var(), h_lat.var(), dx.var(), np.cov(dx, h_lat)[0, 1]))
print('share of stops where |heading lat term| > |dx|: %d of %d' % ((h_lat.abs() > dx.abs()).sum(), len(ST)))
print('|lat|<=10: %d, |lon|<=7.5: %d, both %d' % ((lat.abs() <= 10).sum(), (lon.abs() <= 7.5).sum(), ((lat.abs() <= 10) & (lon.abs() <= 7.5)).sum()))
ax, ay = ST.rx_ref_x_cm, ST.rx_ref_y_cm
ins_ref = (ax.abs() <= 10) & (ay.abs() <= 7.5)
ex, ey = ax + lat, ay + lon
ins_est = (ex.abs() <= 10) & (ey.abs() <= 7.5)
print('inside ref %d, inside est %d, agree %d, est-inside&ref-outside %d, est-outside&ref-inside %d' % (ins_ref.sum(), ins_est.sum(), (ins_ref == ins_est).sum(), (ins_est & ~ins_ref).sum(), (~ins_est & ins_ref).sum()))
print('actual: |x| med %.2f |y| med %.2f' % (ax.abs().median(), ay.abs().median()))
for m, g in ST.groupby('stop_mode'):
    print(m, len(g), '|lat| %.2f |lon| %.2f |dyaw| %.2f' % (g.sfit_rx_dlat_cm.abs().median(), g.sfit_rx_dlon_cm.abs().median(), g.sfit_dyaw.abs().median()))
for m, g in ST.groupby('path'):
    print(m, len(g), '|lat| %.2f |lon| %.2f |dyaw| %.2f' % (g.sfit_rx_dlat_cm.abs().median(), g.sfit_rx_dlon_cm.abs().median(), g.sfit_dyaw.abs().median()))
