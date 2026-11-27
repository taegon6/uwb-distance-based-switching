import pandas as pd, numpy as np
from pathlib import Path
B = Path(r'D:/sandbox/uwb/output/field_20261124')
ST = pd.read_csv(B / 'analysis' / 'stops_v2_seltos_L439.csv')
print([c for c in ST.columns if 'sfit' in c or 'rx_ref' in c])
lat, lon = ST.sfit_rx_dlat_cm, ST.sfit_rx_dlon_cm
print('RX coil estimate error (field offsets, own final mode): |lateral| median %.2f, |longitudinal| median %.2f, 2D median %.2f' % (lat.abs().median(), lon.abs().median(), ST.sfit_rx_err_cm.median()))
print('share with |lat err| <= 10: %d, |lon err| <= 7.5: %d, both: %d of %d' % ((lat.abs() <= 10).sum(), (lon.abs() <= 7.5).sum(), ((lat.abs() <= 10) & (lon.abs() <= 7.5)).sum(), len(ST)))
print('check 2D = hypot:', float((np.hypot(lat, lon) - ST.sfit_rx_err_cm).abs().max()))
# decision: is the stop inside the SAE J2954 tolerance, by the reference and by the UWB estimate
ax, ay = ST.rx_ref_x_cm, ST.rx_ref_y_cm
inside_ref = (ax.abs() <= 10) & (ay.abs() <= 7.5)
for sx in (1, -1):
    ex, ey = ax + sx * lat, ay + lon
    inside_est = (ex.abs() <= 10) & (ey.abs() <= 7.5)
    print('sign', sx, 'inside by reference %d, by estimate %d, agree %d, false inside %d, false outside %d' % (inside_ref.sum(), inside_est.sum(), (inside_ref == inside_est).sum(), (inside_est & ~inside_ref).sum(), (~inside_est & inside_ref).sum()))
print(ST[['trial', 'path', 'stop_mode', 'rx_ref_x_cm', 'rx_ref_y_cm', 'sfit_rx_dlat_cm', 'sfit_rx_dlon_cm', 'sfit_rx_err_cm']].head(6).round(1).to_string())
print(ST.groupby('path')[['sfit_rx_dlat_cm', 'sfit_rx_dlon_cm']].agg(lambda v: v.abs().median()).round(2).to_string())
src = open(B / 'analyze_stops_v2.py', encoding='utf-8').read(); i = src.find('rx_dlat'); print(src[max(0, i - 500): i + 300])

