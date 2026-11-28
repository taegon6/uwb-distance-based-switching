"""Sensitivity of the final-stop results to the two sign corrections of the manual reference (T21 longitudinal, T34 rear lateral).
The evaluation code of analyze_stops_v2.py is reused unchanged; only the reference entries differ."""
import math, numpy as np, pandas as pd
from pathlib import Path
P = Path(r'D:/sandbox/uwb/output/field_20261124/analyze_stops_v2.py')
src = P.read_text(encoding='utf-8').split('# 2) sign-flip check')[0]
g = {'__file__': str(P), '__name__': 'stops'}
exec(compile(src, str(P), 'exec'), g)
st, ref_from, evaluate = g['st'], g['ref_from'], g['evaluate']
def refs(flip21, flip34):
    out = {}
    for s in st.itertuples():
        lat, lon, rear = s.lateral_cm, s.long_cm, s.rear_lat_cm
        if s.trial == 'T21' and flip21: lon = -lon
        if s.trial == 'T34' and flip34: rear = -rear
        out[s.trial] = ref_from(lat, lon, rear)
    return out
rows = []
for name, f21, f34 in (('original entries', False, False), ('T21 corrected only', True, False), ('T34 corrected only', False, True), ('both corrected (manuscript)', True, True)):
    E, _ = evaluate(refs(f21, f34), name)
    r = dict(case=name, tag_median_cm=E.sfit_err_cm.median(), tag_le10=int((E.sfit_err_cm <= 10).sum()), rx_median_cm=E.sfit_rx_err_cm.median(), rx_le10=int((E.sfit_rx_err_cm <= 10).sum()),
             yaw_median_deg=E.sfit_dyaw.abs().median(), n=len(E))
    for t in ('T21', 'T34'):
        e = E[E.trial == t].iloc[0]; r[t + '_tag_cm'] = e.sfit_err_cm; r[t + '_rx_cm'] = e.sfit_rx_err_cm; r[t + '_dyaw'] = e.sfit_dyaw
    ax, ay = E.rx_ref_x_cm, E.rx_ref_y_cm; r['driver_inside_sae'] = int(((ax.abs() <= 10) & (ay.abs() <= 7.5)).sum())
    _ir = (ax.abs() <= 10) & (ay.abs() <= 7.5); _ie = ((ax + E.sfit_rx_dlat_cm).abs() <= 10) & ((ay + E.sfit_rx_dlon_cm).abs() <= 7.5)
    r['estimate_inside'] = int(_ie.sum()); r['estimate_inside_and_reference_inside'] = int((_ie & _ir).sum()); r['reference_inside_missed'] = int((~_ie & _ir).sum()); r['driver_lat_med'] = ax.abs().median(); r['driver_lon_med'] = ay.abs().median()
    rows.append(r)
D = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(D.round(3).to_string(index=False))
D.round(4).to_csv(Path(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/reference_sign_sensitivity.csv'), index=False, encoding='utf-8-sig')
print(st[st.trial.isin(['T21', 'T34'])][['trial', 'lateral_cm', 'long_cm', 'rear_lat_cm']].to_string(index=False))

