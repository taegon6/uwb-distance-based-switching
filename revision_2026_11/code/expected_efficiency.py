"""Expected coil efficiency at the lateral offsets of the stops, by piecewise-linear interpolation of the measured values of one published 3.3 kW system
(Nie et al., IET Power Electron. 2021: 92.97 % aligned, 92.37 % at 10 cm, 90.57 % at 20 cm lateral offset). Illustration only; nothing was measured here."""
import numpy as np, pandas as pd
ST = pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/stops_v2_seltos_L439.csv')
X, E = [0, 10, 20], [92.97, 92.37, 90.57]
loss = lambda o: 92.97 - float(np.interp(o, X, E))
rows = []
for name, lat in (('physical lateral offset of the RX reference point', ST.rx_ref_x_cm), ('lateral error of the RX estimate', ST.sfit_rx_dlat_cm)):
    a = lat.abs(); inside = a[a <= 20]
    rows.append(dict(quantity=name, stops=len(a), median_offset_all_stops_cm=a.median(), loss_at_median_offset_points=loss(a.median()), efficiency_at_median_offset_pct=92.97 - loss(a.median()),
                     stops_within_20cm=len(inside), largest_offset_within_20cm_cm=inside.max(), loss_at_largest_offset_within_20cm_points=loss(inside.max()),
                     median_efficiency_of_stops_within_20cm_pct=float(np.median([92.97 - loss(v) for v in inside])), stops_beyond_20cm=int((a > 20).sum()), largest_offset_cm=a.max()))
D = pd.DataFrame(rows); pd.set_option('display.width', 250); print(D.round(3).T.to_string())
D.round(4).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/expected_efficiency_illustration.csv', index=False)

