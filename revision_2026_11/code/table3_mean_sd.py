"""Table 3 of the revised manuscript: mean and standard deviation over the five trials of each method.

Reads camera_eval/segment_errors.csv, stops/runs_summary_dedup.csv, and stops/stops_v2_seltos_L439.csv
from the derived_data folder of this repository. 'far' is the alignment distance range from 4 to 10.5 m
and 'mid' the range from 0.5 to 4 m. Run with --check to compare the result with
revision_checks/table3_v2_mean_sd.csv.
"""
import argparse
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'derived_data'
NAME = [('FIXED_SS', 'Fixed SS-TWR'), ('FIXED_DS', 'Fixed DS-TWR'), ('FIXED_PHASE', 'Fixed high-precision DS-TWR'), ('STAGED', 'Proposed method')]
PATHS = [('S', 'Straight'), ('C1', 'Curve')]


def pm(v, d=1):
    v = pd.Series(v).dropna()
    if len(v) == 0:
        return '–'
    return '%.*f' % (d, v.mean()) if len(v) == 1 else '%.*f ± %.*f' % (d, v.mean(), d, v.std(ddof=1))


def table3():
    E = pd.read_csv(D / 'camera_eval' / 'segment_errors.csv'); E = E[E.seg.isin(['far', 'mid'])]
    R = pd.read_csv(D / 'stops' / 'runs_summary_dedup.csv')[['trial', 'policy']]
    S = pd.read_csv(D / 'stops' / 'stops_v2_seltos_L439.csv').merge(R, on='trial')
    if 'policy' not in E.columns:
        E = E.merge(R, on='trial')
    rows = []
    for pk, pn in PATHS:
        for k, n in NAME:
            g = E[(E.path == pk) & (E.policy == k)]; s = S[(S.path == pk) & (S.policy == k)]
            if g.empty and s.empty:
                continue
            row = [pn, n, str(len(s))]
            for sg in ('far', 'mid'):
                h = g[g.seg == sg]; row += [pm(h.rate_hz), pm(h.log_lat_rmse), pm(h.log_lon_rmse)]
            row += [pm(s.sfit_dyaw.abs()), pm(s.sfit_rx_err_cm)]
            rows.append(row)
    return pd.DataFrame(rows, columns=['path', 'method', 'runs', 'far_rate', 'far_lat', 'far_lon', 'near_rate', 'near_lat',
                                       'near_lon', 'heading_err_deg', 'coil_err_cm'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--check', action='store_true'); a = ap.parse_args()
    T = table3()
    if a.check:
        ref = pd.read_csv(D / 'revision_checks' / 'table3_v2_mean_sd.csv', dtype=str)
        print('identical to table3_v2_mean_sd.csv:', T.astype(str).values.tolist() == ref.values.tolist())
    else:
        print(T.to_string(index=False))

