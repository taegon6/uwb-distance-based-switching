"""Link T2-A1 on the curved path (Section 4.3 of the manuscript): excess of its ranging error over the mean ranging
error of the other three links while the heading of the vehicle was 30-45 deg away from the axis of the parking space.

Reads revision_checks/link_cause_frames.csv of the analysis computer (written by link_cause.py; a frame-level table of
4 MB, not included) and writes derived_data/revision_checks/link_t2a1_excess_c1_30_45_by_trial.csv with the median
excess of each of the 15 trials on the curved path and, in the row 'all frames', the median over all frames.
"""
from pathlib import Path
import pandas as pd

SRC = Path(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/link_cause_frames.csv')
OUT = Path(__file__).resolve().parents[1] / 'derived_data' / 'revision_checks' / 'link_t2a1_excess_c1_30_45_by_trial.csv'

F = pd.read_csv(SRC)
lc = F[(F.path == 'C1') & (F.psi.abs() >= 30) & (F.psi.abs() < 45)]
per = lc.groupby('trial').excess.agg(frames='size', excess_median_cm='median').reset_index()
T = pd.concat([per, pd.DataFrame([dict(trial='all frames', frames=len(lc), excess_median_cm=lc.excess.median())])], ignore_index=True)
T['excess_median_cm'] = T.excess_median_cm.round(2)
T.to_csv(OUT, index=False)
print(T.to_string(index=False))
