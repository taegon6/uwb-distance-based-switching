"""Disclosure numbers for the relative-displacement analysis (phase_displacement.py): fitted time shifts, share of moving samples,
and the error restricted to the samples in which the camera displacement exceeds 2 mm."""
import numpy as np, pandas as pd, math
R = r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/'
D = pd.read_csv(R + 'phase_relative_displacement.csv'); S = pd.read_csv(R + 'phase_relative_displacement_series.csv')
print('links', len(D), 'stops', D.trial.nunique(), '| RMS all samples: phase median %.2f mm, DS median %.2f mm | travel median %.1f mm | window median %.1f s' % (D.phase_rms_mm.median(), D.ds_rms_mm.median(), D.travel_mm.median(), D.window_s.median()))
print('fitted shift [ms]: median %.0f, at the search bound (|shift| >= 149): %d of %d' % (D.shift_ms.median(), int((D.shift_ms.abs() >= 149).sum()), len(D)))
rows = []
for (t, l), g in S.groupby(['trial', 'link']):
    mv = g.camera_mm.abs() > 2
    rows.append(dict(trial=t, link=l, n=len(g), moving=int(mv.sum()), moving_share=mv.mean(),
                     phase_moving=math.sqrt(np.mean((g.phase_mm - g.camera_mm)[mv] ** 2)) if mv.sum() > 5 else np.nan,
                     ds_moving=math.sqrt(np.mean((g.ds_mm - g.camera_mm)[mv] ** 2)) if mv.sum() > 5 else np.nan))
M = pd.DataFrame(rows); ok = M.dropna(subset=['phase_moving'])
print('share of samples with |camera displacement| > 2 mm: median %.2f %% (mean %.1f %%)' % (100 * M.moving_share.median(), 100 * M.moving_share.mean()))
print('moving samples only: links %d (stops %d), samples per link median %d | phase median %.2f mm, DS median %.2f mm' % (len(ok), ok.trial.nunique(), int(ok.moving.median()), ok.phase_moving.median(), ok.ds_moving.median()))
M.round(4).to_csv(R + 'phase_relative_displacement_moving_only.csv', index=False)

