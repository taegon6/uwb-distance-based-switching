"""Per-link export of the comparison between the fixed-integer range and the DS-TWR median (same code as hp_precision.py)."""
import io, contextlib, numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_precision.py', encoding='utf-8').read()
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(src)
print(buf.getvalue()[-400:])
rows = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy(); late = (tw >= tstop[t] + ANCHOR_S)
        if late.sum() < 30: continue
        rows.append(dict(trial=t, link=l, samples=int(late.sum()), fixed_minus_ds_cycles=(np.nanmedian(g['HP fixed'].to_numpy(float)[late]) - np.nanmedian(g.DS.to_numpy(float)[late])) / LAM))
A = pd.DataFrame(rows); A['more_than_half_cycle'] = A.fixed_minus_ds_cycles.abs() > 0.5
A.round(4).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/hp_fixed_vs_ds_cycles.csv', index=False)
print('links', len(A), 'more than half a cycle', int(A.more_than_half_cycle.sum()), 'stops', A.trial.nunique())

