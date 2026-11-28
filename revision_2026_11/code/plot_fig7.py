"""Figure 7 (revised): motion accuracy vs update rate, final-stop calibration, ranging method at the stop."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'; RC = BASE / 'analysis' / 'revision_checks'; FIG = BASE / 'fig'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9.5, 'mathtext.fontset': 'stix'})
COL = {'FIXED_SS': '#1f5fbf', 'FIXED_DS': '#e06c1b', 'FIXED_PHASE': '#2e8b3a', 'STAGED': 'k'}
LAB = {'FIXED_SS': 'Fixed SS-TWR', 'FIXED_DS': 'Fixed DS-TWR', 'FIXED_PHASE': 'Fixed HP DS-TWR', 'STAGED': 'Distance-based switching'}
MK = {'far': 'o', 'mid': 's'}
fig, ax = plt.subplots(2, 2, figsize=(10, 7.6))

# (a) straight approach while moving: lateral RMSE against the update rate of the same segment
E = pd.read_csv(EV / 'segment_errors.csv'); E = E[E.path == 'S']
a = ax[0, 0]
for pol in COL:
    for sg in ('far', 'mid'):
        g = E[(E.policy == pol) & (E.seg == sg)]
        x = g.rate_hz.median(); y = g.log_lat_rmse.median(); q = g.log_lat_rmse.quantile([.25, .75])
        a.errorbar(x, y, yerr=[[y - q.iloc[0]], [q.iloc[1] - y]], fmt=MK[sg], color=COL[pol], mfc=COL[pol] if sg == 'far' else 'white', ms=7, capsize=3)
a.set_xlabel('Position update rate in the segment [Hz]'); a.set_ylabel('Lateral RMSE vs camera [cm]')
a.set_title('(a) Straight approach, moving (median and IQR, 5 runs)', fontsize=9.5); a.grid(alpha=.3)
handles = [Line2D([], [], marker='o', color=COL[p], ls='', label=LAB[p]) for p in COL]
handles += [Line2D([], [], marker='o', color='0.3', ls='', label='$d_c$ 4–10.5 m'), Line2D([], [], marker='s', mfc='white', color='0.3', ls='', label='$d_c$ 0.5–4 m')]
a.legend(handles=handles, fontsize=7.5, loc='center')

# (b) final stop, all runs: link-offset calibration (3 s link medians solved for the pose, analyze_stops_v2.py)
S = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv')
b = ax[0, 1]; names = [('none', 'No offset'), ('lab', 'Laboratory\noffsets'), ('sfit', 'Field offsets\n(straight stops)')]
vals = [S[f'{k}_err_cm'].to_numpy() for k, _ in names]
b.boxplot(vals, widths=0.5, medianprops=dict(color='k'))
b.set_xticks([1, 2, 3]); b.set_xticklabels([n for _, n in names], fontsize=8)
b.axhline(10, color='crimson', ls=':', lw=1); b.set_ylabel('Final tag-midpoint error [cm]')
b.set_ylim(0, max(v.max() for v in vals) + 4)
for i, v in enumerate(vals):
    b.text(i + 1, max(v) + 0.8, f'{(v <= 10).sum()}/{len(v)} ≤ 10 cm', ha='center', fontsize=8)
b.set_title(f'(b) Final stop, 3 s median, all {len(vals[0])} runs', fontsize=9.5); b.grid(axis='y', alpha=.3)

# (c), (d) the same high-precision stops: ranging method, field offsets
H = pd.read_csv(RC / 'stop_decomposition_hp_stops.csv')
H = H[H.offset == 'field']
methods = [('DS', 'DS-TWR'), ('HP raw', 'HP DS-TWR\nraw'), ('HP fixed', 'HP DS-TWR\nfixed integer')]
c = ax[1, 0]
acc = [H[(H.aggregation == '3 s') & (H.method == m)].rx_cm.to_numpy() for m, _ in methods]
c.boxplot(acc, widths=0.5, medianprops=dict(color='k'))
c.set_xticks([1, 2, 3]); c.set_xticklabels([n for _, n in methods], fontsize=8)
c.axhline(10, color='crimson', ls=':', lw=1); c.set_ylabel('Final RX-coil error [cm]')
c.set_title(f'(c) Accuracy at the stop, 3 s median, same {len(acc[0])} stops', fontsize=9.5); c.grid(axis='y', alpha=.3)
d = ax[1, 1]
jit = H[H.aggregation == 'jitter'].pivot(index='trial', columns='method', values='rx_cm')
for trial, row in jit.iterrows():
    d.plot([1, 2, 3], [row[m] for m, _ in methods], color='0.75', lw=0.8, zorder=1)
d.boxplot([jit[m].to_numpy() for m, _ in methods], widths=0.5, medianprops=dict(color='k'), zorder=2)
d.set_yscale('log'); d.set_xticks([1, 2, 3]); d.set_xticklabels([n for _, n in methods], fontsize=8)
d.set_ylabel('Scatter of the running RX-coil estimate [cm]')
d.set_title('(d) Output stability at the stop, last 3 s, same stops', fontsize=9.5); d.grid(axis='y', which='both', alpha=.3)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'fig7_revised.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)
print('ok', [round(float(np.median(v)), 2) for v in acc], [round(float(jit[m].median()), 2) for m, _ in methods])

