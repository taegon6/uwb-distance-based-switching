"""Figure for cam/c1_link_correction.py: (a) excess of link T2-A1 against the bearing of A1 seen from T2, (b) lateral RMSE of the curved runs before and after the correction."""
import sys, math, json, io
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); OUTR = BASE / 'analysis' / 'revision_checks'
D = pd.read_csv(OUTR / 'c1_link_bearing_frames.csv'); R = pd.read_csv(OUTR / 'c1_link_correction.csv')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9.5, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.7))
bins = np.arange(-50, 51, 10); mid = (bins[:-1] + bins[1:]) / 2
for path, col, mk in (('C1', '#e06c1b', 'o'), ('C2', '#2e8b3a', 's')):
    g = D[D.path == path]; idx = np.digitize(g.beta, bins) - 1; med = []; lo = []; hi = []
    for k in range(len(mid)):
        v = g.excess_cm[idx == k]
        if len(v) >= 50: med.append(v.median()); lo.append(v.quantile(.25)); hi.append(v.quantile(.75))
        else: med.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    med, lo, hi = map(np.array, (med, lo, hi)); ok = np.isfinite(med)
    ax[0].errorbar(mid[ok] + (0.8 if path == 'C2' else -0.8), med[ok], yerr=[med[ok] - lo[ok], hi[ok] - med[ok]], fmt=mk + '-', color=col, ms=4, lw=1, capsize=2, label=f'{path} (median, IQR)')
ax[0].axhline(0, color='0.3', lw=0.8); ax[0].axvline(10, color='0.5', ls=':', lw=0.9)
ax[0].set_xlabel('Bearing of anchor A1 seen from tag T2 [°]'); ax[0].set_ylabel('Excess of the T2–A1 range [cm]'); ax[0].set_title('(a) Excess of link T2–A1 over the other three links', fontsize=9.5); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8, loc='upper left')
x = 0
for path, seg in (('C1', 'far'), ('C1', 'mid'), ('C2', 'far'), ('C2', 'mid')):
    g = R[(R.path == path) & (R.seg == seg)]
    for _, r in g.iterrows(): ax[1].plot([x - 0.18, x + 0.18], [r.resolved, r.corrected], color='0.7', lw=0.8, zorder=1)
    ax[1].plot([x - 0.18] * len(g), g.resolved, 'o', color='#1f5fbf', ms=3.5, zorder=2); ax[1].plot([x + 0.18] * len(g), g.corrected, 'o', color='crimson', ms=3.5, zorder=2); x += 1
ax[1].plot([], [], 'o', color='#1f5fbf', ms=4, label='As recorded'); ax[1].plot([], [], 'o', color='crimson', ms=4, label='T2–A1 corrected by bearing')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(['C1, 4–10.5 m', 'C1, 0.5–4 m', 'C2, 4–10.5 m', 'C2, 0.5–4 m'], fontsize=8.5); ax[1].set_ylabel('Lateral RMSE of a run [cm]'); ax[1].set_ylim(0, None)
ax[1].set_title('(b) Curved runs before and after the correction', fontsize=9.5); ax[1].grid(alpha=.3, axis='y'); ax[1].legend(fontsize=8)
fig.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(BASE / 'fig' / f'c1_link_correction.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.06)
print('saved')

