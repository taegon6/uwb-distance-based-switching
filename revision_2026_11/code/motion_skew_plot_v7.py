"""Figure 8 (v7): annotation "Speeds at alignment distances from 4 to 10.5 m". Figure 8 (v6): legend title "Ranging mode". Figure 8 (v5): band in (b) extended to 4-10.5 m so that it matches the distance range of the paper.
Figure 8: calculated bias from the sequential measurement of four links.
The calculation is preserved from sequential_model.py, which generated the existing
sequential_link_model figure. motion_skew_plot.py generated a different figure.
Creates new files only. Deterministic calculation: SD is not applicable.
"""
import ast
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(r'D:/sandbox/uwb/output/field_20261124')
RC = BASE / 'analysis' / 'revision_checks'
FIG = BASE / 'fig'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def save_figure(fig, stem):
    if '--preview' in sys.argv:
        import tempfile, uuid
        target = Path(tempfile.gettempdir()) / (stem[:-3] + '_preview_' + uuid.uuid4().hex + '_v2.png')
        targets = [(target, 'png')]
    else:
        targets = [(FIG / (stem + '.' + ext), ext) for ext in ('png', 'pdf')]
    for target, _ in targets:
        if target.exists():
            raise FileExistsError(target)
    for target, ext in targets:
        with target.open('xb') as stream:
            fig.savefig(stream, format=ext, dpi=300, bbox_inches='tight', pad_inches=0.08)
        print(target)
    plt.close(fig)

# Reuse the original eight-iteration solver without running camera evaluation.
solver_file = BASE / 'cam' / 'cam_eval.py'
solver_source = solver_file.read_text(encoding='utf-8')
solver_ast = ast.parse(solver_source)
solver_def = next(node for node in solver_ast.body if isinstance(node, ast.FunctionDef) and node.name == 'resolve')
geometry = json.loads((BASE / 'configs' / 'vehicle_geometry_20261124.json').read_text(encoding='utf-8'))
ANC = {key: np.array(value) for key, value in geometry['anchors_m'].items()}
TZ = geometry['tag_heights_m']['T1']
BODY = {'T1': np.array([-0.4, 0.0]), 'T2': np.array([0.4, 0.0])}
LINKS = ['T1_A1', 'T1_A2', 'T2_A1', 'T2_A2']
exec(compile(ast.Module(body=[solver_def], type_ignores=[]), str(solver_file), 'exec'))
timing = pd.read_csv(RC / 'frame_time_budget.csv').rename(columns={'mode': 'ranging_method'})
budget = timing.groupby('ranging_method')[['period_ms', 'same_T1_ms', 'cross_mean_ms']].median()
LABEL = {'SS': 'SS-TWR', 'DS': 'DS-TWR', 'PHASE': 'High-precision DS-TWR'}
COL = {'SS': '#1f5fbf', 'DS': '#e06c1b', 'PHASE': '#2e8b3a'}
rows = []
for method in LABEL:
    same = budget.loc[method, 'same_T1_ms'] / 1e3
    cross = budget.loc[method, 'cross_mean_ms'] / 1e3
    link_time = np.array([0.0, same, same + cross, 2 * same + cross])
    link_time -= link_time.mean()
    for speed in np.arange(0.0, 4.21, 0.1):
        for distance in (4.0, 7.0, 10.5):
            y0 = distance - 2.2
            ranges = np.zeros((1, 4))
            for k, link in enumerate(LINKS):
                tag, anchor = link.split('_')
                body = BODY[tag]
                a = ANC[anchor]
                y = y0 - speed * link_time[k]
                ranges[0, k] = math.sqrt((body[0] - a[0]) ** 2 + (y - a[1]) ** 2 + (a[2] - TZ) ** 2)
            solution, _ = resolve(ranges, np.array([[0.0, y0, 0.0]]))
            heading = solution[0, 2]
            rx_x = solution[0, 0] - 2.2 * math.sin(heading)
            rows.append(dict(ranging_method=method, speed_mps=round(speed, 2),
                speed_kmh=round(3.6 * speed, 2), coil_distance_m=distance,
                tag_lat_cm=100 * solution[0, 0], heading_deg=math.degrees(heading),
                rx_lat_cm=100 * rx_x,
                travel_per_update_cm=100 * speed * budget.loc[method, 'period_ms'] / 1e3))
calculated = pd.DataFrame(rows)
existing = pd.read_csv(RC / 'sequential_link_model.csv').rename(columns={'mode': 'ranging_method'})
same_rows = calculated[calculated.coil_distance_m < 10].reset_index(drop=True)
old_rows = existing[existing.coil_distance_m < 10].reset_index(drop=True)
assert same_rows.ranging_method.tolist() == old_rows.ranging_method.tolist()
numeric = same_rows.select_dtypes(include='number').columns
np.testing.assert_allclose(same_rows[numeric].round(3), old_rows[numeric], atol=1e-9, rtol=0)
out_csv = RC / 'sequential_link_model_10p5.csv'   # written by v5; v6 checks that the calculation is unchanged
prev = pd.read_csv(out_csv)
assert prev.ranging_method.tolist() == calculated.ranging_method.tolist()
np.testing.assert_allclose(calculated.round(3).select_dtypes(include='number').to_numpy(), prev.select_dtypes(include='number').to_numpy(), atol=1e-9, rtol=0)
print('calculation identical to', out_csv.name)
print('half frame time (same_T1 + cross_mean, ms):', (budget.same_T1_ms + budget.cross_mean_ms).round(3).to_dict(), 'period_ms:', budget.period_ms.round(3).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
for method in LABEL:
    group = calculated[(calculated.ranging_method == method) & (calculated.coil_distance_m == 7.0)]
    axes[0].plot(group.speed_mps, group.heading_deg.abs(), color=COL[method], lw=1.5, label=LABEL[method])
    axes[1].plot(group.speed_mps, group.rx_lat_cm.abs(), color=COL[method], lw=1.5)
    lo = calculated[(calculated.ranging_method == method) & (calculated.coil_distance_m == 4.0)]
    hi = calculated[(calculated.ranging_method == method) & (calculated.coil_distance_m == 10.5)]
    axes[1].fill_between(group.speed_mps, lo.rx_lat_cm.abs().to_numpy(),
                        hi.rx_lat_cm.abs().to_numpy(), color=COL[method], alpha=0.15, lw=0)
for ax in axes:
    ax.axvspan(2.5 / 3.6, 5.0 / 3.6, color='0.6', alpha=0.25, lw=0)
    ax.set_xlabel('Vehicle speed (m/s)')
    ax.set_xlim(0, 4.2)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
axes[0].text(0.88, 0.95, 'Speeds at alignment\ndistances from 4 to 10.5 m',
             transform=axes[0].get_xaxis_transform(), fontsize=8,
             ha='center', va='top', color='0.25')
axes[0].set_ylabel('Heading bias (°)')
axes[0].set_title('(a) Heading bias', pad=12)
axes[1].set_ylabel('Lateral bias at the RX coil (cm)')
axes[1].set_title('(b) Lateral bias at the RX coil', pad=12)
axes[1].axhline(10, color='crimson', ls=':', lw=1)
axes[1].text(3.3, 10.8, '10 cm', fontsize=8, color='crimson', va='bottom')
fig.legend(*axes[0].get_legend_handles_labels(), title='Ranging mode', loc='upper center',
           ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=9, title_fontsize=10)
fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.76, wspace=0.29)
save_figure(fig, 'sequential_link_model_v7')
print('The rows at 4 m and 7 m agree with sequential_link_model.csv to its 0.001 precision.')
print(calculated[(calculated.speed_mps == 2.8) & (calculated.coil_distance_m == 7.0)]
      [['ranging_method', 'heading_deg', 'rx_lat_cm']].round(3).to_string(index=False))

