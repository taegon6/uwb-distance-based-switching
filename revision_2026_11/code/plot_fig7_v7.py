"""Figure 7 (v7): legend "at alignment distances from ..." as in the text and Table 3. Figure 7: means and sample SD across runs; fixed manuscript terminology.
Read-only sources: Table 3 CSV, body_v2_full.json, and the original figure's data.
Creates PNG, PDF, numerical CSV and reconciliation report using exclusive writes.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(r'D:/sandbox/uwb/output/field_20261124')
EV = BASE / 'cam' / 'eval'
RC = BASE / 'analysis' / 'revision_checks'
FIG = BASE / 'fig'
BODY = Path(r'D:/sandbox/uwb/output/revision_docs_20261125/body_v2_full.json')
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

body = json.loads(BODY.read_text(encoding='utf-8'))
table3 = pd.read_csv(RC / 'table3_v3_mean_sd.csv')
# Table 3 of build v: update rates from the device-time evaluation (cam/eval_dejit), RMSE from cam/eval
data = pd.read_csv(EV / 'segment_errors.csv')
data_rate = pd.read_csv(BASE / 'cam' / 'eval_dejit' / 'segment_errors.csv')
stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv')
decomposition = pd.read_csv(RC / 'stop_decomposition_hp_stops.csv')
COL = {'FIXED_SS': '#1f5fbf', 'FIXED_DS': '#e06c1b', 'FIXED_PHASE': '#2e8b3a', 'STAGED': 'k'}
LABEL = dict(zip(COL, ['Fixed SS-TWR', 'Fixed DS-TWR', 'Fixed high-precision DS-TWR', 'Proposed method']))
rows = []
def pm(text):
    return tuple(float(v.strip()) for v in text.split('±'))
def stats(values):
    v = np.asarray(values, dtype=float)
    assert np.isfinite(v).all() and len(v) > 1
    return float(v.mean()), float(v.std(ddof=1))
def record(panel, condition, metric, unit, values, source, decimals=1):
    mean, sd = stats(values)
    row = dict(panel=panel, condition=condition, metric=metric, unit=unit,
               runs=len(values), mean=mean, sd=sd,
               reported=f'{mean:.{decimals}f} ± {sd:.{decimals}f}', source=source)
    rows.append(row)
    return mean, sd
fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.9))
a, b, c, d = axes.flat
for method, color in COL.items():
    tab = table3[(table3.path == 'Straight') & (table3.method == LABEL[method])].iloc[0]
    for key, label, marker, filled, prefix in [
        ('far', 'at alignment distances from 4 to 10.5 m', 'o', True, 'far'),
        ('mid', 'at alignment distances from 0.5 to 4 m', 's', False, 'near')]:
        group = data[(data.path == 'S') & (data.policy == method) & (data.seg == key)]
        group_rate = data_rate[(data_rate.path == 'S') & (data_rate.policy == method) & (data_rate.seg == key)]
        condition = LABEL[method] + ', ' + label
        for metric, unit, column, csv_column in [
            ('Update rate', 'Hz', 'rate_hz', prefix + '_rate'),
            ('Lateral RMSE', 'cm', 'log_lat_rmse', prefix + '_lat')]:
            raw_mean, raw_sd = record('a', condition, metric, unit, (group_rate if column == 'rate_hz' else group)[column],
                                      'table3_v2_mean_sd.csv; checked against segment_errors.csv')
            mean, sd = pm(tab[csv_column])
            assert f'{raw_mean:.1f} ± {raw_sd:.1f}' == tab[csv_column]
            rows[-1]['mean'], rows[-1]['sd'] = mean, sd
            rows[-1]['reported'] = tab[csv_column]
        x, xsd = pm(tab[prefix + '_rate'])
        y, ysd = pm(tab[prefix + '_lat'])
        a.errorbar(x, y, xerr=xsd, yerr=ysd, fmt=marker, color=color,
                   mfc=color if filled else 'white', mec=color, ms=7,
                   capsize=3, elinewidth=1.1, zorder=3)
a.set(xlabel='Update rate (Hz)', ylabel='Lateral RMSE (cm)', xlim=(16, 126), ylim=(0, 19))
a.set_title('(a) Straight path during the approach (5 trials per method)', pad=12)
distance_handles = [
    Line2D([], [], marker='o', mfc='0.35', color='0.35', ls='',
           label='at alignment distances from 4 to 10.5 m'),
    Line2D([], [], marker='s', mfc='white', color='0.35', ls='',
           label='at alignment distances from 0.5 to 4 m')]
a.legend(handles=distance_handles, loc='center right', fontsize=8, frameon=False)

calibrations = [('none', 'No calibration'), ('lab', 'Laboratory calibration'), ('sfit', 'Field calibration')]
for i, (key, name) in enumerate(calibrations, 1):
    mean, sd = record('b', name, 'Position error at the final stop', 'cm', stops[key + '_err_cm'],
                      'analysis/stops_v2_seltos_L439.csv')
    b.errorbar(i, mean, yerr=sd, fmt='o', color='#375b82', capsize=5, ms=7)
    b.annotate(f'{mean:.1f} ± {sd:.1f}', (i, mean + sd), xytext=(0, 8),
               textcoords='offset points', ha='center', fontsize=9)
b.set_xticks([1, 2, 3], ['No\ncalibration', 'Laboratory\ncalibration', 'Field\ncalibration'])
b.set(xlim=(0.5, 3.5), ylim=(0, 23), ylabel='Position error at the final stop (cm)')
b.set_title(f'(b) Calibration at the final stop ({len(stops)} trials)', pad=12)

methods = [('DS', 'DS-TWR', '#e06c1b'), ('HP fixed', 'High-precision DS-TWR', '#2e8b3a')]
for i, (key, name, color) in enumerate(methods, 1):
    h = decomposition[(decomposition.offset == 'field') & (decomposition.method == key)]
    accuracy = h[h.aggregation == '3 s']
    precision = h[h.aggregation == 'jitter']
    assert len(accuracy) == len(precision) == 16
    assert set(accuracy.trial) == set(precision.trial)
    mean, sd = record('c', name, 'Coil alignment error', 'cm', accuracy.rx_cm,
                      'stop_decomposition_hp_stops.csv; field calibration; median over 3 s')
    expected = body['table4'][6 + i][3]
    assert f'{mean:.1f} ± {sd:.1f}' == expected
    c.errorbar(i, mean, yerr=sd, fmt='o', color=color, capsize=5, ms=7)
    c.annotate(f'{mean:.1f} ± {sd:.1f}', (i, mean + sd), xytext=(0, 8),
               textcoords='offset points', ha='center', fontsize=9)
    mean, sd = record('d', name, 'Precision', 'cm', precision.rx_cm,
                      'stop_decomposition_hp_stops.csv; SD of RX coil position estimates over last 3 s', 2)
    expected = body['table4'][6 + i][4]
    assert f'{mean:.2f} ± {sd:.2f}' == expected
    d.errorbar(i, mean, yerr=sd, fmt='o', color=color, capsize=5, ms=7)
    d.annotate(f'{mean:.2f} ± {sd:.2f}', (i, mean + sd), xytext=(0, 8),
               textcoords='offset points', ha='center', fontsize=9)
for ax in (c, d):
    ax.set_xticks([1, 2], ['DS-TWR', 'High-precision\nDS-TWR'])
    ax.set_xlim(0.5, 2.5)
c.set(ylim=(0, 19), ylabel='Coil alignment error (cm)')
c.set_title('(c) Accuracy at the final stop (same 16 trials)', pad=12)
d.set(ylim=(0, 3.8), ylabel='Precision (cm)')
d.set_title('(d) Precision at the final stop (same 16 trials)', pad=12)
for ax in axes.flat:
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
fig.legend(handles=[Line2D([], [], marker='o', color=COL[k], ls='', label=LABEL[k]) for k in COL],
           loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False, fontsize=9)
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.09, top=0.90, hspace=0.45, wspace=0.29)

numbers = pd.DataFrame(rows)
print(numbers[['panel', 'condition', 'metric', 'runs', 'reported']].to_string(index=False))
report = """# Figure 7 plotted numbers and figure revision notes

All error bars are sample standard deviations across runs (ddof=1), not standard errors or confidence intervals. Panel (a) uses the published one-decimal means and SDs directly, for both axes; every value was checked against the original run-level data and body_v2_full.json Table 3. Panels (b)–(d) use unrounded run-level means and sample SDs; displayed values are rounded as in the manuscript.

"""
report += '| Panel | Condition | Metric | Runs | Mean ± SD |\n|---|---|---|---:|---|\n'
for row in rows:
    report += f"| ({row['panel']}) | {row['condition']} | {row['metric']} ({row['unit']}) | {row['runs']} | {row['reported']} |\n"
report += """
Panel (b) was recomputed from analysis/stops_v2_seltos_L439.csv, the same 39-run file used by plot_fig7.py. The no-calibration result was absent from the supplied new tables and text. Laboratory calibration was 14.665846 ± 3.773555 cm (displayed as 14.7 ± 3.8 cm), whereas Section 4.4 states 15.1 ± 5.0 cm. Field calibration was 3.528590 ± 2.590525 cm, agreeing with 3.5 ± 2.6 cm in Section 4.4. The source values were preserved; the laboratory-calibration discrepancy remains unresolved.

Panels (c) and (d) compare DS-TWR with high-precision DS-TWR using the fixed integer cycle number, matching Table 4. Coil alignment error used the median link ranges over the last 3 s within each run. Precision was the SD of the estimated RX coil position within the last 3 s of each run; the plotted mean and SD summarize those per-run precision values. Both panels use the same 16 runs with field calibration. The separate recorded high-precision processing comparison in the old figure was omitted to match the two conditions in Table 4. All four panel (c)/(d) values agree with Table 4 and the precision values in Section 4.4.

The median over 3 s is the within-run calculation used in Table 4. It is distinct from the across-run mean ± SD shown in the figure.

Additional figure revision notes:

- Figure 6 retained the original eight panels and runs: (a) T11, (b) T09, (c) T12, (d) T10, (e) T29, (f) T27, (g) T28, and (h) T50. It retained the camera smoothing, coordinate frame, distance circles, transition markers, and interruption connections longer than 0.3 s. The red crosses were removed. Panel (h) is the sharply curved path, which was absent from the brief Figure 6 description in Section 4.2.
- Section 4.3 describes the sharply curved path as driven once with each method, whereas Section 4.1, the experimental design table, and the stop data give four runs: fixed SS-TWR once (T36), fixed DS-TWR once (T37), and the proposed method twice (T38, T50). The figure and caption preserve the existing T50 panel.
- The source specified for Figure 8, motion_skew_plot.py, actually generated motion_skew.png with measured and calculated biases. The existing sequential_link_model figure was generated by sequential_model.py. The new motion_skew_plot_v2.py preserves the latter calculation and the original solver, comparing all 387 rows against sequential_link_model.csv to its stored 0.001 precision. At 2.8 m/s and 7 m, the calculated magnitudes were 1.063°, 2.425°, and 4.619° and 6.773, 15.123, and 28.498 cm, agreeing with the rounded discussion values. Figure 8 uses m/s. Its colored bands show the calculations at 4 and 10 m; they are not standard deviations.
- Figures 6 and 8 are individual-run trajectories and deterministic calculations, respectively. Mean ± SD applies to the across-run summary in Figure 7.
- All three PNGs were visually inspected for overlapping labels. All figures use Arial with the same base, title and axis font sizes (10 pt), 9 pt ticks and legends, and 300 dpi PNGs. PDFs retain vector text and lines.
- Each new script refuses to overwrite existing deliverables. No source data or existing figure was modified.

"""
if '--preview' not in sys.argv:
    targets = [FIG / 'fig7_numbers_v7.csv', FIG / 'fig7_report_v7.md',
               FIG / 'fig7_v7.png', FIG / 'fig7_v7.pdf']
    for target in targets:
        if target.exists():
            raise FileExistsError(target)
save_figure(fig, 'fig7_v7')
if '--preview' not in sys.argv:
    with (FIG / 'fig7_numbers_v7.csv').open('x', encoding='utf-8', newline='') as stream:
        numbers.to_csv(stream, index=False)
    with (FIG / 'fig7_report_v7.md').open('x', encoding='utf-8') as stream:
        stream.write(report)

