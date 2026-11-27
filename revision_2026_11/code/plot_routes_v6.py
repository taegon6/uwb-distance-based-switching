"""Figure 6: RX coil position estimates; same runs and data as plot_routes.py.
Creates new files only. Use --preview for a temporary PNG.
"""
import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerTuple

BASE = Path(r'D:/sandbox/uwb/output/field_20261124')
DAY = BASE / 'results_20261124'
EV = BASE / 'cam' / 'eval'
FIG = BASE / 'fig'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

COL = {'SS': '#1f5fbf', 'DS': '#e06c1b', 'PHASE': '#2e8b3a'}
METHOD = {'FIXED_SS': 'Fixed SS-TWR', 'FIXED_DS': 'Fixed DS-TWR',
          'FIXED_PHASE': 'Fixed high-precision DS-TWR', 'STAGED': 'Proposed method'}
PATH = {'S': 'straight path', 'C1': 'curved path', 'C2': 'sharply curved path'}
jobs = pd.read_csv(BASE / 'cam' / 'jobs.csv').set_index('trial')
references = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
schedule = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')

def rd(path, usecols=None):
    text = path.read_text(encoding='utf-8', errors='replace')
    text = '\n'.join(line for line in text.splitlines() if '\x00' not in line)
    return pd.read_csv(io.StringIO(text), on_bad_lines='skip', low_memory=False, usecols=usecols)

def uwb(trial):
    tables = DAY / jobs.loc[trial, 'run'] / 'tables'
    p = rd(tables / 'poses.csv', ['valid', 'range_source', 'host_pose_monotonic_ns',
            'host_pose_wall_ns', 'rx_coil_x_m', 'rx_coil_y_m'])
    p = p[p.valid.astype(str).str.lower().isin(['true', '1'])].copy()
    source = p.range_source.astype(str)
    p['ranging_method'] = np.where(source == 'SS_RANGE', 'SS',
            np.where(source == 'PHASE_CORRECTED_RANGE', 'PHASE', 'DS'))
    p['t'] = p.host_pose_monotonic_ns.astype('int64') / 1e9
    p['tw'] = p.host_pose_wall_ns.astype('int64') / 1e9
    p['x'] = -p.rx_coil_x_m.astype(float)
    p['y'] = p.rx_coil_y_m.astype(float)
    status = rd(tables / 'runtime_status.csv')
    transitions = (status.host_monotonic_ns.astype('int64') / 1e9)[status.event == 'AUTO_MODE_COMMITTED'].to_numpy()
    return p.sort_values('t').reset_index(drop=True), transitions

def draw(ax, trial, title, xlim, ylim):
    p, transitions = uwb(trial)
    t, x, y, method = (p[c].to_numpy() for c in ('t', 'x', 'y', 'ranging_method'))
    interruption = np.r_[False, np.diff(t) > 0.3]
    coverage = np.zeros(len(t), bool)
    camera_path = EV / f'camtrack_{trial}.csv'
    if camera_path.exists():
        ct = np.sort(pd.read_csv(camera_path, usecols=['t']).t.to_numpy())
        tw = p.tw.to_numpy()
        k = np.clip(np.searchsorted(ct, tw), 1, len(ct) - 1)
        coverage = np.minimum(np.abs(tw - ct[k - 1]), np.abs(ct[k] - tw)) <= 0.1
    starts = np.r_[0, np.flatnonzero(interruption |
        np.r_[False, method[1:] != method[:-1]] | np.r_[False, coverage[1:] != coverage[:-1]])]
    ends = np.r_[starts[1:], len(t)]
    for start, end in zip(starts, ends):
        end2 = min(end + 1, len(t)) if (end < len(t) and not interruption[end] and method[end] == method[start]) else end
        ax.plot(x[start:end2], y[start:end2], '-', color=COL[method[start]],
                lw=0.8, alpha=1.0 if coverage[start] else 0.3, zorder=3)
    for k in np.flatnonzero(interruption):
        ax.plot([x[k - 1], x[k]], [y[k - 1], y[k]], '-', color='0.4', lw=0.8, zorder=2)
    for tt in transitions:
        k = np.searchsorted(t, tt)
        if 0 < k < len(t):
            ax.plot(x[k - 1], y[k - 1], 'D', mfc='k', mec='k', ms=3.5, zorder=7)
            ax.plot(x[k], y[k], 'o', mfc='white', mec='k', ms=5, mew=0.9, zorder=6)
    if camera_path.exists():
        c = pd.read_csv(camera_path).sort_values('t')
        cx = -c.RX_X.rolling(9, center=True, min_periods=1).median()
        cy = c.RX_Y.rolling(9, center=True, min_periods=1).median()
        ax.plot(cx, cy, '--', color='k', lw=1.2, zorder=4)
    if trial in references.index:
        r = references.loc[trial]
        ax.plot(-r.rx_ref_x_cm / 100, r.rx_ref_y_cm / 100, '*', color='k', ms=9, zorder=8)
    for radius in (4.0, 0.5):
        ax.add_patch(Circle((0, 0), radius, fill=False, ls=':', lw=0.8, ec='0.45', zorder=1))
    angle = np.radians(118)
    ax.text(3.55 * np.cos(angle), 3.55 * np.sin(angle), r'4.0 m',
            fontsize=8, color='0.35', ha='center', va='center', rotation=28)
    ax.plot([1, -1], [-3, -3], '^', mfc='white', mec='k', ms=6, zorder=6)
    for xx, label in ((1, 'A$_1$'), (-1, 'A$_2$')):
        ax.annotate(label, (xx, -3), xytext=(0, 6), textcoords='offset points',
            ha='center', va='bottom', fontsize=8,
            bbox=dict(fc='white', ec='none', pad=0.3), zorder=9)
    ax.plot(0, 0, '+', color='k', ms=8, zorder=6)
    ax.set(xlim=xlim, ylim=ylim, xlabel='X (m)', ylabel='Y (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(title, pad=10)


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

order = {key: i for i, key in enumerate(METHOD)}
block = 3
straight = sorted(schedule[(schedule.path == 'S') & (schedule.block == block)].index,
                  key=lambda t: order[schedule.loc[t, 'policy']])
curved = sorted(schedule[(schedule.path == 'C1') & (schedule.block == block)].index,
                key=lambda t: order[schedule.loc[t, 'policy']]) + ['T50']
assert len(straight) == len(curved) == 4
fig = plt.figure(figsize=(10.4, 9.6))
grid = fig.add_gridspec(2, 4, height_ratios=[1.65, 1], left=0.06, right=0.985,
                       bottom=0.06, top=0.84, hspace=0.35, wspace=0.34)
for i, trial in enumerate(straight + curved):
    r = schedule.loc[trial]
    label = METHOD[r.policy]
    if label == 'Fixed high-precision DS-TWR':
        label = 'Fixed high-precision\nDS-TWR'
    title = f'({chr(97 + i)}) {label}\n{PATH[r.path]}'
    draw(fig.add_subplot(grid[i // 4, i % 4]), trial, title,
         (-3, 3) if i < 4 else (-10, 3), (-4.2, 10.8) if i < 4 else (-4.2, 11))
transition = (Line2D([], [], marker='D', color='k', ls='', ms=4),
              Line2D([], [], marker='o', mfc='white', mec='k', ls='', ms=5))
handles = [
    Line2D([], [], color='k', lw=1.2),
    Line2D([], [], color='k', ls='--', lw=1.2),
    transition,
    Line2D([], [], color='0.4', lw=0.8),
    Line2D([], [], color=COL['SS'], lw=1.5),
    Line2D([], [], color=COL['DS'], lw=1.5),
    Line2D([], [], color=COL['PHASE'], lw=1.5),
    Line2D([], [], marker='*', color='k', ls='', ms=9),
    Line2D([], [], marker='+', color='k', ls='', ms=8),
    Line2D([], [], marker='^', mfc='white', mec='k', ls='', ms=6),
    Line2D([], [], color='0.45', ls=':', lw=0.8),
]
labels = ['UWB position estimate', 'Camera reference', 'Switching', 'Interruption',
          'SS-TWR', 'DS-TWR', 'High-precision DS-TWR', 'Manual reference',
          'TX coil center', 'Anchor', r'Alignment distance of 4.0 m and 0.50 m']
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
           ncol=3, frameon=False, fontsize=9, columnspacing=2.5,
           handler_map={tuple: HandlerTuple(ndivide=None)})
save_figure(fig, 'route_fig6_v6')
print('Figure 6 runs:', ', '.join(straight + curved))
