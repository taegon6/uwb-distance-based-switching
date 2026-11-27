"""Paper-style route figures (manuscript frame: x_ms = -x_code, RX coil position)."""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; EV = BASE / 'cam' / 'eval'; FIG = BASE / 'fig'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10, 'mathtext.fontset': 'stix'})
COL = {'SS': '#1f5fbf', 'DS': '#e06c1b', 'PHASE': '#2e8b3a'}
NAME = {'SS': 'SS-TWR', 'DS': 'DS-TWR', 'PHASE': 'High-precision DS-TWR'}
POL = {'FIXED_SS': 'Fixed SS-TWR', 'FIXED_DS': 'Fixed DS-TWR', 'FIXED_PHASE': 'Fixed high-precision DS-TWR', 'STAGED': 'Distance-based switching'}
jobs = pd.read_csv(BASE / 'cam' / 'jobs.csv').set_index('trial')
stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv').set_index('trial')
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')

def rd(p, usecols=None):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False, usecols=usecols)

def uwb(trial):
    run = jobs.loc[trial, 'run']; T = DAY / run / 'tables'
    po = rd(T / 'poses.csv', ['valid', 'range_source', 'host_pose_monotonic_ns', 'host_pose_wall_ns', 'rx_coil_x_m', 'rx_coil_y_m'])
    po = po[po.valid.astype(str).str.lower().isin(['true', '1'])].copy()
    src = po.range_source.astype(str)
    po['mode'] = np.where(src == 'SS_RANGE', 'SS', np.where(src == 'PHASE_CORRECTED_RANGE', 'PHASE', 'DS'))
    po['t'] = po.host_pose_monotonic_ns.astype('int64') / 1e9
    po['tw'] = po.host_pose_wall_ns.astype('int64') / 1e9
    po['x'] = -po.rx_coil_x_m.astype(float); po['y'] = po.rx_coil_y_m.astype(float)
    rs = rd(T / 'runtime_status.csv'); ns = rs.host_monotonic_ns.astype('int64') / 1e9
    tr = ns[rs.event == 'AUTO_MODE_COMMITTED'].to_numpy(); wd = ns[rs.event == 'STREAM_WATCHDOG_RECOVERED'].to_numpy()
    return po.sort_values('t').reset_index(drop=True), tr, wd

def draw(ax, trial, title, xlim, ylim, legend=False):
    po, tr, wd = uwb(trial)
    t = po.t.to_numpy(); x = po.x.to_numpy(); y = po.y.to_numpy(); m = po['mode'].to_numpy()
    gap = np.r_[False, np.diff(t) > 0.3]
    # camera coverage: a camera sample within 0.1 s of the pose time (camera time is aligned to the PC wall time)
    camf = EV / f'camtrack_{trial}.csv'; cov = np.zeros(len(t), bool)
    if camf.exists():
        ct = np.sort(pd.read_csv(camf, usecols=['t']).t.to_numpy()); tw = po.tw.to_numpy(); k = np.clip(np.searchsorted(ct, tw), 1, len(ct) - 1)
        cov = np.minimum(np.abs(tw - ct[k - 1]), np.abs(ct[k] - tw)) <= 0.1
    seg_start = np.r_[0, np.flatnonzero(gap | np.r_[False, m[1:] != m[:-1]] | np.r_[False, cov[1:] != cov[:-1]])]
    seg_end = np.r_[seg_start[1:], len(t)]
    for a, b in zip(seg_start, seg_end):
        b2 = min(b + 1, len(t)) if (b < len(t) and not gap[b] and m[b] == m[a]) else b      # join the pieces where only the coverage changes
        ax.plot(x[a:b2], y[a:b2], '-', color=COL[m[a]], lw=0.8, alpha=1.0 if cov[a] else 0.3, zorder=3)
    for k in np.flatnonzero(gap):                       # output gap > 0.3 s
        ax.plot([x[k - 1], x[k]], [y[k - 1], y[k]], '-', color='0.35', lw=0.6, zorder=2)
    for w in wd:                                        # watchdog restart: last output before it
        k = np.searchsorted(t, w) - 1
        if 0 <= k < len(t): ax.plot(x[k], y[k], 'x', color='crimson', ms=5, mew=1.2, zorder=7)
    for tt in tr:                                       # switch command (last output before) and first new-mode output
        k = np.searchsorted(t, tt)
        if 0 < k < len(t):
            ax.plot(x[k - 1], y[k - 1], 'D', mfc='k', mec='k', ms=3.5, zorder=7)
            ax.plot(x[k], y[k], 'o', mfc='white', mec='k', ms=5, mew=0.9, zorder=6)
    cam = EV / f'camtrack_{trial}.csv'
    if cam.exists():
        c = pd.read_csv(cam).sort_values('t')
        cx = -c.RX_X.rolling(9, center=True, min_periods=1).median(); cy = c.RX_Y.rolling(9, center=True, min_periods=1).median()
        ax.plot(cx, cy, '--', color='k', lw=1.2, zorder=4)
    if trial in stops.index:
        s = stops.loc[trial]; ax.plot(-s.rx_ref_x_cm / 100, s.rx_ref_y_cm / 100, '*', color='k', ms=9, zorder=8)
    for r in (4.0, 0.5):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=':', lw=0.8, ec='0.45', zorder=1))
    a = np.radians(118)
    ax.text(3.55 * np.cos(a), 3.55 * np.sin(a), r'$d_c$ = 4.0 m', fontsize=7, color='0.35', ha='center', va='center', rotation=28)
    ax.plot([1, -1], [-3, -3], '^', mfc='white', mec='k', ms=6, zorder=6)
    for xx, lab_ in ((1, 'A1'), (-1, 'A2')):
        ax.annotate(lab_, (xx, -3), xytext=(0, 6), textcoords='offset points', ha='center', va='bottom', fontsize=7, bbox=dict(fc='white', ec='none', pad=0.3), zorder=9)
    ax.plot(0, 0, '+', color='k', ms=8, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_title(title, fontsize=9.5)

def legend_handles():
    return [Line2D([], [], color=COL['SS'], lw=1.2, label='SS-TWR'), Line2D([], [], color=COL['DS'], lw=1.2, label='DS-TWR'),
            Line2D([], [], color=COL['PHASE'], lw=1.2, label='High-precision DS-TWR'),
            Line2D([], [], color=COL['SS'], lw=1.2, alpha=0.3, label='UWB output without camera reference (not evaluated)'),
            Line2D([], [], color='k', ls='--', lw=1.2, label='Camera reference (RX coil)'),
            Line2D([], [], marker='*', color='k', ls='', ms=9, label='Manual reference (tape + plumb)'),
            Line2D([], [], marker='+', color='k', ls='', ms=8, label='TX coil center (0, 0)'),
            Line2D([], [], marker='D', color='k', ls='', ms=4, label='Switch command (last output before)'),
            Line2D([], [], marker='o', mfc='white', mec='k', ls='', label='First output in the new mode'),
            Line2D([], [], color='0.35', lw=0.6, label='Output gap > 0.3 s'),
            Line2D([], [], marker='x', color='crimson', ls='', label='Watchdog stream restart'),
            Line2D([], [], color='0.45', ls=':', lw=0.8, label=r'Switching boundary $d_c$ = 4.0 / 0.5 m')]

def figure(panels, name, ncol, xlim, ylim, size):
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=size, squeeze=False)
    for ax, (trial, title) in zip(axs.flat, panels):
        draw(ax, trial, title, xlim, ylim)
    for ax in list(axs.flat)[len(panels):]: ax.axis('off')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / size[1]))
    fig.legend(handles=legend_handles(), loc='upper center', ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    for ext in ('png', 'pdf'): fig.savefig(FIG / f'{name}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig); print('saved', name)

def lab(i, trial):
    p = sched.loc[trial]; return f'({chr(97 + i)}) {POL[p.policy]} ({trial}, block {p.block})'

blk = int(sys.argv[1]) if len(sys.argv) > 1 else 3
S = sched[(sched.path == 'S') & (sched.block == blk)].index.tolist()
order = {'FIXED_SS': 0, 'FIXED_DS': 1, 'FIXED_PHASE': 2, 'STAGED': 3}
S = sorted(S, key=lambda t: order[sched.loc[t, 'policy']])
figure([(t, lab(i, t)) for i, t in enumerate(S)], f'route_straight_block{blk}', 4, (-3, 3), (-4.2, 10.8), (11.5, 6.8))
C = sorted(sched[(sched.path == 'C1') & (sched.block == blk)].index.tolist(), key=lambda t: order[sched.loc[t, 'policy']])
panels = [(t, lab(i, t).replace(')', ', C1)', 1) if False else lab(i, t)) for i, t in enumerate(C)] + [('T50', f'({chr(97 + len(C))}) Distance-based switching (T50, C2)')]
figure(panels, f'route_curve_block{blk}', 4, (-10, 3), (-4.2, 11), (14, 6.0))
st = [t for t in sched[(sched.policy == 'STAGED')].index if (EV / f'camtrack_{t}.csv').exists()]
figure([(t, f'({chr(97 + i)}) {sched.loc[t, "path"]} {t}') for i, t in enumerate(st)], 'route_all_switching', 6, (-10, 3), (-4.2, 11), (16, 8.6))

# manuscript Figure 6: straight block (a)-(d) above, curved approaches (e)-(h) below, one legend
fig = plt.figure(figsize=(14, 11.4))
top, bottom = fig.subfigures(2, 1, height_ratios=[7.4, 4.0])
for ax, (i, t) in zip(top.subplots(1, 4), enumerate(S)):
    draw(ax, t, lab(i, t), (-3, 3), (-4.2, 10.8))
curve_panels = [(t, lab(4 + i, t)) for i, t in enumerate(C)] + [('T50', f'({chr(97 + 4 + len(C))}) Distance-based switching (T50, C2)')]
for ax, (t, title) in zip(bottom.subplots(1, 4), curve_panels):
    draw(ax, t, title, (-10, 3), (-4.2, 11))
fig.legend(handles=legend_handles(), loc='upper center', ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.035))
for ext in ('png', 'pdf'): fig.savefig(FIG / f'route_fig6.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.08)
plt.close(fig); print('saved route_fig6')

