"""Independent check of the camera-UWB time alignment with the tablet clock in the camera view (R2 Comment 11).
READ: time shown on the tablet from the tick 'sheet_tick_pts' on, read by eye from eval/tablet_contact_*.png."""
import numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'
READ = {'T01': '21:21:30', 'T03': '21:36:58', 'T04': '21:38:42', 'T02': '21:40:38', 'T05': '21:43:15', 'T06': '21:45:00', 'T07': '21:46:40', 'T08': '21:48:35', 'T09': '21:50:03',
        'T10': '21:52:04', 'T11': '21:53:40', 'T12': '21:55:21', 'T13': '21:56:44', 'T14': '21:58:36', 'T15': '22:00:13', 'T16': '22:02:05', 'T17': '22:03:56', 'T18': '22:05:37',
        'T19': '22:07:11', 'T20': '22:08:43', 'T21': '22:32:10', 'T25': '22:41:30', 'T28': '22:47:38', 'T29': '22:50:16', 'T32': '22:57:31', 'T33': '22:59:45', 'T34': '23:01:59',
        'T35': '23:04:20', 'T36': '23:07:04', 'T37': '23:25:24', 'T38': '23:27:30'}
T = pd.read_csv(EV / 'tablet_clock_ticks_v2.csv').set_index('trial'); E = pd.read_csv(EV / 'segment_errors.csv').drop_duplicates('trial').set_index('trial')
def sec(s): h, m, x = map(int, s.split(':')); return 3600 * h + 60 * m + x
rows = []
for t, shown in READ.items():
    if not (EV / f'camtrack_{t}.csv').exists(): continue
    r = T.loc[t]; name = r.video[9:15]; fsec = sec(f'{name[:2]}:{name[2:4]}:{name[4:]}')
    start_tablet = r.video_start + (sec(shown) - fsec) - r.sheet_tick_pts            # tablet time of video pts 0 (epoch of the file-name second + difference)
    C = pd.read_csv(EV / f'camtrack_{t}.csv', usecols=['t', 'pts']); start_pc = float(np.median(C.t - C.pts))     # PC time of video pts 0 from the range-rate correlation
    rows.append(dict(trial=t, path=E.loc[t, 'path'] if t in E.index else '', policy=E.loc[t, 'policy'] if t in E.index else '', shown=shown, tick_pts=r.sheet_tick_pts, tick_spread=r.tick_spread_s, contrast=r.contrast,
                     start_tablet_minus_filename=start_tablet - r.video_start, start_pc_minus_filename=start_pc - r.video_start, pc_minus_tablet=start_pc - start_tablet,
                     sync_corr=E.loc[t, 'sync_corr'] if t in E.index else np.nan))
D = pd.DataFrame(rows); D.round(4).to_csv(BASE / 'analysis' / 'revision_checks' / 'tablet_clock_check.csv', index=False)
pd.set_option('display.width', 220); print(D.round(3).to_string(index=False))
d = D.pc_minus_tablet
print('\nn', len(d), 'median %.3f s, mean %.3f, SD %.3f, MAD-SD %.3f, min %.3f, max %.3f' % (d.median(), d.mean(), d.std(ddof=1), 1.4826 * (d - d.median()).abs().median(), d.min(), d.max()))
print('within 50 ms of the median:', int(((d - d.median()).abs() <= 0.05).sum()), ' within 100 ms:', int(((d - d.median()).abs() <= 0.1).sum()), 'of', len(d))
print(D.groupby('path').pc_minus_tablet.agg(['median', 'std', 'count']).round(3).to_string())
hrs = (D.trial.map(T.video_start) - T.video_start.min()) / 3600; k = np.polyfit(hrs, d, 1); print('drift over the session: %.3f s per hour' % k[0]); res = d - np.polyval(k, hrs); print('SD after removing the drift %.3f s, max abs %.3f' % (res.std(ddof=2), res.abs().max()))
print('file-name second versus tablet: median %.2f s (range %.2f to %.2f)' % (D.start_tablet_minus_filename.median(), D.start_tablet_minus_filename.min(), D.start_tablet_minus_filename.max()))


