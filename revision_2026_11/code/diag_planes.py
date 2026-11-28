"""Diagnostics for the curve-session registration: per-run plane, board-normal heading, mount offset."""
import sys, math, json
sys.argv = ['x']
from pathlib import Path
import numpy as np, pandas as pd, cv2
exec(open(r'D:/sandbox/uwb/output/field_20261124/cam/cam_eval.py', encoding='utf-8').read().split('# ---------------- main ----------------')[0])
jobs = pd.read_csv(CAM / 'jobs.csv'); stops = pd.read_csv(BASE / 'analysis' / 'stops_v2_seltos_L439.csv')
rows = []
allP = {}
for j in jobs.itertuples():
    f = CAM / 'tracks' / f'{j.trial}.csv'
    if not f.exists(): continue
    c = pd.read_csv(f); c = c[c.cx.notna() & (c.reproj_px < 2.5)].copy(); c['d'] = np.sqrt(c.cx**2 + c.cy**2 + c.cz**2)
    allP[j.trial] = c
    P = c[(c.source == 'DIRECT')][['cx', 'cy', 'cz']].to_numpy()
    m = P.mean(0); _, sv, vt = np.linalg.svd(P - m); n = vt[2]; n = -n if n[1] > 0 else n
    rows.append(dict(trial=j.trial, n_direct=len(P), nx=n[0], ny=n[1], nz=n[2], dplane=float(n @ m), rms_mm=1000 * float(np.sqrt(np.mean((P @ n - n @ m) ** 2))),
                     extent_m=float(np.ptp(P @ np.cross(n, [1, 0, 0]))) ))
D = pd.DataFrame(rows); pd.set_option('display.width', 200)
D['tilt_deg'] = np.degrees(np.arccos(np.clip(-D.ny, -1, 1)))
D['roll_deg'] = np.degrees(np.arctan2(D.nx, -D.ny)); D['pitch_deg'] = np.degrees(np.arctan2(D.nz, -D.ny))
print(D[['trial', 'n_direct', 'rms_mm', 'extent_m', 'dplane', 'roll_deg', 'pitch_deg']].round(3).to_string(index=False))

