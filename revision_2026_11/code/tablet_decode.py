"""Decode the clock region of the tablet for every run video once and store it (eval/tablet_roi/<trial>.npz: pts, frames uint8)."""
import sys, gc
from pathlib import Path
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/tablet_clock.py', encoding='utf-8').read().split("J = pd.read_csv(BASE / 'cam' / 'jobs.csv')")[0]
exec(src)
OUT = EV / 'tablet_roi'; OUT.mkdir(exist_ok=True)
B = pd.read_csv(EV / 'tablet_clock_ticks.csv').set_index('trial'); J = pd.read_csv(BASE / 'cam' / 'jobs.csv')
for j in J.itertuples():
    if (OUT / f'{j.trial}.npz').exists(): continue
    if j.trial in B.index and B.loc[j.trial, 'bx'] == B.loc[j.trial, 'bx']:
        x, y, w, h = [int(B.loc[j.trial, k]) for k in ('bx', 'by', 'bw', 'bh')]
    else:
        box = None
        for pts, img in frames(BASE / 'video' / j.video, 3.0, 3.4, (CW, CH, X0, Y0)):
            box = find_screen(img)
            if box: break
        if box is None: print(j.trial, 'no screen', flush=True); continue
        x, y, w, h = box[:4]
    roi = (int(0.62 * w) // 2 * 2, int(0.44 * h) // 2 * 2, X0 + x + int(0.18 * w), Y0 + y + int(0.28 * h))
    ts, R = [], []
    for pts, img in frames(BASE / 'video' / j.video, TA, TB, roi):
        ts.append(pts); R.append(img.copy())
    if not R:
        print(j.trial, 'no frames decoded', flush=True); continue
    np.savez_compressed(OUT / f'{j.trial}.npz', pts=np.array(ts), frames=np.stack(R), box=np.array([x, y, w, h]), roi=np.array(roi))
    print(j.trial, j.video, len(ts), 'frames', roi, flush=True); del R; gc.collect()
print('done', flush=True)


