"""Tablet clock in the camera view (independent check of the camera-UWB time alignment, R2 Comment 11).
For every run video: locate the tablet screen at 3 s (exposure settled), read the region of the clock digits from 2 to 8 s,
and fit the phase of the one-second changes of the display. A crop half a second after a tick is saved for reading the time.
Output: eval/tablet_clock_ticks.csv and eval/tablet_sheets/<trial>.png"""
import subprocess, threading, re, time, sys, gc
from pathlib import Path
import numpy as np, pandas as pd, cv2
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'; SH = EV / 'tablet_sheets'; SH.mkdir(exist_ok=True)
FF = r'D:/sandbox/uwb/output/laptop_handoff_20261124/ffmpeg/ffmpeg.exe'
X0, Y0, CW, CH = 5200, 2600, 2480, 1720          # lower right part of the 7680 x 4320 frame, where the tablet stands
TA, TB = 2.0, 8.0

def frames(video, t0, t1, crop):
    cw, ch, cx, cy = crop
    cmd = [FF, '-hide_banner', '-threads', '2', '-copyts', '-ss', f'{t0:.3f}', '-i', str(video), '-to', f'{t1:.3f}', '-vf', f'crop={cw}:{ch}:{cx}:{cy},format=gray,showinfo',
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts = []
    def err():
        for line in iter(p.stderr.readline, b''):
            m = re.search(rb'pts_time:\s*(-?[0-9.]+)', line)
            if m: pts.append(float(m.group(1)))
    th = threading.Thread(target=err, daemon=True); th.start()
    n = 0; size = cw * ch
    try:
        while True:
            raw = p.stdout.read(size)
            if len(raw) < size: break
            for _ in range(2000):
                if len(pts) > n or not th.is_alive(): break
                time.sleep(0.001)
            yield (pts[n] if n < len(pts) else float('nan')), np.frombuffer(raw, np.uint8).reshape(ch, cw)
            n += 1
    finally:
        p.stdout.close(); p.kill(); p.wait()

def find_screen(img):
    b = cv2.GaussianBlur(img[::2, ::2], (0, 0), 1.5); cands = []
    for thr in (240, 215, 190, 165):
        n, lab, st, _ = cv2.connectedComponentsWithStats((b > thr).astype(np.uint8))
        for k in range(1, n):
            x, y, w, h = [int(2 * v) for v in st[k][:4]]; a = int(4 * st[k][4])
            if 60000 < a < 700000 and a / (w * h) > 0.75 and 1.45 < w / h < 2.0: cands.append((x, y, w, h, a))
        if cands: break
    return max(cands, key=lambda c: c[4]) if cands else None

J = pd.read_csv(BASE / 'cam' / 'jobs.csv'); only = set(sys.argv[1:])
OUTF = EV / ('tablet_clock_ticks.csv' if not only else 'tablet_clock_ticks_test.csv'); rows = []; last = None
for j in J.itertuples():
    if only and j.trial not in only: continue
    video = BASE / 'video' / j.video; box = None
    for pts, img in frames(video, 3.0, 3.4, (CW, CH, X0, Y0)):
        box = find_screen(img)
        if box: break
    if box is None: box = last
    if box is None:
        print(j.trial, j.video, 'no screen', flush=True); rows.append(dict(trial=j.trial, video=j.video)); continue
    last = box; x, y, w, h, a = box
    roi = (int(0.56 * w) // 2 * 2, int(0.36 * h) // 2 * 2, X0 + x + int(0.22 * w), Y0 + y + int(0.32 * h))     # clock digits without the suffix
    ts, R = [], []
    for pts, img in frames(video, TA, TB, roi):
        f = img.astype(np.float32); lo, hi = np.percentile(f, 1), np.percentile(f, 70)
        ts.append(pts); R.append(np.clip((f - lo) / max(hi - lo, 1.0), 0, 1))
    t = np.array(ts); R = np.stack(R)
    contrast = float(np.median([np.percentile(img_, 70) - np.percentile(img_, 1) for img_ in (R[::15] * 255)]))
    D = np.r_[np.nan, np.mean(np.abs(R[1:] - R[:-1]), axis=(1, 2))]; tm = np.r_[np.nan, 0.5 * (t[1:] + t[:-1])]; ok = np.isfinite(D)
    D0 = D - np.nanmedian(D)
    phis = np.arange(0, 1, 0.002)
    def comb(ph):
        q = np.arange(np.ceil(tm[ok][0] - ph), np.floor(tm[ok][-1] - ph) + 1) + ph
        return float(np.mean(np.interp(q, tm[ok], D0[ok]))), q
    score = np.array([comb(ph)[0] for ph in phis]); kbest = int(np.argmax(score)); phi = float(phis[kbest]); ticks = comb(phi)[1]
    noise = 1.4826 * np.nanmedian(np.abs(D0)) + 1e-9; snr = float(score[kbest] / noise)
    # second best phase at least 0.15 s away: if it is nearly as good, the phase is ambiguous
    away = np.minimum(np.abs(phis - phi), 1 - np.abs(phis - phi)) > 0.15; ratio = float(score[kbest] / max(score[away].max(), 1e-9))
    # refine every tick as the frame boundary with the largest change within +-0.1 s, phase = median of the residuals
    fine = []
    for q in ticks:
        m = ok & (np.abs(tm - q) <= 0.1)
        if m.any(): fine.append(tm[m][np.argmax(D[m])] - q)
    phi_f = phi + float(np.median(fine)) if fine else phi; spread = float(np.std(fine)) if len(fine) > 1 else np.nan
    sheet_tick = [q for q in ticks if q + 0.55 <= t[-1]][-1] + (phi_f - phi)
    ks = int(np.argmin(np.abs(t - (sheet_tick + 0.5))))
    tile = (255 * R[ks]).astype(np.uint8); tile = cv2.resize(tile, (480, int(tile.shape[0] * 480 / tile.shape[1])), interpolation=cv2.INTER_CUBIC)
    tile = cv2.copyMakeBorder(tile, 30, 4, 0, 0, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(tile, f'{j.trial} {j.video[9:15]} from {sheet_tick:.2f}s snr{snr:.0f}', (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    cv2.imwrite(str(SH / f'{j.trial}.png'), tile)
    print(j.trial, j.video, 'box', box[:4], 'frames', len(t), 'phase %.3f' % (phi_f % 1), 'snr %.1f' % snr, 'ratio %.2f' % ratio, 'tick spread %.3f' % spread, 'contrast %.0f' % contrast, flush=True)
    rows.append(dict(trial=j.trial, video=j.video, video_start=j.video_start, tick_phase=phi_f % 1, sheet_tick_pts=sheet_tick, snr=snr, ratio=ratio, tick_spread_s=spread, n_ticks=len(ticks),
                     contrast=contrast, bx=x, by=y, bw=w, bh=h))
    pd.DataFrame(rows).to_csv(OUTF, index=False)
    del R; gc.collect()
print('done', flush=True)

