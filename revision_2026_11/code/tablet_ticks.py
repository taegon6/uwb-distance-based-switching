"""Tablet clock ticks from the stored clock regions (see tablet_decode.py).
A tick is a persistent change of the seconds digit: mean image of the 8 frames after a frame boundary minus the 8 frames before it.
(A plain frame difference is dominated by the one-second key-frame period of the video codec.)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); EV = BASE / 'cam' / 'eval'; ROI = EV / 'tablet_roi'; SH = EV / 'tablet_sheets'; SH.mkdir(exist_ok=True)
J = pd.read_csv(BASE / 'cam' / 'jobs.csv'); M = 8
rows = []
for j in J.itertuples():
    f = ROI / f'{j.trial}.npz'
    if not f.exists(): continue
    Z = np.load(f); t = Z['pts']; F = Z['frames'].astype(np.float32)
    lo = np.percentile(F, 1, axis=(1, 2))[:, None, None]; hi = np.percentile(F, 70, axis=(1, 2))[:, None, None]
    R = np.clip((F - lo) / np.maximum(hi - lo, 1.0), 0, 1); n = len(t)
    # text of the clock = dark pixels of the time-median image; HH:MM:SS has eight character slots, the seconds are the last two
    dark = (1 - np.median(R, axis=0)) > 0.35
    rowc = dark.sum(axis=1); rr = np.where(rowc >= 0.3 * rowc.max())[0]; r0, r1 = int(rr.min()), int(rr.max()) + 1
    colc = dark[r0:r1].sum(axis=0); cc = np.where(colc >= 2)[0]
    groups = [g for g in np.split(cc, np.where(np.diff(cc) > 3)[0] + 1) if colc[g].max() >= 0.45 * colc.max()]      # tall strokes only (drops the small suffix)
    xa, xb = int(min(g[0] for g in groups)), int(max(g[-1] for g in groups)) + 1
    c0, c1 = int(xa + 0.74 * (xb - xa)), xb + 2; s0, s1 = xa, int(xa + 0.60 * (xb - xa))            # seconds digits, static part (HH:MM)
    def change(region, m):
        A = R[:, r0:r1, region[0]:region[1]]
        if m == 1: return np.r_[np.nan, np.mean(np.abs(A[1:] - A[:-1]), axis=(1, 2))]
        cs = np.concatenate([np.zeros((1,) + A.shape[1:], np.float32), np.cumsum(A, axis=0)]); out = np.full(n, np.nan)
        for k in range(m, n - m + 1): out[k] = np.mean(np.abs((cs[k + m] - cs[k]) - (cs[k] - cs[k - m]))) / m
        return out
    sM = change((c0, c1), M) - change((s0, s1), M); d1 = change((c0, c1), 1) - change((s0, s1), 1)
    tm = np.r_[np.nan, 0.5 * (t[1:] + t[:-1])]; ok = np.isfinite(sM)
    phis = np.arange(0, 1, 0.002)
    def comb(ph):
        q = np.arange(np.ceil(tm[ok][0] - ph), np.floor(tm[ok][-1] - ph) + 1) + ph
        return float(np.mean(np.interp(q, tm[ok], sM[ok]))), q
    sc = np.array([comb(ph)[0] for ph in phis]); kb = int(np.argmax(sc)); phi = float(phis[kb]); ticks = comb(phi)[1]
    contrast = float((sc.max() - np.median(sc)) / (np.median(np.abs(sc - np.median(sc))) + 1e-9))
    fine = []
    for q in ticks:
        m = np.isfinite(d1) & (np.abs(tm - q) <= 0.15)
        if m.any(): fine.append(tm[m][np.nanargmax(d1[m])])
    fine = np.array(fine); frac = np.mod(fine - phi + 0.5, 1) - 0.5 + phi
    phase = float(np.median(frac)) % 1; spread = float(np.std(frac)) if len(frac) > 1 else np.nan
    usable = [q for q in fine if q + 0.55 <= t[-1]]; sheet_tick = usable[-1] if usable else fine[-1]
    k = int(np.argmin(np.abs(t - (sheet_tick + 0.5))))
    tile = (255 * R[k]).astype(np.uint8); cv2.rectangle(tile, (int(c0), int(r0)), (int(c1) - 1, int(r1) - 1), 128, 1)
    tile = cv2.resize(tile, (420, int(tile.shape[0] * 420 / tile.shape[1])), interpolation=cv2.INTER_CUBIC)
    tile = cv2.copyMakeBorder(tile, 26, 4, 0, 4, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(tile, f'{j.trial} {j.video[9:15]} tick {sheet_tick:.2f} c{contrast:.0f}', (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 2)
    cv2.imwrite(str(SH / f'{j.trial}.png'), tile)
    rows.append(dict(trial=j.trial, video=j.video, video_start=j.video_start, n_frames=n, n_ticks=len(fine), phase=phase, tick_spread_s=spread, contrast=contrast,
                     sheet_tick_pts=sheet_tick, digit_w=int(c1 - c0), phase_comb=phi, ticks=';'.join(f'{q:.3f}' for q in fine)))
    print(j.trial, 'frames', n, 'ticks', np.round(fine, 3), 'phase %.3f spread %.3f contrast %.1f digit cols %d-%d' % (phase, spread, contrast, c0, c1), flush=True)
D = pd.DataFrame(rows); D.to_csv(EV / 'tablet_clock_ticks_v2.csv', index=False)
# contact sheets
tiles = [cv2.imread(str(SH / f'{t}.png'), 0) for t in D.trial]
h = max(x.shape[0] for x in tiles); tiles = [cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=255) for x in tiles]
for i in range(0, len(tiles), 14):
    part = tiles[i:i + 14]; part += [np.full_like(tiles[0], 255)] * (-len(part) % 2)
    sheet = np.vstack([np.hstack(part[k:k + 2]) for k in range(0, len(part), 2)])
    cv2.imwrite(str(EV / f'tablet_contact_{i // 14 + 1}.png'), sheet)
print('done', len(D))

