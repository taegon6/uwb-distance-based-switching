"""Camera board track for one run (low memory): direct ChArUco near, chunked backward LK far.

    python cam_track.py VIDEO T0_S T1_S OUT.csv
Times are video seconds. One row per frame: pts (video s), source DIRECT/LKF/LKB/NONE, pose.
"""
from __future__ import annotations
import sys, subprocess, threading, re, csv, time
from pathlib import Path
import numpy as np, cv2

FF = r'D:/sandbox/uwb/output/laptop_handoff_20261124/ffmpeg/ffmpeg.exe'
W, H = 7680, 4320
K = np.array([[6340.010092229866 * 1.0237, 0, 3840], [0, 6369.812813790984 * 1.0237, 2160], [0, 0, 1]])
D = np.array([0.2979395022572093, 0.5592051964165025, -6.247384436685632, 19.678310018949134])
SQ, MK = 0.035, 0.026
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
BOARD = cv2.aruco.CharucoBoard((5, 7), SQ, MK, DICT)
OBJ = BOARD.getChessboardCorners().astype(np.float64)
CENTER = np.array([5 * SQ / 2, 7 * SQ / 2, 0.0])
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def detector(full):
    p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = 0.005 if full else 0.02
    p.adaptiveThreshWinSizeMax = 53
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(BOARD, cv2.aruco.CharucoParameters(), p)
DET_FULL, DET_ROI = detector(True), detector(False)

def frames(video, t0, t1):
    # -copyts keeps the original pts, so -to is an absolute end time and pts_time is the video time
    cmd = [FF, '-hide_banner', '-threads', '4', '-copyts', '-ss', f'{max(t0, 0):.3f}', '-i', str(video), '-to', f'{t1:.3f}',
           '-vf', 'format=gray,showinfo', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts = []
    def err():
        for line in iter(p.stderr.readline, b''):
            m = re.search(rb'pts_time:\s*(-?[0-9.]+)', line)
            if m: pts.append(float(m.group(1)))
    th = threading.Thread(target=err, daemon=True); th.start()
    n = 0; size = W * H
    try:
        while True:
            raw = p.stdout.read(size)
            if len(raw) < size: break
            for _ in range(2000):
                if len(pts) > n or not th.is_alive(): break
                time.sleep(0.001)
            yield (pts[n] if n < len(pts) else float('nan')), np.frombuffer(raw, np.uint8).reshape(H, W)
            n += 1
    finally:
        p.stdout.close(); p.kill(); p.wait()

def pose(img_pts, ids):
    if len(ids) < 6: return None
    und = cv2.fisheye.undistortPoints(img_pts.reshape(-1, 1, 2).astype(np.float64), K, D, P=K)
    o = OBJ[ids].reshape(-1, 1, 3)
    ok, rv, tv = cv2.solvePnP(o, und, K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok: return None
    rv, tv = cv2.solvePnPRefineLM(o, und, K, None, rv, tv)
    proj, _ = cv2.projectPoints(o, rv, tv, K, None)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - und.reshape(-1, 2)) ** 2, axis=1))))
    c = cv2.Rodrigues(rv)[0] @ CENTER + tv.ravel()
    return rv.ravel(), tv.ravel(), c, err

def detect(gray, roi=None):
    if roi is None:
        cc, cid, _, _ = DET_FULL.detectBoard(gray); off = np.zeros(2)
    else:
        x0, y0, x1, y1 = roi; cc, cid, _, _ = DET_ROI.detectBoard(np.ascontiguousarray(gray[y0:y1, x0:x1])); off = np.array([x0, y0])
    if cid is None or len(cid) < 6: return None
    return cc.reshape(-1, 2) + off, cid.ravel()

def complete24(pts, ids):
    Hm, _ = cv2.findHomography(OBJ[ids, :2], pts, cv2.RANSAC, 2.0)
    if Hm is None: return None
    return cv2.perspectiveTransform(OBJ[:, :2].reshape(-1, 1, 2), Hm).reshape(-1, 2)

def bbox(pts, margin, extra=0):
    x0, y0 = pts.min(0); x1, y1 = pts.max(0); s = max(x1 - x0, y1 - y0) * margin + extra
    return [int(max(0, x0 - s)), int(max(0, y0 - s)), int(min(W, x1 + s)), int(min(H, y1 + s))]

def row(pt, src, n, p):
    r = {'pts': round(pt, 5), 'source': src, 'n': n}
    keys = ('rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'cx', 'cy', 'cz', 'reproj_px', 'u', 'v')
    if p is None:
        r.update({k: '' for k in keys})
    else:
        (rv, tv, c, e), uv = p
        r.update(rx=rv[0], ry=rv[1], rz=rv[2], tx=tv[0], ty=tv[1], tz=tv[2], cx=c[0], cy=c[1], cz=c[2],
                 reproj_px=round(e, 3), u=round(uv[0], 1), v=round(uv[1], 1))
    return r

def lk_step(a, b, pts, off):
    """Track pts (full-res coords) from crop a to crop b (same crop box at offset off)."""
    p = (pts - off).astype(np.float32).reshape(-1, 1, 2)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None, **LK)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **LK)
    fb = np.linalg.norm(back.reshape(-1, 2) - p.reshape(-1, 2), axis=1)
    q = nxt.reshape(-1, 2) + off
    hb, wb = b.shape
    inside = (q[:, 0] - off[0] > 3) & (q[:, 1] - off[1] > 3) & (q[:, 0] - off[0] < wb - 4) & (q[:, 1] - off[1] < hb - 4)
    return q, (st.ravel() > 0) & (st2.ravel() > 0) & (fb < 0.7) & inside

def forward(video, t0, t1):
    rows, first = [], None
    prev = None; g_prev = None
    for i, (pt, g) in enumerate(frames(video, t0, t1)):
        if first is None:
            if i % 3 == 0:
                d = detect(g)
                if d is not None and len(d[1]) >= 18:
                    first = pt; prev = complete24(*d); g_prev = g
                    rows.append(row(pt, 'DIRECT', len(d[1]), (pose(*d), d[0].mean(0)))); continue
            rows.append(row(pt, 'NONE', 0, None)); continue
        pts24 = prev
        roi = bbox(pts24, 0.8, 100)
        d = detect(g, roi)
        if d is not None:
            prev = complete24(*d); g_prev = g
            rows.append(row(pt, 'DIRECT', len(d[1]), (pose(*d), d[0].mean(0)))); continue
        a = g_prev[roi[1]:roi[3], roi[0]:roi[2]]; b = g[roi[1]:roi[3], roi[0]:roi[2]]
        q, good = lk_step(a, b, pts24, np.array(roi[:2]))
        g_prev = g
        if good.sum() >= 8:
            full = complete24(q[good], np.flatnonzero(good))
            if full is not None:
                prev = full
                rows.append(row(pt, 'LKF', int(good.sum()), (pose(q[good], np.flatnonzero(good)), q[good].mean(0)))); continue
        rows.append(row(pt, 'NONE', 0, None))
    return rows, first

def backward(video, t_first, t0, pts24, chunk=1.0):
    """Track from the first direct detection back to t0 in 1 s chunks (full-res crops)."""
    out = []; t_end = t_first; cur = pts24
    while t_end > t0 + 1e-3:
        ts = max(t0, t_end - chunk)
        box = bbox(cur, 1.2, 250)
        # decode the chunk plus the already tracked frame at t_end, all cropped with one box
        buf = [(pt, g[box[1]:box[3], box[0]:box[2]].copy()) for pt, g in frames(video, ts, t_end + 0.02) if pt <= t_end + 1e-3]
        off = np.array(box[:2])
        if len(buf) < 2 or abs(buf[-1][0] - t_end) > 0.005:
            print('chunk boundary mismatch', t_end, buf[-1][0] if buf else None, flush=True); break
        cur_crop = buf[-1][1]
        lost = False
        for pt, crop in reversed(buf[:-1]):
            q, good = lk_step(cur_crop, crop, cur, off)
            if good.sum() < 8: lost = True; break
            full = complete24(q[good], np.flatnonzero(good))
            if full is None: lost = True; break
            out.append(row(pt, 'LKB', int(good.sum()), (pose(q[good], np.flatnonzero(good)), q[good].mean(0))))
            cur, cur_crop = full, crop
        if lost: break
        t_end = buf[0][0]
    return out

def run(video, t0, t1, out):
    a = time.time()
    rows, first = forward(video, t0, t1)
    if first is not None:
        f0 = next(r for r in rows if r['pts'] == round(first, 5))
        # rebuild the 24 corners of the first detection from its pose (projected) for a clean seed
        rv = np.array([f0['rx'], f0['ry'], f0['rz']], float); tv = np.array([f0['tx'], f0['ty'], f0['tz']], float)
        und, _ = cv2.projectPoints(OBJ.reshape(-1, 1, 3), rv, tv, K, None)
        seed = cv2.fisheye.distortPoints(cv2.undistortPoints(und, K, None).reshape(-1, 1, 2), K, D).reshape(-1, 2)
        back = backward(video, first, t0, seed)
        got = {r['pts']: r for r in back}
        rows = [got.get(r['pts'], r) if r['source'] == 'NONE' and r['pts'] < first else r for r in rows]
    if not rows or not any(r['source'] == 'DIRECT' for r in rows):
        raise RuntimeError(f'no direct detection ({len(rows)} frames)')
    tmp = str(out) + '.tmp'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    Path(tmp).replace(out)
    src = {}
    for r in rows: src[r['source']] = src.get(r['source'], 0) + 1
    print(Path(out).name, len(rows), src, f'first {first}', f'{time.time() - a:.0f} s', flush=True)

if __name__ == '__main__':
    run(Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4])

