import numpy as np, pandas as pd, cv2
from pathlib import Path
E = Path(r'D:/sandbox/uwb/output/field_20261124/cam/eval')
panels = []
for t, title in [('T10', 'S T10 STAGED'), ('T28', 'C1 T28 STAGED'), ('T50', 'C2 T50 STAGED')]:
    p = pd.read_csv(E / f'pairs_{t}.csv')
    W, H, m = 520, 760, 50
    img = np.full((H, W, 3), 255, np.uint8)
    xs = np.r_[p.tag_midpoint_x_m, p.cam_X]; ys = np.r_[p.tag_midpoint_y_m, p.cam_Y]
    xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
    x0, x1, y0, y1 = xs.min() - 0.3, xs.max() + 0.3, ys.min() - 0.3, ys.max() + 0.3
    s = min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0))
    f = lambda x, y: (int(m + (x - x0) * s), int(H - m - (y - y0) * s))
    for gx in np.arange(np.ceil(x0), x1, 1.0):
        cv2.line(img, f(gx, y0), f(gx, y1), (230, 230, 230), 1); cv2.putText(img, f'{gx:.0f}', f(gx, y0), 0, 0.4, (120, 120, 120), 1)
    for gy in np.arange(np.ceil(y0), y1, 1.0):
        cv2.line(img, f(x0, gy), f(x1, gy), (230, 230, 230), 1); cv2.putText(img, f'{gy:.0f}', f(x0, gy), 0, 0.4, (120, 120, 120), 1)
    col = {'SS': (0, 140, 255), 'DS': (200, 100, 0), 'PHASE': (0, 160, 0)}
    for r in p.itertuples():
        if np.isfinite(r.tag_midpoint_x_m): cv2.circle(img, f(r.tag_midpoint_x_m, r.tag_midpoint_y_m), 2, col.get(r.mode, (0, 0, 0)), -1)
    c = p[['cam_X', 'cam_Y']].dropna().to_numpy()
    for a, b in zip(c[:-1], c[1:]): cv2.line(img, f(*a), f(*b), (0, 0, 0), 2)
    cv2.putText(img, title, (10, 25), 0, 0.7, (0, 0, 0), 2)
    for i, (k, v) in enumerate([('camera', (0, 0, 0)), ('UWB SS', col['SS']), ('UWB DS', col['DS']), ('UWB HP', col['PHASE'])]):
        cv2.putText(img, k, (W - 130, 30 + 22 * i), 0, 0.5, v, 2)
    panels.append(img)
cv2.imwrite(str(E / 'example_trajectories.png'), np.hstack(panels)); print('ok')

