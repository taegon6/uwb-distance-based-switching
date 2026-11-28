"""Validation of the optical-flow (LK) part of the camera reference on straight runs: start from a direct
ChArUco detection near the camera and track backward through frames that also have a direct detection,
then compare the two poses frame by frame.

    python lk_validate.py T10 1.6
"""
import sys, math
import numpy as np, pandas as pd, cv2
from pathlib import Path
CAM = Path(r'D:/sandbox/uwb/output/field_20261124/cam')
src = open(CAM / 'cam_track.py', encoding='utf-8').read().split('def run(video, t0, t1, out):')[0]
exec(src)
trial = sys.argv[1]; seed_distance = float(sys.argv[2])
job = pd.read_csv(CAM / 'jobs.csv').set_index('trial').loc[trial]
video = Path(r'D:/sandbox/uwb/output/field_20261124/video') / job.video
T = pd.read_csv(CAM / 'tracks' / f'{trial}.csv'); T = T[T.cx.notna()].copy()
T['d'] = np.sqrt(T.cx ** 2 + T.cy ** 2 + T.cz ** 2)
direct = T[T.source == 'DIRECT'].sort_values('pts')
seed_row = direct[direct.d <= seed_distance].iloc[0]          # first direct frame at or inside the seed distance
t_first = float(direct.pts.iloc[0])
rv = seed_row[['rx', 'ry', 'rz']].to_numpy(float); tv = seed_row[['tx', 'ty', 'tz']].to_numpy(float)
und, _ = cv2.projectPoints(OBJ.reshape(-1, 1, 3), rv, tv, K, None)
seed = cv2.fisheye.distortPoints(cv2.undistortPoints(und, K, None).reshape(-1, 1, 2), K, D).reshape(-1, 2)
print(trial, video.name, 'seed pts', seed_row.pts, 'd', round(seed_row.d, 2), 'back to', t_first, flush=True)
rows = backward(str(video), float(seed_row.pts), t_first, seed)
B = pd.DataFrame(rows); B = B[B.cx != ''].astype({k: float for k in ('rx', 'ry', 'rz', 'cx', 'cy', 'cz')})
M = B.merge(direct, on='pts', suffixes=('_lk', '_direct'))
def normal_angle(rx, ry, rz):
    n = cv2.Rodrigues(np.array([rx, ry, rz]))[0][:, 2]
    return math.degrees(math.atan2(n[0], -n[2]))               # horizontal angle of the board normal in the camera frame
M['lateral_mm'] = 1000 * (M.cx_lk - M.cx_direct); M['height_mm'] = 1000 * (M.cy_lk - M.cy_direct); M['depth_mm'] = 1000 * (M.cz_lk - M.cz_direct)
M['yaw_lk'] = [normal_angle(*v) for v in M[['rx_lk', 'ry_lk', 'rz_lk']].to_numpy()]
M['yaw_direct'] = [normal_angle(*v) for v in M[['rx_direct', 'ry_direct', 'rz_direct']].to_numpy()]
M['yaw_diff'] = (M.yaw_lk - M.yaw_direct + 180) % 360 - 180
M['bin'] = pd.cut(M.d, [0, 2, 3, 4, 5, 6])
out = CAM / 'eval' / f'lk_validation_{trial}.csv'
M[['pts', 'd', 'lateral_mm', 'height_mm', 'depth_mm', 'yaw_lk', 'yaw_direct', 'yaw_diff']].round(3).to_csv(out, index=False)
print(M.groupby('bin', observed=True).agg(n=('d', 'size'), lateral_med=('lateral_mm', 'median'), lateral_abs_p95=('lateral_mm', lambda v: v.abs().quantile(.95)),
      depth_med=('depth_mm', 'median'), depth_abs_p95=('depth_mm', lambda v: v.abs().quantile(.95)),
      yaw_med=('yaw_diff', 'median'), yaw_abs_p95=('yaw_diff', lambda v: v.abs().quantile(.95))).round(1).to_string())
print('frames compared', len(M), 'of', len(B), 'tracked')

