
import math
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/pose_offset_calibration.py', encoding='utf-8').read()
exec(src.split('def features(p)')[0])
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
FEATURES = {'constant': lambda p: [1.0], 'heading': lambda p: [1.0, p[2]], 'lateral': lambda p: [1.0, p[0]], 'lateral + heading': lambda p: [1.0, p[0], p[2]],
            'lateral + longitudinal + heading': lambda p: [1.0, p[0], p[1] - STOP[1], p[2]]}
def run(train_of, feats):
    out = []
    for t in trials:
        train = train_of(t)
        if t in train: train = [u for u in train if u != t]
        constant = {l: float(np.median([implied[u][l] for u in train])) for l in LINKS}
        A = np.array([feats(pose[u]) for u in train]); coef = {l: np.linalg.lstsq(A, np.array([implied[u][l] for u in train]), rcond=None)[0] for l in LINKS}
        m = meas[t][1]; x, y, yaw = pose[t]; R0 = rx(x, y, yaw)
        p = solve({l: m[l] - constant[l] for l in LINKS}, (x, y, yaw))
        if len(feats(p)) > 1:
            for _ in range(4):
                p = solve({l: m[l] - float(coef[l] @ np.array(feats(p))) for l in LINKS}, tuple(p))
        out.append(dict(trial=t, tag=100 * math.hypot(p[0] - x, p[1] - y), yaw=abs(p[2] - yaw), rx=100 * float(np.linalg.norm(rx(*p) - R0))))
    return pd.DataFrame(out).set_index('trial')
def parity(t):
    return int(sched.loc[t, 'block']) % 2 if t in sched.index and pd.notna(sched.loc[t, 'block']) else int(t[1:]) % 2
splits = {'leave one stop out': lambda t: trials, 'two-fold by block (train on the other half)': lambda t: [u for u in trials if parity(u) != parity(t)]}
rows = []
for sname, train_of in splits.items():
    base = run(train_of, FEATURES['constant'])
    for fname, feats in FEATURES.items():
        D = run(train_of, feats); d = D.rx - base.rx
        rows.append(dict(split=sname, model=fname, tag_median=D.tag.median(), tag_le10=int((D.tag <= 10).sum()), yaw_median=D.yaw.median(), rx_median=D.rx.median(), rx_le10=int((D.rx <= 10).sum()), rx_max=D.rx.max(),
                         rx_minus_constant=f'{d.mean():+.2f} +- {2.024 * d.std(ddof=1) / math.sqrt(len(d)):.2f}', lower=int((d < 0).sum())))
T = pd.DataFrame(rows); pd.set_option('display.width', 250); print(T.round(2).to_string(index=False))
T.round(3).to_csv(OUT / 'revision_checks' / 'pose_dependent_offset_models.csv', index=False)
print('train sizes two-fold:', sum(parity(t) == 0 for t in trials), sum(parity(t) == 1 for t in trials))

