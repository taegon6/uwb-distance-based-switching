
import math
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/pose_offset_models.py', encoding='utf-8').read()
exec(src.split('def run(train_of, feats):')[0])
def run(train_of, feats, passes):
    out = []
    for t in trials:
        train = [u for u in train_of(t) if u != t]
        constant = {l: float(np.median([implied[u][l] for u in train])) for l in LINKS}
        A = np.array([feats(pose[u]) for u in train]); coef = {l: np.linalg.lstsq(A, np.array([implied[u][l] for u in train]), rcond=None)[0] for l in LINKS}
        m = meas[t][1]; x, y, yaw = pose[t]; R0 = rx(x, y, yaw)
        p = solve({l: m[l] - constant[l] for l in LINKS}, (x, y, yaw))
        for _ in range(passes):
            p = solve({l: m[l] - float(coef[l] @ np.array(feats(p))) for l in LINKS}, tuple(p))
        out.append(dict(trial=t, tag=100 * math.hypot(p[0] - x, p[1] - y), yaw=abs(p[2] - yaw), rx=100 * float(np.linalg.norm(rx(*p) - R0))))
    return pd.DataFrame(out).set_index('trial')
def parity(t):
    return int(sched.loc[t, 'block']) % 2 if t in sched.index and pd.notna(sched.loc[t, 'block']) else int(t[1:]) % 2
splits = {'leave one stop out': lambda t: trials, 'two-fold by block': lambda t: [u for u in trials if parity(u) != parity(t)]}
full = FEATURES['lateral + longitudinal + heading']; rows = []
for sname, train_of in splits.items():
    base = run(train_of, FEATURES['constant'], 0)
    for passes in (1, 4):
        D = run(train_of, full, passes); d = D.rx - base.rx
        rows.append(dict(split=sname, passes=passes, tag_median=D.tag.median(), tag_le10=int((D.tag <= 10).sum()), yaw_median=D.yaw.median(), rx_median=D.rx.median(), rx_le10=int((D.rx <= 10).sum()), rx_max=D.rx.max(),
                         rx_minus_constant=f'{d.mean():+.2f} +- {2.024 * d.std(ddof=1) / math.sqrt(len(d)):.2f}', lower=int((d < 0).sum()), base_rx_median=base.rx.median(), base_rx_le10=int((base.rx <= 10).sum())))
pd.set_option('display.width', 250); print(pd.DataFrame(rows).round(2).to_string(index=False))

