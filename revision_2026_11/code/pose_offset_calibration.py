"""Pose-dependent link-offset calibration from the existing stops.
Baseline: one constant offset per link (median of the other stops).
Model: offset of a link as a linear function of the stop pose (lateral x, longitudinal y, heading).
Evaluation: leave one stop out, the model sees only the UWB estimate of the left-out stop (iterated), never its reference."""
import math
import numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/analyze_stops_v2.py', encoding='utf-8').read()
src = src.replace('BASE = Path(__file__).parent', "BASE = Path(r'D:/sandbox/uwb/output/field_20261124')")
exec(src.split('# 1) original entries')[0])
S = pd.read_csv(OUT / 'stops_v2_seltos_L439.csv').set_index('trial')
trials = list(S.index)
pose = {t: (S.loc[t, 'ref_x'], S.loc[t, 'ref_y'], S.loc[t, 'ref_yaw']) for t in trials}
implied = {t: {l: meas[t][1][l] - slant(*pose[t])[l] for l in LINKS} for t in trials}
def features(p): return np.array([1.0, p[0], p[1] - STOP[1], p[2]])        # intercept, lateral [m], longitudinal [m], heading [deg]
def evaluate(train_of, label):
    rows = []
    for t in trials:
        train = train_of(t)
        constant = {l: float(np.median([implied[u][l] for u in train])) for l in LINKS}
        A = np.array([features(pose[u]) for u in train])
        coef = {l: np.linalg.lstsq(A, np.array([implied[u][l] for u in train]), rcond=None)[0] for l in LINKS}
        m = meas[t][1]; x, y, yaw = pose[t]; R0 = rx(x, y, yaw)
        p = solve({l: m[l] - constant[l] for l in LINKS}, (x, y, yaw)); base = p.copy()
        for _ in range(4):                                   # offsets from the current UWB estimate, then solve again
            offs = {l: float(coef[l] @ features(p)) for l in LINKS}
            p = solve({l: m[l] - offs[l] for l in LINKS}, tuple(p))
        for name, q in (('constant', base), ('pose-dependent', p)):
            rows.append(dict(trial=t, path=S.loc[t, 'path'], model=name, tag_cm=100 * math.hypot(q[0] - x, q[1] - y), yaw_deg=abs(q[2] - yaw), rx_cm=100 * float(np.linalg.norm(rx(*q) - R0))))
    D = pd.DataFrame(rows); D['split'] = label
    return D
loo = evaluate(lambda t: [u for u in trials if u != t], 'leave one stop out, all 39 stops')
straight = [t for t in trials if S.loc[t, 'path'] == 'S']
s2c = evaluate(lambda t: [u for u in straight if u != t], 'trained on straight stops only')
pd.set_option('display.width', 220)
for D in (loo, s2c):
    print('==', D.split.iloc[0])
    print(D.groupby('model').agg(tag_median=('tag_cm', 'median'), tag_le10=('tag_cm', lambda v: int((v <= 10).sum())), yaw_median=('yaw_deg', 'median'), rx_median=('rx_cm', 'median'),
                                rx_le10=('rx_cm', lambda v: int((v <= 10).sum())), rx_p90=('rx_cm', lambda v: v.quantile(.9)), n=('trial', 'count')).round(2).to_string())
    print(D.groupby(['path', 'model']).agg(tag_median=('tag_cm', 'median'), rx_median=('rx_cm', 'median'), rx_le10=('rx_cm', lambda v: int((v <= 10).sum())), n=('trial', 'count')).round(2).to_string())
    W = D.pivot(index='trial', columns='model', values='rx_cm'); d = W['pose-dependent'] - W['constant']
    print('RX error, pose-dependent minus constant: %+.2f +- %.2f cm (n %d), lower in %d' % (d.mean(), 2.024 * d.std(ddof=1) / math.sqrt(len(d)), len(d), int((d < 0).sum())))
    Wt = D.pivot(index='trial', columns='model', values='tag_cm'); dt = Wt['pose-dependent'] - Wt['constant']
    print('tag midpoint, pose-dependent minus constant: %+.2f +- %.2f cm, lower in %d' % (dt.mean(), 2.024 * dt.std(ddof=1) / math.sqrt(len(dt)), int((dt < 0).sum())))
pd.concat([loo, s2c]).round(3).to_csv(OUT / 'revision_checks' / 'pose_dependent_offset_calibration.csv', index=False)
# how much of the offset variation the pose explains (all stops, in-sample R^2 per link)
A = np.array([features(pose[u]) for u in trials])
for l in LINKS:
    yv = np.array([implied[u][l] for u in trials]); c = np.linalg.lstsq(A, yv, rcond=None)[0]; r = yv - A @ c
    print(l, 'offset SD %.1f cm, residual SD after the pose model %.1f cm, R2 %.2f, slopes: lateral %.1f cm/m, longitudinal %.1f cm/m, heading %.2f cm/deg' % (100 * yv.std(), 100 * r.std(), 1 - r.var() / yv.var(), 100 * c[1], 100 * c[2], 100 * c[3]))

