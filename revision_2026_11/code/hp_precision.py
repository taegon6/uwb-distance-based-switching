"""High-precision DS-TWR at the stop: precision versus accuracy on the same 16 stops.
(1) per-stop RX error by method, with one exclusion rule applied to every method,
(2) link precision of DS-TWR for growing averaging windows against fixed-integer HP."""
import numpy as np, pandas as pd, math
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_fixed.py', encoding='utf-8').read()
exec(src.split("MODE = {'DS': 'DS', 'HP raw': 'PHASE', 'HP fixed': 'PHASE'}")[0])
pd.set_option('display.width', 220)
D = pd.read_csv(OUTR / 'stop_decomposition_hp_stops.csv')
F = D[(D.offset == 'field') & (D.aggregation == '3 s')].pivot(index='trial', columns='method', values='rx_cm')
T = D[(D.offset == 'field') & (D.aggregation == '3 s')].pivot(index='trial', columns='method', values='tag_cm')
Y = D[(D.offset == 'field') & (D.aggregation == '3 s')].pivot(index='trial', columns='method', values='yaw_deg')
print('== RX coil error per stop [cm], field offsets, 3 s'); print(F.round(1).to_string())
def summary(P, label):
    d = (P['HP fixed'] - P['DS']).dropna(); se = d.std(ddof=1) / math.sqrt(len(d))
    tq = {16: 2.131, 15: 2.145, 14: 2.160, 13: 2.179, 12: 2.201, 11: 2.228, 10: 2.262}[len(d)]
    print(f'{label}: n {len(d)} median DS {P["DS"].median():.2f}, HP raw {P["HP raw"].median():.2f}, HP fixed {P["HP fixed"].median():.2f} | HP fixed - DS {d.mean():+.2f} +- {tq * se:.2f}, lower in {int((d < 0).sum())}/{len(d)}')
summary(F, 'RX, all stops'); summary(T, 'tag midpoint, all stops'); summary(Y, 'heading [deg], all stops')
# one rule for every method: drop a stop when its error exceeds median + 3 scaled MAD in ANY method
bad = set()
for m in F.columns:
    v = F[m]; mad = 1.4826 * (v - v.median()).abs().median()
    bad |= set(v.index[v > v.median() + 3 * mad])
print('stops removed by the common rule:', sorted(bad))
keep = [t for t in F.index if t not in bad]
summary(F.loc[keep], 'RX, common rule'); summary(T.loc[keep], 'tag midpoint, common rule'); summary(Y.loc[keep], 'heading, common rule')

# (2) link precision against the averaging window, last part of the stop after the integer is fixed
rows = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy(); late = (tw >= tstop[t] + ANCHOR_S)
        ds = pd.Series(g.DS.to_numpy(float)[late]); hp = pd.Series(g['HP fixed'].to_numpy(float)[late]); dur = tw[late].max() - tw[late].min() if late.sum() > 1 else 0
        if late.sum() < 60: continue
        r = dict(trial=t, link=l, n=int(late.sum()), dur_s=dur, hp_fixed_sd_mm=1000 * hp.std())
        for w in (1, 5, 25, 50):
            r[f'ds_w{w}_sd_mm'] = 1000 * ds.rolling(w).median().std()
        rows.append(r)
P = pd.DataFrame(rows)
print('== link precision after the stop [mm], median over', len(P), 'links; windows in samples at 26 Hz')
print(P[['dur_s', 'n', 'ds_w1_sd_mm', 'ds_w5_sd_mm', 'ds_w25_sd_mm', 'ds_w50_sd_mm', 'hp_fixed_sd_mm']].median().round(2).to_string())
print('ratio DS single / HP fixed %.0f, DS 25-sample median / HP fixed %.0f' % ((P.ds_w1_sd_mm / P.hp_fixed_sd_mm).median(), (P.ds_w25_sd_mm / P.hp_fixed_sd_mm).median()))
P.round(3).to_csv(OUTR / 'hp_precision_vs_window.csv', index=False)
# integer anchoring: distance between the fixed-integer range and the DS median of the same window, in cycles
anch = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy(); late = (tw >= tstop[t] + ANCHOR_S)
        if late.sum() < 30: continue
        anch.append((np.nanmedian(g['HP fixed'].to_numpy(float)[late]) - np.nanmedian(g.DS.to_numpy(float)[late])) / LAM)
anch = np.array(anch)
print('fixed-integer range minus DS median [cycles]: median %.2f, p05 %.2f, p95 %.2f, |x| > 0.5 cycle in %d of %d links' % (np.median(anch), np.percentile(anch, 5), np.percentile(anch, 95), int((np.abs(anch) > 0.5).sum()), len(anch)))

