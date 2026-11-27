"""Does outlier rejection plus a median (as in Ref. 14) turn the phase precision into accuracy at the stop?
Per stop and link: range with outliers rejected and the median taken, for DS-TWR and for fixed-integer HP.
Compared: scatter inside a stop (precision) and scatter of the stop-implied offset across stops (what limits accuracy)."""
import numpy as np, pandas as pd, math
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_fixed.py', encoding='utf-8').read()
exec(src.split("MODE = {'DS': 'DS', 'HP raw': 'PHASE', 'HP fixed': 'PHASE'}")[0])
def robust_median(v):
    v = v[np.isfinite(v)]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med))
    keep = np.abs(v - med) <= 3 * mad if mad > 0 else np.ones(len(v), bool)
    return float(np.median(v[keep])), float(1 - keep.mean())
rows = []
for t in runs:
    x, y, yaw = ref(t); true = slant(x, y, yaw)
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy(); late = tw >= tstop[t] + ANCHOR_S
        if late.sum() < 60: continue
        halves = np.array_split(np.flatnonzero(late), 2)
        r = dict(trial=t, path=sched.loc[t, 'path'], link=l)
        for m, name in (('DS', 'ds'), ('HP fixed', 'hp')):
            v = g[m].to_numpy(float)
            value, rejected = robust_median(v[late])
            first, _ = robust_median(v[halves[0]]); second, _ = robust_median(v[halves[1]])
            r[name + '_offset_cm'] = 100 * (value - true[l]); r[name + '_rejected_pct'] = 100 * rejected; r[name + '_half_diff_mm'] = 1000 * (second - first)
        rows.append(r)
D = pd.DataFrame(rows); pd.set_option('display.width', 200)
print('inside one stop, first half vs second half of the standstill (median of |difference|) [mm]: DS %.2f, HP fixed %.2f' % (D.ds_half_diff_mm.abs().median(), D.hp_half_diff_mm.abs().median()))
print('samples rejected as outliers [%%]: DS %.1f, HP fixed %.1f' % (D.ds_rejected_pct.median(), D.hp_rejected_pct.median()))
S = D[D.path == 'S']
spread = S.groupby('link')[['ds_offset_cm', 'hp_offset_cm']].agg(lambda v: 1.4826 * (v - v.median()).abs().median())
print('straight stops (%d): scatter of the stop-implied offset across stops, per link [cm]' % S.trial.nunique()); print(spread.round(2).to_string())
print('range of the implied offset across straight stops [cm]:'); print((S.groupby('link')[['ds_offset_cm', 'hp_offset_cm']].max() - S.groupby('link')[['ds_offset_cm', 'hp_offset_cm']].min()).round(1).to_string())
allspread = D.groupby('link')[['ds_offset_cm', 'hp_offset_cm']].agg(lambda v: 1.4826 * (v - v.median()).abs().median())
print('all 16 stops, scatter across stops [cm]: DS %.2f, HP fixed %.2f (median over links)' % (allspread.ds_offset_cm.median(), allspread.hp_offset_cm.median()))
D.round(3).to_csv(OUTR / 'hp_median_within_vs_across_stops.csv', index=False)

