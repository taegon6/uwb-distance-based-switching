"""Fixed-integer high-precision processing with a causal stop detector (UWB only).
The stop is declared from the phase itself: the resultant length of the last 25 wrapped-phase samples of a link
stays at or above R_ON for HOLD consecutive samples, and the declaration is withdrawn when it falls below R_OFF.
The stop time of a run is the latest declaration over the four links. Everything after that is hp_fixed.py."""
import numpy as np, pandas as pd, math
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_fixed.py', encoding='utf-8').read()
prelude, rest = src.split('step_rows = []')
exec(prelude)
R_ON, R_OFF, HOLD = 0.6, 0.4, 13
tstop_camera = dict(tstop); tstop_uwb = {}; stats = []
for t in runs:
    declared = []
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy()
        phasor = np.exp(1j * 2 * np.pi * np.mod(g['HP raw'].to_numpy(float), LAM) / LAM)
        resultant = np.abs(pd.Series(phasor.real).rolling(W, min_periods=W).mean().to_numpy() + 1j * pd.Series(phasor.imag).rolling(W, min_periods=W).mean().to_numpy())
        stats.append(dict(trial=t, link=l, r_moving=np.nanmedian(resultant[tw < tstop_camera[t] - 1.0]) if (tw < tstop_camera[t] - 1.0).sum() > W else np.nan,
                          r_stopped_p05=np.nanpercentile(resultant[tw > tstop_camera[t] + 1.0], 5), r_stopped_median=np.nanmedian(resultant[tw > tstop_camera[t] + 1.0])))
        stationary, run_length, since = False, 0, np.nan
        for i in range(len(tw)):
            r = resultant[i]
            if not stationary:
                run_length = run_length + 1 if r >= R_ON else 0
                if run_length >= HOLD: stationary, since = True, tw[i]
            elif r < R_OFF:
                stationary, run_length, since = False, 0, np.nan
        declared.append(since)
    tstop_uwb[t] = float(np.nanmax(declared)) if not np.all(np.isnan(declared)) else np.nan
S = pd.DataFrame(stats)
print('resultant length of 25 phase samples: moving median %.2f, stopped median %.2f, stopped p05 %.2f' % (S.r_moving.median(), S.r_stopped_median.median(), S.r_stopped_p05.median()))
delay = pd.Series({t: tstop_uwb[t] - tstop_camera[t] for t in runs})
print('declared stop minus camera stop [s]:', delay.round(2).to_dict())
print('median %.2f s, range %.2f to %.2f, not declared: %d' % (delay.median(), delay.min(), delay.max(), int(delay.isna().sum())))
runs = [t for t in runs if np.isfinite(tstop_uwb[t])]
tstop = tstop_uwb
rest = rest.replace('hp_fixed_links.csv', 'hp_fixed_causal_links.csv').replace("'stop_decomposition_hp_stops.csv'", "'stop_decomposition_hp_stops_causal.csv'").replace('stop_decomposition_hp_stops_summary.csv', 'stop_decomposition_hp_stops_causal_summary.csv')
exec('step_rows = []' + rest)

