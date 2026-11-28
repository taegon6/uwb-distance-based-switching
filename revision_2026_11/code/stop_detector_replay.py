"""Behaviour of the phase-based stop detector over the whole high-precision part of each record (not only the last declaration).
Counts the declarations of each link and those that came before the stop indicated by the camera."""
import numpy as np, pandas as pd, io, contextlib
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_fixed.py', encoding='utf-8').read()
prelude, rest = src.split('step_rows = []')
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(prelude)
R_ON, R_OFF, HOLD = 0.6, 0.4, 13
rows = []
for t in runs:
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy()
        phasor = np.exp(1j * 2 * np.pi * np.mod(g['HP raw'].to_numpy(float), LAM) / LAM)
        res = np.abs(pd.Series(phasor.real).rolling(W, min_periods=W).mean().to_numpy() + 1j * pd.Series(phasor.imag).rolling(W, min_periods=W).mean().to_numpy())
        stationary, run_length, decl, withdrawn = False, 0, [], 0
        for i in range(len(tw)):
            r = res[i]
            if not stationary:
                run_length = run_length + 1 if r >= R_ON else 0
                if run_length >= HOLD: stationary = True; decl.append(tw[i])
            elif r < R_OFF:
                stationary, run_length = False, 0; withdrawn += 1
        early = [d for d in decl if d < tstop[t]]
        rows.append(dict(trial=t, link=l, record_s=tw.max() - tw.min(), before_stop_s=tstop[t] - tw.min(), declarations=len(decl), withdrawn=withdrawn, declared_before_camera_stop=len(early),
                         earliest_lead_s=(tstop[t] - min(early)) if early else np.nan, final_delay_s=(decl[-1] - tstop[t]) if decl else np.nan, stationary_at_end=stationary))
D = pd.DataFrame(rows); D.round(3).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/stop_detector_replay.csv', index=False)
pd.set_option('display.width', 220)
R = D.groupby('trial').agg(declarations=('declarations', 'sum'), withdrawn=('withdrawn', 'sum'), early=('declared_before_camera_stop', 'sum'), before_stop_s=('before_stop_s', 'first'), record_s=('record_s', 'first'), final_delay=('final_delay_s', 'max'), all_end=('stationary_at_end', 'all'))
print(R.round(2).to_string())
print('links %d, stops %d | links with one declaration only: %d | links with a withdrawn declaration: %d (stops %d) | links declared before the camera stop: %d (stops %d) | all four links stationary at the end: %d stops' % (
    len(D), D.trial.nunique(), int((D.declarations == 1).sum()), int((D.withdrawn > 0).sum()), D[D.withdrawn > 0].trial.nunique(), int((D.declared_before_camera_stop > 0).sum()), D[D.declared_before_camera_stop > 0].trial.nunique(), int(R.all_end.sum())))
print('high-precision record before the camera stop [s]: median %.1f, range %.1f to %.1f' % (R.before_stop_s.median(), R.before_stop_s.min(), R.before_stop_s.max()))
print('final delay per stop [s]: median %.2f range %.2f to %.2f' % (R.final_delay.median(), R.final_delay.min(), R.final_delay.max()))

