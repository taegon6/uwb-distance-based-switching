"""Early declarations of the phase-based stop detector: where was the vehicle according to the camera?"""
import numpy as np, pandas as pd, io, contextlib
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/hp_fixed.py', encoding='utf-8').read()
prelude, rest = src.split('step_rows = []')
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(prelude)
R_ON, R_OFF, HOLD = 0.6, 0.4, 13
rows = []
for t in runs:
    c = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t')
    X = c.X.rolling(9, center=True, min_periods=3).median(); Y = c.Y.rolling(9, center=True, min_periods=3).median(); fin = np.array([X.iloc[-30:].median(), Y.iloc[-30:].median()])
    dist = np.hypot(X - fin[0], Y - fin[1]).to_numpy(); tt = c.t.to_numpy()
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy()
        phasor = np.exp(1j * 2 * np.pi * np.mod(g['HP raw'].to_numpy(float), LAM) / LAM)
        res = np.abs(pd.Series(phasor.real).rolling(W, min_periods=W).mean().to_numpy() + 1j * pd.Series(phasor.imag).rolling(W, min_periods=W).mean().to_numpy())
        stationary, run_length, since = False, 0, None
        for i in range(len(tw)):
            r = res[i]
            if not stationary:
                run_length = run_length + 1 if r >= R_ON else 0
                if run_length >= HOLD: stationary, since = True, tw[i]
            elif r < R_OFF:
                d0 = float(np.interp(since, tt, dist)); d1 = float(np.interp(tw[i], tt, dist))
                rows.append(dict(trial=t, link=l, declared_rel_s=since - tstop[t], withdrawn_rel_s=tw[i] - tstop[t], held_s=tw[i] - since, dist_at_declaration_mm=1000 * d0, dist_at_withdrawal_mm=1000 * d1, moved_while_declared_mm=1000 * abs(d0 - d1)))
                stationary, run_length = False, 0
D = pd.DataFrame(rows); pd.set_option('display.width', 220)
print(D.round(2).to_string(index=False))
D.round(3).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/stop_detector_withdrawn_declarations.csv', index=False)
early = D[D.declared_rel_s < 0]; late = D[D.declared_rel_s >= 0]
print('withdrawn declarations: %d (links %d, stops %d) | declared before the camera stop: %d, distance to the final position at the declaration: median %.0f mm, max %.0f mm; movement while declared: median %.0f mm, max %.0f mm' % (
    len(D), D.groupby(['trial', 'link']).ngroups, D.trial.nunique(), len(early), early.dist_at_declaration_mm.median(), early.dist_at_declaration_mm.max(), early.moved_while_declared_mm.median(), early.moved_while_declared_mm.max()))
print('declared after the camera stop and withdrawn: %d (stops %s), held median %.1f s' % (len(late), sorted(late.trial.unique()), late.held_s.median() if len(late) else float('nan')))

