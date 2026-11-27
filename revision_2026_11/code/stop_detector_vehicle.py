"""Vehicle-level state of the phase-based stop detector (all four links declared) over the high-precision part of each record."""
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
    grid = np.arange(min(data[t][l].tw.min() for l in LINKS), min(data[t][l].tw.max() for l in LINKS), 0.02); state = np.ones(len(grid), bool); nlinks = np.zeros(len(grid), int); end_state = {}
    for l in LINKS:
        g = data[t][l]; tw = g.tw.to_numpy()
        phasor = np.exp(1j * 2 * np.pi * np.mod(g['HP raw'].to_numpy(float), LAM) / LAM)
        res = np.abs(pd.Series(phasor.real).rolling(W, min_periods=W).mean().to_numpy() + 1j * pd.Series(phasor.imag).rolling(W, min_periods=W).mean().to_numpy())
        s = np.zeros(len(tw), bool); stationary, run_length = False, 0
        for i in range(len(tw)):
            r = res[i]
            if not stationary:
                run_length = run_length + 1 if r >= R_ON else 0
                if run_length >= HOLD: stationary = True
            elif r < R_OFF: stationary, run_length = False, 0
            s[i] = stationary
        idx = np.clip(np.searchsorted(tw, grid, side='right') - 1, 0, len(tw) - 1); sg = s[idx] & (grid >= tw[0]); state &= sg; nlinks += sg; end_state[l] = bool(s[-1])
    on = np.flatnonzero(state); edges = np.flatnonzero(np.diff(state.astype(int)) == 1) + 1
    starts = ([0] if state[0] else []) + edges.tolist()
    first = grid[on[0]] if len(on) else np.nan
    pre = grid < tstop[t]
    rows.append(dict(trial=t, vehicle_declarations=len(starts), first_all_four_rel_s=first - tstop[t], dist_at_first_mm=1000 * float(np.interp(first, tt, dist)) if len(on) else np.nan,
                     all_four_at_end=bool(state[-1]), links_declared_at_end=sum(end_state.values()), time_all_four_before_stop_s=0.02 * int((state & pre).sum()),
                     max_links_while_far=int(nlinks[pre & (np.interp(grid, tt, dist) > 0.1)].max()) if (pre & (np.interp(grid, tt, dist) > 0.1)).any() else 0))
D = pd.DataFrame(rows); pd.set_option('display.width', 220); print(D.round(2).to_string(index=False))
D.round(3).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/stop_detector_vehicle_level.csv', index=False)
e = D[D.first_all_four_rel_s < -0.5]
print('stops with all four links declared more than 0.5 s before the camera stop: %d %s, lead %s s, distance to the final position %s mm' % (len(e), e.trial.tolist(), e.first_all_four_rel_s.round(1).tolist(), e.dist_at_first_mm.round(0).tolist()))
print('stops with all four links declared at the end: %d of %d | never all four: %d | largest number of links declared while more than 0.1 m from the final position: %d' % (int(D.all_four_at_end.sum()), len(D), int(D.first_all_four_rel_s.isna().sum()), int(D.max_links_while_far.max())))

