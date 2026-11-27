"""Replay of the phase-based stop detector over ALL high-precision samples of each record (no crop before the camera stop).
Per link: resultant length of the last 25 wrapped-phase samples, declared at >= 0.6 for 13 consecutive samples, withdrawn below 0.4.
Camera gives the state of the vehicle: speed (smoothed) and distance to the final position."""
import io, contextlib, math, numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/stop_convergence.py', encoding='utf-8').read().split('runs = [t for t in st.index')[0]
exec(src)
LAM = 299792458.0 / (2 * 7.9872e9); W = 25; R_ON, R_OFF, HOLD = 0.6, 0.4, 13; V_MOVE = 0.05
runs = [t for t in st.index if st.loc[t, 'stop_mode'] == 'PHASE' and (EV / f'camtrack_{t}.csv').exists()]
link_rows, veh_rows, ev_rows = [], [], []
for t in runs:
    c = pd.read_csv(EV / f'camtrack_{t}.csv').sort_values('t'); tt = c.t.to_numpy()
    X = c.X.rolling(9, center=True, min_periods=3).median(); Y = c.Y.rolling(9, center=True, min_periods=3).median()
    fin = np.array([X.iloc[-30:].median(), Y.iloc[-30:].median()]); dist = np.hypot(X - fin[0], Y - fin[1]).to_numpy()
    far = np.flatnonzero(dist > 0.01); tstop = float(tt[far[-1] + 1]) if len(far) and far[-1] + 1 < len(c) else float(tt[0])
    Xs = c.X.rolling(15, center=True, min_periods=5).mean(); Ys = c.Y.rolling(15, center=True, min_periods=5).mean()
    speed = np.hypot(np.gradient(Xs, tt), np.gradient(Ys, tt))
    lk = rd(DAY / jobs.loc[t, 'run'] / 'tables' / 'links.csv', ['host_wall_ns', 'link_id', 'success', 'phase_corrected_range_m', 'phase_valid'])
    lk = lk[lk.success.astype(str).isin(['1', 'True', 'true']) & lk.phase_valid.astype(str).isin(['1', 'True', 'true'])].copy(); lk['tw'] = lk.host_wall_ns.astype('int64') / 1e9
    t0 = max(lk.tw.min(), tt.min()); t1 = min(lk.groupby('link_id').tw.max().min(), tt.max())
    grid = np.arange(t0, t1, 0.02); nl = np.zeros(len(grid), int)
    gs = np.interp(grid, tt, speed); gd = np.interp(grid, tt, dist)
    for l in LINKS:
        g = lk[lk.link_id == l].sort_values('tw'); tw = g.tw.to_numpy(); ph = np.exp(1j * 2 * np.pi * np.mod(g.phase_corrected_range_m.to_numpy(float), LAM) / LAM)
        res = np.abs(pd.Series(ph.real).rolling(W, min_periods=W).mean().to_numpy() + 1j * pd.Series(ph.imag).rolling(W, min_periods=W).mean().to_numpy())
        s = np.zeros(len(tw), bool); stationary, run_length, since = False, 0, None; ndecl = 0
        for i in range(len(tw)):
            r = res[i]
            if not stationary:
                run_length = run_length + 1 if r >= R_ON else 0
                if run_length >= HOLD: stationary, since = True, tw[i]; ndecl += 1
            elif r < R_OFF:
                v0 = float(np.interp(since, tt, speed)); d0 = float(np.interp(since, tt, dist))
                seg = (tt >= since) & (tt <= tw[i]); vmax = float(speed[seg].max()) if seg.any() else v0
                ev_rows.append(dict(trial=t, link=l, declared_rel_s=since - tstop, held_s=tw[i] - since, speed_at_declaration=v0, max_speed_while_declared=vmax, dist_at_declaration_m=d0, moved_m=abs(d0 - float(np.interp(tw[i], tt, dist))), withdrawn=True))
                stationary, run_length = False, 0
            s[i] = stationary
        if stationary:
            v0 = float(np.interp(since, tt, speed)); d0 = float(np.interp(since, tt, dist)); seg = (tt >= since)
            ev_rows.append(dict(trial=t, link=l, declared_rel_s=since - tstop, held_s=tw[-1] - since, speed_at_declaration=v0, max_speed_while_declared=float(speed[seg].max()) if seg.any() else v0, dist_at_declaration_m=d0, moved_m=d0, withdrawn=False))
        idx = np.clip(np.searchsorted(tw, grid, side='right') - 1, 0, len(tw) - 1); nl += (s[idx] & (grid >= tw[0]))
        link_rows.append(dict(trial=t, link=l, samples=len(tw), declarations=ndecl, declared_at_end=bool(s[-1])))
    moving = gs > V_MOVE; four = nl == 4
    on = np.flatnonzero(four); first = grid[on[0]] if len(on) else np.nan
    # first four-link declaration after the vehicle has started its last approach to the final position (within 0.1 m)
    veh_rows.append(dict(trial=t, policy=sched.loc[t, 'policy'], hp_record_s=t1 - t0, hp_before_stop_s=tstop - t0, moving_time_s=0.02 * int(moving.sum()),
                         time_any_link_while_moving_s=0.02 * int((moving & (nl >= 1)).sum()), max_links_while_moving=int(nl[moving].max()) if moving.any() else 0,
                         time_four_while_moving_s=0.02 * int((moving & four).sum()), max_links_beyond_0p1m=int(nl[gd > 0.1].max()) if (gd > 0.1).any() else 0,
                         time_four_beyond_0p1m_s=0.02 * int(((gd > 0.1) & four).sum()), speed_max_while_four=float(gs[four].max()) if four.any() else np.nan,
                         first_four_near_rel_s=(grid[np.flatnonzero(four & (gd <= 0.1))[0]] - tstop) if (four & (gd <= 0.1)).any() else np.nan,
                         dist_at_first_four_near_mm=1000 * float(gd[np.flatnonzero(four & (gd <= 0.1))[0]]) if (four & (gd <= 0.1)).any() else np.nan,
                         all_four_at_end=bool(four[-1])))
V = pd.DataFrame(veh_rows); E = pd.DataFrame(ev_rows); L = pd.DataFrame(link_rows)
V.round(3).to_csv(OUTR / 'stop_detector_fullstream_vehicle.csv', index=False); E.round(4).to_csv(OUTR / 'stop_detector_fullstream_declarations.csv', index=False); L.to_csv(OUTR / 'stop_detector_fullstream_links.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(V.round(2).to_string(index=False))
mv = E[E.speed_at_declaration > V_MOVE]
print('\ndeclarations %d (links %d, stops %d) | withdrawn %d (links %d, stops %d)' % (len(E), E.groupby(['trial', 'link']).ngroups, E.trial.nunique(), int(E.withdrawn.sum()), E[E.withdrawn].groupby(['trial', 'link']).ngroups, E[E.withdrawn].trial.nunique()))
print('declared while the camera speed exceeded %.2f m/s: %d declarations in stops %s' % (V_MOVE, len(mv), sorted(mv.trial.unique())))
print(mv.round(3).to_string(index=False))
print('\nstops with four links declared while moving: %d | beyond 0.1 m: %d | max links while moving %d | all four at end %d of %d' % (int((V.time_four_while_moving_s > 0).sum()), int((V.time_four_beyond_0p1m_s > 0).sum()), int(V.max_links_while_moving.max()), int(V.all_four_at_end.sum()), len(V)))
print('HP record before the camera stop [s]:', V.set_index('trial').hp_before_stop_s.round(1).to_dict())

