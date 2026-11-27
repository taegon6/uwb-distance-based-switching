import io, numpy as np, pandas as pd
from pathlib import Path
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE/'results_20261124'
jobs = pd.read_csv(BASE/'cam'/'jobs.csv').set_index('trial'); st = pd.read_csv(BASE/'analysis'/'stops_v2_seltos_L439.csv').set_index('trial')
sched = pd.read_csv(r'D:/sandbox/uwb/output/exp_ready_20261124/repo/docs/revision_schedule_20261124.csv').set_index('trial')
def rd(p, cols=None):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False, usecols=cols)
rows=[]
for t in [x for x in sched.index if sched.loc[x,'policy'] in ('FIXED_PHASE','FIXED_DS') and sched.loc[x,'path']=='S']:
    lk = rd(DAY/jobs.loc[t,'run']/'tables'/'links.csv', ['host_monotonic_ns','link_id','success','ds_range_m','phase_corrected_range_m','firmware_n','phase_valid'])
    lk = lk[lk.success.astype(str).isin(['1','True','true'])]
    end = lk.host_monotonic_ns.astype('int64').max(); w = lk[lk.host_monotonic_ns.astype('int64') > end-3e9]
    r = dict(trial=t, policy=sched.loc[t,'policy'])
    for l, g in w.groupby('link_id'):
        r[l+'_ds_sd_mm'] = 1000*g.ds_range_m.astype(float).std()
        if sched.loc[t,'policy']=='FIXED_PHASE':
            n = g.firmware_n.astype(float)
            r[l+'_N_vals'] = '/'.join(str(int(v)) for v in sorted(n.unique()))
            r[l+'_hp_sd_mm'] = 1000*g.phase_corrected_range_m.astype(float).std()
    cam = pd.read_csv(BASE/'cam'/'eval'/f'camtrack_{t}.csv'); cam = cam[cam.t > cam.t.max()-1.5]
    r['plumb_yaw'] = round(st.loc[t,'ref_yaw'],2); r['uwb_yaw'] = round(st.loc[t,'sfit_yaw'],2); r['cam_yaw'] = round(float(np.median(cam.psi)),2)
    r['rx_err'] = round(st.loc[t,'sfit_rx_err_cm'],1)
    rows.append(r)
D = pd.DataFrame(rows); pd.set_option('display.width',260); pd.set_option('display.max_columns',40)
print(D[['trial','policy','plumb_yaw','cam_yaw','uwb_yaw','rx_err']].to_string(index=False))
hp = D[D.policy=='FIXED_PHASE']
print(hp[['trial']+[c for c in D.columns if c.endswith('_N_vals')]].to_string(index=False))
print(hp[['trial']+[c for c in D.columns if c.endswith('_hp_sd_mm') or c.endswith('_ds_sd_mm')]].round(2).to_string(index=False))

