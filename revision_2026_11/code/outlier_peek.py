
import pandas as pd, numpy as np, glob, os
E=pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/segment_errors.csv')
sc=E.drop_duplicates('trial').set_index('trial')
rows=[]
for f in glob.glob(r'D:/sandbox/uwb/output/field_20261124/cam/eval/pairs_T*.csv'):
    t=os.path.basename(f)[6:-4]; p=pd.read_csv(f); p['trial']=t; rows.append(p)
P=pd.concat(rows); P['policy']=P.trial.map(sc.policy); P['path']=P.trial.map(sc.path)
print('residual rms quantiles (m):', P.range_residual_rms_m.quantile([.5,.9,.99,.999]).round(3).to_dict())
P['abs_lat']=P.log_lat.abs()*100
print(P.groupby(['path','policy']).abs_lat.quantile([.5,.99]).unstack().round(1))
print('sync corr:', sc.sync_corr.round(3).sort_values().head(5).to_dict())
print(P.groupby(['path','policy']).apply(lambda g: pd.Series({'n':len(g),'res>0.10':(g.range_residual_rms_m>0.10).sum(),'res>0.05':(g.range_residual_rms_m>0.05).sum()})))

