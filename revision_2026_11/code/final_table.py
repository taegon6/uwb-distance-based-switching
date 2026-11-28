
import pandas as pd, numpy as np, glob
E=pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/segment_errors.csv')
t=E.groupby(['path','seg','policy']).agg(runs=('trial','nunique'),rate=('rate_hz','median'),lat_bias=('log_lat_bias','median'),lat_sd=('log_lat_sd','median'),lat_rmse=('log_lat_rmse','median'),lon_rmse=('log_lon_rmse','median'),cmn_lon=('cmn_lon_rmse','median'),cmn_lat=('cmn_lat_rmse','median')).round(1)
pd.set_option('display.width',200); print(t.to_string())
for path in ['S','C1','C2']:
    mx=[]
    for t_ in E[E.path==path].trial.unique():
        p=pd.read_csv(r'D:/sandbox/uwb/output/field_20261124/cam/eval/pairs_%s.csv'%t_); p=p[p.cam_X.notna()]
        mx.append(p.d_c.max())
    print(path,'camera-covered max d_c median %.1f min %.1f'%(np.median(mx),np.min(mx)))

