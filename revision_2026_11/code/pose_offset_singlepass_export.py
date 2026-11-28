"""Per-stop export of the single-pass pose-dependent offset calibration (same code as pose_offset_singlepass.py)."""
import io, contextlib, numpy as np, pandas as pd
src = open(r'D:/sandbox/uwb/output/field_20261124/cam/pose_offset_singlepass.py', encoding='utf-8').read()
buf = io.StringIO()
with contextlib.redirect_stdout(buf): exec(src)
print(buf.getvalue()[-1500:])
out = []
for sname, train_of in splits.items():
    base = run(train_of, FEATURES['constant'], 0); D = run(train_of, full, 1)
    for t in base.index:
        out.append(dict(split=sname, trial=t, constant_tag_cm=base.tag[t], constant_yaw_deg=base.yaw[t], constant_rx_cm=base.rx[t], model_tag_cm=D.tag[t], model_yaw_deg=D.yaw[t], model_rx_cm=D.rx[t]))
    print(sname, '| constant: tag %.3f yaw %.3f rx %.3f le10 %d | model: tag %.3f yaw %.3f rx %.3f le10 %d max %.1f | n %d' % (base.tag.median(), base.yaw.median(), base.rx.median(), int((base.rx <= 10).sum()),
          D.tag.median(), D.yaw.median(), D.rx.median(), int((D.rx <= 10).sum()), D.rx.max(), len(D)), '| tag le10 constant %d model %d' % (int((base.tag <= 10).sum()), int((D.tag <= 10).sum())))
pd.DataFrame(out).round(4).to_csv(r'D:/sandbox/uwb/output/field_20261124/analysis/revision_checks/pose_dependent_offset_singlepass.csv', index=False)

