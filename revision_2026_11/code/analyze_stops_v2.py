"""Stop analysis v2: rear-plumb yaw with Kia Seltos length, sign-flip check of tape entries,
field link offsets (LOO / straight->curve), RX coil errors.

    python analyze_stops_v2.py
"""
from __future__ import annotations
import io, json, math
from pathlib import Path
import numpy as np, pandas as pd

BASE = Path(__file__).parent
DAY = BASE / "results_20261124"; OUT = BASE / "analysis"
GEOM = json.loads((BASE / "configs" / "vehicle_geometry_20261124.json").read_text(encoding="utf-8"))
ANC = {k: np.array(v) for k, v in GEOM["anchors_m"].items()}
TZ = GEOM["tag_heights_m"]["T1"]
BODY = {"T1": np.array([-0.4, 0.0]), "T2": np.array([0.4, 0.0])}
RX_BODY = np.array([0.0, 2.2])
LINKS = ["T1_A1", "T1_A2", "T2_A1", "T2_A2"]
LAB = GEOM["link_offsets_m"]
L_REAR = 4.39     # Kia Seltos (2022- facelift) overall length 4,390 mm: front tag midpoint plumb -> rear bumper plumb
STOP = (0.0, -2.2)

def rot(yaw): c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)); return np.array([[c, -s], [s, c]])
def slant(x, y, yaw):
    R = rot(yaw); out = {}
    for l in LINKS:
        t, a = l.split("_"); p = np.array([x, y]) + R @ BODY[t]
        out[l] = float(np.sqrt(np.sum((p - ANC[a][:2]) ** 2) + (ANC[a][2] - TZ) ** 2))
    return out
def solve(r, x0):
    p = np.array(x0, float)
    for _ in range(40):
        f = slant(*p); fv = np.array([f[l] for l in LINKS]); res = fv - np.array([r[l] for l in LINKS])
        J = np.zeros((4, 3)); h = [1e-5, 1e-5, 1e-3]
        for k in range(3):
            q = p.copy(); q[k] += h[k]; g = slant(*q); J[:, k] = (np.array([g[l] for l in LINKS]) - fv) / h[k]
        dp = np.linalg.lstsq(J, -res, rcond=None)[0]; p += dp
        if np.abs(dp).max() < 1e-8: break
    return p
def rd(p):
    t = p.read_text(encoding="utf-8", errors="replace"); t = "\n".join(l for l in t.splitlines() if "\x00" not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines="skip", low_memory=False)
def ref_from(lat, lon, rear):
    x, y = STOP[0] + lat / 100, STOP[1] + lon / 100
    s = (x - rear / 100) / L_REAR
    return x, y, math.degrees(math.asin(max(-1, min(1, s))))
def rx(x, y, yaw): return np.array([x, y]) + rot(yaw) @ RX_BODY

st = pd.read_csv(DAY / "stops_20261124.csv").drop_duplicates("trial", keep="last").reset_index(drop=True)
meas = {}
for _, s in st.iterrows():
    lk = rd(DAY / s["run_dir"] / "tables" / "links.csv")
    lk = lk[lk["success"].astype(str).isin(["1", "True", "true"])]
    end = lk["host_monotonic_ns"].astype("int64").max(); last = lk[lk["host_monotonic_ns"].astype("int64") > end - 3e9]
    src = str(last["range_source"].iloc[-1]); fm = "SS" if src.startswith("SS") else ("DS" if src.startswith("DS") else "PHASE")
    m = {}
    for l in LINKS:
        g = last[last["link_id"] == l]
        if fm == "PHASE":
            g = g[g["phase_valid"].astype(str).isin(["1", "True", "true"])]; m[l] = float(g["phase_corrected_range_m"].astype(float).median())
        else:
            m[l] = float(g["range_corrected_m"].astype(float).median())
    meas[s["trial"]] = (fm, m, s["run_dir"].split("_REV_")[1].split("_")[0])

def implied(trials, refs):
    rows = []
    for t in trials:
        fm, m, path = meas[t]; x, y, yaw = refs[t]; e = slant(x, y, yaw)
        for l in LINKS: rows.append({"trial": t, "path": path, "mode": fm, "link": l, "off": m[l] - e[l]})
    return pd.DataFrame(rows)

def evaluate(refs, tag):
    I = implied(list(refs), refs); out = []
    for t in refs:
        fm, m, path = meas[t]
        oth = I[(I.trial != t)]
        loo = oth[oth["mode"] == fm].groupby("link")["off"].median()
        sfit = oth[oth.path == "S"].groupby("link")["off"].median()
        x, y, yaw = refs[t]; R = rx(x, y, yaw)
        row = {"trial": t, "path": path, "stop_mode": fm, "ref_x": x, "ref_y": y, "ref_yaw": yaw,
               "rx_ref_x_cm": 100 * R[0], "rx_ref_y_cm": 100 * R[1]}
        zero = pd.Series({l: 0.0 for l in LINKS})
        for name, off in (("none", zero), ("lab", pd.Series(LAB[fm])), ("loo", loo), ("sfit", sfit)):
            p = solve({l: m[l] - off[l] for l in LINKS}, (x, y, yaw))
            row[f"{name}_dx_cm"] = 100 * (p[0] - x); row[f"{name}_dy_cm"] = 100 * (p[1] - y)
            row[f"{name}_err_cm"] = 100 * math.hypot(p[0] - x, p[1] - y); row[f"{name}_yaw"] = p[2]
            row[f"{name}_dyaw"] = p[2] - yaw
            Ru = rx(*p); row[f"{name}_rx_err_cm"] = 100 * float(np.linalg.norm(Ru - R))
            row[f"{name}_rx_dlat_cm"] = 100 * (Ru[0] - R[0]); row[f"{name}_rx_dlon_cm"] = 100 * (Ru[1] - R[1])
        out.append(row)
    return pd.DataFrame(out), I

# 1) original entries
orig = {s.trial: ref_from(s.lateral_cm, s.long_cm, s.rear_lat_cm) for s in st.itertuples()}
E0, _ = evaluate(orig, "orig")
# 2) sign-flip check against the cross-validated UWB estimate (curves: straight-fit, straight: LOO)
flips = []
for s in st.itertuples():
    e = E0[E0.trial == s.trial].iloc[0]; key = "sfit" if e.path != "S" else "loo"
    ux = e.ref_x + e[f"{key}_dx_cm"] / 100; uy = e.ref_y + e[f"{key}_dy_cm"] / 100; uyaw = e[f"{key}_yaw"]
    base_pos = math.hypot(ux - e.ref_x, uy - e.ref_y) * 100; base_yaw = abs(uyaw - e.ref_yaw)
    for f, (lat, lon, rear) in {"lateral": (-s.lateral_cm, s.long_cm, s.rear_lat_cm),
                                "long": (s.lateral_cm, -s.long_cm, s.rear_lat_cm),
                                "rear": (s.lateral_cm, s.long_cm, -s.rear_lat_cm)}.items():
        if (f == "lateral" and s.lateral_cm == 0) or (f == "long" and s.long_cm == 0) or (f == "rear" and s.rear_lat_cm == 0):
            continue
        x, y, yaw = ref_from(lat, lon, rear)
        # a lateral flip also changes yaw (yaw uses lateral - rear), so judge it on position AND yaw
        pos = math.hypot(ux - x, uy - y) * 100; dyaw = abs(uyaw - yaw)
        if f == "rear":   # UWB yaw error is ~2-3 deg typical, so only a clear disagreement counts
            better = base_yaw > 6 and dyaw < 2
        else:
            better = base_pos > 8 and pos < base_pos / 2 and dyaw <= base_yaw + 1
        flips.append({"trial": s.trial, "field": f, "pos_err_before_cm": round(base_pos, 1), "pos_err_after_cm": round(pos, 1),
                      "yaw_err_before_deg": round(base_yaw, 1), "yaw_err_after_deg": round(dyaw, 1), "flip": better})
F = pd.DataFrame(flips)
# T34 rear: UWB yaw (4.1 deg) and the camera board heading (4.0 deg, same direction) both disagree
# with the entered rear-plumb sign; flipping makes both agree -> treated as an entry error.
F.loc[(F.trial == "T34") & (F.field == "rear"), "flip"] = True
chosen = F[F.flip]
# 3) apply flips, re-evaluate (offsets re-fit with corrected references)
fixed = {}
for s in st.itertuples():
    lat, lon, rear = s.lateral_cm, s.long_cm, s.rear_lat_cm
    for f in chosen[chosen.trial == s.trial]["field"]:
        if f == "lateral": lat = -lat
        if f == "long": lon = -lon
        if f == "rear": rear = -rear
    fixed[s.trial] = ref_from(lat, lon, rear)
E1, I1 = evaluate(fixed, "fixed")
OUT.mkdir(exist_ok=True)
F.to_csv(OUT / "stop_flip_check.csv", index=False, encoding="utf-8-sig")
E1.round(3).to_csv(OUT / "stops_v2_seltos_L439.csv", index=False, encoding="utf-8-sig")
I1.to_csv(OUT / "stop_implied_link_offsets_v2.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("flips chosen:"); print(chosen.to_string(index=False) if len(chosen) else "none")
print("\nall flip candidates with before>8cm or yaw>4deg:")
print(F[(F.pos_err_before_cm > 8) | (F.yaw_err_before_deg > 4)].to_string(index=False))
print("\nfield offsets (median of stop-implied, cm) by mode:")
print((I1.groupby(["mode", "link"])["off"].median() * 100).round(1).unstack())
cols = ["trial", "path", "stop_mode", "ref_yaw", "sfit_yaw", "sfit_dyaw", "lab_err_cm", "loo_err_cm", "sfit_err_cm", "sfit_rx_err_cm", "rx_ref_x_cm", "rx_ref_y_cm"]
print(E1[cols].round(1).to_string(index=False))
for k in ("lab", "loo", "sfit"):
    g = E1.groupby("path")
    print(f"{k}: tag mid median {g[k+'_err_cm'].median().round(1).to_dict()}  <=10 {int((E1[k+'_err_cm']<=10).sum())}/{len(E1)} | "
          f"RX median {g[k+'_rx_err_cm'].median().round(1).to_dict()} <=10 {int((E1[k+'_rx_err_cm']<=10).sum())}/{len(E1)} | yaw |d| median {E1[k+'_dyaw'].abs().median():.2f} deg")
a = E1[["rx_ref_x_cm", "rx_ref_y_cm"]]
print("\nactual RX-TX alignment (tape+plumb): lateral |x| median %.1f cm max %.1f, longitudinal |y| median %.1f max %.1f; within SAE +-10 lat & +-7.5 lon: %d/%d" % (
    a.rx_ref_x_cm.abs().median(), a.rx_ref_x_cm.abs().max(), a.rx_ref_y_cm.abs().median(), a.rx_ref_y_cm.abs().max(),
    int(((a.rx_ref_x_cm.abs() <= 10) & (a.rx_ref_y_cm.abs() <= 7.5)).sum()), len(a)))

