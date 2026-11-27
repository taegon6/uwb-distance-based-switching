"""Batch camera tracking for all evaluated runs.
    python batch_track.py jobs            # write jobs.csv
    python batch_track.py work K N        # process jobs i % N == K
"""
import sys, csv, io, datetime, subprocess, json, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
BASE = Path(r'D:/sandbox/uwb/output/field_20261124'); DAY = BASE / 'results_20261124'; VID = BASE / 'video'; OUT = BASE / 'cam' / 'tracks'
FFP = r'D:/sandbox/uwb/output/laptop_handoff_20261124/ffmpeg/ffprobe.exe'

def rd(p):
    t = p.read_text(encoding='utf-8', errors='replace'); t = '\n'.join(l for l in t.splitlines() if '\x00' not in l)
    return pd.read_csv(io.StringIO(t), on_bad_lines='skip', low_memory=False)

def jobs():
    st = pd.read_csv(DAY / 'stops_20261124.csv').drop_duplicates('trial', keep='last')
    vids = []
    for p in sorted(VID.glob('*.mp4')):
        m = re.search(r'(\d{8})_(\d{6})', p.stem)
        s = datetime.datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S').timestamp()
        dur = float(json.loads(subprocess.run([FFP, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(p)], capture_output=True).stdout)['format']['duration'])
        vids.append((p.name, s, dur))
    rows = []
    for r in st.itertuples():
        fr = rd(DAY / r.run_dir / 'tables' / 'frames.csv')
        w0, w1 = fr.frame_start_host_wall_ns.min() / 1e9, fr.frame_end_host_wall_ns.max() / 1e9
        best = max(vids, key=lambda v: min(w1, v[1] + v[2]) - max(w0, v[1]))
        t0 = max(0.0, w0 - best[1] - 1.5); t1 = min(best[2] - 0.05, w1 - best[1] + 0.5)
        rows.append(dict(trial=r.trial, run=r.run_dir, video=best[0], video_start=best[1], t0=round(t0, 3), t1=round(t1, 3), uwb_w0=w0, uwb_w1=w1))
    pd.DataFrame(rows).to_csv(BASE / 'cam' / 'jobs.csv', index=False)
    print(pd.DataFrame(rows)[['trial', 'video', 't0', 't1']].to_string(index=False))

def work(k, n):
    import cam_track
    OUT.mkdir(parents=True, exist_ok=True)
    J = pd.read_csv(BASE / 'cam' / 'jobs.csv')
    for i, j in J.iterrows():
        if i % n != k: continue
        out = OUT / f'{j.trial}.csv'
        if out.exists(): continue
        try:
            cam_track.run(VID / j.video, float(j.t0), float(j.t1), str(out))
        except Exception as e:
            print(j.trial, 'ERROR', repr(e), flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'jobs': jobs()
    else: work(int(sys.argv[2]), int(sys.argv[3]))

