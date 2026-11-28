"""Figure 6 with a neutral legend entry for option A (the figure itself is unchanged): fig/route_fig6_neutral.png. The per-run figures go to fig/neutral_tmp and are not used."""
from pathlib import Path
p = Path(r'D:/sandbox/uwb/output/field_20261124/cam/plot_routes.py'); s = p.read_text(encoding='utf-8')
a, b, c = "label='Watchdog stream restart'", "FIG / f'route_fig6.{ext}'", "FIG / f'{name}.{ext}'"
assert s.count(a) == 1 and s.count(b) == 1 and s.count(c) == 1
(Path(r'D:/sandbox/uwb/output/field_20261124/fig') / 'neutral_tmp').mkdir(exist_ok=True)
s = s.replace(a, "label='Tags started again by the PC'").replace(b, "FIG / f'route_fig6_neutral.{ext}'").replace(c, "FIG / 'neutral_tmp' / f'{name}.{ext}'").replace("print('saved route_fig6')", "print('saved route_fig6_neutral')")
exec(compile(s, str(p), 'exec'), {'__name__': '__main__', '__file__': str(p)})

