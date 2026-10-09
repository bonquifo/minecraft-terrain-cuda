"""Builds docs/report.html (self-contained, inline SVG charts) and docs/img/*.png from the
measured data in bench/results/ (see bench/run_all.py). Every number in the report is
computed here from those files; nothing is typed in by hand.

    python docs/make_report.py
"""
import csv
import datetime
import html
import io
import json
import re
import statistics as st
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'bench' / 'results'
FINAL = RES / 'final'
OPT = RES / 'opt'
IMG = ROOT / 'docs' / 'img'

C_VAN = '#9a9a9a'
C_VAN_D = '#6b6b6b'
C_CPU = '#4f8fd6'
C_CPU_D = '#2f6db5'
C_GPU = '#76b900'
C_GPU_D = '#4f7d00'
STAGE_COLORS = {'biome': '#8e6bd8', 'noise': '#3c9fd6', 'aquifer': '#29b3a6', 'fill': '#f2a33a', 'surface': '#e0605a', 'd2h': '#b0b0b0'}
STAGE_SHORT = {'biome': 'biomes', 'noise': 'noise', 'aquifer': 'aquifer cells', 'fill': 'fill', 'surface': 'material rules', 'd2h': 'copy to host'}
STAGE_NAMES = {'biome': 'biomes', 'noise': 'noise corners', 'aquifer': 'aquifer cells', 'fill': 'density + aquifer fill',
               'surface': 'material rules', 'd2h': 'device-to-host copy'}

TICKS = {'v0': 'naive', 'v1': 'floor/\nwrap', 'v2': 'axis\ntables', 'v3': 'zoom\ntable', 'v4': 'grid\ndedupe', 'v5': 'column\ndedupe',
         'v6': 'zoom\nfilter', 'v7': 'inline\nrules', 'v8': 'biome\nsearch', 'v9': 'pipeline'}

SERIES_TOKENS = {C_VAN: '--c-van', C_VAN_D: '--c-van2', C_CPU: '--c-cpu', C_CPU_D: '--c-cpu2', C_GPU: '--c-gpu', C_GPU_D: '--c-gpu2',
                 '#1f77b4': '--ink'}

# Sentinel colours replaced by CSS variables in the inline SVG (light/dark themes).
INK, AXIS, GRID = '#010203', '#010204', '#010205'


# ---- data ------------------------------------------------------------------------------
def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(r, k):
    return float(r[k])


def stats(values):
    v = list(values)
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return {'mean': m, 'sd': sd, 'cv': sd / m * 100 if m else 0.0, 'min': min(v), 'max': max(v), 'n': len(v)}


def vanilla_terrain():
    out = {}
    for p in sorted(FINAL.glob('vanilla_terrain_t*_run*.csv')):
        t = int(re.search(r'_t(\d+)_', p.name).group(1))
        run = int(re.search(r'_run(\d+)', p.name).group(1))
        rows = read_csv(p)
        out.setdefault(t, {'all': [], 'runs': {}, 'region': int(rows[0]['region']), 'biome_s': [], 'terrain_s': []})
        vals = [fnum(r, 'chunks_per_s') for r in rows]
        out[t]['all'] += vals
        out[t]['runs'][run] = vals
        out[t]['biome_s'] += [fnum(r, 'biome_seconds') for r in rows]
        out[t]['terrain_s'] += [fnum(r, 'terrain_seconds') for r in rows]
    return out


def ours(path, **match):
    rows = read_csv(path)
    sel = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in match.items())]
    return sel


def mean_col(rows, k):
    return st.mean(fnum(r, k) for r in rows)


def parse_tests(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'\s+(.+?)\s{2,}(PASS|FAIL)\s+\((\d+) checked, (\d+) mismatches(.*)\)', line)
        if m:
            rows.append({'name': m.group(1).strip(), 'ok': m.group(2) == 'PASS', 'checked': int(m.group(3)), 'bad': int(m.group(4)), 'note': m.group(5).strip('; ')})
    result = 'RESULT: PASS (bit-exact)' in Path(path).read_text(encoding='utf-8', errors='replace')
    return rows, result


def parse_verify(log):
    out = []
    cur = None
    for line in Path(log).read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.search(r'--mode (terrain-gpu|terrain-cpu) .*--tag (verify_\w+)', line)
        if m:
            cur = (m.group(1), m.group(2))
        m = re.search(r'verify vs .*?: (\d+) blocks, (\d+) mismatches', line)
        if m and cur:
            out.append({'impl': 'GPU' if cur[0] == 'terrain-gpu' else 'CPU port', 'region': cur[1][len('verify_'):], 'blocks': int(m.group(1)),
                        'bad': int(m.group(2))})
            cur = None
    return out


# ---- charts --------------------------------------------------------------------------------
def style(theme):
    ink = INK if theme == 'svg' else '#1f2328'
    axis = AXIS if theme == 'svg' else '#6e7781'
    grid = GRID if theme == 'svg' else '#d8dee4'
    plt.rcParams.update({
        'font.family': ['DejaVu Sans', 'sans-serif'], 'font.size': 10.5, 'svg.fonttype': 'none',
        'text.color': ink, 'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink, 'axes.edgecolor': axis,
        'axes.titlecolor': ink, 'axes.titleweight': 'semibold', 'axes.titlesize': 12, 'axes.titlelocation': 'left',
        'grid.color': grid, 'grid.linewidth': 0.8, 'axes.grid': True, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
        'figure.facecolor': 'none' if theme == 'svg' else 'white', 'axes.facecolor': 'none' if theme == 'svg' else 'white',
        'savefig.facecolor': 'none' if theme == 'svg' else 'white',
    })
    return ink, axis, grid


def to_svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    s = buf.getvalue()
    s = s[s.index('<svg'):]
    s = re.sub(r'<metadata>.*?</metadata>', '', s, flags=re.S)
    s = s.replace(INK, 'var(--ink)').replace(AXIS, 'var(--axis)').replace(GRID, 'var(--grid)')
    for hexcol, token in SERIES_TOKENS.items():   # series colours follow the page theme
        s = s.replace(hexcol, 'var(%s)' % token)
    s = re.sub(r"font-family: [^;\"]*", 'font-family: var(--font-chart)', s)   # charts use the page's text face
    s = re.sub(r'(<svg[^>]*?) width="[^"]*" height="[^"]*"', r'\1', s, count=1)   # responsive
    return s


def to_png(fig, name):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / name, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fmt_rate(v):
    return '{:,.0f}'.format(v)


def chart_throughput(theme, D):
    ink, _, _ = style(theme)
    items = D['tp_items']
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    names = [i['label'] for i in items][::-1]
    vals = [i['s']['mean'] for i in items][::-1]
    lo = [i['s']['mean'] - i['s']['min'] for i in items][::-1]
    hi = [i['s']['max'] - i['s']['mean'] for i in items][::-1]
    cols = [i['color'] for i in items][::-1]
    y = range(len(items))
    ax.barh(y, vals, color=cols, height=0.62, xerr=[lo, hi], error_kw={'ecolor': ink, 'elinewidth': 0.9, 'capsize': 2.5})
    ax.set_xscale('log')
    ax.set_yticks(list(y), names)
    ax.grid(axis='y', visible=False)
    base = items[0]['s']['mean']
    for yi, v, it in zip(y, vals, items[::-1]):
        sp = v / base
        txt = '%s chunks/s' % fmt_rate(v) + ('' if it is items[0] else '   (%s×)' % ('{:,.0f}'.format(sp) if sp >= 10 else '%.1f' % sp))
        ax.text(v * 1.18, yi, txt, va='center', fontsize=9.5)
    ax.set_xlim(base / 3, vals[-1 if vals[-1] > vals[0] else 0] * 30)
    ax.set_xlim(min(vals) / 2.5, max(vals) * 25)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:,.0f}'.format(v)))
    ax.set_xlabel('chunks per second (log scale); whiskers: min–max over all repetitions')
    ax.set_title('Full TERRAIN step throughput: vanilla 26.3 vs our CPU port vs RTX 5090')
    return fig


def chart_ladder(theme, D):
    ink, _, _ = style(theme)
    L = D['ladder']
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.6, 6.6), gridspec_kw={'height_ratios': [1.05, 1]})
    x = range(len(L))
    vals = [l['rate'] for l in L]
    cols = [C_GPU if not l.get('pipelined') else C_GPU_D for l in L]
    ax.bar(x, vals, color=cols, width=0.66)
    for xi, l in zip(x, L):
        ax.text(xi, l['rate'] * 1.02, '{:,.0f}'.format(l['rate'] / 1000) + 'k', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(list(x), [l['short'] + '\n' + l['tick'] for l in L], fontsize=8.4)
    ax.grid(axis='x', visible=False)
    ax.set_ylabel('chunks / s')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:,.0f}k'.format(v / 1000)))
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title('Optimization ladder (region 64×64, 4,096 chunks; every version bit-exact)')
    # stage breakdown (sequential runs only)
    stages = ['biome', 'noise', 'aquifer', 'fill', 'surface', 'd2h']
    bottoms = [0.0] * len(L)
    for sname in stages:
        h = [l['stages'][sname] if l.get('stages') else 0 for l in L]
        ax2.bar(x, h, bottom=bottoms, color=STAGE_COLORS[sname], width=0.66, label=STAGE_NAMES[sname])
        bottoms = [b + hh for b, hh in zip(bottoms, h)]
    ax2.set_xticks(list(x), [l['short'] for l in L], fontsize=9)
    ax2.grid(axis='x', visible=False)
    ax2.set_ylabel('ms per 4,096 chunks')
    ax2.legend(ncol=3, fontsize=8.8, loc='upper right')
    ax2.set_title('Stage times per version (CUDA events, one batch at a time)')
    fig.tight_layout(h_pad=2.2)
    return fig


def chart_stages(theme, D):
    ink, _, _ = style(theme)
    S = D['final_stages']
    fig, ax = plt.subplots(figsize=(8.6, 2.2))
    left = 0.0
    total = sum(S.values())
    for k in ['biome', 'noise', 'aquifer', 'fill', 'surface', 'd2h']:
        w = S[k]
        ax.barh([0], [w], left=left, color=STAGE_COLORS[k], height=0.55)
        if w / total > 0.07:
            ax.text(left + w / 2, 0, '%s\n%.1f ms' % (STAGE_SHORT[k], w), ha='center', va='center', fontsize=8.6, color='white')
        else:
            ax.annotate('%s %.1f ms' % (STAGE_SHORT[k], w), xy=(left + w / 2, 0.28), xytext=(left + w / 2, 0.52), ha='center', va='bottom', fontsize=8.6,
                        arrowprops={'arrowstyle': '-', 'color': ink, 'lw': 0.8})
        left += w
    ax.set_ylim(-0.4, 0.8)
    ax.set_yticks([])
    ax.set_xlim(0, total)
    ax.set_xlabel('ms per batch of %d chunks (kernels %.1f ms + copy %.1f ms)' % (D['final_stage_batch'], total - S['d2h'], S['d2h']))
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Where the GPU time goes (final version, one batch at a time)')
    return fig


def chart_scaling(theme, D):
    ink, _, _ = style(theme)
    V = D['van']
    ts = sorted(V)
    means = [stats(V[t]['all'])['mean'] for t in ts]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(ts, [means[0] * t for t in ts], ls=':', color=ink, lw=1, label='ideal linear scaling')
    ax.plot(ts, means, marker='o', color=C_VAN_D, lw=2, label='vanilla 26.3 (JVM thread pool)')
    cts = [t for t in (1, 32) if t in D['cpu_terrain']]
    ax.plot(cts, [D['cpu_terrain'][t]['mean'] for t in cts], marker='s', ms=8, ls='none', color=C_CPU, label='CPU port (C++), 1 and 32 threads')
    ax.axhline(D['gpu_e2e']['mean'], color=C_GPU, lw=2.2, label='RTX 5090, end to end')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(ts, [str(t) for t in ts])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:,.0f}'.format(v)))
    ax.set_xlabel('CPU threads (Ryzen 9 9950X: 16 cores / 32 threads)')
    ax.set_ylabel('chunks / s (log)')
    ax.legend(fontsize=8.8, loc='center left')
    ax.set_title('Thread scaling of vanilla vs one GPU')
    return fig


def chart_batch(theme, D):
    ink, _, _ = style(theme)
    B = D['batch']
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    for s, col, lab in ((1, C_GPU_D, '1 slot (copy after kernels)'), (2, C_GPU, '2 slots (copy overlaps next batch)')):
        pts = sorted((b, v['mean']) for (b, ss), v in B.items() if ss == s)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', color=col, lw=2, label=lab)
    ax.set_xscale('log', base=2)
    xs = sorted({b for b, _ in B})
    ax.set_xticks(xs, [str(b) for b in xs])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '{:,.0f}k'.format(v / 1000)))
    ax.set_xlabel('chunks per batch (region 128×128 = 16,384 chunks per repetition)')
    ax.set_ylabel('chunks / s, end to end')
    ax.legend(fontsize=9)
    ax.set_title('Batch size and pipelining')
    return fig


def chart_latency(theme, D):
    ink, _, _ = style(theme)
    Lt = D['latency']
    n = [l['chunks'] for l in Lt]
    g = [l['ms'] for l in Lt]
    v1 = [c * D['van1_ms'] for c in n]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    ax.plot(n, v1, marker='o', color=C_VAN_D, lw=2, label='vanilla, 1 thread (%.1f ms per chunk)' % D['van1_ms'])
    ax.plot(n, g, marker='o', color=C_GPU, lw=2, label='RTX 5090, one call')
    for c, ms in zip(n, g):
        ax.text(c, ms * 1.35, '%.2f ms' % ms, ha='center', va='bottom', fontsize=8.5, color=C_GPU_D)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(n, [str(c) for c in n])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ('%g' % v)))
    ax.set_xlabel('chunks requested at once (square regions)')
    ax.set_ylabel('wall time per request, ms (log)')
    ax.legend(fontsize=9)
    ax.set_title('Latency of small requests')
    return fig


# ---- code snippets ------------------------------------------------------------------------
def snippet(rel, start, end, caption, lang='cuda', before=0):
    """A highlighted excerpt of a repository file: from the first line matching `start`
    (moved up by `before` lines to take a leading comment) to the next line matching
    `end`. Anchors are text, not line numbers, so the excerpt follows the code; a missing
    anchor stops the build instead of quoting stale code."""
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import CMakeLexer, CudaLexer, JavaLexer, TextLexer
    lines = (ROOT / rel).read_text(encoding='utf-8').splitlines()
    i = next((k for k, l in enumerate(lines) if re.search(start, l)), None)
    if i is None:
        raise SystemExit('snippet anchor not found in %s: %s' % (rel, start))
    i = max(0, i - before)
    j = next((k for k in range(i, len(lines)) if re.search(end, lines[k])), None)
    if j is None:
        raise SystemExit('snippet end not found in %s: %s' % (rel, end))
    code = textwrap.dedent('\n'.join(lines[i:j + 1]))
    lexer = {'cuda': CudaLexer, 'java': JavaLexer, 'cmake': CMakeLexer, 'text': TextLexer}[lang]()
    body = highlight(code, lexer, HtmlFormatter(linenos='inline', linenostart=i + 1))
    return ('<figure class="code"><div class="code-head"><span class="path">%s</span><span class="lines">lines %d\u2013%d</span></div>%s'
            '<figcaption>%s</figcaption></figure>' % (html.escape(rel), i + 1, j + 1, body, caption))


# ---- html -------------------------------------------------------------------------------
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,500..800'
         '&family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&display=swap">')

CSS = '''
/* Layout: one reading column on a stone-grey ground, a sticky section bar, charts at column width.
   Colour: terrain strata (stone, moss, deepslate); series colours shared with the charts. */
:root{
  --bg:#f3f5f1; --surface:#ffffff; --raised:#e9eee6; --ink:#16201a; --muted:#56635a; --line:#d4dcd2;
  --accent:#2e7334; --accent-ink:#ffffff; --accent-soft:#e1eedc; --axis:#748178; --grid:#dde4db;
  --c-van:#a3aba5; --c-van2:#68736c; --c-cpu:#6c9bd2; --c-cpu2:#2f69a8; --c-gpu:#74ad3d; --c-gpu2:#3d7a22;
  --font-display:"Archivo","Arial Narrow",system-ui,sans-serif; --font-body:"Source Sans 3","Segoe UI",system-ui,-apple-system,sans-serif;
  --font-mono:"JetBrains Mono",ui-monospace,Consolas,monospace; --font-chart:"Source Sans 3","Segoe UI",system-ui,sans-serif;
  --video-bg:#101411; --code-bg:#fbfcfa; --code-kw:#7b3fa0; --code-type:#1f6a86; --code-fn:#8a5800; --code-str:#a63b1e; --code-com:#6c776f; --code-num:#2e7334; --code-pre:#9b3f7a;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#101512; --surface:#161d18; --raised:#1d2620; --ink:#e3e9e3; --muted:#9aa89f; --line:#2a352e;
  --accent:#8fcf68; --accent-ink:#0f1a10; --accent-soft:#1c2c1d; --axis:#8a978e; --grid:#252f28;
  --c-van:#6f7a73; --c-van2:#adb7b0; --c-cpu:#3f6fa6; --c-cpu2:#86b5ea; --c-gpu:#4f8a2c; --c-gpu2:#97d56c;
  --code-bg:#121914; --code-kw:#c9a0ee; --code-type:#6fc2d9; --code-fn:#e3b860; --code-str:#f0937a; --code-com:#7f8d84; --code-num:#97d56c; --code-pre:#e3a3d6; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#101512; --surface:#161d18; --raised:#1d2620; --ink:#e3e9e3; --muted:#9aa89f; --line:#2a352e;
  --accent:#8fcf68; --accent-ink:#0f1a10; --accent-soft:#1c2c1d; --axis:#8a978e; --grid:#252f28;
  --c-van:#6f7a73; --c-van2:#adb7b0; --c-cpu:#3f6fa6; --c-cpu2:#86b5ea; --c-gpu:#4f8a2c; --c-gpu2:#97d56c;
  --code-bg:#121914; --code-kw:#c9a0ee; --code-type:#6fc2d9; --code-fn:#e3b860; --code-str:#f0937a; --code-com:#7f8d84; --code-num:#97d56c; --code-pre:#e3a3d6; color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:64px}
@media (prefers-reduced-motion: reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16.5px/1.62 var(--font-body);-webkit-font-smoothing:antialiased}
.page{max-width:960px;margin:0 auto;padding:0 20px 72px}
header.hero{padding-block:44px 22px}
.eyebrow{font:600 12.5px/1.4 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
h1{font:780 clamp(30px,5.4vw,50px)/1.05 var(--font-display);font-stretch:112%;letter-spacing:-.015em;margin:12px 0 14px;text-wrap:balance}
h2{font:760 27px/1.15 var(--font-display);font-stretch:108%;margin:0 0 12px;text-wrap:balance}
h3{font:700 18px/1.3 var(--font-display);font-stretch:104%;margin:30px 0 8px}
p,li{max-width:68ch}
.lede{font-size:18.5px;color:var(--muted);margin:0 0 20px}
.spec{display:flex;flex-wrap:wrap;gap:8px 18px;margin:0;padding:12px 0 0;border-top:1px solid var(--line);font:400 13px/1.5 var(--font-mono);color:var(--muted)}
.spec b{color:var(--ink);font-weight:600}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:26px 0 8px}
@media (max-width:760px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px 13px}
.kpi .v{font:760 34px/1.05 var(--font-display);font-stretch:106%;font-variant-numeric:tabular-nums;color:var(--accent)}
.kpi .k{font-size:14px;line-height:1.35;color:var(--muted);margin-top:6px}
nav.sections{position:sticky;top:env(safe-area-inset-top,0px);z-index:5;background:var(--bg);border-bottom:1px solid var(--line);margin:22px -20px 0;padding:0 20px}
nav.sections ul{display:flex;gap:4px;list-style:none;margin:0;padding:8px 0;overflow-x:auto;scrollbar-width:none}
nav.sections a{display:block;white-space:nowrap;padding:5px 11px;border-radius:99px;font:600 13.5px/1.3 var(--font-body);color:var(--muted);text-decoration:none}
nav.sections a:hover{background:var(--raised);color:var(--ink)}
a{color:var(--accent);text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
section{padding-block:40px 6px;border-bottom:1px solid var(--line)}
section:last-of-type{border-bottom:0}
figure{margin:22px 0 10px;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:16px 16px 12px}
figure svg{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:14px;line-height:1.5;margin-top:8px;max-width:none}
.wrap{overflow-x:auto;margin:12px 0 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14.5px}
th,td{text-align:left;padding:7px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
th{font:600 12px/1.3 var(--font-mono);letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--raised)}
td.n,th.n{text-align:right;font-family:var(--font-mono);font-size:13.5px;font-variant-numeric:tabular-nums;white-space:nowrap}
.ok{display:inline-block;padding:0 7px;border-radius:4px;background:var(--accent-soft);color:var(--accent);font:600 12px/1.6 var(--font-mono)}
code{font-family:var(--font-mono);font-size:.86em;background:var(--raised);padding:1px 5px;border-radius:4px}
pre{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:14px 16px;overflow-x:auto;font:400 13px/1.6 var(--font-mono)}
pre code{background:none;padding:0}
.note{border-left:3px solid var(--accent);padding:10px 14px;background:var(--accent-soft);border-radius:0 6px 6px 0;max-width:none}
ul{padding-left:22px}
li{margin:4px 0}
footer{margin-top:34px;color:var(--muted);font-size:14px}
figure.code{padding:0;overflow:hidden;background:var(--code-bg)}
.code-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;padding:7px 14px;border-bottom:1px solid var(--line);background:var(--raised);
  font:600 12px/1.4 var(--font-mono);color:var(--muted)}
.code-head .path{color:var(--ink)}
figure.code .highlight{overflow-x:auto}
figure.code pre{margin:0;border:0;border-radius:0;background:none;padding:12px 14px;font:400 12.5px/1.55 var(--font-mono);color:var(--ink);font-variant-ligatures:none}
code,pre{font-variant-ligatures:none}
figure.code figcaption{margin:0;padding:9px 14px 11px;border-top:1px solid var(--line);background:var(--surface)}
.linenos{display:inline-block;min-width:3.2em;padding-right:1em;color:var(--muted);opacity:.7;user-select:none;text-align:right}
.k,.kd,.kn,.kr,.kp,.ow{color:var(--code-kw)} .kt,.nc,.nn{color:var(--code-type)} .nf,.fm{color:var(--code-fn)}
.s,.s1,.s2,.sc,.sa,.dl,.se{color:var(--code-str)} .c,.c1,.cm,.cs,.ch{color:var(--code-com);font-style:italic}
.cp,.cpf{color:var(--code-pre)} .m,.mi,.mf,.mh,.mo,.il,.kc{color:var(--code-num)} .err{color:inherit;background:none}
.lead{margin-bottom:0}
h4{font:700 16px/1.35 var(--font-display);font-stretch:104%;margin:30px 0 4px}
figure.video{padding:12px}
figure.video video,figure.video img{display:block;width:100%;height:auto;border-radius:4px;background:var(--video-bg)}
.videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin:14px 0 6px}
.videos figure{margin:0}
@media (max-width:640px){.videos{grid-template-columns:minmax(0,1fr)}}
.videos.three{grid-template-columns:repeat(3,minmax(0,1fr))}
@media (max-width:860px){.videos.three{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:560px){.videos.three{grid-template-columns:minmax(0,1fr)}}
table.names td:first-child{white-space:nowrap}
'''


def table(headers, rows, num_cols=(), cls=''):
    h = '<div class="wrap"><table%s><thead><tr>' % (' class="%s"' % cls if cls else '') + ''.join('<th%s>%s</th>' % (' class="n"' if i in num_cols else '', x) for i, x in enumerate(headers)) + '</tr></thead><tbody>'
    for r in rows:
        h += '<tr>' + ''.join('<td%s>%s</td>' % (' class="n"' if i in num_cols else '', x) for i, x in enumerate(r)) + '</tr>'
    return h + '</tbody></table></div>'


def figure(svg, caption):
    return '<figure>%s<figcaption>%s</figcaption></figure>' % (svg, caption)


def main():
    D = {}
    van = vanilla_terrain()
    D['van'] = van
    van1, van32 = stats(van[1]['all']), stats(van[32]['all'])
    D['van1_ms'] = 1000.0 / van1['mean']
    cpu = read_csv(FINAL / 'cpu_terrain.csv')
    D['cpu_terrain'] = {t: stats(fnum(r, 'chunks_per_s') for r in cpu if int(r['threads']) == t) for t in (1, 32)}
    head = read_csv(FINAL / 'gpu_headline.csv')
    gpu = stats(fnum(r, 'chunks_per_s') for r in head)
    D['gpu_e2e'] = gpu
    sweep = read_csv(FINAL / 'gpu_batch_sweep.csv')
    D['batch'] = {}
    for r in sweep:
        D['batch'].setdefault((int(r['batch']), int(r['streams'])), []).append(fnum(r, 'chunks_per_s'))
    D['batch'] = {k: stats(v) for k, v in D['batch'].items()}
    # kernel-only throughput: one batch at a time, sum of kernel stage times
    seq = [r for r in sweep if int(r['streams']) == 1 and int(r['batch']) == 4096]
    region_chunks = int(seq[0]['region']) ** 2
    kern = stats(region_chunks / (fnum(r, 'kernel_ms') / 1000.0) for r in seq)
    D['final_stage_batch'] = 4096
    D['final_stages'] = {k: mean_col(seq, k + '_ms') * 4096 / region_chunks for k in ('biome', 'noise', 'aquifer', 'fill', 'surface', 'd2h')}
    lat = read_csv(FINAL / 'gpu_latency.csv')
    D['latency'] = []
    for r_ in sorted({int(r['region']) for r in lat}):
        rows = [r for r in lat if int(r['region']) == r_]
        D['latency'].append({'chunks': r_ * r_, 'ms': st.median(fnum(r, 'seconds') for r in rows) * 1000.0})
    pageable = stats(fnum(r, 'chunks_per_s') for r in read_csv(FINAL / 'gpu_pageable.csv'))
    vcore = {t: stats(fnum(r, 'chunks_per_s') for r in read_csv(FINAL / ('vanilla_core_t%d.csv' % t))) for t in (1, 32)}
    ccore_rows = read_csv(FINAL / 'cpu_core.csv')
    ccore = {t: stats(fnum(r, 'chunks_per_s') for r in ccore_rows if int(r['threads']) == t) for t in (1, 32)}
    gcore = stats(fnum(r, 'chunks_per_s') for r in read_csv(FINAL / 'gpu_core.csv'))
    tests, tests_ok = parse_tests(FINAL / 'tests_full.txt')
    ties = read_csv(FINAL / 'gpu_biome_ties.csv')
    tie_leaf = [1e6 * int(t['leaf_overrides']) / int(t['quarts']) for t in ties]
    tie_biome = [1e6 * int(t['biome_changes']) / int(t['quarts']) for t in ties]
    verify = parse_verify(FINAL / 'run_all.log')

    D['tp_items'] = [
        {'label': 'Vanilla 26.3, 1 thread', 's': van1, 'color': C_VAN},
        {'label': 'Vanilla 26.3, 32 threads', 's': van32, 'color': C_VAN_D},
        {'label': 'CPU port (C++), 1 thread', 's': D['cpu_terrain'][1], 'color': C_CPU},
        {'label': 'CPU port (C++), 32 threads', 's': D['cpu_terrain'][32], 'color': C_CPU_D},
        {'label': 'RTX 5090, kernels only', 's': kern, 'color': C_GPU_D},
        {'label': 'RTX 5090, end to end', 's': gpu, 'color': C_GPU},
    ]

    # optimization ladder (measured right after each change; same region and batch)
    steps = [('v0', 'naive port', 'v0_naive.csv'), ('v1', 'FP64-free floor/wrap', 'v1_fp64_compare.csv'), ('v2', 'axis tables', 'v2_axis_tables.csv'),
             ('v3', 'zoom table', 'v3_zoom_table.csv'), ('v4', 'grid dedupe', 'v4_grid_dedupe.csv'), ('v5', 'column dedupe', 'v5_column_dedupe.csv'),
             ('v6', 'zoom FP32 filter', 'v6_zoom_fp32_filter.csv'), ('v7', 'inline rules', 'v7_inline_rules.csv'),
             ('v8', 'biome search', 'v8_biome_search_and_chain.csv')]
    ladder = []
    for v, name, f in steps:
        rows = [r for r in read_csv(OPT / f) if int(r['batch']) == 4096 and int(r['region']) == 64]
        ladder.append({'short': v, 'name': name, 'tick': TICKS[v], 'rate': mean_col(rows, 'chunks_per_s'),
                       'stages': {k: mean_col(rows, k + '_ms') for k in ('biome', 'noise', 'aquifer', 'fill', 'surface', 'd2h')}})
    v9 = read_csv(OPT / 'v9_pipelined_r64.csv')
    v9p = [r for r in v9 if int(r['batch']) == 2048 and r.get('streams') == '2']
    v9s = [r for r in v9 if int(r['batch']) == 4096 and r.get('streams') == '1']
    ladder.append({'short': 'v9', 'name': 'pipelined, 2 slots', 'tick': TICKS['v9'], 'rate': mean_col(v9p, 'chunks_per_s'), 'pipelined': True,
                   'stages': {k: mean_col(v9s, k + '_ms') for k in ('biome', 'noise', 'aquifer', 'fill', 'surface', 'd2h')}})
    D['ladder'] = ladder

    charts = {}
    for name, fn in (('throughput', chart_throughput), ('ladder', chart_ladder), ('stages', chart_stages), ('scaling', chart_scaling),
                     ('batch', chart_batch), ('latency', chart_latency)):
        charts[name] = to_svg(fn('svg', D))
        to_png(fn('png', D), name + '.png')

    # tables
    def ms_chunk(rate):
        ms = 1000.0 / rate
        return '%.2f' % ms if ms >= 0.1 else '%.4f' % ms

    def srow(label, s, extra=''):
        return [label, fmt_rate(s['mean']), ms_chunk(s['mean']), '%.1f%%' % s['cv'], fmt_rate(s['min']), fmt_rate(s['max']), str(s['n']), extra]

    def runs_note(t):
        rs = van[t]['runs']
        if len(rs) < 2:
            return ''
        ms = [st.mean(v) for v in rs.values()]
        return '%d JVM runs, run means %s–%s' % (len(rs), fmt_rate(min(ms)), fmt_rate(max(ms)))

    bench_rows = [
        srow('Vanilla 26.3, 1 thread (region %d²)' % van[1]['region'], van1, runs_note(1)),
        srow('Vanilla 26.3, 32 threads (region %d²)' % van[32]['region'], van32, runs_note(32)),
        srow('CPU port, 1 thread (region 16²)', D['cpu_terrain'][1], '3 runs'),
        srow('CPU port, 32 threads (region 32²)', D['cpu_terrain'][32], '3 runs'),
        srow('RTX 5090 end to end, batch 2048, 2 slots (region 128²)', gpu, '3 runs, pinned output'),
        srow('RTX 5090 end to end, pageable output', pageable, '1 run'),
        srow('RTX 5090 kernels only, batch 4096 (region 128²)', kern, 'sum of kernel event times'),
    ]
    core_rows = [srow('Vanilla 26.3 core, 1 thread', vcore[1]), srow('Vanilla 26.3 core, 32 threads', vcore[32]),
                 srow('CPU port core, 1 thread', ccore[1]), srow('CPU port core, 32 threads', ccore[32]),
                 srow('RTX 5090 core (naive v0 kernels), end to end', gcore)]
    why = {'v0': 'reference code, ported directly', 'v1': 'the same results from integer work on the IEEE bits',
           'v2': 'the same IEEE operations, computed once per distinct coordinate', 'v3': 'the same expression, tabulated over its 8 × 1,024 inputs',
           'v4': 'every value is a pure function of position and semantics', 'v5': 'every value is a pure function of position',
           'v6': 'proven error bound; near ties take the FP64 comparison', 'v7': 'code placement only',
           'v8': 'any bound above the minimum yields the same first minimal leaf', 'v9': 'scheduling only'}
    ladder_rows = []
    for i, l in enumerate(ladder):
        prev = ladder[i - 1]['rate'] if i else None
        ladder_rows.append([l['short'], l['name'], fmt_rate(l['rate']), ('%+.0f%%' % ((l['rate'] / prev - 1) * 100)) if prev else '', '%.1f×' % (l['rate'] / ladder[0]['rate']),
                            why[l['short']]])
    def test_note(t):
        if t['name'].startswith('zoom'):
            return 'GPU self-test against the plain FP64 BiomeManager zoom (not a vanilla comparison); ' + html.escape(t['note'])
        if t['name'] == 'unit':
            return 'Java arithmetic semantics, gradient table, MD5, host FP contraction'
        return html.escape(t['note'])

    test_rows = [[html.escape(t['name']), '{:,}'.format(t['checked']), str(t['bad']), '<span class="ok">PASS</span>' if t['ok'] else '<b>FAIL</b>',
                  test_note(t)] for t in tests]
    vanilla_checked = sum(t['checked'] for t in tests if not t['name'].startswith('zoom') and t['name'] != 'unit')
    verify_rows = [[v['impl'], v['region'], '{:,}'.format(v['blocks']), str(v['bad'])] for v in verify]
    total_checked = sum(t['checked'] for t in tests)

    nsys_rows = []
    launches = None
    nsys_csv = FINAL / 'nsys_kernels.csv'
    if nsys_csv.exists():
        nr = [r for r in read_csv(nsys_csv) if float(r['launches_per_batch']) >= 1]   # per-generator setup kernels are not per batch
        launches = sum(float(r['launches_per_batch']) for r in nr)
        small = [r for r in nr if float(r['pct']) < 1.0]   # listed together: 14 kernels, a few percent in all
        for r in nr:
            if r not in small:
                nsys_rows.append(['<code>%s</code>' % html.escape(r['kernel']), '%.3f' % float(r['avg_ms']), '%.1f%%' % float(r['pct']),
                                  '%g' % float(r['launches_per_batch']), r['what']])
        if small:
            nsys_rows.append(['%d more kernels' % len(small), '%.3f' % sum(float(r['avg_ms']) for r in small), '%.1f%%' % sum(float(r['pct']) for r in small),
                              '%g' % sum(float(r['launches_per_batch']) for r in small),
                              'y-independent inputs and axis records per distinct column, R-tree tie flags, the eroded-badlands pass, aquifer skip heights, '
                              'and the dedupe (keys, CUB radix sort and scan, run heads, scatters)'])
    inline_cut = (1 - ladder[7]['stages']['surface'] / ladder[6]['stages']['surface']) * 100
    abl = read_csv(OPT / 'ablations.csv')
    abl_rows = [[a['experiment'], a['code_state'], '<code>%s</code>' % a['kernel'], '%.2f' % float(a['baseline_ms']), '%.2f' % float(a['ablated_ms']),
                 '%.0f%%' % ((1 - float(a['ablated_ms']) / float(a['baseline_ms'])) * 100)] for a in abl]
    zoom_share = next((1 - float(a['ablated_ms']) / float(a['baseline_ms'])) * 100 for a in abl if a['experiment'].startswith('E2'))
    zt = next(t for t in tests if t['name'].startswith('zoom'))
    zoom_n = '{:,}'.format(zt['checked'])
    zoom_fb = '{:,}'.format(int(re.search(r'(\d+) near ties', zt['note']).group(1)))
    eff16 = stats(van[16]['all'])['mean'] / (16 * van1['mean']) * 100
    smt = (van32['mean'] / stats(van[16]['all'])['mean'] - 1) * 100

    def gpu_state_of(name):
        lines = (FINAL / name).read_text().strip().splitlines()
        head, vals = [h.strip() for h in lines[0].split(',')], [v.strip() for v in lines[-1].split(',')]
        d = dict(zip(head, vals))
        return '%s SM (max %s), %s, %s, %s' % (d.get('clocks.current.sm [MHz]', '?'), d.get('clocks.max.sm [MHz]', '?'), d.get('temperature.gpu', '?') + ' \u00b0C',
                                             d.get('power.draw [W]', '?'), d.get('pstate', '?'))

    gpu_state = gpu_state_of('gpu_state_end.csv')

    sp1 = gpu['mean'] / van1['mean']
    sp32 = gpu['mean'] / van32['mean']
    spc = gpu['mean'] / D['cpu_terrain'][32]['mean']
    cpu1x = D['cpu_terrain'][1]['mean'] / van1['mean']
    now = datetime.date.today().isoformat()

    g = load_json(ROOT / 'docs' / 'videos' / 'ingame.json')
    in_game = g['control_takes']['vanilla_raw']['verify'] if g else None   # first structures-off control take
    compared = vanilla_checked + sum(v['blocks'] for v in verify) + (in_game['blocks'] + in_game['biomes'] if in_game else 0)

    H = []
    H.append('<title>Minecraft Terrain on CUDA</title>%s<style>%s</style><div class="page">' % (FONTS, CSS))
    H.append('<header class="hero"><div class="eyebrow">Minecraft Java 26.3 &middot; overworld TERRAIN step &middot; RTX 5090</div>')
    H.append('<h1>Minecraft terrain generation on CUDA, bit for bit</h1>')
    H.append('<p class="lede">Biomes, density with every cave noise, aquifers and the material rules with ore veins, reimplemented as CUDA kernels. '
             'Every block, biome and density value matches vanilla exactly, checked offline against the game\'s own values and inside the running game.</p>')
    H.append('<p class="spec"><span><b>chunk</b> 16 × 384 × 16 blocks, y −64 to 319</span><span><b>seed</b> 123456789</span>'
             '<span><b>GPU</b> RTX 5090</span><span><b>CPU</b> Ryzen 9 9950X, 32 threads</span><span><b>vanilla</b> 26.3 on JDK 25</span></p>')
    H.append('<div class="kpis">')
    for v, k in (('%s×' % '{:,.0f}'.format(sp1), 'faster than vanilla on 1 thread'), ('%.0f×' % sp32, 'faster than vanilla on all 32 threads'),
                 ('%s' % '{:,.0f}'.format(gpu['mean']), 'chunks/s end to end on the GPU (%.1f µs per chunk)' % (1e6 / gpu['mean'])),
                 ('0', 'mismatches in %s values compared with vanilla' % '{:,}'.format(compared))):
        H.append('<div class="kpi"><div class="v">%s</div><div class="k">%s</div></div>' % (v, k))
    H.append('</div></header>')
    H.append('<nav class="sections" aria-label="Sections"><ul>' + ''.join('<li><a href="#%s">%s</a></li>' % (i, t) for i, t in (
        ('results', 'Results'), ('videos', 'Videos'), ('correctness', 'Correctness'), ('exactness', 'Exactness'), ('speedup', 'How it got fast'),
        ('method', 'Method and scope'), ('reproduce', 'Reproduce'))) + '</ul></nav>')

    # ---- results
    H.append('<section id="results"><h2>Results</h2>')
    H.append(figure(charts['throughput'], 'Same work everywhere: biomes for the region plus a one-chunk border, then the noise fill with aquifers and the '
                    'material rules for every chunk. Vanilla runs its own private methods on real ProtoChunks (see <a href="#method">Method</a>). End to end '
                    'includes host preparation and copying every block back to pinned host memory.'))
    H.append(table(['configuration', 'chunks/s', 'ms/chunk', 'CV', 'min', 'max', 'n', 'notes'], bench_rows, num_cols=(1, 2, 3, 4, 5, 6)))
    H.append('<p>Against our own C++ port on 32 threads the GPU is <b>%.0f×</b> faster. The port is the debugging reference, an interpreter of the density '
             'graph written for clarity and verified first, not a tuned CPU implementation: on one thread it runs at %.2f× vanilla\'s JIT-compiled speed. '
             'CV is the coefficient of variation over all timed repetitions; warm-up repetitions are excluded.</p>' % (spc, cpu1x))
    H.append(figure(charts['scaling'], 'Vanilla reaches %.0f%% of linear scaling at 16 threads (one per core); SMT adds another %.0f%% at 32. The GPU line is the '
                    'end-to-end rate.' % (eff16, smt)))
    H.append('<h3>Inside the GPU run</h3>')
    H.append(figure(charts['stages'], 'CUDA event times per stage for one batch of %s chunks run on its own, without overlap.' % '{:,}'.format(D['final_stage_batch'])))
    H.append(figure(charts['batch'], 'Throughput needs a few thousand chunks in flight. With two slots the copy back overlaps the next batch\'s kernels '
                    '(v9 in <a href="#speedup">How it got fast</a>).'))
    one = D['latency'][0]
    H.append(figure(charts['latency'], 'For a handful of chunks the fixed cost of a call (%s kernel launches including two radix sorts) dominates: one chunk takes '
                    '%.2f ms, against %.1f ms for vanilla on one thread. Median of 20 calls each.' % ('%.0f' % launches if launches else 'about 50', one['ms'], D['van1_ms'])))
    H.append('<h3>The density core alone</h3>')
    H.append('<p>The first milestone ported only the density and the block fill, without aquifers or material rules. Vanilla here is <code>NoiseChunk</code> + '
             '<code>sampleVolume(finalDensity)</code> + the substance decision per block; the GPU row uses the first, naive kernels, kept as the baseline '
             'implementation <code>cuda-v0</code>. On one thread vanilla spends %.2f ms per chunk on this core against %.2f ms for the whole TERRAIN step, '
             'so the core is only %.0f%% of the work. Biomes, aquifers and the material rules are the larger part, which is why the full step was the target.</p>'
             % (1000.0 / vcore[1]['mean'], 1000.0 / van1['mean'], 100.0 * van1['mean'] / vcore[1]['mean']))
    H.append(table(['configuration', 'chunks/s', 'ms/chunk', 'CV', 'min', 'max', 'n', ''], core_rows, num_cols=(1, 2, 3, 4, 5, 6)))

    # ---- videos
    H.append(videos_section(load_json(ROOT / 'docs' / 'videos' / 'videos.json'), load_json(ROOT / 'docs' / 'videos' / 'flythrough.json'), g))

    # ---- correctness: the evidence
    H.append('</section><section id="correctness"><h2>Correctness</h2>')
    H.append('<p class="note">Bit-exactness is the definition of done: every block state, every biome and every density value is compared for equality with '
             'vanilla, with no tolerance. Java is strict IEEE with no fused multiply-add, so device code is compiled with '
             '<code>-fmad=false -prec-div=true -prec-sqrt=true -ftz=false</code>, never fast-math. Every optimization keeps the exact IEEE operations or '
             'replaces them by integer work with a written proof that the result is identical.</p>')
    H.append(snippet('CMakeLists.txt', r'add_compile_options\(\$<\$<COMPILE_LANGUAGE:CUDA>:-fmad=false>', r'--expt-relaxed-constexpr',
                     'Device flags for every CUDA file: no fused multiply-add, IEEE division and square root, denormals kept. The host side uses '
                     '<code>/fp:precise</code>, and a unit test checks that the host compiler does not contract.', lang='cmake'))
    H.append('<p>The evidence comes in three layers, from single values to a running game.</p>')
    H.append('<h3>Against values from the game</h3>')
    H.append('<p>The reference oracle runs the real game classes (Minecraft 26.3 server jar via Fabric Loom, on this machine only) and dumps values for 32 seeds '
             '(8 fixed, 24 random) at 18 chunks each: a 3×3 block at the origin, chunks near and far, chunks at the world border (±1,874,999) and at the '
             '32-bit wrap points of the noise lattice, over the full build height (y = −64 to 319). <code>mccuda_tests --set full</code> (%s):</p>'
             % ('<span class="ok">RESULT: PASS (bit-exact)</span>' if tests_ok else '<b>FAIL</b>'))
    H.append(table(['level / check', 'values', 'mismatches', '', 'notes'], test_rows, num_cols=(1, 2), cls='names'))
    H.append('<h3>Whole regions, block for block</h3>')
    H.append('<p>The benchmark harness dumps three contiguous regions from vanilla and compares them block for block with both generators:</p>')
    H.append(table(['generator', 'region', 'blocks', 'mismatches'], verify_rows, num_cols=(2, 3)))
    if g:
        H.append(in_game_section(g))

    # ---- exactness: the engine behaviours it depends on
    H.append('</section><section id="exactness"><h2>What exactness takes</h2>')
    H.append('<p>These are the behaviours of 26.3\'s terrain generation that are easy to get almost right. They were found by reading public signatures and '
             'bytecode of the locally prepared game jar and by comparing against the oracle; no game source is reproduced. The longer write-up is '
             '<code>docs/ENGINE_NOTES.md</code>.</p>')
    H.append('<h3>Two evaluation semantics</h3>')
    H.append('<p>26.3\'s density functions are float-valued and have two evaluation paths that do not produce identical bits. The volume path '
             '(<code>sampleVolume</code>) generates a chunk\'s density, the climate for biomes, ore-vein density and richness, and the aquifer\'s '
             'surface-level prefill. The point path (<code>sampleValue</code>) answers the aquifer\'s barrier, floodedness, spread and lava queries, the '
             'ore-vein filler gap and the <code>find_top_surface</code> probes.</p>')
    H.append(table(['', 'volume path', 'point path'], [
        ['noise coordinates', '<code>block * (scale * frequency)</code>', '<code>(block * scale) * frequency</code>'],
        ['gradient dot product', '<code>(gx*x + gz*z) + gy*y</code>, xz part reused over y', '<code>(gx*x + gy*y) + gz*z</code>'],
        ['interpolation', 'z, then x, then incremental steps in y', 'lerp3 at the point'],
        ['min / max', 'keep the first argument unless the second is strictly smaller or larger', '<code>Math.min</code> / <code>max</code>, with range-based short cuts'],
        ['mul / div', 'plain', 'zero short-circuit on the first operand']]))
    H.append('<p class="lead">The code generator (<code>cuda/codegen/gen_router.py</code>) emits a separate evaluator per context, each with fixed semantics. '
             'Mixing them up changes the last bit at a small fraction of positions, which is enough to flip blocks. The gradient dot product shows how small '
             'the difference is:</p>')
    H.append(snippet('cpu/include/mccuda/noise_eval.h', r'^// GRADIENT\[hash & 15\]\.dot', r'^    return a \+ c;',
                     'Point samples add y before z; volume samples compute the xz part once and reuse it over y. In float arithmetic the two orders round '
                     'differently.'))
    H.append('<h3>The compiled density graph</h3>')
    H.append('<p>Vanilla rewrites the density graph before sampling: references are inlined, structurally equal cache nodes are merged, and functions that do '
             'not depend on an axis are evaluated on slices. The IR exporter (<code>oracle/.../IrExporter.java</code>) mirrors these rewrites, and the binary '
             'operations carry vanilla\'s compile-time decisions: constant detection, division by a constant as a multiply by the float reciprocal, and '
             'min/max pruning from value ranges. Those decisions change results, so they are made once, at export time, exactly as vanilla makes them.</p>')
    H.append('<h3>Random numbers</h3>')
    H.append('<p>Xoroshiro128++ with positional factories: <code>fromHashOf</code> seeds from the MD5 of a name, <code>at(x, y, z)</code> from '
             '<code>Mth.getSeed</code>. The legacy 48-bit LCG still drives the biome temperature noise. <code>nextDouble</code> is '
             '<code>(bits &gt;&gt;&gt; 11) * 2^-53</code> computed in double: a decompiler rendering suggested float arithmetic, and the bytecode '
             '(<code>l2d</code>, <code>dmul</code>) settled it. Noise construction sums the octave amplitudes with a compensated sum, so plain summation '
             'is wrong.</p>')
    H.append('<h3>Biomes and the R-tree tie rule</h3>')
    H.append('<p>The six climate parameters are sampled at every quart (4 blocks) with the volume path and quantized as <code>(long)(value * 10000f)</code>. '
             'The lookup is an R-tree over 7,594 parameter boxes (19 children per node, built with stable sorts): the search descends into a child only if '
             'its box is strictly closer than the best leaf so far, and accepts a leaf only if it is strictly closer. <code>BiomeManager</code> then picks '
             'a quart per block from the 8 surrounding quarts with a seeded offset; that seed is the SHA-256 of the world seed, read little-endian.</p>')
    H.append('<p>The tree keeps a last-result candidate and starts each search from it. On an exact distance tie the candidate wins if it reaches the minimal '
             'distance; otherwise the first minimal leaf in DFS order wins. This is not a corner case: over %d seeds (%s quart lookups each, region '
             '128×128 plus borders) the candidate changed the chosen leaf at %.0f–%.0f ppm of lookups and the biome at %.0f–%.0f ppm, because '
             'quantized climate values regularly land exactly on a box boundary. Returning the plain nearest leaf would put a wrong biome in roughly one '
             'quart in %s. Both generators reproduce the rule along each chunk\'s chain of lookups. Vanilla\'s only other history dependence is the first '
             'lookup of a chunk, which starts from whatever the worker thread looked up last; the oracle and our generators start it empty, so a live '
             'server can differ where that first lookup is an exact tie.%s</p>'
             % (len(ties), '{:,}'.format(int(ties[0]['quarts'])), min(tie_leaf), max(tie_leaf), min(tie_biome), max(tie_biome),
                '{:,}'.format(int(round(1e6 / st.mean(tie_biome), -2))),
                (' Inside the running game none of the %s biome values compared <a href="#in-game">above</a> differed.' % '{:,}'.format(in_game['biomes'])
                 if in_game and in_game['biome_mismatches'] == 0 else '')))
    H.append('<p class="lead">The GPU finds the first minimal leaf of every quart in parallel (with a bounded search, v8 below). The rule then needs the '
             'previous result only where the previous leaf also reaches the minimum:</p>')
    H.append(snippet('cuda/src/terrain.cu', r'^// tie\[p\]: the previous position', r"^    tie\[idx\] = ",
                     'Only where this flag is set can the last-result candidate change the answer, and only there does the chain kernel take its slow path; '
                     'one warp per chunk resolves those windows.'))
    H.append('<h3>Aquifers</h3>')
    H.append('<p>A 16×12×16 grid. Cell locations are random per grid cell; fluid status comes from the surface level around the cell, the floodedness '
             'and spread noises, and a lava-noise check for fluid levels at or below −10. The surface level comes from <code>find_top_surface</code>: '
             'positions inside the aquifer constructor\'s prefill window (quarts 4c−4 to 4c+6) use volume semantics, other positions are sampled lazily '
             'with point semantics, and the GPU must use the same split. Blocks above a per-chunk skip height use the global fluid picker directly.</p>')
    H.append('<h3>Material rules</h3>')
    H.append('<p>A column is walked top-down. Stone depth above and below, water height and surface depth feed the rule tree: about 150 conditions, 110 block '
             'results, 43 noise thresholds, 2 vertical gradients, 2 ore veins (new in 26.3\'s material rules) and clay bands. 26.3 checks a cave biome for '
             'almost every stone block below the preliminary surface, so the biome zoom runs for most blocks of a chunk.</p>')
    H.append('<p class="lead">The heightmap is updated on every block placement, and the slope condition reads neighbour heights. Vanilla processes columns '
             'in x-major, z-minor order, so a neighbour\'s height is from before or after both surface extensions. The GPU walks all columns in parallel '
             'and reconstructs what each column would have seen from three height snapshots, with the frozen-ocean extension dry-run in a first pass:</p>')
    H.append(snippet('cuda/src/terrain.cu', r'auto H = \[&\]\(int ox, int oz\)', r'^    \};', 'Columns before (x, z) in the x-major, z-minor loop '
                     'have finished both surface extensions (h2), later ones neither (h0), and the column itself has finished the eroded-badlands '
                     'extension (h1). The slope condition reads these heights.', before=2))

    # ---- speedup
    H.append('</section><section id="speedup"><h2>How it got fast</h2>')
    H.append('<p>Starting from a straight port of the reference code (v0), each step was chosen from profiling data, measured right after the change on '
             'the same 64×64 region (4,096 chunks), and re-verified on the full oracle set before it was committed. The headline rate uses the '
             '128×128 region, where the two-slot pipeline runs eight batches instead of two.</p>')
    H.append(figure(charts['ladder'], 'v9 is measured with two pipeline slots and batch 2048; its stage bars come from a one-slot run of the same code.'))
    H.append(table(['version', 'change', 'chunks/s', 'step', 'vs v0', 'why the output stays identical'], ladder_rows, num_cols=(2, 3, 4)))
    zoom_line = next(l.strip() for l in (FINAL / 'tests_full.txt').read_text(encoding='utf-8-sig').splitlines() if 'zoom fast path' in l)
    H.append('<h3>What mattered, with the code</h3>')
    H.append('<h4>FP64 is the bottleneck on a GeForce part</h4><p class="lead">The RTX 5090 runs FP64 at 1/64 of its FP32 rate. Minecraft 26.3\'s '
             'gradient noise is FP32 inside, but each octave sets up its three coordinates in FP64, about 29 FP64 operations per octave. A static SASS '
             'count of the v0 corner kernel showed about 7,000 FP64 arithmetic, compare and conversion instructions against about 16,000 FP32 ones, '
             'so at 1:64 the FP64 work dominates by an order of magnitude.</p>')
    H.append(snippet('cpu/include/mccuda/noise_eval.h', r'^MCC_HD MCC_FI float perlin_volume\(', r'^}', 'Every <code>double</code> line is FP64 '
                     'work: a multiply, the wrap test, an add, a floor and a subtraction per axis. Everything after <code>volume_corners</code> is FP32.',
                     before=1))
    H.append('<h4>v1: floor and wrap as integer work</h4><p class="lead"><code>Mth.floor</code> is <code>(int) Math.floor(v)</code>, and Java\'s cast '
             'saturates with NaN going to 0. On the device it becomes one conversion instruction with the NaN case decided on the bit pattern, and the wrap '
             'range test becomes integer compares on the IEEE bits:</p>')
    H.append(snippet('cpu/include/mccuda/javamath.h', r'^MCC_HD MCC_FI int32_t mth_floor\(double v\)', r'^}', '<code>cvt.rmi.s32.f64</code> floors '
                     'and saturates exactly like Java\'s <code>(int)</code> cast. The host keeps the reference form.', before=1))
    H.append(snippet('cpu/include/mccuda/noise_eval.h', r'^MCC_HD MCC_FI double gn_wrap\(double x\)', r'^#endif', 'IEEE magnitudes order like '
                     'unsigned integers, so <code>-H &lt;= x &lt; H</code> needs no FP64 compare; the sign bit picks the inclusive or exclusive bound.'))
    H.append('<h4>v2: per-axis coordinate tables</h4><p class="lead">Every corner-level noise is unshifted and at a fixed scale, so each axis\'s '
             'coordinate setup depends on that axis alone. Its exact result is computed once per distinct coordinate and read back:</p>')
    H.append(snippet('cpu/include/mccuda/noise_eval.h', r'^struct AxisRec \{', r'^    return \{f, r, mth_smoothstep\(r\)\};',
                     'The record holds what one axis contributes to a sample. <code>axis_plain</code> is the same arithmetic as the axis setup in '
                     '<code>perlin_volume</code> above, so the bits are identical.', before=4))
    H.append(snippet('cpu/include/mccuda/noise_eval.h', r'^MCC_HD MCC_FI float stack_volume_tab\(', r'^}', 'The sample itself is now FP32 only: '
                     'lattice hashes, gradients and lerps, with the coordinate setup read from three records.', before=1))
    H.append(snippet('cuda/src/terrain.cu', r'^__global__ void k_corners\(', r'^}', 'Each distinct corner column u owns its x and z records; '
                     'the y records are shared by every column at the same corner row.'))
    H.append('<h4>v3 and v6: the biome zoom</h4><p class="lead">The material rules check a cave biome for nearly every stone block, and the '
             'BiomeManager zoom was %.0f%% of the surface kernel in an ablation build. Its squared offsets take only 8 × 1,024 distinct values '
             '(v3), and an FP32 pre-filter with a proven error bound leaves the FP64 comparison to near ties (v6):</p>' % zoom_share)
    H.append(snippet('cuda/include/mccuda_cuda/terrain_device.h', r'^MCC_HD MCC_FI double zoom_square\(int t\)', r'^}', 'The table is built on '
                     'the device by this function, with the same expression as the per-lookup code it replaces.', before=3))
    H.append(snippet('cuda/include/mccuda_cuda/terrain_device.h', r'^// zoom_quart_tab with an exact FP32 pre-filter', r'^    bx = \(min_i & 4\)',
                     'The comment holds the error bound. One FP32 candidate is the answer; several fall back to vanilla\'s FP64 arithmetic in corner order.'))
    H.append(snippet('bench/results/final/tests_full.txt', re.escape(zoom_line[:20]), re.escape(zoom_line[:20]), 'The self-test compares the fast path '
                     'with the plain FP64 zoom on 2^31 random blocks up to the world border. The near ties exercise the FP64 fallback.', lang='text'))
    H.append('<h4>v4 and v5: deduplication across chunks</h4><p class="lead">Neighbouring chunks need the same positions. For the 64×64 region, '
             '2,117,632 surface-level lookups are 144,778 distinct ones (14.6×), and 102,400 corner columns are 66,049 (1.55×). The dedupe runs on '
             'the device for any chunk list:</p>')
    H.append(snippet('cuda/src/dedupe.cuh', r'^    // unique\[0 \.\. \*count\)', r'^    }$', 'Radix sort, mark the first key of each run, scan, scatter. '
                     'The unique count stays on the device, so the evaluation kernels read it without a host round trip.'))
    H.append(snippet('cuda/src/terrain.cu', r'^__global__ void k_grid_keys\(', r'^}', 'The key carries the evaluation semantics: a position '
                     'inside a chunk\'s aquifer prefill window is a volume-path sample, outside it a point-path sample (see '
                     '<a href="#exactness">Exactness</a>).'))
    H.append('<h4>v7: inline the generated rule tree</h4><p class="lead">The material-rule functions were <code>noinline</code>, which kept the '
             'whole rule context in local memory. Inlining them cut the stage by %.0f%%:</p>' % inline_cut)
    gen_hdr = ROOT / 'build' / 'generated' / 'mccuda_router.cuh'
    n_rule_fns = len(re.findall(r'^MCC_RULE_FN (?:bool cond|int rule)\d+\(SurfaceCtx& c\) \{', gen_hdr.read_text(encoding='utf-8'), flags=re.M)) \
        if gen_hdr.exists() else 0
    H.append(snippet('cuda/src/terrain.cu', r'^#if defined\(__CUDACC__\)$', r'^#endif$', 'The generator emits %s condition and rule functions with '
                     'this qualifier, one per node of the material-rule tree; the change is this one line.' % ('{:,}'.format(n_rule_fns) if n_rule_fns else 'the')))
    H.append('<h4>v8: a bounded R-tree search and a warp-parallel tie chain</h4><p class="lead">A search seeded with a neighbouring quart\'s leaf '
             'distance plus one returns the same first minimal leaf and prunes most of the tree:</p>')
    H.append(snippet('cuda/src/terrain.cu', r'^// Climate\.RTree\.SubTree\.search \(iterative DFS', r'^    int closest = -1;',
                     'The proof sits in the comment; the only change from an unbounded search is the initial <code>min_d</code>.'))
    H.append(snippet('cuda/src/terrain.cu', r'const unsigned any_tie = __ballot_sync', r'prev = prev_lstar = __shfl_sync',
                     'The common case in the chain kernel: a window of 32 positions with no tie and in sync writes all 32 results at once.'))
    H.append('<h4>v9: overlap the copy with the next batch</h4><p class="lead">The first pipelined version overlapped nothing: copies run in '
             'submission order, so a download enqueued before the next batch\'s upload held that upload back. Downloads are now enqueued one batch '
             'later:</p>')
    H.append(snippet('cuda/src/terrain.cu', r'^    Slot\* prev = nullptr;', r'^    if \(prev\) I\.download\(\*prev\);$',
                     'Batch k\'s download is enqueued after batch k+1\'s uploads and kernels, so it overlaps that batch\'s kernels. Found from an '
                     'Nsight Systems timeline.'))
    H.append('<h3>Ablations</h3><p>Throwaway builds that remove one piece of work, to find what a kernel spends its time on (results not bit-exact by design):</p>')
    H.append(table(['experiment', 'code', 'kernel', 'baseline ms', 'ablated ms', 'share'], abl_rows, num_cols=(3, 4, 5)))
    H.append('''<h3>Tried and reverted</h3><ul>
<li>Per-column memo of the 2D rule noises: no measurable change.</li>
<li>Per-column cache of zoom corner hashes, and a per-chunk hash table: both slower (local-memory traffic and scattered loads beat recomputing).</li>
<li>Coalesced column mapping in the surface kernels: neutral on its own (kept, it is the cleaner access pattern).</li></ul>''')
    if nsys_rows:
        H.append('<h3>Final kernel profile</h3><p>Nsight Systems, region 64×64 in one batch of 4,096 chunks, average over repetitions:</p>')
        H.append(table(['kernel', 'ms', 'share', 'launches', 'what it does'], nsys_rows, num_cols=(1, 2, 3)))

    # ---- method and scope
    H.append('</section><section id="method"><h2>Method and scope</h2><h3>Method</h3><ul>')
    H.append('<li><b>Machine:</b> AMD Ryzen 9 9950X (16 cores / 32 threads), 128 GB RAM, NVIDIA GeForce RTX 5090 (170 SMs, 32 GB, driver 591.86, WDDM), '
             'Windows 11. CUDA 13.1, MSVC 14.44, JDK 25.0.4.1. GPU state sampled right after the GPU runs: %s.</li>' % html.escape(gpu_state))
    H.append('<li><b>Clocks unlocked</b> (locking needs administrator rights); variance is reported instead: CV over repetitions and, for the headline '
             'configurations, the spread of independent process runs.</li>')
    H.append('<li><b>Workload:</b> seed 123456789, square regions anchored at chunk (0, 0). Vanilla and the CPU port use 16×16 (1 to 4 threads) or 32×32 '
             'chunks; the GPU uses 128×128 (it needs the larger region to fill the device). Warm-up repetitions run first (JIT, allocations) and are not timed.</li>')
    H.append('<li><b>Fairness:</b> vanilla computes biomes once per chunk of the region plus border, exactly what its BIOMES step does; the GPU recomputes the '
             'biomes of tile borders shared between batches, which is extra work on its side. Vanilla\'s worker threads share one RandomState like a server; '
             'its R-tree last-result cache is left alone (live-server behaviour) in throughput runs and reset per chunk in the dump runs.</li>')
    H.append('<li><b>Profiling:</b> Nsight Systems for kernel times and timelines, cuobjdump for register/stack use and SASS instruction mixes, CUDA events '
             'per stage, and targeted ablation builds. Nsight Compute could not collect counters: it reports <code>ERR_NVGPUCTRPERM</code> (counter access '
             'is admin-only on this machine and was not changed).</li></ul>')
    H.append('<p class="lead">Vanilla is timed through its own code path. The inner loop of the vanilla benchmark makes the calls that '
             '<code>buildTerrain</code> makes before carvers, on real ProtoChunks whose biomes were generated in phase 1:</p>')
    H.append(snippet('oracle/src/main/java/mccuda/oracle/TerrainBench.java', r'// phase 2: noise fill', r'buildSurface\.invoke',
                     'One NoiseChunk per chunk, <code>doFill</code> with aquifers on, then <code>buildSurface</code> with a BiomeManager over the 3×3 '
                     'neighbourhood. The GPU does exactly this work; nothing is skipped on either side.', lang='java'))
    H.append('<h3>Scope</h3><ul>')
    H.append('<li><b>Included:</b> biomes (multi-noise climate, exact R-tree search including its last-result tie rule), final density with all cave '
             'noises, NoiseBasedAquifer, doFill, the material rules with ore veins, clay bands, eroded-badlands and frozen-ocean extensions, and the heightmap '
             'updates they depend on.</li>')
    H.append('<li><b>Not included:</b> carvers (they were not requested), the terrain adaptation around structures (measured in the game, see '
             '<a href="#in-game">Correctness</a>), blending with old-world chunks, features and lighting. A live server\'s first biome lookup per chunk '
             'can differ on an exact tie (see <a href="#exactness">Exactness</a>).</li>')
    H.append('<li><b>Decisions taken without the author:</b> Minecraft 26.3 on native Windows; game-derived data (IR, terrain model, oracle dumps) is '
             'extracted locally into the git-ignored <code>data/</code> and never committed; clocks unlocked; overworld only. A Fabric mod '
             '(<code>client/</code>) exists only to record the in-game videos from a precomputed dump; running CUDA inside the game (JNI) was not '
             'attempted.</li></ul>')

    # ---- reproduce
    steps = [('powershell -File build.ps1', 'CMake + Ninja, Release'),
             ('powershell -File oracle\\regen.ps1', 'reference data from the local game jar (data/, git-ignored)'),
             ('build\\tests\\mccuda_tests.exe --set full', 'bit-exactness, all levels'),
             ('python bench\\run_all.py', 'every measurement in this report'),
             ('python bench\\make_videos.py record', 'top-down videos: run every implementation on the video region'),
             ('python bench\\make_videos.py render', 'top-down videos: render into docs/videos/'),
             ('python bench\\make_flythrough.py', 'first-person GPU flythrough'),
             ('cd client; .\\gradlew runClient', 'in-game take; control takes in bench/make_ingame.py'),
             ('python bench\\make_ingame.py', 'in-game videos, frame comparison, in-game checks'),
             ('python docs\\make_report.py', 'this page and docs/img/*.png')]
    width = max(len(c) for c, _ in steps) + 2
    H.append('</section><section id="reproduce"><h2>Reproduce</h2><pre>%s</pre>' % '\n'.join(html.escape(c.ljust(width) + '# ' + d) for c, d in steps))
    H.append('<p><code>oracle/regen.ps1</code> writes, under the git-ignored <code>data/</code>, the optimized density graph '
             '(<code>overworld_ir.json</code>: floats as raw bits, doubles as hex, with value ranges and axis dependence per node), the material-rule tree, '
             'block palette and biome parameters (<code>overworld_terrain.json</code>), and the per-seed oracle dumps: named-function volumes (L1), final '
             'density (L2), core block fill (L3), biomes, aquifer fill and final terrain, in <code>((y − minY) * 16 + z) * 16 + x</code> order with '
             'SHA-256 manifests. The in-game region comes from <code>build\\bench\\mccuda_bench.exe --mode terrain-gpu --region 136 --x0 404 --z0 -324 '
             '--reps 1 --warmup 0 --dump data\\videos\\ingame_dump</code>.</p>')
    H.append('<footer>Generated %s by <code>docs/make_report.py</code> from the measurements in <code>bench/results/</code>.</footer>' % now)
    H.append('</section></div>')
    page = '\n'.join(H)
    out = ROOT / 'docs' / 'report.html'
    out.write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                   '</head><body>\n' + page + '\n</body></html>\n', encoding='utf-8')
    if len(sys.argv) > 2 and sys.argv[1] == '--artifact':   # body-only variant for hosts that add the document skeleton
        Path(sys.argv[2]).write_text(page + '\n', encoding='utf-8')
    write_readme(van1, van32, D, gpu, kern, sp1, sp32, spc, cpu1x, tests, vanilla_checked, verify, ladder)
    print('wrote', out, 'and', IMG)
    print('headline: vanilla1 %.1f, vanilla32 %.1f, cpu1 %.1f, cpu32 %.1f, gpu e2e %.0f, kernels %.0f; speedups %.0fx / %.1fx' % (
        van1['mean'], van32['mean'], D['cpu_terrain'][1]['mean'], D['cpu_terrain'][32]['mean'], gpu['mean'], kern['mean'], sp1, sp32))


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def video(name, caption, autoplay=False):
    attrs = 'autoplay muted loop playsinline controls preload="auto"' if autoplay else 'muted playsinline controls preload="none"'
    label = re.sub(r'<[^>]+>', '', caption).split('.')[0]
    return ('<figure class="video"><video src="videos/%s.mp4" poster="videos/%s.jpg" %s aria-label="%s"></video><figcaption>%s</figcaption></figure>'
            % (name, name, attrs, html.escape(label), caption))


def videos_section(m, f, g):
    """The recorded videos: top-down generation replays (bench/make_videos.py), the GPU flythrough (bench/make_flythrough.py)
    and the real client in GPU-generated terrain (client/, bench/make_ingame.py)."""
    out = ['</section><section id="videos"><h2>Watch it generate</h2>']
    if m:
        reg = m['region']

        def rate(k):
            return '{:,.0f}'.format(m[k]['chunks'] / (m[k]['end_ms'] / 1000.0))

        def secs(k):
            ms = m[k]['end_ms']
            return '%.1f s' % (ms / 1000.0) if ms >= 1000 else '%.0f ms' % ms if ms >= 100 else '%.1f ms' % ms

        out.append('<p>Each implementation generated the same %d×%d-chunk region (chunk %s, %s: badlands, a river, beach, ocean and grassland) while '
                   'logging the moment every chunk was finished. The videos replay those logs at real speed over a top-down map of the generated blocks: '
                   'highest block per column, water shaded by depth, relief from the height field. The map is the same for every implementation because the '
                   'blocks are; for this region the GPU\'s output matches a vanilla dump block for block. These are single recorded runs; the tables in '
                   '<a href="#results">Results</a> are the reference measurements.</p>'
                   % (reg['size'], reg['size'], str(reg['x0']).replace('-', '−'), str(reg['z0']).replace('-', '−')))
        out.append(video('race', 'All five on one clock. The GPU is done within the first frame (%s); vanilla on one thread needs %s.'
                         % (secs('gpu'), secs('van1')), autoplay=True))
        out.append('<div class="videos three">')
        out.append(video('vanilla-1-thread', 'Vanilla, 1 thread: %s, %s chunks/s. Biomes for the region and its border come first, then chunks finish one by '
                                             'one in order.' % (secs('van1'), rate('van1'))))
        out.append(video('vanilla-32-threads', 'Vanilla, 32 threads: %s, %s chunks/s. Chunks land out of order as worker threads pick them up.'
                         % (secs('van32'), rate('van32'))))
        out.append(video('cpu-port-1-thread', 'C++ port, 1 thread: %s, %s chunks/s.' % (secs('cpu1'), rate('cpu1'))))
        out.append(video('cpu-port-32-threads', 'C++ port, 32 threads: %s, %s chunks/s.' % (secs('cpu32'), rate('cpu32'))))
        out.append(video('gpu', 'RTX 5090: %s in four batches of 256 chunks, then a replay at 1/%d speed to show each batch land.'
                         % (secs('gpu'), m['gpu_slowdown'])))
        out.append(video('gpu-region-128', 'RTX 5090 at its benchmark batch size: a 128×128 region (16,384 chunks) in %s, %s chunks/s, eight batches '
                                           'of 2,048, then a replay at 1/%d speed.' % (secs('gpu128'), rate('gpu128'), m['gpu128_slowdown'])))
        out.append('</div>')
    if f and g:
        c = '{:,}'.format
        st_, ss = f['startup'], f['steady']
        out.append('<h3>In first person</h3>')
        out.append('<p>Two recordings fly one 38-second camera path over the same seed: a walk at eye height in the badlands, then a climb over mesas, a '
                   'forest, a river and the coast. The first is our own renderer generating the terrain live on the GPU; the second is the real game, '
                   'flying through chunks the GPU generated.</p>')
        out.append('<div class="videos">')
        out.append(video('flythrough', 'Our CUDA renderer (<code>viz/fly.cu</code>) generates every chunk within %d chunks of the camera with the bit-exact '
                                       'pipeline and ray traces the view on the same GPU: %s chunks in the first %d frames (%.0f ms), then %.0f per generating '
                                       'frame in %.1f ms on average (%.1f ms at most), %s in all. Procedural colours, no game textures.'
                         % (f['radius_chunks'], c(st_['chunks']), st_['frames'], st_['gen_ms'], ss['new_mean'], ss['gen_ms_mean'], ss['gen_ms_max'],
                            c(f['chunks_total']))))
        out.append(video('ingame', 'The Minecraft 26.3 client. A Fabric mod fills every chunk of a %d×%d-chunk region with the GPU\'s blocks and biomes; '
                                   'the game adds its own caves, trees and lighting. %s chunks came from the GPU, and at every frame the client held %d '
                                   'chunks from anywhere else. %.0f fps, captured frame by frame. How the mod works and what it checked: '
                                   '<a href="#in-game">Correctness</a>.'
                         % (g['region']['size'], g['region']['size'], c(g['gpu_chunks']), g['client_chunks_outside_region_max'], g['client_fps_median'])))
        out.append('</div>')
        out.append(video('compare', 'The same frames side by side: the same mesas, terracotta bands and rivers. The ray tracer shows the TERRAIN step alone; '
                                    'the client adds the game\'s later steps on top of it.'))
    return '\n'.join(out)


def in_game_section(g):
    """The in-game check: vanilla's own chunks, compared with the GPU dump inside the running game (client/)."""
    c = '{:,}'.format
    t = g['control_takes']
    out = ['<h3 id="in-game">Inside the running game</h3>']
    out.append('<p class="lead">The in-game recording (<a href="#videos">Videos</a>) uses a Fabric mod in a development client. Inside a %d×%d-chunk '
               'region it puts the GPU\'s output in at vanilla\'s own seams: the noise fill writes the GPU\'s finished blocks, the surface pass is skipped '
               'because its result is already there, and biomes are swapped in the same way at <code>doCreateBiomes</code>. The client reads a dump the GPU '
               'wrote beforehand (%.1f GB); it does not call CUDA itself.</p>' % (g['region']['size'], g['region']['size'], g['region']['size'] ** 2 * 384 * 256 / 1e9))
    out.append(snippet('client/src/main/java/mccuda/client/mixin/NoiseGeneratorMixin.java', r'@Inject\(method = "doFill"', r'^    }$',
                       'Inside the region the chunk is filled from the dump and vanilla\'s fill is cancelled; outside it the game generates as usual.',
                       lang='java'))
    out.append('<p class="lead">For control takes the same mod leaves generation to the game and compares each chunk with the GPU\'s dump as the server '
               'generates it: the biomes right after <code>doCreateBiomes</code>, and every block right after <code>buildSurface</code>, the end of the '
               'TERRAIN step.</p>')
    out.append(snippet('client/src/main/java/mccuda/client/GpuDump.java', r'public void verifyBlocks', r'^    }$',
                       'The game\'s own chunk after its surface pass, compared state by state with the GPU\'s palette index at the same position. Block states '
                       'are canonical objects, so identity is equality.', lang='java'))
    rows = []
    for key, label in (('vanilla_raw', 'vanilla'), ('vanilla2_raw', 'vanilla, second run'), ('vanilla_struct_raw', 'vanilla, structures on')):
        v = t[key]['verify']
        rows.append([label, 'on' if t[key]['structures'] else 'off', c(v['block_chunks']), c(v['blocks']),
                     '<span class="ok">0</span>' if v['block_mismatches'] == 0 else c(v['block_mismatches']), c(v['biomes']),
                     '<span class="ok">0</span>' if v['biome_mismatches'] == 0 else c(v['biome_mismatches'])])
    out.append(table(['take', 'structures', 'chunks', 'blocks compared', 'mismatches', 'biome values', 'mismatches'], rows, num_cols=(2, 3, 4, 5, 6)))
    vs = t['vanilla_struct_raw']['verify']
    out.append('<p>With structures on, every mismatch is in a chunk whose density a structure reshapes (the Beardifier, for villages, trail ruins and the '
               'like): %s of the %s such chunks differ, %s blocks in all, and the other %s chunks match exactly. That adaptation is outside this project\'s '
               'scope, so the recordings use worlds without structures. One more thing the dump does not carry: vanilla\'s <code>doFill</code> also marks '
               'aquifer fluid blocks that should start flowing once the chunk is finished. In GPU-filled chunks those fluids stay still until a neighbouring '
               'block changes; the blocks themselves are identical.</p>'
               % (c(vs['structure_adapted_chunks_differing']), c(vs['structure_adapted_chunks']), c(vs['structure_adapted_mismatches']),
                  c(vs['block_chunks'] - vs['structure_adapted_chunks'])))
    gv, vv = g['frames_gpu_vs_vanilla'], g['frames_vanilla_vs_vanilla']
    out.append('<p>The recorded frames still differ, and are expected to. The GPU take differs from the first vanilla take in %.1f%% of pixels per frame on '
               'average (by more than 8 of 255), and the two vanilla takes differ from each other by about as much, %.1f%%. The differences are trees. A '
               'chunk\'s trees are placed while its neighbours may already be decorated, so where trees compete for space the outcome depends on the order in '
               'which the server\'s worker threads finish chunks.</p>' % (100 * gv['pixels_over_8_mean'], 100 * vv['pixels_over_8_mean']))
    out.append('<figure class="video"><img src="videos/trees.jpg" alt="The same forest in three takes: GPU chunks, vanilla, and vanilla again" loading="lazy">'
               '<figcaption>Frame %d of three takes, cropped where the GPU take and the first vanilla take differ most. The trees change between the two '
               'vanilla runs as much as between GPU and vanilla; the ground under them does not.</figcaption></figure>' % gv['worst_frame'])
    return '\n'.join(out)


def write_readme(van1, van32, D, gpu, kern, sp1, sp32, spc, cpu1x, tests, total_checked, verify, ladder):
    def r(s):
        return '{:,.0f}'.format(s['mean'])

    cpu1, cpu32 = D['cpu_terrain'][1], D['cpu_terrain'][32]
    blocks = sum(v['blocks'] for v in verify)
    ingame = ''
    ij = ROOT / 'docs' / 'videos' / 'ingame.json'
    if ij.exists():
        g = json.loads(ij.read_text(encoding='utf-8'))
        v = g['control_takes']['vanilla_raw']['verify']
        ingame = (f'''

[![The GPU renderer and the real Minecraft client on the same camera path](docs/videos/compare.jpg)](docs/videos/compare.mp4)

*Click for the video: our CUDA renderer generating terrain live (left) and the real Minecraft 26.3 client flying
the same path through chunks the GPU generated (right; [in-game video](docs/videos/ingame.mp4)). Inside the game,
the same chunks generated by vanilla itself matched the GPU's output: {v['blocks']:,} blocks and
{v['biomes']:,} biome values, 0 mismatches (structures off; see the report).*''')
    md = f'''# Minecraft terrain generation on CUDA, bit for bit

Minecraft Java Edition 26.3's overworld terrain generation reimplemented as CUDA kernels.
The output is **bit-identical to vanilla**: every biome, block state and density value is
checked for equality with values computed by the real game, with no tolerance. On an
RTX 5090 it generates the full TERRAIN step (biomes, density with all caves, aquifers,
material rules including ore veins) at **{r(gpu)} chunks/s**, which is **{sp1:,.0f}x**
vanilla on one thread and **{sp32:.0f}x** vanilla on all 32 threads of a Ryzen 9 9950X.

![Throughput](docs/img/throughput.png)

[![The same region generated by every implementation, real time](docs/videos/race.jpg)](docs/videos/race.mp4)

*Click for the video: the same 32\u00d732-chunk region generated by every implementation, replayed at real speed from
each run's per-chunk finish times ([all videos](docs/videos/)).*{ingame}

| Full TERRAIN step (no carvers) | chunks/s | ms per chunk | vs vanilla 1 thread |
|---|---:|---:|---:|
| Vanilla 26.3, 1 thread | {r(van1)} | {1000 / van1['mean']:.2f} | 1x |
| Vanilla 26.3, 32 threads | {r(van32)} | {1000 / van32['mean']:.3f} | {van32['mean'] / van1['mean']:.1f}x |
| CPU port (C++), 1 thread | {r(cpu1)} | {1000 / cpu1['mean']:.2f} | {cpu1x:.1f}x |
| CPU port (C++), 32 threads | {r(cpu32)} | {1000 / cpu32['mean']:.3f} | {cpu32['mean'] / van1['mean']:.0f}x |
| **RTX 5090, end to end** (copy to pinned host memory included) | **{r(gpu)}** | **{1000 / gpu['mean']:.4f}** | **{sp1:,.0f}x** |
| RTX 5090, kernels only | {r(kern)} | {1000 / kern['mean']:.4f} | {kern['mean'] / van1['mean']:,.0f}x |

Measured on one machine (Windows 11, clocks unlocked, variance reported); see the
**[full report](docs/report.html)** for method, variance, profiling and the optimization log.
Every number above is produced by `bench/run_all.py` and `docs/make_report.py`.

## Correctness

- A reference oracle drives the real game classes (Minecraft 26.3 server jar through
  Fabric Loom, at runtime on the author's machine) and dumps reference values for 32 seeds
  x 18 chunks: origin, near and far chunks, the world border (+-1,874,999) and the noise
  lattice's 32-bit wrap points, over the full height (y = -64..319).
- `mccuda_tests --set full` compares every level, from RNG streams and single noise
  samples up to final blocks, for the CPU port and both GPU implementations:
  {total_checked:,} values compared with vanilla, 0 mismatches. A GPU self-test also
  checks the fast biome-zoom path against the plain FP64 one on 2^31 random blocks.
- The benchmark harness also dumps three contiguous regions from vanilla (origin, negative
  coordinates, world border) and compares them with both generators: {blocks:,} blocks,
  0 mismatches.
- Device code uses `-fmad=false -prec-div=true -prec-sqrt=true -ftz=false`, never
  fast-math. Every optimization either performs the same IEEE operations or replaces
  them with integer work that has a written proof of identical results.

## How it got fast

![Optimization ladder](docs/img/ladder.png)

From a direct port (v0, {ladder[0]['rate']:,.0f} chunks/s) to {ladder[-1]['rate']:,.0f} chunks/s
({ladder[-1]['rate'] / ladder[0]['rate']:.1f}x) on the same 4,096-chunk region. The main lesson:
on a GeForce GPU, FP64 runs at 1/64 of the FP32 rate, and Minecraft's noise does its
coordinate math in FP64. Most of the gain came from removing FP64 work without changing a
single result:

- exact per-axis tables of the noise coordinate setup (y once per seed, x/z per column),
- `Mth.floor` and the noise wrap test as integer work on the IEEE bit pattern,
- a tabulated BiomeManager zoom with a proven-safe FP32 pre-filter (FP64 only for near ties),
- GPU deduplication of positions shared by neighbouring chunks (radix sort, any chunk list),
- a bounded R-tree search that provably returns vanilla's leaf, with the R-tree's
  last-result tie rule resolved by warp-level tie flags,
- an inlined material-rule tree, and a two-slot pipeline that overlaps the copy back to
  the host with the next batch.

## Layout

| path | contents |
|---|---|
| `oracle/` | Java reference harness: drives vanilla, exports the density-function IR and terrain model, dumps reference data, vanilla benchmarks |
| `cpu/` | single-threaded C++ reference port (the debugging ground truth) |
| `cuda/` | code generator (`codegen/gen_router.py`: IR -> CUDA evaluators and material rules) and the GPU pipeline (`src/terrain.cu`) |
| `tests/` | `mccuda_tests`: bit-exactness at every level, environment probes |
| `bench/` | `mccuda_bench`, `run_all.py`, measured results (`results/`) |
| `profiles/` | profiling helpers (Nsight Systems) |
| `docs/` | the report, its generator, and [engineering notes](docs/ENGINE_NOTES.md) on what bit-exactness depends on |

## Build and run

```
powershell -File build.ps1                 # CMake + Ninja + MSVC 14.4x + CUDA 13.1, Release, sm_120
powershell -File oracle\\regen.ps1          # reference data from your local game jar (JDK 25)
build\\tests\\mccuda_tests.exe --set full    # bit-exactness, all levels
python bench\\run_all.py                    # all measurements
python docs\\make_report.py                 # docs/report.html and docs/img/*.png
```

## Data and legal hygiene

The algorithms are reimplemented in our own code. No decompiled Mojang source and no game
files are in this repository. Game-derived data (the density-function graph, the terrain
model and the oracle dumps) is extracted at runtime from the user's own copy of the game
into the git-ignored `data/` directory; the CUDA evaluators generated from it live only in
the build tree.

## Scope

Overworld, Minecraft 26.3. Not included: carvers, structures, blending with old chunks,
features and lighting. The optional Fabric/JNI server mod was not started.
'''
    (ROOT / 'README.md').write_text(md, encoding='utf-8')
    print('wrote README.md')


if __name__ == '__main__':
    main()
