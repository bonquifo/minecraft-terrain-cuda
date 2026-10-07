"""Videos of terrain generation for the report: a top-down map of a region that fills in
chunk by chunk at the moments each implementation finished them.

The map pixels come from the generated blocks (mccuda_bench --topdown; the generators are
bit-identical, and the region is checked against a vanilla dump). The timing of every chunk
comes from that implementation's own run (vanilla bench-terrain timeline=1, mccuda_bench
--timeline), so each video plays the recorded generation at real speed.

    python bench/make_videos.py preview <topdown> <out.png>
    python bench/make_videos.py record            # run every implementation on the video region
    python bench/make_videos.py render            # all videos into docs/videos/
"""
import csv
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'videos'
OUT = ROOT / 'docs' / 'videos'
FPS = 30
FONT_DIR = Path('C:/Windows/Fonts')

COLORS = {
    'minecraft:stone': (126, 126, 126), 'minecraft:water': (38, 84, 172), 'minecraft:lava': (222, 98, 22),
    'minecraft:terracotta': (152, 94, 67), 'minecraft:white_terracotta': (209, 178, 161), 'minecraft:orange_terracotta': (161, 83, 37),
    'minecraft:yellow_terracotta': (186, 133, 36), 'minecraft:brown_terracotta': (77, 51, 36), 'minecraft:red_terracotta': (143, 61, 47),
    'minecraft:light_gray_terracotta': (135, 107, 98), 'minecraft:packed_ice': (141, 180, 250), 'minecraft:snow_block': (240, 246, 248),
    'minecraft:bedrock': (60, 60, 60), 'minecraft:copper_ore': (124, 125, 120), 'minecraft:raw_copper_block': (154, 105, 79),
    'minecraft:granite': (149, 103, 86), 'minecraft:deepslate_iron_ore': (92, 90, 90), 'minecraft:raw_iron_block': (166, 136, 107),
    'minecraft:tuff': (108, 109, 102), 'minecraft:coarse_dirt': (119, 85, 59), 'minecraft:grass_block': (96, 146, 58),
    'minecraft:dirt': (134, 96, 67), 'minecraft:red_sandstone': (181, 98, 31), 'minecraft:red_sand': (190, 102, 33),
    'minecraft:gravel': (131, 127, 126), 'minecraft:ice': (145, 183, 253), 'minecraft:powder_snow': (248, 253, 253),
    'minecraft:calcite': (223, 224, 220), 'minecraft:sandstone': (216, 203, 155), 'minecraft:sand': (219, 207, 163),
    'minecraft:cinnabar': (158, 52, 46), 'minecraft:sulfur': (205, 196, 74), 'minecraft:mud': (60, 57, 61),
    'minecraft:podzol': (91, 63, 24), 'minecraft:mycelium': (111, 99, 105), 'minecraft:deepslate': (80, 80, 84), 'minecraft:air': (0, 0, 0),
}
BG = np.array((22, 27, 24), np.float32)          # not yet generated
GRID = np.array((40, 48, 43), np.float32)


def palette_colors():
    tm = json.loads((ROOT / 'data' / 'gamedata' / '26.3' / 'overworld_terrain.json').read_text(encoding='utf-8'))
    out = []
    for name in tm['palette']:
        base = name.split('[')[0]
        if base not in COLORS:
            raise SystemExit('no colour for ' + name)
        out.append(COLORS[base])
    return np.array(out, np.float32), [n.split('[')[0] for n in tm['palette']]


def load_topdown(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b'MCTD':
        raise SystemExit('not a topdown file: %s' % path)
    version, region, x0, z0, min_y = struct.unpack('<5i', raw[4:24])
    side = region * 16
    rec = np.frombuffer(raw, dtype=np.dtype([('top', 'u1'), ('floor', 'u1'), ('depth', 'u1'), ('pad', 'u1'), ('top_y', '<i2'), ('floor_y', '<i2')]),
                        offset=24, count=side * side).reshape(side, side)
    return {'region': region, 'x0': x0, 'z0': z0, 'min_y': min_y, 'rec': rec}


def render_map(td):
    """RGB float image of the region: block colours, water by depth, hillshade."""
    cols, names = palette_colors()
    rec = td['rec']
    top, floor_s, depth = rec['top'].astype(int), rec['floor'].astype(int), rec['depth'].astype(np.float32)
    h = rec['top_y'].astype(np.float32)
    water = np.array([n == 'minecraft:water' for n in names])[top]
    img = cols[top].copy()
    # water: floor colour seen through water, darker with depth
    a = (1.0 - np.exp(-depth / 5.0))[..., None]
    under = cols[floor_s] * 0.75
    img = np.where(water[..., None], under * (1 - a) + cols[top] * a, img)
    # hillshade from the solid surface (water uses its floor, damped)
    hs = np.where(water, rec['floor_y'].astype(np.float32), h)
    gx = np.zeros_like(hs)
    gz = np.zeros_like(hs)
    gx[:, 1:-1] = (hs[:, 2:] - hs[:, :-2]) * 0.5
    gz[1:-1, :] = (hs[2:, :] - hs[:-2, :]) * 0.5
    n = np.stack([-gx, np.full_like(hs, 1.0), -gz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.6, 0.8, -0.45], np.float32)
    L /= np.linalg.norm(L)
    lit = np.clip((n @ L), 0, 1)
    shade = 0.55 + 0.6 * lit
    shade = np.where(water, 0.85 + 0.15 * lit, shade)
    relief = 1.0 + np.clip((h - 64.0) / 400.0, -0.08, 0.25)       # higher ground slightly lighter
    img = img * (shade * relief)[..., None]
    return np.clip(img, 0, 255)


def preview(topdown, out_png, window=32):
    td = load_topdown(topdown)
    img = render_map(td).astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    side = td['region'] * 16
    font = ImageFont.truetype(str(FONT_DIR / 'segoeuib.ttf'), 26)
    step = window * 16
    for gz in range(0, side, step):
        for gx in range(0, side, step):
            d.rectangle([gx, gz, gx + step - 1, gz + step - 1], outline=(255, 255, 255))
            d.text((gx + 8, gz + 6), '%d,%d' % (td['x0'] + gx // 16, td['z0'] + gz // 16), fill=(255, 255, 255), font=font)
    im.save(out_png)
    print('wrote', out_png)


# ---- video -----------------------------------------------------------------------------------
def read_timeline(path):
    rows = list(csv.DictReader(open(path, newline='', encoding='utf-8')))
    phase1 = sorted(float(r['done_ms']) for r in rows if r['phase'] == '1')
    chunks = {(int(r['cx']), int(r['cz'])): float(r['done_ms']) for r in rows if r['phase'] == '2'}
    wall = [float(r['done_ms']) for r in rows if r['phase'] == '9']
    end = max(max(chunks.values()), wall[0] if wall else 0.0)
    return {'phase1': phase1, 'chunks': chunks, 'end': end}


class Fonts:
    def __init__(self, scale=1.0):
        self.title = ImageFont.truetype(str(FONT_DIR / 'segoeuib.ttf'), int(22 * scale))
        self.sub = ImageFont.truetype(str(FONT_DIR / 'segoeui.ttf'), int(15 * scale))
        self.clock = ImageFont.truetype(str(FONT_DIR / 'consolab.ttf'), int(26 * scale))
        self.mono = ImageFont.truetype(str(FONT_DIR / 'consola.ttf'), int(15 * scale))


def fmt_t(ms):
    return '%6.3f s' % (ms / 1000.0) if ms < 100000 else '%6.1f s' % (ms / 1000.0)


class Panel:
    """One implementation's view: the map, revealed at the recorded chunk times."""

    def __init__(self, title, sub, timeline, mapimg, td, size):
        self.title, self.sub, self.tl = title, sub, timeline
        self.td = td
        region = td['region']
        self.n = region * region
        side = region * 16
        self.done = np.full((region, region), np.inf, np.float32)
        for (cx, cz), t in timeline['chunks'].items():
            self.done[cz - td['z0'], cx - td['x0']] = t
        if np.isinf(self.done).any():
            raise SystemExit('timeline does not cover the region: ' + title)
        im = Image.fromarray(mapimg.astype(np.uint8)).resize((size, size), Image.LANCZOS)
        self.map = np.asarray(im, np.float32)
        bg = np.tile(BG, (size, size, 1))
        cell = size / region
        for k in range(region + 1):                                    # faint chunk grid on the empty ground
            p = min(size - 1, int(round(k * cell)))
            bg[p, :] = GRID
            bg[:, p] = GRID
        self.bg = bg
        self.size, self.region, self.side = size, region, side
        idx = (np.arange(size) * region // size).astype(int)
        self.cell_of = (idx[:, None], idx[None, :])

    def frame(self, t_ms):
        d = self.done[self.cell_of]                                    # per-pixel finish time of its chunk
        shown = d <= t_ms
        flash = np.clip(1.0 - (t_ms - d) / 120.0, 0, 1) * shown          # brief highlight when a chunk lands
        img = np.where(shown[..., None], self.map, self.bg)
        img = img + flash[..., None] * 70.0
        return np.clip(img, 0, 255).astype(np.uint8), int((self.done <= t_ms).sum())

    def status(self, t_ms):
        if self.tl['phase1'] and t_ms < self.tl['phase1'][-1]:
            n1 = sum(1 for x in self.tl['phase1'] if x <= t_ms)
            return 'phase 1: biomes %d / %d' % (n1, len(self.tl['phase1']))
        return None


def draw_panel(canvas, draw, panel, t_ms, x, y, fonts, show_final=True):
    img, n = panel.frame(t_ms)
    canvas.paste(Image.fromarray(img), (x, y + 62))
    draw.text((x, y), panel.title, fill=(236, 241, 236), font=fonts.title)
    draw.text((x, y + 30), panel.sub, fill=(150, 165, 155), font=fonts.sub)
    by = y + 62 + panel.size + 8
    clock = min(t_ms, panel.tl['end'])
    draw.text((x, by), fmt_t(clock).strip(), fill=(236, 241, 236), font=fonts.clock)
    st = panel.status(t_ms)
    right = '%s chunks' % '{:,} / {:,}'.format(n, panel.n)
    draw.text((x + panel.size - draw.textlength(right, font=fonts.mono), by + 8), right, fill=(150, 165, 155), font=fonts.mono)
    if st:
        draw.text((x, by + 34), st, fill=(214, 190, 120), font=fonts.mono)
    elif n == panel.n and show_final:
        rate = panel.n / (panel.tl['end'] / 1000.0)
        draw.text((x, by + 34), 'done: %s chunks/s' % '{:,.0f}'.format(rate), fill=(143, 207, 104), font=fonts.mono)


def encode(frames_iter, w, h, path, nframes):
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h), '-r', str(FPS), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = None
    for k, fr in enumerate(frames_iter):
        p.stdin.write(fr.tobytes())
        last = fr
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit('ffmpeg failed for %s' % path)
    return last


def single_video(name, panel, extra=None, hold_s=1.6):
    """Real time, then (optionally) a slowed replay for runs too fast to see."""
    fonts = Fonts()
    pad = 20
    w = panel.size + 2 * pad
    h = panel.size + 62 + 70 + 2 * pad
    w += w % 2
    h += h % 2
    segments = [(1.0, '')]
    if extra:
        segments.append(extra)                                         # (slowdown factor, label)

    def frames():
        for factor, label in segments:
            dur = panel.tl['end'] * factor / 1000.0
            total = int(math.ceil(dur * FPS)) + int(hold_s * FPS)
            for k in range(total + 1):
                t_ms = (k / FPS) * 1000.0 / factor
                cv = Image.new('RGB', (w, h), (16, 20, 18))
                dr = ImageDraw.Draw(cv)
                draw_panel(cv, dr, panel, t_ms, pad, pad, fonts)
                tag = label if label else 'real time'
                dr.text((w - pad - dr.textlength(tag, font=fonts.mono), pad + 6), tag, fill=(214, 190, 120) if label else (150, 165, 155),
                        font=fonts.mono)
                yield np.asarray(cv)

    nframes = sum(int(math.ceil(panel.tl['end'] * f / 1000.0 * FPS)) + int(hold_s * FPS) + 1 for f, _ in segments)
    last = encode(frames(), w, h, OUT / (name + '.mp4'), nframes)
    Image.fromarray(last).save(OUT / (name + '.jpg'), quality=88)
    print('wrote', OUT / (name + '.mp4'))


def race_video(name, panels, cols=3, hold_s=2.0):
    fonts = Fonts(0.85)
    pad, gap = 18, 22
    size = panels[0].size
    cell_h = size + 62 + 66
    rows = int(math.ceil(len(panels) / cols))
    w = cols * size + (cols - 1) * gap + 2 * pad
    h = rows * cell_h + (rows - 1) * gap + 2 * pad + 40
    w += w % 2
    h += h % 2
    end = max(p.tl['end'] for p in panels)
    total = int(math.ceil(end / 1000.0 * FPS)) + int(hold_s * FPS)

    def frames():
        big = ImageFont.truetype(str(FONT_DIR / 'consolab.ttf'), 30)
        for k in range(total + 1):
            t_ms = k / FPS * 1000.0
            cv = Image.new('RGB', (w, h), (16, 20, 18))
            dr = ImageDraw.Draw(cv)
            head = 'same %d\u00d7%d-chunk region, real time   t = %s' % (panels[0].region, panels[0].region, fmt_t(min(t_ms, end)).strip())
            dr.text((pad, pad - 4), head, fill=(236, 241, 236), font=big)
            for i, p in enumerate(panels):
                cx = pad + (i % cols) * (size + gap)
                cy = pad + 40 + (i // cols) * (cell_h + gap)
                draw_panel(cv, dr, p, t_ms, cx, cy, fonts)
            yield np.asarray(cv)

    last = encode(frames(), w, h, OUT / (name + '.mp4'), total + 1)
    Image.fromarray(last).save(OUT / (name + '.jpg'), quality=88)
    print('wrote', OUT / (name + '.mp4'))


def render():
    td = load_topdown(DATA / 'region32_topdown.bin')
    mapimg = render_map(td)
    region = td['region']
    tl = {k: read_timeline(DATA / f) for k, f in (
        ('van1', 'vanilla_t1_timeline.csv'), ('van32', 'vanilla_t32_timeline.csv'), ('cpu1', 'cpu_t1_timeline.csv'),
        ('cpu32', 'cpu_t32_timeline.csv'), ('gpu', 'gpu_timeline.csv'))}
    spec = {
        'van1': ('Vanilla 26.3, 1 thread', 'JVM, biomes then noise + surface per chunk'),
        'van32': ('Vanilla 26.3, 32 threads', 'JVM thread pool on a Ryzen 9 9950X'),
        'cpu1': ('CPU port (C++), 1 thread', 'reference port, verified first'),
        'cpu32': ('CPU port (C++), 32 threads', 'same port on every core'),
        'gpu': ('RTX 5090', 'batches of 256 chunks, copy to host included'),
    }
    big = {}
    for key, (title, sub) in spec.items():
        big[key] = Panel(title, sub, tl[key], mapimg, td, 512)
    files = {'van1': 'vanilla-1-thread', 'van32': 'vanilla-32-threads', 'cpu1': 'cpu-port-1-thread', 'cpu32': 'cpu-port-32-threads'}
    for key, fname in files.items():
        single_video(fname, big[key])
    gpu_end = tl['gpu']['end']
    slow = max(1, int(round(3000.0 / gpu_end / 10.0)) * 10)            # slowed so the replay lasts about 3 s
    single_video('gpu', big['gpu'], extra=(slow, 'replay at 1/%d speed' % slow))
    small = [Panel(spec[k][0], spec[k][1], tl[k], mapimg, td, 300) for k in ('van1', 'van32', 'gpu', 'cpu1', 'cpu32')]
    race_video('race', small)
    # the GPU on a 128 x 128 region at its benchmark batch size
    td128 = load_topdown(DATA / 'region128_topdown.bin')
    tl128 = read_timeline(DATA / 'gpu128_timeline.csv')
    p128 = Panel('RTX 5090, 16,384 chunks', 'region 128\u00d7128, batches of 2,048, copy to host included', tl128, render_map(td128), td128, 512)
    slow128 = max(1, int(round(4000.0 / tl128['end'] / 5.0)) * 5)
    single_video('gpu-region-128', p128, extra=(slow128, 'replay at 1/%d speed' % slow128))
    meta = {k: {'end_ms': v['end'], 'chunks': len(v['chunks'])} for k, v in tl.items()}
    meta['gpu128'] = {'end_ms': tl128['end'], 'chunks': len(tl128['chunks'])}
    meta['region'] = {'x0': td['x0'], 'z0': td['z0'], 'size': region}
    meta['gpu_slowdown'] = slow
    meta['gpu128_slowdown'] = slow128
    (OUT / 'videos.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print('wrote', OUT / 'videos.json')


SEED, X0, Z0, R = 123456789, 462, -296, 32           # badlands, beach, ocean and grassland in one window
X128, Z128 = 400, -300


def record():
    """Runs every implementation on the video region and collects the timelines."""
    import os
    import shutil
    DATA.mkdir(parents=True, exist_ok=True)
    bench = ROOT / 'build' / 'bench' / 'mccuda_bench.exe'
    env = dict(os.environ)
    env.setdefault('JAVA_HOME', str(ROOT / '.tools' / 'jdk-25'))

    def run(cmd, cwd=ROOT):
        print('$', ' '.join(str(c) for c in cmd), flush=True)
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in p.stdout.splitlines():
            if 'chunks/s' in line or 'verify' in line or 'wrote' in line:
                print('   ', line.strip())
        if p.returncode != 0:
            print(p.stdout[-3000:])
            raise SystemExit('failed: %s' % cmd)

    def oracle(args):
        run([ROOT / 'oracle' / 'gradlew.bat', '--console=plain', '-q', 'oracle', '-Pcmd=' + args], cwd=ROOT / 'oracle')

    common = 'region=%d seed=%d x0=%d z0=%d' % (R, SEED, X0, Z0)
    oracle('bench-terrain threads=1 reps=1 warmup=0 reset=1 dump=1 tag=video_verify ' + common)
    for t, warm in ((1, 2), (32, 3)):
        oracle('bench-terrain threads=%d reps=1 warmup=%d timeline=1 tag=video_van %s' % (t, warm, common))
        shutil.copy(ROOT / 'data' / 'bench' / ('video_van_timeline_t%d.csv' % t), DATA / ('vanilla_t%d_timeline.csv' % t))
    region = ['--region', R, '--x0', X0, '--z0', Z0, '--seed', SEED, '--reps', 1, '--out', DATA / 'timing.csv']
    for t, warm in ((1, 1), (32, 2)):
        run([bench, '--mode', 'terrain-cpu', '--threads', t, '--warmup', warm, '--timeline', DATA / ('cpu_t%d_timeline.csv' % t)] + region)
    run([bench, '--mode', 'terrain-gpu', '--batch', 256, '--streams', 2, '--warmup', 3, '--timeline', DATA / 'gpu_timeline.csv',
         '--topdown', DATA / 'region32_topdown.bin', '--verify', ROOT / 'data' / 'bench' / 'video_verify_terrain.u16'] + region)
    run([bench, '--mode', 'terrain-gpu', '--region', 128, '--x0', X128, '--z0', Z128, '--seed', SEED, '--reps', 1, '--warmup', 2, '--batch', 2048,
         '--streams', 2, '--timeline', DATA / 'gpu128_timeline.csv', '--topdown', DATA / 'region128_topdown.bin', '--out', DATA / 'timing.csv'])


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'record':
        record()
    elif len(sys.argv) >= 4 and sys.argv[1] == 'preview':
        preview(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'render':
        render()
    else:
        print(__doc__)
