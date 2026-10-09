"""First-person flythrough video of terrain generated live on the GPU (viz/fly.cu).

mccuda_fly generates the chunks around the camera every frame with the bit-exact
pipeline and ray traces the view; this script writes the camera path, adds the
overlay from each frame's measured timings and encodes docs/videos/flythrough.mp4.

    python bench/make_flythrough.py
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'videos'
DATA = ROOT / 'data' / 'videos'
FLY = ROOT / 'build' / 'viz' / 'mccuda_fly.exe'
W, H, FPS, RADIUS = 1280, 720, 30, 30
VANILLA_32T = 2057.0          # chunks/s, bench/results/final (vanilla 26.3, 32 threads)
FONTS = Path('C:/Windows/Fonts')

# t, x, z, mode (g: height above ground, a: absolute y), y, pitch (degrees)
PATH = """
0.0  7480 -4060 g 1.62   2
3.0  7468 -4072 g 1.62   3
6.0  7455 -4085 g 1.62   4
9.0  7400 -4170 a 125   -8
12.0 7310 -4320 a 165  -16
15.0 7250 -4470 a 180  -20
18.0 7400 -4590 a 180  -20
21.0 7640 -4570 a 175  -22
24.0 7800 -4410 a 160  -22
27.0 7840 -4170 a 140  -20
30.0 7720 -3930 a 120  -18
33.0 7600 -3700 a 105  -15
36.0 7520 -3520 a 95   -12
38.0 7480 -3440 a 95   -10
"""


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path_file = DATA / 'flythrough_path.txt'
    path_file.write_text('# t x z mode y pitch\n' + PATH.strip() + '\n', encoding='utf-8')
    fly = subprocess.Popen([str(FLY), '--path', str(path_file), '--width', str(W), '--height', str(H), '--fps', str(FPS), '--radius', str(RADIUS)],
                           stdout=subprocess.PIPE, cwd=ROOT)
    enc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-',
                            '-c:v', 'libx264', '-preset', 'slow', '-crf', '29', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(OUT / 'flythrough.mp4')],
                           stdin=subprocess.PIPE)
    title = ImageFont.truetype(str(FONTS / 'segoeuib.ttf'), 22)
    small = ImageFont.truetype(str(FONTS / 'segoeui.ttf'), 15)
    mono = ImageFont.truetype(str(FONTS / 'consola.ttf'), 16)
    monob = ImageFont.truetype(str(FONTS / 'consolab.ttf'), 18)
    stats, frame_bytes, poster = [], W * H * 3, None
    last = None   # latest frame that generated chunks
    while True:
        head = fly.stdout.read(256)
        if len(head) < 256:
            break
        kv = dict(item.split('=', 1) for item in head.decode('ascii').split() if '=' in item)
        raw = fly.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        st = {k: (v if k == 'mode' else float(v)) for k, v in kv.items()}
        stats.append(st)
        im = Image.frombuffer('RGB', (W, H), raw, 'raw', 'RGB', 0, 1).copy()
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle([0, 0, W, 54], fill=(10, 14, 12, 150))
        d.rectangle([0, H - 64, W, H], fill=(10, 14, 12, 150))
        im = Image.alpha_composite(im.convert('RGBA'), ov)
        d = ImageDraw.Draw(im)
        d.text((18, 8), 'Terrain generated live on the RTX 5090, seen in first person', fill=(236, 241, 236), font=title)
        d.text((18, 33), 'bit-exact Minecraft 26.3 TERRAIN step \u00b7 ray traced on the GPU \u00b7 flat procedural colours, no game textures',
               fill=(170, 185, 175), font=small)
        moving = 'walking (eye height 1.62)' if st.get('mode') == 'g' else 'flying'
        tag = '%s   t = %5.2f s' % (moving, st['t'])
        d.text((W - 18 - d.textlength(tag, font=mono), 18), tag, fill=(214, 190, 120), font=mono)
        n = int(st['new'])
        if n:
            last = st
        if last is not None:
            ago = st['t'] - last['t']
            when = 'this frame' if ago < 1e-9 else '%.1f s ago' % ago
            line = 'latest batch (%s): %d chunks generated in %.1f ms' % (when, int(last['new']), last['gen_ms'])
            van = 'vanilla, 32 threads, at its measured rate: %.0f ms for those chunks' % (last['new'] / VANILLA_32T * 1000.0)
        else:
            line, van = 'generating...', ''
        d.text((18, H - 54), line, fill=(143, 207, 104) if n else (200, 215, 205), font=monob)
        if van:
            d.text((18, H - 28), van, fill=(170, 185, 175), font=mono)
        right = 'chunks generated so far: {:,}'.format(int(st['total']))
        d.text((W - 18 - d.textlength(right, font=monob), H - 54), right, fill=(236, 241, 236), font=monob)
        pos = 'x %.0f  y %.0f  z %.0f' % (st['x'], st['y'], st['z'])
        d.text((W - 18 - d.textlength(pos, font=mono), H - 28), pos, fill=(170, 185, 175), font=mono)
        rgb = np.asarray(im.convert('RGB'))
        enc.stdin.write(rgb.tobytes())
        if poster is None and st['t'] >= 16.0:
            poster = im.convert('RGB')
    enc.stdin.close()
    if fly.wait() != 0 or enc.wait() != 0:
        raise SystemExit('flythrough failed')
    if poster is not None:
        poster.save(OUT / 'flythrough.jpg', quality=88)
    gen = [s for s in stats if s['new'] > 0]
    steady = [s for s in gen if s['t'] >= 1.0]
    summary = {
        'frames': len(stats), 'fps': FPS, 'width': W, 'height': H, 'radius_chunks': RADIUS,
        'chunks_total': int(stats[-1]['total']) if stats else 0,
        'startup': {'frames': sum(1 for s in stats if s['t'] < 1.0 and s['new'] > 0), 'chunks': int(sum(s['new'] for s in stats if s['t'] < 1.0)),
                    'gen_ms': sum(s['gen_ms'] for s in stats if s['t'] < 1.0)},
        'steady': {'frames_with_new': len(steady), 'chunks': int(sum(s['new'] for s in steady)),
                   'gen_ms_max': max((s['gen_ms'] for s in steady), default=0.0),
                   'gen_ms_mean': (sum(s['gen_ms'] for s in steady) / len(steady)) if steady else 0.0,
                   'new_mean': (sum(s['new'] for s in steady) / len(steady)) if steady else 0.0},
        'render_ms_mean': sum(s['render_ms'] for s in stats) / len(stats) if stats else 0.0,
        'render_ms_max': max((s['render_ms'] for s in stats), default=0.0),
    }
    (OUT / 'flythrough.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
