"""In-game videos: the real Minecraft client flying through GPU-generated terrain.

client/ is a Fabric mod for a dev client. With the GPU dump of a 136 x 136-chunk region
(mccuda_bench --dump), every chunk of the region gets its blocks and biomes from the GPU
instead of vanilla's noise fill, surface rules and biome placement; carvers, features
and light stay the game's. The recorder flies the GPU flythrough's camera path
(bench/make_flythrough.py) and captures every frame. Four takes, structures off unless
noted (the GPU pipeline leaves out the terrain adaptation around structures):

    cd client
    gradlew runClient                                                   GPU chunks -> ingame_raw
    gradlew runClient -Pserve=false -Pout=vanilla_raw                   vanilla, checked against the dump
    gradlew runClient -Pserve=false -Pout=vanilla2_raw                  vanilla again (run-to-run baseline)
    gradlew runClient -Pserve=false -Pstructures=true -Pout=vanilla_struct_raw
    cd ..
    python bench/make_flythrough.py          (also writes data/videos/flythrough_raw.mp4)
    python bench/make_ingame.py

This script adds the overlay (docs/videos/ingame.mp4), puts the GPU ray tracer and the
real client side by side on the same frames (compare.mp4), measures how far the frames of
the takes differ, and writes the evidence to ingame.json.
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'docs' / 'videos'
DATA = ROOT / 'data' / 'videos'
FONTS = Path('C:/Windows/Fonts')
W, H, FPS = 1280, 720, 30


def frames(path, w=W, h=H):
    """Decoded RGB frames of a video, scaled to w x h."""
    p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-vf', 'scale=%d:%d:flags=lanczos' % (w, h), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         stdout=subprocess.PIPE)
    n = w * h * 3
    while True:
        b = p.stdout.read(n)
        if len(b) < n:
            break
        yield Image.frombuffer('RGB', (w, h), b, 'raw', 'RGB', 0, 1)
    p.wait()


def frame(path, k):
    b = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-vf', 'select=eq(n\\,%d)' % k, '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(H, W, 3)


def encoder(path, w, h, crf):
    return subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (w, h), '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)],
                            stdin=subprocess.PIPE)


def ab(x, y):
    """How far two takes of the same path differ, frame by frame (max channel difference per pixel)."""
    mean, over = [], []
    for a, b in zip(frames(DATA / x), frames(DATA / y)):
        d = np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).max(axis=2)
        mean.append(float(d.mean()))
        over.append(float((d > 8).mean()))
    return {'frames': len(mean), 'identical_frames': sum(1 for v in mean if v == 0.0), 'mean_abs_diff': float(np.mean(mean)),
            'pixels_over_8_mean': float(np.mean(over)), 'pixels_over_8_max': float(np.max(over)), 'worst_frame': int(np.argmax(over))}


def trees_still(k, path):
    """GPU take, vanilla take and a second vanilla take at frame k, cropped where the first two differ most."""
    takes = [frame(DATA / f, k) for f in ('ingame_raw.mp4', 'vanilla_raw.mp4', 'vanilla2_raw.mp4')]
    d = np.abs(takes[0].astype(np.int16) - takes[1].astype(np.int16)).max(axis=2) > 8
    cw, ch, best = 400, 225, (-1, 0, 0)
    for y in range(0, H - ch + 1, 15):
        for x in range(0, W - cw + 1, 20):
            s = int(d[y:y + ch, x:x + cw].sum())
            if s > best[0]:
                best = (s, x, y)
    _, x, y = best
    pw, ph, bar = cw * 3 // 2, ch * 3 // 2, 34
    font = ImageFont.truetype(str(FONTS / 'segoeuib.ttf'), 17)
    im = Image.new('RGB', (3 * pw + 16, ph + bar), (14, 18, 16))
    d = ImageDraw.Draw(im)
    for i, (t, label) in enumerate(zip(takes, ('GPU chunks', 'vanilla', 'vanilla, second run'))):
        im.paste(Image.fromarray(t[y:y + ch, x:x + cw]).resize((pw, ph), Image.NEAREST), (i * (pw + 8), bar))
        d.text((i * (pw + 8) + 8, 7), label, fill=(236, 241, 236), font=font)
    im.save(path, quality=90)


def overlay_ingame(meta, rows):
    title = ImageFont.truetype(str(FONTS / 'segoeuib.ttf'), 22)
    small = ImageFont.truetype(str(FONTS / 'segoeui.ttf'), 15)
    mono = ImageFont.truetype(str(FONTS / 'consola.ttf'), 16)
    monob = ImageFont.truetype(str(FONTS / 'consolab.ttf'), 18)
    region = meta['region']
    enc = encoder(OUT / 'ingame.mp4', W, H, 27)
    poster = None
    for k, im in enumerate(frames(DATA / 'ingame_raw.mp4')):
        r = rows[min(k, len(rows) - 1)]
        ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        d.rectangle([0, 0, W, 54], fill=(10, 14, 12, 150))
        d.rectangle([0, H - 64, W, H], fill=(10, 14, 12, 150))
        im = Image.alpha_composite(im.convert('RGBA'), ov)
        d = ImageDraw.Draw(im)
        d.text((18, 8), 'The real Minecraft 26.3 client, flying through terrain generated on the RTX 5090', fill=(236, 241, 236), font=title)
        d.text((18, 33), 'blocks and biomes of every chunk in view come from the GPU \u00b7 caves, trees and lighting are the game\'s own \u00b7 structures off',
               fill=(170, 185, 175), font=small)
        tag = 't = %5.2f s' % float(r['t'])
        d.text((W - 18 - d.textlength(tag, font=mono), 18), tag, fill=(214, 190, 120), font=mono)
        line = 'chunks filled from the GPU output: {:,} \u00b7 chunks in view from anywhere else: {}'.format(int(r['gpu_chunks']),
                                                                                                          int(r['client_chunks_outside_region']))
        d.text((18, H - 54), line, fill=(143, 207, 104), font=monob)
        sub = 'region %d x %d chunks from (%d, %d) \u00b7 seed %d \u00b7 render distance %d' % (region['size'], region['size'], region['x0'], region['z0'],
                                                                                             meta['seed'], meta['render_distance'])
        d.text((18, H - 28), sub, fill=(170, 185, 175), font=mono)
        pos = 'x %.0f  y %.0f  z %.0f' % (float(r['x']), float(r['y']), float(r['z']))
        d.text((W - 18 - d.textlength(pos, font=mono), H - 28), pos, fill=(170, 185, 175), font=mono)
        enc.stdin.write(np.asarray(im.convert('RGB')).tobytes())
        if poster is None and float(r['t']) >= 16.0:
            poster = im.convert('RGB')
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit('ingame encode failed')
    poster.save(OUT / 'ingame.jpg', quality=88)


def side_by_side():
    """GPU ray tracer (left) and the real client (right) on the same frames of the same path."""
    hw, hh, bar = 960, 540, 40
    monob = ImageFont.truetype(str(FONTS / 'consolab.ttf'), 18)
    mono = ImageFont.truetype(str(FONTS / 'consola.ttf'), 16)
    enc = encoder(OUT / 'compare.mp4', 2 * hw, hh + bar, 29)
    poster = None
    for k, (a, b) in enumerate(zip(frames(DATA / 'flythrough_raw.mp4', hw, hh), frames(DATA / 'ingame_raw.mp4', hw, hh))):
        im = Image.new('RGB', (2 * hw, hh + bar), (14, 18, 16))
        im.paste(a, (0, bar))
        im.paste(b, (hw, bar))
        d = ImageDraw.Draw(im)
        d.text((14, 9), 'our CUDA ray tracer: TERRAIN step generated live', fill=(236, 241, 236), font=monob)
        d.text((hw + 14, 9), 'Minecraft 26.3 client: GPU chunks + the game\'s later steps', fill=(236, 241, 236), font=monob)
        t = 't = %5.2f s' % (k / FPS)
        d.text((2 * hw - 14 - d.textlength(t, font=mono), 11), t, fill=(214, 190, 120), font=mono)
        d.line([(hw, 0), (hw, hh + bar)], fill=(14, 18, 16), width=4)
        enc.stdin.write(np.asarray(im).tobytes())
        if poster is None and k / FPS >= 16.0:
            poster = im
    enc.stdin.close()
    if enc.wait() != 0:
        raise SystemExit('compare encode failed')
    poster.save(OUT / 'compare.jpg', quality=88)


def main():
    meta = json.loads((DATA / 'ingame_raw.json').read_text(encoding='utf-8'))
    with open(DATA / 'ingame_raw.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    overlay_ingame(meta, rows)
    side_by_side()
    gv = ab('ingame_raw.mp4', 'vanilla_raw.mp4')
    vv = ab('vanilla_raw.mp4', 'vanilla2_raw.mp4')
    trees_still(gv['worst_frame'], OUT / 'trees.jpg')
    takes = {n: json.loads((DATA / (n + '.json')).read_text(encoding='utf-8')) for n in ('vanilla_raw', 'vanilla2_raw', 'vanilla_struct_raw')}
    fps = [int(r['client_fps']) for r in rows]
    summary = dict(meta)
    summary.update({
        'client_fps_median': float(np.median(fps)), 'client_fps_min': min(fps),
        'client_chunks_max': max(int(r['client_chunks']) for r in rows),
        'client_chunks_outside_region_max': max(int(r['client_chunks_outside_region']) for r in rows),
        'ingame_mb': round((OUT / 'ingame.mp4').stat().st_size / 1e6, 1), 'compare_mb': round((OUT / 'compare.mp4').stat().st_size / 1e6, 1),
        'frames_gpu_vs_vanilla': gv, 'frames_vanilla_vs_vanilla': vv,
        'control_takes': {n: {k: t[k] for k in ('structures', 'vanilla_chunks', 'verify')} for n, t in takes.items()},
    })
    (OUT / 'ingame.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
