"""Profiles the final GPU pipeline with Nsight Systems and writes a per-kernel summary to
bench/results/final/nsys_kernels.csv (used by docs/make_report.py).

    python profiles/summarize_nsys.py
"""
import csv
import sqlite3
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NSYS = r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64\nsys.exe'
REP = ROOT / 'profiles' / 'final_r64_b4096'
OUT = ROOT / 'bench' / 'results' / 'final' / 'nsys_kernels.csv'
REPS = 3

WHAT = {
    'k_biome_columns': 'climate inputs that do not depend on y, per quart column',
    'k_biome_quarts': 'climate at every quart + bounded R-tree search (first minimal leaf)',
    'k_biome_ties': 'flags where the previous leaf ties the minimum',
    'k_biome_chain': "R-tree last-result rule (one warp per biome chunk)",
    'k_col_keys': 'corner-column keys for the dedupe',
    'k_columns': 'y-independent density inputs per distinct corner column',
    'k_axis_xz': 'x/z noise coordinate records per distinct column',
    'k_corners': 'density and ore channels at the 4x8x4 cell corners',
    'k_grid_keys': 'surface-level cell keys for the dedupe',
    'k_fts_unique': 'find_top_surface per distinct surface-level cell',
    'k_grid_scatter': 'surface levels back to each chunk',
    'k_aquifer_skip': 'aquifer skip height per chunk',
    'k_aquifer_cells': 'aquifer cell locations and fluid status',
    'k_blocks': 'interpolation, final density and aquifer substance per block',
    'k_heights': 'initial heightmap',
    'k_surface1': 'eroded-badlands extension, frozen-ocean heights',
    'k_surface2': 'material rules (incl. ore veins) and frozen-ocean extension',
    'k_iota': 'dedupe: indices', 'k_heads': 'dedupe: run heads', 'k_scatter': 'dedupe: unique keys and map',
    'DeviceRadixSort': 'dedupe: CUB radix sort', 'DeviceScan': 'dedupe: CUB scan',
}


def short(name):
    for k in WHAT:
        if k in name:
            return k
    return name.split('(')[0][-40:]


def main():
    subprocess.run([NSYS, 'profile', '--trace=cuda', '--force-overwrite=true', '--export=sqlite', '-o', str(REP), str(ROOT / 'build' / 'bench' / 'mccuda_bench.exe'),
                    '--mode', 'terrain-gpu', '--region', '64', '--seed', '123456789', '--reps', str(REPS), '--warmup', '1', '--batch', '4096', '--streams', '1',
                    '--out', str(ROOT / 'profiles' / 'nsys_bench.csv')], check=True, stdout=subprocess.DEVNULL)
    con = sqlite3.connect(str(REP) + '.sqlite')
    names = dict(con.execute('select id, value from StringIds').fetchall())
    rows = con.execute('select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    batches = REPS + 1
    agg = defaultdict(lambda: [0.0, 0])
    for s, e, n in rows:
        k = short(names.get(n, '?'))
        agg[k][0] += (e - s) / 1e6
        agg[k][1] += 1
    total = sum(v[0] for v in agg.values())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'avg_ms', 'pct', 'launches_per_batch', 'what'])
        for k, (ms, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            w.writerow([k, '%.4f' % (ms / batches), '%.2f' % (ms / total * 100), '%.1f' % (cnt / batches), WHAT.get(k, '')])
    print('kernel launches per batch: %.1f; kernel time per batch %.2f ms -> %s' % (len(rows) / batches, total / batches, OUT))


if __name__ == '__main__':
    main()
