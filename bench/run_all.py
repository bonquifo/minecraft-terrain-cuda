"""Runs every benchmark behind docs/index.html and README.md, writing CSVs and logs to
bench/results/final/. Needs the Release build (build.ps1), the oracle data (oracle/regen.ps1)
and JDK 25 (.tools/jdk-25 or JAVA_HOME).

    python bench/run_all.py [--quick]

Everything is measured on this machine with unlocked clocks; repeated runs give the variance.
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'bench' / 'results' / 'final'
BENCH = ROOT / 'build' / 'bench' / 'mccuda_bench.exe'
TESTS = ROOT / 'build' / 'tests' / 'mccuda_tests.exe'
DATA_BENCH = ROOT / 'data' / 'bench'
SEED = 123456789
LOG = None

# Regions dumped from vanilla (threads=1, R-tree cache reset per chunk) and compared
# block for block with our generators: origin, negative coordinates, world border.
VERIFY = [('origin', 0, 0, 16), ('negative', -300, -220, 16), ('border', 1874992, 1874992, 8)]


HOME = os.path.expanduser('~')


def redact(text):
    return text.replace(HOME, '%USERPROFILE%')   # keep local user names out of the committed logs


def log(msg):
    line = redact('[%s] %s' % (datetime.datetime.now().strftime('%H:%M:%S'), msg))
    print(line, flush=True)
    LOG.write(line + '\n')
    LOG.flush()


def run(cmd, cwd=ROOT, env=None):
    log('$ ' + ' '.join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    for line in p.stdout.splitlines():
        if 'WARNING' in line or 'sun.misc.Unsafe' in line or line.startswith('[') and 'INFO' in line and 'STDOUT' not in line:
            continue
        LOG.write('    ' + redact(line) + '\n')
    LOG.flush()
    tail = [l for l in p.stdout.splitlines() if l.strip()][-3:]
    for l in tail:
        print('    ' + l, flush=True)
    if p.returncode != 0:
        raise SystemExit('command failed (%d): %s' % (p.returncode, cmd))
    return p.stdout


def oracle(cmd):
    env = dict(os.environ)
    if not env.get('JAVA_HOME'):
        env['JAVA_HOME'] = str(ROOT / '.tools' / 'jdk-25')
    return run([ROOT / 'oracle' / 'gradlew.bat', '--console=plain', '-q', 'oracle', '-Pcmd=' + cmd], cwd=ROOT / 'oracle', env=env)


def bench(*args):
    return run([BENCH, *args])


def gpu_info(tag):
    q = 'name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,pstate'
    out = subprocess.run(['nvidia-smi', '--query-gpu=' + q, '--format=csv'], stdout=subprocess.PIPE, text=True).stdout
    (OUT / ('gpu_state_%s.csv' % tag)).write_text(out)
    log('gpu state %s: %s' % (tag, out.strip().splitlines()[-1]))


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='fewer repetitions (smoke test of the script)')
    args = ap.parse_args()
    runs = 1 if args.quick else 3
    reps = 2 if args.quick else 5
    OUT.mkdir(parents=True, exist_ok=True)
    LOG = open(OUT / 'run_all.log', 'w', encoding='utf-8')
    log('run_all.py start, quick=%s' % args.quick)
    for f in OUT.glob('*.csv'):
        f.unlink()
    gpu_info('start')
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | Format-List; '
                    'Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version | Format-List; '
                    '(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],
                   stdout=open(OUT / 'machine.txt', 'w'), text=True)

    # 1. correctness: full suite on the full oracle set
    out = run([TESTS, '--set', 'full'])
    (OUT / 'tests_full.txt').write_text(out)

    # 2. vanilla region dumps and cross-checks of both generators
    for name, x0, z0, r in VERIFY:
        tag = 'verify_' + name
        oracle('bench-terrain threads=1 region=%d reps=1 warmup=0 seed=%d x0=%d z0=%d tag=%s reset=1 dump=1' % (r, SEED, x0, z0, tag))
        dump = DATA_BENCH / (tag + '_terrain.u16')
        bench('--mode', 'terrain-gpu', '--region', r, '--x0', x0, '--z0', z0, '--seed', SEED, '--reps', 1, '--warmup', 0, '--batch', 256,
              '--verify', dump, '--tag', tag, '--out', OUT / 'verify_gpu.csv')
        bench('--mode', 'terrain-cpu', '--region', r, '--x0', x0, '--z0', z0, '--seed', SEED, '--reps', 1, '--warmup', 0, '--threads', 32,
              '--verify', dump, '--tag', tag, '--out', OUT / 'verify_cpu.csv')

    # 3. vanilla full TERRAIN throughput (thread scaling; repeated JVM runs at 1 and 32 threads)
    for t in (1, 2, 4, 8, 16, 32):
        region = 16 if t <= 4 else 32
        for run_i in range(runs if t in (1, 32) else 1):
            tag = 'vanilla_terrain_run%d' % run_i
            oracle('bench-terrain threads=%d region=%d reps=%d warmup=2 seed=%d x0=0 z0=0 tag=%s' % (t, region, reps, SEED, tag))
            shutil.copy(DATA_BENCH / ('%s_t%d.csv' % (tag, t)), OUT / ('vanilla_terrain_t%d_run%d.csv' % (t, run_i)))

    # 4. vanilla core (final density + fill, aquifers off), the Phase 3 comparison
    for t, region in ((1, 16), (32, 32)):
        oracle('bench-vanilla threads=%d region=%d reps=%d warmup=2 seed=%d x0=0 z0=0 tag=vanilla_core' % (t, region, reps, SEED))
        shutil.copy(DATA_BENCH / ('vanilla_core_t%d.csv' % t), OUT / ('vanilla_core_t%d.csv' % t))

    # 5. CPU port (single-threaded C++ reference port; also run on all threads)
    for t, region in ((1, 16), (32, 32)):
        for run_i in range(runs):
            bench('--mode', 'terrain-cpu', '--region', region, '--seed', SEED, '--reps', reps, '--warmup', 1, '--threads', t,
                  '--tag', 'run%d' % run_i, '--out', OUT / 'cpu_terrain.csv')
        bench('--mode', 'core-cpu', '--region', region, '--seed', SEED, '--reps', reps, '--warmup', 1, '--threads', t, '--out', OUT / 'cpu_core.csv')

    # 6. GPU, final version
    gpu_info('before_gpu')
    for run_i in range(runs):   # headline: end to end, transfers overlapped
        bench('--mode', 'terrain-gpu', '--region', 128, '--seed', SEED, '--reps', reps, '--warmup', 2, '--batch', 2048, '--streams', 2,
              '--tag', 'run%d' % run_i, '--out', OUT / 'gpu_headline.csv')
    for b in (64, 256, 1024, 2048, 4096):   # batch sweep, both pipeline depths
        for s in (1, 2):
            bench('--mode', 'terrain-gpu', '--region', 128, '--seed', SEED, '--reps', reps, '--warmup', 2, '--batch', b, '--streams', s,
                  '--out', OUT / 'gpu_batch_sweep.csv')
    for r in (1, 2, 3, 4, 8, 16):   # small requests: latency of one call
        bench('--mode', 'terrain-gpu', '--region', r, '--seed', SEED, '--reps', 20, '--warmup', 3, '--batch', r * r, '--streams', 1,
              '--out', OUT / 'gpu_latency.csv')
    bench('--mode', 'terrain-gpu', '--region', 128, '--seed', SEED, '--reps', reps, '--warmup', 2, '--batch', 2048, '--streams', 2, '--pageable',
          '--tag', 'pageable', '--out', OUT / 'gpu_pageable.csv')
    bench('--mode', 'core-gpu', '--region', 64, '--seed', SEED, '--reps', reps, '--warmup', 2, '--batch', 1024, '--version', 0,
          '--out', OUT / 'gpu_core.csv')
    for s in (SEED, 1, 2, 3, 42):   # how often the R-tree's last-result rule decides a lookup (writes gpu_biome_ties.csv)
        bench('--mode', 'terrain-gpu', '--region', 128, '--seed', s, '--reps', 1, '--warmup', 0, '--batch', 2048, '--tag', 'ties',
              '--out', OUT / 'gpu_ties_timing.csv')
    gpu_info('end')
    log('run_all.py done')


if __name__ == '__main__':
    main()
