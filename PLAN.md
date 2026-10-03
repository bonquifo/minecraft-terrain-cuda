# PLAN: GPU-accelerated Minecraft terrain generation (CUDA)

Phase 0 deliverable, written 2026-10-02 on the target machine (Ryzen 9 9950X, 128 GB,
RTX 5090). Every command in section 3 was run here, and its real output is recorded.
Phase 0 adds only environment checks under [`tests/env/`](tests/env/); no project code
has been written.

---

## 0. Summary

**What I found**

1. **The latest stable Java release is 26.3** (2026-09-15, per Mojang's version manifest).
   It needs **Java 25** and ships **unobfuscated**: Mojang no longer publishes mapping
   files because the names are in the jar. Fabric supports it with Loader 0.19.5 and Loom 1.18.2.
2. **26.3 replaced the terrain engine your spec describes.** In 1.18–1.21 terrain was
   `ImprovedNoise`, `NoiseChunk` cell interpolation and all-`double` density. In 26.3 it is a
   *compiled, batched, float-valued* density engine:
   - Density values are `float` (`DensityBuffer` holds a `float[]`, and
     `DensitySampler.sampleValue` returns `float`).
   - Perlin octaves do their coordinate work in `double` but the lattice math (smoothstep,
     gradients, lerps) in `float`.
   - Vanilla evaluates whole `DensityVolume`s at once, through a `DensityFunctionCompiler`
     with rewrite rules.
   - Interpolation is now a density-function node (`InterpolatedFunction`, 4×8 cells, used
     5 times in the overworld router).
   - NOISE, SURFACE and CARVERS are merged into one `TERRAIN` chunk step.
   - Surface rules became "material rules", and ore veins are one of them.

   `ImprovedNoise` no longer exists.
3. **FP64 is still the GPU bottleneck in 26.3**, but now it sits only in the per-octave
   coordinate pipeline (scale, wrap, offset, floor, fraction, convert to float). The measured
   consumer-Blackwell FP64:FP32 rate is about **1:90** for the unfused mul+add that Java
   semantics force. At that rate, roughly 20–35 double ops per octave outweigh roughly 40–70
   float ops by a wide margin. That is a well-contained place to apply exact integer and
   fixed-point tricks (section 4.7).
4. **Vanilla has two evaluation paths, and they disagree.** Per-point sampling and per-volume
   sampling differ bitwise on **23–30% of blocks**: by up to about 2e-7, with corners agreeing
   and no solid/air flips in the 6 chunks tested. Chunk generation uses the volume path, so
   that is the one we must match.
5. **Environment:** CUDA 13.1, MSVC 14.44, CMake, both Nsight tools, Python and git are
   installed and working. The sm_120 hello world passes, including a host/device FP parity
   check. **Missing:** JDK 25, Ninja, and Nsight Compute counter permission. I ran the Java
   harness smoke test and the CMake CUDA build with *portable* JDK 25 and Ninja copies in a
   temp folder. Both pass. Install steps are in section 3.3.

**What I recommend:** pin **Minecraft 26.3** and develop **natively on Windows**. Make the
core definition of done: bit-exact final density per block, plus bit-exact block fill from
vanilla's own `buildTerrain` with aquifers disabled and an empty material rule. Treat aquifers
as stretch S1. Section 7 lists the decisions I need from you.

---

## 1. Platform and version

### 1.1 Minecraft version: recommend 26.3, with 1.21.11 as the alternative

| | **26.3** (recommended) | **1.21.11** (alternative) |
|---|---|---|
| Released | 2026-09-15 (latest release) | 2025-12-09 (last 1.x release) |
| Java | 25 | 21 |
| Names | Unobfuscated jar; no mappings needed | Obfuscated; official mappings via Loom |
| Density engine | Compiled, batched, `float` values; `double` only in coordinate handling | Per-point, `double` everywhere (`ImprovedNoise`, `NoiseChunk`) |
| FP64 story on a 5090 | Concentrated in the coordinate pipeline, which is good ground for exact integer tricks | Everywhere; a bigger FP64 tax and a lower ceiling |
| Vanilla CPU baseline | Already batched; a stronger, more honest baseline | Classic per-point evaluator |
| Documentation | New in the 26.x line (I didn't check which 26.x release introduced it); little community documentation, so I learn it from the local jar | Well documented by the community |
| Matches spec wording | Partly (class names changed) | Yes |

26.3 is what a server runs today. It's also the version you asked me to recommend, and its
float engine keeps the FP64 problem in one part of the pipeline where it can be attacked.
The cost is more Phase 1 study of a young engine. Changing version after Phase 2 would mean
redoing most of the work, so **please confirm** (Q1).

### 1.2 Native Windows (recommended) vs. WSL2

**Native Windows**, because:

- Every tool is already installed and verified natively (section 3). WSL2 would mean a
  second Linux CUDA toolchain to keep in sync.
- WSL2 doesn't escape WDDM. It reaches the GPU through the same Windows driver, so it shares
  the display adapter and the TDR watchdog. A GeForce card can't use TCC either way. WSL2
  adds restrictions on pinned memory, profiler caveats, and slow `/mnt/c` I/O.
- Vanilla, Fabric and the stretch JNI mod all run natively, with no VM boundary in the
  end-to-end numbers.

Costs I'll manage:

- WDDM launch overhead and the roughly 2 s TDR limit: size batches so each launch stays well
  under it.
- The 5090 also drives the desktop. During these checks it already showed 7.4 GB used and
  22% utilization from background apps, so the benchmark protocol closes them and reports
  variance.
- MSVC instead of GCC for the CPU port. x64 MSVC uses SSE2 scalar FP with no x87 excess
  precision, which is fine with the flags in section 4.7.

---

## 2. Dependencies

| Tool | Minimum | Why | Installed here | Status |
|---|---|---|---|---|
| NVIDIA driver | Reports CUDA ≥ 13.1 | Runs the CUDA 13.1 runtime on sm_120 | 591.86 (CUDA 13.1) | OK |
| CUDA Toolkit | 12.8 (first with sm_120) | nvcc, cuobjdump, compute-sanitizer | **13.1.115** | OK |
| MSVC host compiler | VS 2022 toolset 14.2x–14.4x | CUDA 13.1 `host_config.h` rejects `_MSC_VER >= 1950` | 14.44 (`cl` 19.44.35228) | OK. **VS 2026 (14.50) is also installed and is rejected**, so builds must import the VS 2022 `vcvars64.bat` |
| CMake | 3.28 | Build system. Always pass `CMAKE_CUDA_ARCHITECTURES=120`, because CUDA 13 dropped pre-Turing targets that older CMake defaults probe | 4.4.2 | OK |
| Ninja | 1.11 | CMake generator. The VS generator can't build CUDA here because CUDA's MSBuild integration isn't installed in any VS | none | **MISSING** |
| JDK | **25** (required by 26.3's `javaVersion`) | Runs vanilla code for the oracle, and Gradle | Only Android Studio's JDK 21.0.8 and the launcher's Java 21 runtime | **MISSING** |
| Gradle | 9.7.1 via wrapper | Builds the oracle; the version the Fabric template uses for 26.3 | Wrapper checked in, SHA-256 verified | OK (needs JDK 25) |
| Fabric Loom | 1.18.2 (`net.fabricmc.fabric-loom`, the non-remapping plugin) | Downloads the 26.3 jar and libraries; can decompile into the Gradle cache for local reading | Fetched by Gradle | OK |
| Fabric Loader / API | 0.19.5 / 0.161.0+26.3 | Only for stretch S2 (server mod) | Not needed yet | N/A |
| Nsight Compute | 2025.x with sm_120 | Kernel profiling (Phase 4) | 2025.4.1 | **Needs counter permission** (`ERR_NVGPUCTRPERM`) |
| Nsight Systems | 2025.x | Timeline, overlap and transfer analysis | 2025.5.2 | OK (not on PATH; scripts find it) |
| Python | 3.10, plus numpy and matplotlib | Charts, codegen, test driver | 3.11.9, numpy 2.4.4, matplotlib 3.10.8 | OK |
| git | 2.x | Version control | 2.53.0 | OK |
| JMH | 1.37 (Gradle plugin) | Rigorous vanilla Java benchmarks (Phase 5) | Fetched by Gradle | Later |
| Admin shell (optional) | | `nvidia-smi -lgc` for locked GPU clocks; changing the counter policy | Session not elevated | See Q4 |

---

## 3. Verification

### 3.1 Commands and results

Everything below runs with one command:
`powershell -ExecutionPolicy Bypass -File tests\env\check_env.ps1 [-JavaHome <jdk25>] [-NinjaDir <dir>]`.
The script imports the VS 2022 x64 environment first.

| # | Check | Command | Result on this machine |
|---|---|---|---|
| 1 | Driver/GPU | `nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv` | `NVIDIA GeForce RTX 5090, 591.86, 12.0, 32607 MiB`: **PASS** |
| 2 | CUDA | `nvcc --version` | `Cuda compilation tools, release 13.1, V13.1.115`: **PASS** |
| 3 | MSVC | `vcvars64.bat` (VS 2022), then `cl` | `Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64`: **PASS** |
| 4 | CUDA accepts MSVC | `grep _MSC_VER "%CUDA_PATH%\include\crt\host_config.h"` | `#if _MSC_VER < 1920 \|\| _MSC_VER >= 1950 → #error unsupported`. 19.44 OK; VS 2026's 19.50 rejected |
| 5 | CMake | `cmake --version` | `cmake version 4.4.2`: **PASS** |
| 6 | Ninja | `ninja --version` | not found: **MISSING** (portable 1.13.2 used for #8) |
| 7 | CUDA hello on sm_120 | `nvcc -O3 -arch=sm_120 -fmad=false hello_sm120.cu && cuobjdump --list-elf && hello_sm120.exe` | sm_120 cubins; hello prints; FMA and subnormal parity match; `RESULT: PASS` (output below) |
| 8 | CMake + Ninja + CUDA | `cmake -S tests/env -B <dir> -G Ninja && cmake --build <dir>` | `CUDA compiler identification is NVIDIA 13.1.115 with host compiler MSVC 19.44.35228.0`, links: **PASS** (with portable Ninja) |
| 9 | Nsight Systems | `nsys --version`; `nsys profile --trace=cuda hello_sm120.exe` | `2025.5.2.266`; kernel summary produced: **PASS** |
| 10 | Nsight Compute | `ncu --version`; `ncu --kernel-name contraction --section SpeedOfLight hello_sm120.exe` | `2025.4.1.0`; then `ERR_NVGPUCTRPERM`: **FAIL** (permission) |
| 11 | JDK 25 | `java -version` | `'java' is not recognized`: **MISSING** (portable Temurin 25.0.4.1 used for #12) |
| 12 | Gradle + oracle smoke | `tests\env\oracle-smoke\gradlew.bat --version` and `gradlew.bat -q runSmoke` | `Gradle 9.7.1`, `Fabric Loom: 1.18.2`, vanilla 26.3 bootstraps, `SMOKE: PASS` (output below) |
| 13 | Python libs | `python -c "import numpy, matplotlib; ..."` | `numpy 2.4.4 matplotlib 3.10.8`: **PASS** |
| 14 | git | `git --version` | `git version 2.53.0.windows.1`: **PASS** |

Summary from the run on the machine as it is (`check_env.ps1`, no extra arguments). Output is
verbatim. The `CMake MISSING` row was a bug in my script (a truncated pipeline clobbered the exit
code; `cmake version 4.4.2` printed fine). It's fixed, and the second run shows the correct row:
```
MSVC                    PASS    Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64
Driver/GPU              PASS    NVIDIA GeForce RTX 5090, 591.86, 12.0, 32607 MiB
CUDA nvcc               PASS    Cuda compilation tools, release 13.1, V13.1.115
CMake                   MISSING
Ninja                   MISSING winget install Ninja-build.Ninja
CUDA sm_120 run         PASS    hello + FMA/subnormal parity checks passed
CMake CUDA build        SKIPPED needs Ninja (no CUDA MSBuild integration in VS either)
Nsight Systems          PASS    NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0 (not on PATH: ...)
Nsight Compute          FAIL    Version 2025.4.1.0 (build 37053803) (public-release); counters are admin-only (ERR_NVGPUCTRPERM)
JDK 25                  MISSING found: 'java' is not recognized as an internal or external command,
Oracle smoke (MC 26.3)  SKIPPED needs JDK 25
Python+numpy+matplotlib PASS    Python 3.11.9; numpy 2.4.4 matplotlib 3.10.8
git                     PASS    git version 2.53.0.windows.1
```
Summary with the portable JDK 25 and Ninja (`-JavaHome <temp>\jdk-25.0.4.1+1 -NinjaDir <temp>\ninja`).
The only remaining failure is the counter permission:
```
MSVC                    PASS   Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35228 for x64
Driver/GPU              PASS   NVIDIA GeForce RTX 5090, 591.86, 12.0, 32607 MiB
CUDA nvcc               PASS   Cuda compilation tools, release 13.1, V13.1.115
CMake                   PASS   cmake version 4.4.2
Ninja                   PASS   ninja 1.13.2
CUDA sm_120 run         PASS   hello + FMA/subnormal parity checks passed
CMake CUDA build        PASS   Ninja generator, sm_120, -fmad=false
Nsight Systems          PASS   NVIDIA Nsight Systems version 2025.5.2.266-255236693005v0 (not on PATH: ...)
Nsight Compute          FAIL   Version 2025.4.1.0 (build 37053803) (public-release); counters are admin-only (ERR_NVGPUCTRPERM)
JDK 25                  PASS   openjdk version "25.0.4.1" 2026-08-18 LTS
Oracle smoke (MC 26.3)  PASS   Gradle 9.7.1, Loom 1.18.2
Python+numpy+matplotlib PASS   Python 3.11.9; numpy 2.4.4 matplotlib 3.10.8
git                     PASS   git version 2.53.0.windows.1
```

### 3.2 Key outputs

<details>
<summary><b>CUDA hello world / FP parity / throughput probe</b> (<code>tests/env/hello_sm120.cu</code>)</summary>

```
ELF file    1: hello_sm120.1.sm_120.cubin
ELF file    2: hello_sm120.2.sm_120.cubin
hello from sm_120 block 0
hello from sm_120 block 1
device: NVIDIA GeForce RTX 5090  cc 12.0  SMs 170  clock 2437 MHz  mem 31.8 GiB  FP32:FP64 ratio 64
runtime 13.1  driver 13.1  smem/block opt-in 99 KiB  regs/SM 65536  maxThreads/SM 1536
a*b+c     host 0 (0x0000000000000000)  device 0 (0x0000000000000000)  MATCH
subnormal double 0x0000000400000000 vs host 0x0000000400000000 MATCH | float 0x00080000 vs host 0x00080000 MATCH
throughput (dependent x = x*a + b chains, 8 chains/thread):
  fp64       344.6 G mul+add pairs/s  (best of 5: 24.831 ms)
  fp32     30465.4 G mul+add pairs/s  (best of 5: 1.123 ms)
  int32    31420.3 G mul+add pairs/s  (best of 5: 1.089 ms)
  int64     7748.3 G mul+add pairs/s  (best of 5: 1.104 ms)
  ratios: fp32/fp64 88.4x  int32/fp64 91.2x  int64/fp64 22.49x
RESULT: PASS
```
The same source built with `-fmad=true` fails the parity check. This is why `-fmad=false`
is mandatory:
```
a*b+c     host 0 (0x0000000000000000)  device 1.1102230246251563e-16 (0x3c9ffffffffffffe)  MISMATCH (contracted to FMA?)
  fp64       745.6 G mul+add pairs/s      <- fused DFMA: 2.2x faster, but not Java semantics
  fp32     59485.4 G mul+add pairs/s
RESULT: FAIL
```
What this means: unfused FP64 runs at about 0.69 T instructions/s, against about 61 T for
FP32 and about 31 T IMAD/s for INT32. Exact Java semantics cost a further 2.2× on FP64
mul-add chains compared with fused DFMA. Across 5 runs the FP32/FP64 ratio ranged 88–96×,
with background desktop load on the GPU.
</details>

<details>
<summary><b>Java reference harness smoke test</b> (<code>tests/env/oracle-smoke</code>, vanilla 26.3, headless)</summary>

```
minecraft 26.3 (id 26.3), java 25.0.4.1, bootstrap 3.7 s
overworld: minY=-64 height=384 seaLevel=63 defaultBlock=Block{minecraft:stone} defaultFluid=Block{minecraft:water} aquifers=true legacyRandom=false
final_density graph: 4523 nodes as a tree, 227 distinct objects
  (reference)                                1285
  BinaryFunction:ADD                         425
  BinaryFunction:MAX                         5
  BinaryFunction:MIN                         6
  BinaryFunction:MUL                         245
  BlendDensityFunction                       1
  BlendedNoise                               3
  CacheFunction                              1091
  ClampFunction                              5
  ConstantFunction                           1164
  GradientFunction                           12
  InterpolatedFunction:cell 4x8              5
  IntervalSelectFunction                     5
  LerpFunction                               11
  NoiseFunction                              408
  RangeChoiceFunction                        7
  ShiftNoiseFunction$ShiftA                  354
  ShiftNoiseFunction$ShiftB                  354
  SimpleDensityFunction                      13
  SplineFunction                             9
  UnaryFunction:ABS                          389
  UnaryFunction:CUBE                         3
  UnaryFunction:HALF_NEGATIVE                3
  UnaryFunction:QUARTER_NEGATIVE             3
  UnaryFunction:SQUARE                       1
  UnaryFunction:SQUEEZE                      1
seed 0
  final_density(0, 64, 0) = -0.100785263  bits 0xbdce6881
  final_density(0, -64, 0) = 0.0374824181  bits 0x3d19872a
  final_density(0, 319, 0) = -0.0249947906  bits 0xbcccc1e0
  final_density(-1, 63, -1) = -0.0742429346  bits 0xbd980cae
  final_density(15, 40, 16) = 0.177183390  bits 0x3e356f90
  final_density(-123456, 12, 654321) = 0.135186777  bits 0x3e0a6e67
  final_density(29999999, 70, -29999999) = -0.0380979590  bits 0xbd1c0c9b
  final_density(-29999984, -60, 29999984) = 0.0464016646  bits 0x3d3e0fac
  chunk (0, 0) caches=true: volume vs point: 22666/98304 bitwise mismatches (0 at cell corners), max |diff| 1.94e-07, max 4096 ulp, 0 solid/air flips; 31899 blocks with density > 0
  chunk (0, 0) caches=false: volume vs point: 22666/98304 bitwise mismatches (0 at cell corners), max |diff| 1.94e-07, max 4096 ulp, 0 solid/air flips; 31899 blocks with density > 0
  chunk (-1874999, 1874998) caches=true: volume vs point: 27997/98304 bitwise mismatches (0 at cell corners), max |diff| 1.49e-07, max 1792 ulp, 0 solid/air flips; 28067 blocks with density > 0
seed -4172144997902289642
  ... (8 point samples)
  chunk (0, 0) caches=true: volume vs point: 23755/98304 bitwise mismatches (0 at cell corners), max |diff| 1.49e-07, max 5120 ulp, 0 solid/air flips; 26102 blocks with density > 0
  chunk (0, 0) caches=false: volume vs point: 23755/98304 bitwise mismatches (0 at cell corners), max |diff| 1.49e-07, max 5120 ulp, 0 solid/air flips; 26102 blocks with density > 0
  chunk (-1874999, 1874998) caches=true: volume vs point: 29354/98304 bitwise mismatches (4 at cell corners), max |diff| 1.49e-07, max 8192 ulp, 0 solid/air flips; 39579 blocks with density > 0
vanilla final_density volume sample: 2.141 ms per 16x384x16 chunk (1 thread, after warm-up)
SMOKE: PASS
```
The large ulp counts come from values near zero. In absolute terms the differences are float
rounding (about 1e-7), consistent with a different interpolation order between the two paths.
The 4 corner mismatches in the far-out chunk aren't explained yet. They're the first item
for Phase 1 (risk R2).
</details>

<details>
<summary><b>How I checked the 26.3 engine</b></summary>

No decompilation was needed. I read public signatures with `javap` on the Loom-prepared jar,
counted bytecode opcodes, and evaluated the router through vanilla's API.

- `javap` shows `NoiseRouter` has 8 fields: temperature, vegetation, continents, erosion,
  depth, ridges, chunkSurfaceLevel, finalDensity.
- `NoiseSettings(minY, height)` no longer has cell sizes.
- `NoiseGeneratorSettings` has `materialRule` and `Optional<Aquifer.Config> aquifers`.
- Chunk statuses are `EMPTY, STRUCTURE_STARTS, STRUCTURE_REFERENCES, BIOMES, TERRAIN, FEATURES, INITIALIZE_LIGHT, LIGHT, SPAWN, FULL`.
- Bytecode opcode counts in `synth.PerlinNoise`: `9 dadd, 9 dsub, 3 dmul, 9 d2f` against
  `48 fsub, 14 fadd, 12 fmul`. That means double for the coordinate side and float for the
  lattice math.
- `Math.pow`/`sqrt` appear only in setup code (`NormalNoise.buildOctaves`,
  `computeNormalizationFactor`, `BlendedNoise.createFbm`) and in density ops the overworld
  router doesn't use (`PowFunction`, `LOG`).
- `InterpolatedFunction$Sampler` uses `floorMod`, which suggests cells are anchored to world
  coordinates.
- Seeding still goes through `RandomSupport.seedFromHashOf` (Guava MD5), `mixStafford13`,
  Xoroshiro positional factories, and `GradientNoise` (3 `nextDouble` and `nextInt` for the
  permutation).
</details>

### 3.3 Missing or broken: install steps (yours to run)

1. **JDK 25:** `winget install EclipseAdoptium.Temurin.25.JDK` (or `Microsoft.OpenJDK.25`).
   In the MSI, enable "Set JAVA_HOME" and "Add to PATH". Verify with `java -version`, which
   should print `25.0.x`.
2. **Ninja:** `winget install Ninja-build.Ninja`. Open a new shell and run `ninja --version`.
3. **Nsight Compute counters (admin):** open NVIDIA Control Panel, then Desktop menu → *Enable
   Developer Settings*. Under Developer → *Manage GPU Performance Counters*, choose *Allow
   access to the GPU performance counters to all users* and apply. Alternatively, run `ncu`
   from an elevated prompt. Verify by re-running `check_env.ps1`; the Nsight Compute row
   should show PASS.
4. *(Optional)* Add `C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.2\target-windows-x64` to PATH.
5. Then run `powershell -ExecutionPolicy Bypass -File tests\env\check_env.ps1`. It should
   print all PASS. The portable JDK and Ninja I used live in a temp folder and will go away.

---

## 4. Design

### 4.1 Vanilla 26.3 stages and what gets ported

| Stage | Vanilla 26.3 classes (official names) | Port? |
|---|---|---|
| World seed → per-noise RNG | `RandomState.create`, `RandomSupport.seedFromHashOf` (MD5 of identifier), `mixStafford13`, `upgradeSeedTo128bit`, `Xoroshiro128PlusPlus`, `XoroshiroRandomSource` (`nextLong/nextInt(bound)/nextDouble/nextFloat`), `PositionalRandomFactory` (`fromHashOf`, `at(x,y,z)`) | Yes, host side (CPU) |
| Legacy LCG | `LegacyRandomSource` (Java `java.util.Random` LCG), the `LegacyFbmInitializer` skip pattern (`consumeCount`) | Yes, unit-tested. Used only on legacy paths; Phase 1 confirms whether the overworld `BlendedNoise` uses it |
| Noise primitives | `GradientNoise` (perm table, 3 `double` offsets, `wrap`), `PerlinNoise` (double coordinates to float lattice math; `addToVolume`), `NoiseStack` (octave layers), `NormalNoise` (octave parameters, normalization), `BlendedNoise` (legacy 3D terrain noise), `SimplexNoise` (if used) | Yes: host setup, device evaluation |
| Density graph | `DensityFunction` (`compileSampler`, `rewriteChildren`, `range`, `domainAxes`), `DensityFunctionCompiler`, `DfRewriteRule` (`INLINE_REFERENCE`, `SLICE_UNIFORM_AXES`), node types `op.*` and `generator.*` (histogram in 3.2) | Yes, as an IR exported from the oracle (4.4) |
| Batched evaluation | `DensitySampler.sampleVolume`, `DensityVolume`, `DensityBuffer` (`float[]`), `SamplerContext`, `CacheFunction`/`CachingDensitySampler` | Semantics replicated exactly; structure is our own |
| Interpolation | `op.InterpolatedFunction(input, cellSizeXz=4, cellSizeY=8)` ×5 in `final_density` | Yes, using the *volume-path* arithmetic (R2) |
| Splines | `op.SplineFunction` over `util.CubicSpline` | Yes |
| Density → block | `NoiseBasedChunkGenerator.buildTerrain`, `NoiseChunk(randomState, beardifier, settings, fluidPicker, blender, volume)`, `Aquifer.createDisabled(fluidPicker)` / `Aquifer.computeSubstance(x,y,z,density)` | Yes (aquifers disabled); aquifers enabled is stretch S1 |
| Aquifers | `Aquifer.Config` (barrier, floodedness, spread, lava, exclusion, surfaceLevel functions), `FluidPicker` | Stretch S1 |
| Material rules | `material.MaterialSystem.buildSurface`, `material.rule.*` incl. `OreVeinRule`, `BandlandsRule` | **Out of scope** (surface rules and ore veins) |
| Structures / blending | `Beardifier` (use `Beardifier.EMPTY`), `blending.Blender` (empty for new worlds) | Out of scope; oracle runs without them |

### 4.2 Definition of done, by level

All comparisons use raw bits (`float`/`double` bit patterns, including the sign of zero), not tolerances.

| Level | What is compared | Source of truth |
|---|---|---|
| L0 | RNG streams, MD5-derived seeds, per-noise octave parameters (perms, offsets, frequencies, amplitudes, normalization), point noise values | Oracle dumps from vanilla classes |
| L1 | Each distinct node of the `final_density` graph, sampled over small volumes | Vanilla `rs.getSampler(node).sampleVolume(...)` |
| L2 | **Final density per block** (`float`) for full chunks (16×384×16), via the volume path chunk generation uses | Oracle, through the same entry point `buildTerrain` uses |
| L3 | **Block fill per block** for the core config: vanilla `buildTerrain` on a fresh `ProtoChunk` with **aquifers disabled and an empty material rule**, `Beardifier.EMPTY`, empty blender. Output is stone, water, lava or air | Oracle |
| L3-S1 | The same with aquifers enabled (stretch) | Oracle |

L3 is the 26.3 equivalent of "the pre-surface-rules block fill". It is vanilla's own fill code
with only the surface/material step switched off. With aquifers off, solid vs. non-solid is
exactly `density > 0`. With aquifers on, aquifer barriers can turn non-positive density into
stone. That's why aquifers can't be added as an afterthought and get a stage of their own.

**Do aquifers fit cleanly?** Partly. They are deterministic per position, with no sequential
state across chunks, so they parallelize. But they mean 6 extra density functions,
positional-RNG aquifer centers on a grid, a nearest-4 search, and pressure/barrier logic.
That is irregular, branchy work needing its own kernel stage. I recommend **stretch S1**,
done after the core passes (Q2).

### 4.3 Reference oracle (`oracle/`)

- **Build:** a Gradle project using Loom 1.18.2's non-remapping plugin with
  `serverOnlyMinecraftJar()` (as in the smoke test). It runs as a **plain JVM main**: no
  Fabric Loader, no server, no world on disk. It calls `SharedConstants.tryDetectVersion()`,
  `Bootstrap.bootStrap()` and `VanillaRegistries.createWorldLookup()`, then uses vanilla
  classes directly. This is deterministic and single-threaded, and a run takes seconds.
- **Settings variants:** the core config is built with vanilla's own
  `NoiseGeneratorSettings` record constructor: the same router and noise settings,
  `aquifers = Optional.empty()`, and a no-op material rule. Self-check: final density from
  the variant equals final density from stock settings bit for bit.
- **Commands** (one Gradle task each, plus `oracle/regen.ps1` to run them all):
  - `export-ir` writes the overworld settings as JSON through vanilla's own codecs (router
    graph, noise parameters, splines), plus the post-rewrite graph if the compiler rewrites
    it. Output goes to `data/gamedata/26.3/` (gitignored, see 4.4).
  - `dump-rng` writes Xoroshiro/LCG streams, `seedFromHashOf` for many identifiers, and
    positional forks.
  - `dump-noise` writes per-noise octave parameters and point values for many seeds.
  - `dump-nodes` writes L1 per-node volumes.
  - `dump-density` writes L2. `dump-fill` writes L3 (and L3-S1).
- **Format:** little-endian binary payloads with a JSON manifest per dump. The manifest holds
  the MC version, JDK version, seed list, chunk list, array layouts, block palette and payload
  SHA-256. Density is `float32[chunk][y][z][x]`. Fill is `uint8[chunk][y][z][x]` with a
  palette. The layout is y-major, then z, then x, which matches vanilla section ordering. Raw
  sizes are 393 KB (density) and 98 KB (fill) per chunk. Files live in `data/oracle/26.3/`.
- **Determinism check:** `regen.ps1` run twice must give identical manifests (same hashes).

### 4.4 Legal hygiene

- The repo never contains decompiled code, the game jar, Mojang data files, data exported
  from the game, code generated from that data, or oracle dumps. `.gitignore` already covers
  `data/`, `build/`, `*.jar` (except the Gradle wrapper), `.gradle/` and `logs/`.
- **Game-derived constants** (router graph, spline tables, noise parameters) are exported at
  build time from your local jar by the oracle (`export-ir`). The C++ and CUDA code consume
  that IR: the CPU port interprets it, and the CUDA build runs it through a codegen step. So
  nothing Mojang-authored is ever committed or redistributed (Q3).
- **Reading vs. copying:** bit-exactness means matching operation order, so I'll read the 26.3
  classes locally (Loom can decompile into its Gradle cache, outside the repo) to learn the
  exact arithmetic. I'll write implementations in my own structure and naming, in C++/CUDA,
  never pasting or transliterating Java code. Small numeric kernels (a lerp, a smoothstep)
  will necessarily compute the same operations in the same order; that is the behavior being
  reproduced.

### 4.5 CPU port (`cpu/`, Phase 2)

- **Toolchain and flags:** C++20, MSVC 14.44, `/O2 /fp:precise`, with `/fp:strict` also
  tested. Never `/fp:fast` or `/fp:contract`. A host build of the contraction probe is part
  of the test suite.
- **`javamath`:** Java-semantics helpers.
  - JLS `d2i`/`d2l`/`f2i` (saturating, NaN → 0).
  - `Math.floor`, `Mth.floor`, `floorDiv`, `floorMod`.
  - `Math.min`/`max` (NaN propagation, −0.0 < +0.0).
  - Wrapping `int`/`long` arithmetic via unsigned types, and Java shift masking.
- **`rng`:** Xoroshiro128++, `nextInt(bound)` exactly as vanilla, MD5 (own implementation),
  Stafford13 mixing, positional factories, LCG.
- **`noise`:** gradient/Perlin, noise stacks, normal noise, blended noise.
- **`df`:** loads the exported IR and evaluates it with **vanilla's volume-path semantics**
  (interpolation order, caches, domain-axes slicing, range/interval selection). Every node
  can dump its volume, so a mismatch can be bisected to the first differing node.
- **`terrain`:** density to block id (disabled-aquifer fluid picker; S1 adds aquifers).
- **Scope:** single-threaded by design, as the debugging baseline. An OpenMP-over-chunks
  build is added only as an extra benchmark row.

### 4.6 CUDA design (`cuda/`)

**Code generation.** A Python generator (`cuda/codegen/`) turns the exported IR into
straight-line `__host__ __device__` functions, one per stage, written to
`build/generated/` (gitignored). The same generated code also compiles for the host. That
gives a debugging ladder: oracle → CPU interpreter (independent implementation) →
generated code on the CPU → generated code on the GPU. A GPU-only mismatch is then a
compiler-flag or intrinsic problem, not a logic problem.

**Lattice.** Cells are 4×8×4, anchored to world coordinates (to be confirmed in Phase 1).
- Corner columns sit at x, z ≡ 0 (mod 4).
- Corner rows run y = −64, −56, …, 320, which is 49 rows.
- Per chunk that is 5×5 corner columns and 1,225 corners. In region mode, neighbouring
  chunks share edges, giving 16 columns and 784 unique corners per chunk.

**One launch = one batch of N chunks** (N about 256–4096, tuned so a launch stays far below
the WDDM TDR limit). The input is either a list of chunk positions (list mode) or a rectangle
(region mode, which enables corner sharing). Three kernels run in one stream:

| Kernel | Work item → thread | Threads per chunk | Evaluates | Writes |
|---|---|---|---|---|
| A `columns` | one corner column (and one block column, if any per-block node is XZ-only) | 25 (list) / 16 (region) | Nodes whose `domainAxes` exclude Y (continents, erosion, ridges, splines, shifts) | `colval[node][col]` (SoA, float) |
| B `corners` | one cell corner; thread index → (column, row) with the row fastest, so a warp walks up a column | 1,225 / 784 | XYZ nodes under the 5 `InterpolatedFunction` inputs; shared subexpressions once per corner; range/interval branches as vanilla | `corner[channel][col][row]` (SoA, float, about 24 KB per chunk) |
| C `blocks` | one (x, z, cell-row) column segment, 8 blocks, so incremental interpolation is handled if vanilla's volume path uses it | 12,288 | Exact volume-path interpolation of the 5 channels, the per-block nodes above the interpolators, then density → block id (fluid picker) | `fill[chunk][y][z][x]` uint8 (+ optional `density` float for L2 tests) |

**Seed tables.** Built once per seed on the host and uploaded once: per octave, a 256-byte
permutation table, three `double` offsets and frequency/amplitude constants; plus spline and
BlendedNoise tables, in total tens of KB. They start in global memory (L1-cached); shared
memory and constant memory are Phase 4 experiments.

**Transfers.**
- Host → device is tiny (chunk coordinates).
- Device → host is 98 KB of fill per chunk, so 1,024 chunks is about 100 MB.
- Pinned ring buffers and 2–3 streams overlap copies of batch k with compute of batch k+1.
- A bit-packed solid mask (12 KB per chunk) is an optional output for transfer-bound comparisons.

**Naive version (Phase 3):** kernels A/B/C as above, all native FP64 for the double parts,
tables in global memory, list mode only.

### 4.7 Precision strategy (exactness first)

**Exactness rules, enforced in CMake for every target:**
- nvcc: `-fmad=false`. `-ftz=false -prec-div=true -prec-sqrt=true` are the defaults, but set explicitly.
- Never `--use_fast_math` or approximate intrinsics.
- Host: `/fp:precise`, no contraction.
- A test inspects the SASS: any `FFMA`/`DFMA` must come only from correctly-rounded div/sqrt
  library routines.

**Where FP64 remains in 26.3.** It is the per-octave coordinate pipeline: scale by frequency,
`wrap` (mod 2^25), add the double offset, floor, fraction, then convert to float. That is
roughly 20–35 FP64-pipe operations per 3D octave against roughly 40–70 FP32 operations. At
the measured unfused rates (about 0.69 T FP64 instructions/s vs. 61 T FP32), **FP64 is
likely over 90% of arithmetic issue time in a naive port.** Phase 3 replaces this estimate
with SASS instruction counts.

**Exactness-preserving reductions to evaluate in Phase 4.** Each one is gated by the full
oracle suite *and* by op-level randomized equivalence tests (billions of inputs plus edge cases):

1. *IEEE facts that hold exactly:*
   - multiplying by 2^k (no over/underflow) is an exponent add on the integer pipe;
   - `x − floor(x)` is exact;
   - `wrap(x) == x` for |x| comfortably inside ±2^24, which gives a fast path (the exact
     bound gets derived in Phase 1, because `GradientNoise` uses `Math.nextDown` in its
     constants);
   - double → float round-to-nearest-even can be done with integer ops.
2. *Fixed-point coordinate path.* Block coordinates are integers. If the per-octave
   frequencies are dyadic (to check against the exported parameters), `coord·freq + offset`
   can be computed exactly in 64/128-bit fixed point and rounded once to binary64. That is
   bit-identical to IEEE, since IEEE `dadd` is the correctly-rounded exact sum, and it runs
   entirely on the INT pipe.
3. *Correctly-rounded software `dadd`/`dmul`* on the INT32 pipe (about 20–40 instructions
   against a roughly 90× slower FP64 pipe) for the remaining ops, with native FP64 and
   emulated ops mixed so both pipes stay busy.
4. *Less work, same results:* corner sharing in region mode (36% fewer corner evaluations).
   Vanilla-style range-based skipping is kept only where it can't change a result.
5. *Measuring the FP64 tax:* a deliberately inexact FP32-coordinate build, **measurement only,
   never a deliverable**, reported with its mismatch rate, to bound what methods 1–3 can win.

### 4.8 Test plan (one command)

- **Command:** `powershell -File tests\run_all.ps1 [-Quick]`. It imports the VS 2022 env,
  configures and builds with CMake/Ninja, regenerates oracle data if it is missing or stale
  (version and hash check), runs the CTest suites (CPU, then CUDA) for levels L0–L3, and
  prints the first mismatch per level: position, node, expected and actual bits, ulp distance.
  The exit code is non-zero on any mismatch.
- **Seeds:**
  - fixed: 0, 1, −1, `Long.MIN_VALUE`, `Long.MAX_VALUE`, ±2^31;
  - about 40 random seeds from a fixed meta-seed (the list is recorded in the manifest).
- **Chunks per seed:**
  - the 3×3 around the origin (negative coordinates and the −1/0 chunk boundaries);
  - random chunks within ±100k and ±10M blocks;
  - **world-border chunks**: x, z chunk ±1,874,999, all four corners and the edge midpoints;
  - **wrap-boundary chunks**, chosen from the exported frequencies so that x·freq crosses
    ±2^24 for the lowest-frequency octaves;
  - for S1, coordinates where positional-hash multiplications overflow `int`.
- **Height:** every dump covers the full −64…319 range (plus the y = 320 corner row).
- **Corpus sizes:** "quick" is about 5 seeds × 9 chunks, for iteration. "Full" is about
  50 seeds × 30 chunks, roughly 1,500 chunks, for milestone sign-off.
- **Also:** `compute-sanitizer` memcheck and racecheck on the quick set, and the SASS
  FMA lint.

### 4.9 Benchmark method (Phase 5)

- **Workloads:** region mode (64×64 chunks at several locations) and list mode (4,096
  scattered chunks, no sharing).
- **Vanilla:** `buildTerrain` in the core config on 1 thread and on 16/32 threads, plus
  density-only `sampleVolume`. Measured with JMH-style warm-up, at least 10 iterations.
- **CPU port:** 1 thread (plus OpenMP across chunks).
- **Each GPU version:** kernel-only time (CUDA events), **end-to-end wall clock** (chunk list
  in, filled host buffer out, transfers included), and pipelined steady state.
- **Statistics:** median and IQR over at least 10 runs, plus min/max.
- **Conditions:** locked GPU clocks if an admin shell is available (Q4); otherwise clocks are
  logged through NVML during runs. Background GPU apps closed. Results written to CSV, with
  matplotlib charts.

### 4.10 Repo layout

```
oracle/    Java harness (Gradle + Loom), regen.ps1, docs/dump-format.md
cpu/       C++ reference port (library + CLI)
cuda/      CUDA versions v0..vN, codegen/ (IR -> generated .cuh)
tests/     run_all.ps1, CTest suites, env/ (Phase 0 checks: check_env.ps1, hello_sm120.cu, oracle-smoke/)
bench/     benchmark drivers, plot.py, results/*.csv
profiles/  Nsight exports (text/CSV/screenshots) + per-version notes; raw .ncu-rep/.nsys-rep gitignored
docs/      design notes, findings, engine notes for 26.3
data/      (gitignored) exported IR + oracle dumps
```

---

## 5. Milestones

| # | Milestone | Pass/fail check |
|---|---|---|
| M0 | Plan + environment | `check_env.ps1` all PASS after the steps in 3.3; you approve this plan |
| M1 | Reference oracle | `oracle\regen.ps1` produces the quick and full corpora (L0–L3) **twice with identical SHA-256 manifests**. The variant-vs-stock final-density self-check passes. The R2 corner-mismatch question is answered and documented |
| M2 | CPU C++ port | `tests\run_all.ps1` shows **0 mismatches** at L0–L3 on the full corpus (CPU suite) |
| M3 | Naive CUDA | **0 mismatches** at L0–L3 on the full corpus, against both the oracle and the CPU port; compute-sanitizer clean; SASS FMA lint clean |
| M4 | Optimization (iterative) | Each version vN passes all of M3, ships an Nsight Compute report plus a changelog entry (what changed, measured kernel and end-to-end effect), and stays buildable (`-DMCCUDA_VERSION=N`) |
| M5 | Benchmarks | `bench\run_bench.ps1` gives CSV plus charts for vanilla 1T/NT, the CPU port and every GPU version, kernel-only and end-to-end, with variance |
| M6 | Write-up | `README.md`: design, correctness method, charts, profiler findings, limitations |
| S1 | *Stretch:* aquifers | L3-S1, 0 mismatches on the full corpus |
| S2 | *Stretch:* Fabric server via JNI | A server generates chunks through the GPU path; the in-game fill matches the vanilla config; in-game throughput measured |

Each milestone ends with a commit and with the test output pasted into the commit or the docs.

---

## 6. Risks to bit-exactness, and how I'll isolate them

| # | Risk | Evidence / why | Isolation and mitigation |
|---|---|---|---|
| R1 | 26.3's engine is new and could change again in 26.4 | Major refactor seen via `javap` (3.2) | Pin exactly 26.3 (jar SHA-1 recorded in the manifest). Never chase snapshots. Write engine notes in `docs/` |
| R2 | **Two vanilla paths disagree** (point vs. volume) | 23–30% of blocks differ by about 1e-7; 4 corner mismatches in the far chunk | Match the volume path only. Confirm the exact volume geometry and entry point `buildTerrain` uses. Test volumes of different origin and size to detect origin dependence. Explain the 4 corners in M1 |
| R3 | Compile-time graph rewrites (`DfRewriteRule`, `DensityFunctionCompiler`) change structure or operation order | Rewrite rules exist (`INLINE_REFERENCE`, `SLICE_UNIFORM_AXES`) | Export the post-rewrite graph; per-node L1 dumps; implement the rewrites identically or consume the rewritten IR |
| R4 | Float/double mixing | `d2f`, `i2f` (rounds above 2^24, which matters near the world border), float splines and lerps | Op-level vectors produced by the JVM; strict flag set (4.7); host-compiled generated code vs. interpreter |
| R5 | Java integer semantics | Overflow, shift masking, `d2i` saturation, `floorMod` on negatives | `javamath` helpers tested against JVM-generated edge-case tables (L0) |
| R6 | `Math.min`/`max` and signed zeros | Java propagates NaN and orders −0 < +0; CUDA `fminf` and `std::min` don't | Explicit Java-semantics helpers; raw-bit comparison catches sign-of-zero drift |
| R7 | Transcendentals in setup | `Math.pow` in `NormalNoise.buildOctaves` and `BlendedNoise.createFbm` (HotSpot intrinsic, not guaranteed correctly rounded) | Compare our octave parameters against L0 dumps. If any differ, use exact construction for that argument class, or as a documented last resort load the oracle-computed parameters. JDK version recorded |
| R8 | Far coordinates | `wrap` jumps at ±2^24/freq; `i2f` rounding; positional-hash overflow | Dedicated border and wrap-boundary chunks in every corpus |
| R9 | GPU compilation drift | Contraction, FTZ, or an approximate intrinsic slipping in | Target-wide flags in CMake, the SASS lint, and the parity probe run in CI-style tests |
| R10 | Measurement noise | The GPU drives the display (7.4 GB / 22% in use during checks); WDDM; TDR | Benchmark protocol (4.9); batch sizing; report variance |
| R11 | Oracle fidelity | Settings variants or harness code could perturb vanilla | Variant-vs-stock self-check; determinism (two identical regen runs); entry points identical to `buildTerrain` |
| R12 | Performance ceiling | FP64 coordinate pipeline estimated at over 90% of arithmetic time | Phase 4 precision work (4.7). If a technique doesn't pay off, report that honestly |
| R13 | Scope gaps vs. real worlds | Beardifier (terrain near structures) and blending (old chunks) are excluded | Documented limitation. For S2, fall back to vanilla for chunks with structure pieces unless the beardifier is ported later |

---

## 7. Open questions (yours to decide)

1. **Version (Q1):** pin **26.3** (recommended; latest, float engine, unobfuscated) or
   **1.21.11** (the classic all-double engine your spec describes)?
2. **Definition of done (Q2):** OK for the core L3 to be vanilla `buildTerrain` with
   **aquifers disabled and the material-rule step off**, with **aquifers as stretch S1**? In
   26.3, ore veins are a material rule, so they'd fall out of scope along with surface rules.
3. **Game-derived data (Q3):** OK to extract the router, splines and noise parameters from
   your local jar at build time and generate code from them, never committing either? The
   alternative is hand-porting those constants into committed source, which is weaker legal
   hygiene.
4. **Locked clocks (Q4):** may benchmark runs use an elevated shell for
   `nvidia-smi -lgc`/`-rgc`? If not, I'll use unlocked clocks, log them, and report variance
   over repeated runs.
5. **Settings coverage (Q5):** overworld only? Amplified and Large Biomes would come almost
   free through the same IR as extra test coverage. Nether and End stay out.

Before Phase 1, I need: answers to the above, the three installs in 3.3, and your "go".
