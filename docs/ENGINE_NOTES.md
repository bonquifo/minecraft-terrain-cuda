# Engineering notes: what bit-exactness depends on

Notes on the behaviours of Minecraft 26.3's terrain generation that had to be reproduced
exactly, and on how the GPU code stays exact while being fast. Everything here was found by
reading public signatures/bytecode of the locally prepared game jar and by comparing against
the oracle; no game source is reproduced in this repository.

## 1. Two evaluation semantics

26.3's density functions are float-valued and have two evaluation paths, and they do not
produce identical bits:

- **Volume path** (`sampleVolume`): used to generate a chunk's density, the climate for
  biomes, ore-vein density and richness, and the aquifer surface-level prefill.
- **Point path** (`sampleValue`): used for aquifer barrier/floodedness/spread/lava queries,
  the ore-vein filler gap, and the `find_top_surface` probes.

Differences that matter:

| | volume | point |
|---|---|---|
| noise coordinates | `block * (scale * frequency)` | `(block * scale) * frequency` |
| gradient dot product | `(gx*x + gz*z) + gy*y` (xz part reused over y) | `(gx*x + gy*y) + gz*z` |
| interpolation | z, then x, then incremental steps in y | `lerp3` at the point |
| min / max | keep the first argument unless the second is strictly smaller/larger | `Math.min/max`, with range-based short cuts |
| mul / div | plain | zero short-circuit on the first operand |

The code generator (`cuda/codegen/gen_router.py`) emits a separate evaluator per context,
and each context has fixed semantics. Mixing them up produces values that differ in the
last bit for a small fraction of positions, enough to flip blocks.

## 2. The compiled graph

Vanilla rewrites the density graph before sampling: references are inlined, structurally
equal cache nodes are merged, and functions that do not depend on an axis are evaluated
on slices. The IR exporter (`oracle/.../IrExporter.java`) mirrors these rewrites, and the
binary operations carry vanilla's compile-time decisions: constant detection, division by
a constant as a multiply by the float reciprocal, and min/max pruning from value ranges.
Those decisions change results, so they are made once, at export time, exactly as vanilla
makes them.

## 3. Random numbers

- Xoroshiro128++ with positional factories: `fromHashOf` seeds from MD5 of the name, `at(x, y, z)`
  from `Mth.getSeed`. The legacy 48-bit LCG is still used for biome temperature noise.
- `nextDouble` is `(bits >>> 11) * 2^-53` computed in double. A decompiler rendering suggested
  float arithmetic, and the bytecode (`l2d`, `dmul`) settled it. Lesson: verify literals and
  casts against bytecode.
- Noise construction sums amplitudes with a compensated stream sum, so plain summation is wrong.

## 4. Biomes

- The six climate parameters are sampled at quart resolution (every 4 blocks) with the volume
  path and quantized as `(long)(value * 10000f)`.
- Lookup is an R-tree over 7,594 parameter boxes (19 children per node, built with stable
  sorts). Search descends into a child only if its box is strictly closer than the best leaf
  so far, and accepts a leaf only if strictly closer.
- The tree keeps a **last-result candidate per thread** and starts each search from it. On an
  exact distance tie this makes the result depend on the previous lookup. The rule is: the
  candidate wins if it reaches the minimal distance, otherwise the first minimal leaf in DFS
  order. A live server's result therefore depends on which chunk a worker thread generated
  before. The oracle resets the candidate per chunk, and so do the CPU port and the GPU.
- `BiomeManager` picks a quart per block from the 8 surrounding quarts with a seeded "fiddle".
  The obfuscated seed is SHA-256 of the world seed, read little-endian.

GPU: L* (the first minimal leaf) is computed in parallel with a DFS whose initial bound is the
distance of a neighbouring quart's leaf plus one; that provably yields the same L*. The tie rule
then needs the previous result only where the previous L* also reaches the minimum. Those flags
are computed in parallel, and one warp per chunk resolves the rare windows that contain one.

## 5. Aquifers

- 16x12x16 grid. Cell locations are random per grid cell, and fluid status is derived from
  the surface level around the cell, the floodedness and spread noises, and a lava-noise check
  for fluid levels at or below -10.
- The surface level comes from `find_top_surface`. Positions inside the aquifer's
  constructor prefill window (quarts `4c-4 .. 4c+6`) use **volume** semantics. Other
  positions are sampled lazily with **point** semantics. Both orders must match.
- Blocks above a per-chunk skip height use the global fluid picker directly.

## 6. Material rules

- A column is walked top-down. Stone depth above and below, water height and surface depth
  feed the rule tree: about 150 conditions, 110 block results, 43 noise thresholds, 2 vertical
  gradients, 2 ore veins (new in 26.3's material rules) and clay bands.
- The heightmap is updated on every block placement. The slope condition reads neighbour
  heights, and vanilla processes columns in x-major, z-minor order, so a neighbour's height
  is either before or after both extensions. The GPU reproduces this with three height
  snapshots: initial, after the eroded-badlands extension, and after the frozen-ocean
  extension, which is dry-run in a first pass.
- 26.3 checks a cave biome for almost every stone block below the preliminary surface, so the
  zoom (section 4) runs for most blocks of a chunk.

## 7. Floating point

- Java has no FMA contraction and is strict IEEE. Device code: `-fmad=false -prec-div=true
  -prec-sqrt=true -ftz=false`. Host code: `/fp:precise`, with contraction verified by a unit test.
- `Mth.floor` is `(int) Math.floor(v)`, and Java's cast saturates with NaN going to 0. On the
  device that is a single `cvt.rmi.s32.f64` plus a NaN test on the bit pattern.

## 8. GPU exactness techniques

| technique | why it is exact |
|---|---|
| per-axis noise coordinate tables (`AxisRec`) | the same IEEE operations, computed once per distinct coordinate and reused |
| zoom square table | same expression, tabulated over its 8 x 1,024 possible inputs |
| zoom FP32 pre-filter | error bound proof: corners more than 8e-6 above the FP32 minimum cannot be (or tie) the minimum; survivors use the exact FP64 comparison |
| distance compares on bit patterns | non-negative doubles order like their bit patterns |
| position dedupe | each value is a pure function of position |
| bounded R-tree search | any bound above the minimum yields the same first minimal leaf |

## 9. Oracle data

`oracle/regen.ps1` writes, under the git-ignored `data/`:

- `gamedata/26.3/overworld_ir.json`: the optimized density graph (floats as raw bits,
  doubles as hex), with value ranges and axis dependence per node.
- `gamedata/26.3/overworld_terrain.json`: the material-rule tree, block palette, biome
  parameter list and biome climate.
- `oracle/26.3/{quick,full}/`: per seed, named-function volumes (L1), final density (L2),
  core block fill (L3), and biomes, aquifer fill and final terrain. The layout is
  `((y - minY) * 16 + z) * 16 + x`, with SHA-256 manifests.
