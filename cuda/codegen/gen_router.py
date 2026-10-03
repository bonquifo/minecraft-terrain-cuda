"""Generate CUDA/C++ evaluators for the overworld TERRAIN step from the local IR and
terrain model (data/gamedata/<ver>/overworld_{ir,terrain}.json).

The generated header lives in the build tree and is never committed: it embeds
game-derived constants. It reproduces two vanilla evaluation semantics:

  volume  DensitySampler.sampleVolume, per element (chunk density, climate,
          ore-vein density/richness, surface-level prefill)
  point   DensitySampler.sampleValue (aquifer queries, ore filler gap,
          find_top_surface probes)

Evaluators are structs with one memoised method per node; branches are lazy
(every node is a pure function of position, so skipping an unused side cannot
change a result). Contexts:

  ColumnA / ColumnB  y = 0 values of slice inputs on the 4-block lattice
                     (A: read by the 4x8 corner contexts, B: by the climate context)
  CornerEval / OreCornerEval  interpolated-channel inputs at cell corners
  CornerEvalTab / OreCornerEvalTab  the same, with every noise's coordinate setup
                     read from per-axis AxisRec tables (kAxisSites) instead of computed
  BlockEval / OreBlockEval    per-block roots; interpolated nodes read channels
  QuartEval          the six climate roots at quart positions
  VolY0Eval          find_top_surface upper bound, volume semantics, y = 0
  PointY0 / PointEval  point semantics; PointEval delegates slice inputs to PointY0
"""
import argparse
import json
import struct
import sys

import numpy as np

from ir import IR, f32, f64, spline_coords


def flit(bits):
    v = f32(bits)
    if v != v or v in (float('inf'), float('-inf')):
        return 'mcc::bits2f(0x%08Xu)' % (bits & 0xFFFFFFFF)
    return '(%sf)' % v.hex()


def flit_v(v32):
    return flit(struct.unpack('<I', struct.pack('<f', float(v32)))[0])


def dlit(hexbits):
    v = f64(hexbits)
    if v != v or v in (float('inf'), float('-inf')):
        return 'mcc::bits2d(0x%016Xull)' % int(hexbits, 16)
    return '(%s)' % v.hex()


def dlit_v(v):
    return '(%s)' % float(v).hex()


class Ctx:
    def __init__(self, name, sem, slices, interp, chans=None, col_nodes=None, axis=False):
        self.name = name
        self.axis = axis          # noises read tabulated coordinates (members ax, ay, az)
        self.sem = sem            # 'vol' | 'pt'
        self.slices = slices      # 'col' | 'ident_y0' | 'sub' | 'subst'
        self.interp = interp      # 'chan' | 'pass' | 'err'
        self.chans = chans or []
        self.col_nodes = col_nodes or []


class Gen:
    def __init__(self, ir, tm):
        self.ir = ir
        self.tm = tm
        self.subs = {}
        self.pre = []
        self.spline_fns = set()
        self.noise_keys = sorted(ir.noises.keys())
        self.noise_index = {k: i for i, k in enumerate(self.noise_keys)}
        self.blended_ids = sorted(n['id'] for n in ir.nodes.values() if n['t'] == 'blended_noise')
        self.axis_sites = []      # (kind, index, xz, y, smeared)
        R = ir.roots
        self.root = R['final_density']
        self.ore_rules = []
        self.collect_ores(tm['material_rule'])
        self.ore_roots = []
        for o in self.ore_rules:
            self.ore_roots += [R[o['density']], R[o['richness']]]
        self.climate_roots = [R['climate/' + k] for k in ('temperature', 'vegetation', 'continents', 'erosion', 'depth', 'ridges')]
        fts = [n for n in ir.nodes.values() if n['t'] == 'find_top_surface']
        if len(fts) != 1:
            raise SystemExit('expected exactly one find_top_surface node')
        self.fts = fts[0]
        self.point_roots = [R['aquifer/' + k] for k in ('barrier', 'fluid_level_floodedness', 'fluid_level_spread', 'lava', 'exclusion')]
        self.point_roots += [R[o['filler_gap']] for o in self.ore_rules]
        self.point_roots.append(self.fts['density'])
        self.check_surface_roots()

        self.chans = self.leaves([self.root], 'interpolated')
        self.ore_chans = [c for c in self.leaves(self.ore_roots, 'interpolated') if c not in self.chans]
        for c in self.chans + self.ore_chans:
            n = ir.nodes[c]
            if (n['cell_xz'], n['cell_y']) != (4, 8):
                raise SystemExit('only 4x8 interpolation channels are supported')
        self.col_a = self.leaves([ir.nodes[c]['in'] for c in self.chans + self.ore_chans], 'slice', stop_at=('interpolated',))
        self.col_b = self.leaves(self.climate_roots, 'slice')
        self.pt_slices = self.leaves(self.point_roots, 'slice')
        y0 = lambda i: ir.nodes[i]['axis'] == 'y' and ir.nodes[i]['coord'] == 0
        self.col_a = [c for c in self.col_a if y0(c)]
        self.col_b = [c for c in self.col_b if y0(c)]
        self.pt_slices = [c for c in self.pt_slices if y0(c)]

    def collect_ores(self, r):
        t = r['t']
        if t == 'ref':
            self.collect_ores(r['rule'])
        elif t == 'sequence':
            for x in r['rules']:
                self.collect_ores(x)
        elif t == 'condition':
            self.collect_ores(r['then'])
        elif t == 'ore_vein':
            if r not in self.ore_rules:
                self.ore_rules.append(r)

    def check_surface_roots(self):
        R, N = self.ir.roots, self.ir.nodes
        a = N[R['aquifer/surface_level']]
        if not (a['t'] == 'slice' and a['in'] == self.fts['id']):
            raise SystemExit('unexpected aquifer/surface_level structure')
        c = N[R['surface/chunk_surface_level']]
        i = N[c['in']] if c['t'] == 'slice' else None
        if not (i and i['t'] == 'interpolated' and i['in'] == self.fts['id'] and (i['cell_xz'], i['cell_y']) == (16, 1)):
            raise SystemExit('unexpected surface/chunk_surface_level structure')

    def leaves(self, roots, leaf_type, stop_at=()):
        out, seen = [], set()

        def walk(i):
            if i in seen:
                return
            seen.add(i)
            n = self.ir.nodes[i]
            if n['t'] == leaf_type:
                out.append(i)
                return
            if n['t'] in stop_at:
                return
            for c in self.ir.children(n):
                walk(c)

        for r in roots:
            walk(r)
        return out

    # ---- vanilla compile-time decisions ----------------------------------------
    def binary_plan(self, n):
        L, R = self.ir.nodes[n['l']], self.ir.nodes[n['r']]
        lc, rc = L['t'] == 'constant', R['t'] == 'constant'
        op = n['op']
        lmin, lmax = f32(L['range'][0]), f32(L['range'][1])
        rmin, rmax = f32(R['range'][0]), f32(R['range'][1])
        if op in ('add', 'sub', 'mul'):
            return (op, n['l'], n['r'], None, lc, rc)
        if op == 'div':
            if not lc and rc:
                return ('divc', n['l'], n['r'], np.float32(1.0) / np.float32(f32(R['v'])), lc, rc)
            return ('div', n['l'], n['r'], None, lc, rc)
        if op == 'min':
            if lmax < rmin:
                return ('pass', n['l'], None, None, lc, rc)
            if rmax < lmin:
                return ('pass', n['r'], None, None, lc, rc)
            return ('min', n['r'], n['l'], None, lc, rc) if lc else ('min', n['l'], n['r'], None, lc, rc)
        if op == 'max':
            if lmin > rmax:
                return ('pass', n['l'], None, None, lc, rc)
            if rmin > lmax:
                return ('pass', n['r'], None, None, lc, rc)
            return ('max', n['r'], n['l'], None, lc, rc) if lc else ('max', n['l'], n['r'], None, lc, rc)
        raise SystemExit('bad binary op ' + op)

    def gradient_plan(self, n):
        rng = n['to'] - n['from']
        fv, tv = np.float32(f32(n['from_value'])), np.float32(f32(n['to_value']))
        factor = (tv - fv) / np.float32(rng)
        mode = {'clamp_to_edge': 0, 'repeat': 1, 'mirrored_repeat': 2}[n['tiling']]
        return mode, n['from'], min(n['from'], n['to']), max(n['from'], n['to']), rng, fv, factor

    def stack(self, key):
        return 'T.stacks[%d]' % self.noise_index[key]

    def axis_site(self, kind, index, xz, y, smeared):
        key = (kind, index, float(xz).hex(), float(y).hex(), smeared)
        if key not in self.axis_sites:
            self.axis_sites.append(key)
        return self.axis_sites.index(key)

    @staticmethod
    def axis_args(k):
        return 'ax + T.axis_off[%d], ay + T.axis_off[%d], az + T.axis_off[%d]' % (k, k, k)

    # ---- contexts ----------------------------------------------------------------
    def reach(self, ctx, roots):
        out, seen = [], set()

        def walk(i):
            if i in seen:
                return
            seen.add(i)
            n = self.ir.nodes[i]
            t = n['t']
            stop = (t == 'interpolated' and ctx.interp == 'chan') or (t == 'slice' and not self.slice_is_identity(n, ctx))
            if not stop:
                for c in self.ir.children(n):
                    walk(c)
            out.append(i)

        for r in roots:
            walk(r)
        return out

    def ensure_spline(self, node_id, coords):
        name = 'spline_%d' % node_id
        if name in self.spline_fns:
            return name
        self.spline_fns.add(name)
        self.spline_coord_index = {c: k for k, c in enumerate(coords)}
        L = ['MCC_SPLINE_FN float %s(const float* cs) {' % name, 'float v;']
        self.emit_spline(self.ir.nodes[node_id]['spline'], 'v', L, [0])
        L.append('return v;')
        L.append('}')
        self.pre += L
        return name

    def slice_is_identity(self, n, ctx):
        return ctx.slices == 'ident_y0' and n['axis'] == 'y' and n['coord'] == 0

    def ensure_sub(self, sem, root):
        key = (sem, root)
        if key in self.subs:
            return self.subs[key]
        name = 'Sub%s_%d' % ('V' if sem == 'vol' else 'P', root)
        self.subs[key] = name
        code = self.emit_struct(Ctx(name, sem, 'subst', 'err'), [root])   # may recursively add dependencies first
        self.pre += code
        return name

    def emit_struct(self, ctx, roots, extra_member=None):
        reach = self.reach(ctx, roots)
        o = ['struct %s {' % ctx.name, '    const RouterTables& T;', '    int32_t x, y, z;']
        if ctx.slices == 'col':
            o.append('    const float* col;')
        if ctx.interp == 'chan':
            o.append('    const float* ch;')
        if extra_member:
            o.append('    ' + extra_member)
        for i in reach:
            o.append('    float m%d; bool h%d;' % (i, i))
        for i in reach:
            o.append('    MCC_GEN_FN float n%d() {' % i)
            o.append('        if (h%d) return m%d;' % (i, i))
            for line in self.node_body(i, ctx):
                o.append('        ' + line)
            o.append('        m%d = v; h%d = true; return v;' % (i, i))
            o.append('    }')
        o.append('    MCC_HD void dump(float* v, bool* h) const {')
        for k, i in enumerate(reach):
            o.append('        v[%d] = m%d; h[%d] = h%d;' % (k, i, k, i))
        o.append('    }')
        o.append('};')
        o.append('static constexpr int kReach_%s[] = {%s};' % (ctx.name, ', '.join(str(i) for i in reach)))
        o.append('static constexpr int kReachCount_%s = %d;' % (ctx.name, len(reach)))
        return o

    def node_body(self, i, ctx):
        n = self.ir.nodes[i]
        t = n['t']
        L = []
        vol = ctx.sem == 'vol'
        if t == 'interpolated':
            if ctx.interp == 'chan':
                L.append('float v = ch[%d];' % ctx.chans.index(i))
            elif ctx.interp == 'pass':
                L.append('float v = n%d();' % n['in'])     # aligned cell volume: vanilla passes through
            else:
                raise SystemExit('interpolated node %d not supported in %s' % (i, ctx.name))
            return L
        if t == 'slice':
            y0 = n['axis'] == 'y' and n['coord'] == 0
            if self.slice_is_identity(n, ctx):
                L.append('float v = n%d();  // y is already 0' % n['in'])
            elif y0 and ctx.slices == 'col':
                L.append('float v = col[%d];' % ctx.col_nodes.index(i))
            elif y0 and ctx.slices == 'sub':
                L.append('float v = sub->n%d();' % n['in'])
            else:
                # Generic SliceFunction: evaluate the input at the substituted coordinate.
                name = self.ensure_sub(ctx.sem, n['in'])
                cx = str(n['coord']) if n['axis'] == 'x' else 'x'
                cy = str(n['coord']) if n['axis'] == 'y' else 'y'
                cz = str(n['coord']) if n['axis'] == 'z' else 'z'
                L.append('%s s{T, %s, %s, %s}; float v = s.n%d();' % (name, cx, cy, cz, n['in']))
            return L
        if t in ('cache', 'blend_density'):
            L.append('float v = n%d();' % n['in'])
        elif t == 'constant':
            L.append('float v = %s;' % flit(n['v']))
        elif t == 'simple':
            L.append('float v = %s;' % ('1.0f' if n['kind'] == 'blend_alpha' else '0.0f'))
        elif t == 'gradient':
            mode, frm, mn, mx, rng, fv, factor = self.gradient_plan(n)
            L.append('const mcc::GradientPlan g{%d, %d, %d, %d, %d, %s, %s};' % (mode, frm, mn, mx, rng, flit_v(fv), flit_v(factor)))
            L.append('float v = mcc::gradient_compute(g, %s);' % n['axis'])
        elif t == 'noise':
            xz, ys = dlit(n['xz_scale']), dlit(n['y_scale'])
            L.append('const mcc::StackRef2& s = %s;' % self.stack(n['noise']))
            if n['shift_x_zero'] and n['shift_y_zero'] and n['shift_z_zero']:
                if vol and ctx.axis:
                    k = self.axis_site(0, self.noise_index[n['noise']], f64(n['xz_scale']), f64(n['y_scale']), 0)
                    L.append('float v = mcc::stack_volume_tab(0.0f, T.layers + s.first, s.count, T.octaves, %s, 1.0f);' % self.axis_args(k))
                elif vol:
                    L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, x, y, z, %s, %s, 1.0f);' % (xz, ys))
                else:
                    L.append('float v = mcc::stack_get(T.layers + s.first, s.count, T.octaves, static_cast<double>(x) * %s, static_cast<double>(y) * %s, '
                             'static_cast<double>(z) * %s);' % (xz, ys, xz))
            else:
                L.append('double nx = static_cast<double>(x) * %s + static_cast<double>(n%d());' % (xz, n['shift_x']))
                if n['shift_y_zero']:
                    L.append('double ny = static_cast<double>(y) * %s;' % ys)
                else:
                    L.append('double ny = static_cast<double>(y) * %s + static_cast<double>(n%d());' % (ys, n['shift_y']))
                L.append('double nz = static_cast<double>(z) * %s + static_cast<double>(n%d());' % (xz, n['shift_z']))
                L.append('float v = mcc::stack_get(T.layers + s.first, s.count, T.octaves, nx, ny, nz);')
        elif t in ('shift_a', 'shift', 'shift_b'):
            L.append('const mcc::StackRef2& s = %s;' % self.stack(n['noise']))
            if vol:
                if t == 'shift_b':
                    L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, z, x, 0, 0.25, 0.25, 4.0f);')
                else:
                    yscale = '0.0' if t == 'shift_a' else '0.25'
                    L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, x, y, z, 0.25, %s, 1.0f) * 4.0f;' % yscale)
            else:
                if t == 'shift_b':
                    L.append('float v = mcc::stack_get(T.layers + s.first, s.count, T.octaves, static_cast<double>(z) * 0.25, static_cast<double>(x) * 0.25, 0.0) * 4.0f;')
                else:
                    yscale = '0.0' if t == 'shift_a' else '0.25'
                    L.append('float v = mcc::stack_get(T.layers + s.first, s.count, T.octaves, static_cast<double>(x) * 0.25, static_cast<double>(y) * %s, '
                             'static_cast<double>(z) * 0.25) * 4.0f;' % yscale)
        elif t == 'blended_noise':
            bi = self.blended_ids.index(i)
            xzm = 684.412 * f64(n['xz_scale'])
            ym = 684.412 * f64(n['y_scale'])
            L.append('const mcc::BlendedRef& b = T.blended[%d];' % bi)
            L.append('const double xzm = %s, ym = %s;' % (dlit_v(xzm), dlit_v(ym)))
            L.append('const double mxz = xzm / %s, my = ym / %s;' % (dlit(n['xz_factor']), dlit(n['y_factor'])))
            if vol and ctx.axis:
                scales = {'mxz': xzm / f64(n['xz_factor']), 'my': ym / f64(n['y_factor']), 'xzm': xzm, 'ym': ym}
                kinds = {'main': 1, 'min_limit': 2, 'max_limit': 3}

                def sample(st, sx, sy):
                    k = self.axis_site(kinds[st], bi, scales[sx], scales[sy], 1)
                    return 'mcc::stack_volume_tab(0.0f, T.layers + b.%s.first, b.%s.count, T.octaves, %s, 1.0f)' % (st, st, self.axis_args(k))
            elif vol:
                def sample(st, sx, sy):
                    return 'mcc::stack_add_volume(0.0f, T.layers + b.%s.first, b.%s.count, T.octaves, true, x, y, z, %s, %s, 1.0f)' % (st, st, sx, sy)
            else:
                def sample(st, sx, sy):
                    return ('mcc::stack_get_smeared(T.layers + b.%s.first, b.%s.count, T.octaves, static_cast<double>(x) * %s, '
                            'static_cast<double>(y) * %s, static_cast<double>(z) * %s)' % (st, st, sx, sy, sx))
            L.append('float a = %s;' % sample('main', 'mxz', 'my'))
            L.append('a = mcc::mth_clamp(a + 0.5f, 0.0f, 1.0f);')
            L.append('float v;')
            L.append('if (a == 0.0f) v = %s;' % sample('min_limit', 'xzm', 'ym'))
            L.append('else if (a == 1.0f) v = %s;' % sample('max_limit', 'xzm', 'ym'))
            L.append('else { float lo = %s; float hi = %s; v = mcc::mth_lerp(a, lo, hi); }' % (sample('min_limit', 'xzm', 'ym'), sample('max_limit', 'xzm', 'ym')))
        elif t == 'binary':
            kind, a, b, inv, lc, rc = self.binary_plan(n)
            if kind == 'pass':
                L.append('float v = n%d();' % a)
            elif vol:
                L.append('float a = n%d();' % a)
                L.append('float b = n%d();' % b)
                expr = {'add': 'a + b', 'sub': 'a - b', 'mul': 'a * b', 'div': 'a / b', 'min': 'mcc::vol_min(a, b)', 'max': 'mcc::vol_max(a, b)'}.get(kind)
                if kind == 'divc':
                    expr = 'a * %s' % flit_v(inv)
                L.append('float v = %s;' % expr)
            else:
                Rn = self.ir.nodes[n['r']]
                if kind in ('add', 'sub'):
                    L.append('float a = n%d(); float b = n%d(); float v = a %s b;' % (n['l'], n['r'], '+' if kind == 'add' else '-'))
                elif kind == 'mul':
                    if lc or rc:
                        L.append('float a = n%d(); float b = n%d(); float v = a * b;' % (n['l'], n['r']))
                    else:
                        L.append('float a = n%d(); float v = a == 0.0f ? 0.0f : a * n%d();' % (n['l'], n['r']))
                elif kind == 'divc':
                    L.append('float v = n%d() * %s;' % (n['l'], flit_v(inv)))
                elif kind == 'div':
                    if lc:
                        L.append('float v = %s / n%d();' % (flit(self.ir.nodes[n['l']]['v']), n['r']))
                    else:
                        L.append('float a = n%d(); float v = a == 0.0f ? 0.0f : a / n%d();' % (n['l'], n['r']))
                elif kind in ('min', 'max'):
                    fn = 'mcc::java_fmin' if kind == 'min' else 'mcc::java_fmax'
                    if lc or rc:
                        L.append('float s = n%d(); float c = n%d(); float v = %s(s, c);' % (a, b, fn))
                    elif kind == 'min':
                        L.append('float a = n%d(); float v = a <= %s ? a : mcc::java_fmin(a, n%d());' % (n['l'], flit(Rn['range'][0]), n['r']))
                    else:
                        L.append('float a = n%d(); float v = a >= %s ? a : mcc::java_fmax(a, n%d());' % (n['l'], flit(Rn['range'][1]), n['r']))
        elif t == 'unary':
            f = {'abs': 'mcc::un_abs(u)', 'square': 'mcc::un_square(u)', 'cube': 'mcc::un_cube(u)', 'sqrt': 'mcc::un_sqrt(u)',
                 'half_negative': 'mcc::un_leaky(u, 0.5f)', 'quarter_negative': 'mcc::un_leaky(u, 0.25f)', 'reciprocal': 'mcc::un_reciprocal(u)',
                 'negate': 'mcc::un_negate(u)', 'squeeze': 'mcc::un_squeeze(u)', 'sign': 'mcc::un_sign(u)'}.get(n['op'])
            if f is None:
                raise SystemExit('unsupported unary ' + n['op'])
            L.append('float u = n%d();' % n['in'])
            L.append('float v = %s;' % f)
        elif t == 'clamp':
            L.append('float v = mcc::mth_clamp(n%d(), %s, %s);' % (n['in'], flit(n['min']), flit(n['max'])))
        elif t == 'lerp':
            L.append('float a = n%d();' % n['alpha'])
            L.append('float v;')
            L.append('if (a == 0.0f) v = n%d();' % n['first'])
            L.append('else if (a == 1.0f) v = n%d();' % n['second'])
            L.append('else { float f0 = n%d(); float f1 = n%d(); v = mcc::mth_lerp(a, f0, f1); }' % (n['first'], n['second']))
        elif t == 'range_choice':
            L.append('float c = n%d();' % n['in'])
            L.append('float v = mcc::in_range(c, %s, %s) ? n%d() : n%d();' % (flit(n['min_inclusive']), flit(n['max_exclusive']), n['when_in'], n['when_out']))
        elif t == 'interval_select':
            L.append('float c = n%d();' % n['in'])
            L.append('float v;')
            fns = n['functions']
            for k, th in enumerate(n['thresholds']):
                L.append('%sif (c < %s) v = n%d();' % ('' if k == 0 else 'else ', flit(th), fns[k]))
            L.append('else v = n%d();' % fns[-1])
        elif t == 'spline':
            # Splines are emitted once as shared functions; coordinates are evaluated
            # eagerly here (pure functions, so the value equals vanilla's lazy sampling).
            coords = list(dict.fromkeys(spline_coords(n['spline'])))
            fname = self.ensure_spline(i, coords)
            L.append('const float cs[%d] = {%s};' % (len(coords), ', '.join('n%d()' % c for c in coords)))
            L.append('float v = %s(cs);' % fname)
        else:
            raise SystemExit('unsupported node type %s in %s' % (t, ctx.name))
        return L

    def emit_spline(self, s, target, L, counter):
        """CubicSpline.Multipoint.sample into `target`; every local carries a unique
        suffix so nested splines never shadow their parent's target."""
        if 'k' in s:
            L.append('%s = %s;' % (target, flit(s['k'])))
            return
        k = counter[0]
        counter[0] += 1
        loc = [f32(b) for b in s['loc']]
        der = [f32(b) for b in s['der']]
        n = len(loc)
        q = '_%d' % k
        L.append('{')
        L.append('const float in%s = cs[%d];' % (q, self.spline_coord_index[s['coord']]))
        L.append('int st%s;' % q)
        L.append('{')
        L.append('int from%s = 0, len%s = %d;' % (q, q, n))
        L.append('while (len%s > 0) {' % q)
        L.append('int half%s = len%s / 2, mid%s = from%s + half%s;' % (q, q, q, q, q))
        L.append('float lm%s;' % q)
        L.append('switch (mid%s) {' % q)
        for i in range(n):
            L.append('case %d: lm%s = %s; break;' % (i, q, flit_v(loc[i])))
        L.append('default: lm%s = 0.0f; break;' % q)
        L.append('}')
        L.append('if (in%s < lm%s) len%s = half%s; else { from%s = mid%s + 1; len%s -= half%s + 1; }' % (q, q, q, q, q, q, q, q))
        L.append('}')
        L.append('st%s = from%s - 1;' % (q, q))
        L.append('}')
        L.append('switch (st%s) {' % q)
        L.append('case -1: {')
        L.append('float y0%s;' % q)
        self.emit_spline(s['val'][0], 'y0' + q, L, counter)
        if der[0] == 0.0:
            L.append('%s = y0%s;' % (target, q))
        else:
            L.append('float dx%s = in%s - %s; %s = y0%s + %s * dx%s;' % (q, q, flit_v(loc[0]), target, q, flit_v(der[0]), q))
        L.append('break;')
        L.append('}')
        for st in range(n - 1):
            L.append('case %d: {' % st)
            x1, x2 = np.float32(loc[st]), np.float32(loc[st + 1])
            d1, d2 = np.float32(der[st]), np.float32(der[st + 1])
            L.append('const float x1%s = %s, x2%s = %s, d1%s = %s, d2%s = %s;' % (q, flit_v(x1), q, flit_v(x2), q, flit_v(d1), q, flit_v(d2)))
            L.append('float t%s = (in%s - x1%s) / (x2%s - x1%s);' % (q, q, q, q, q))
            L.append('float y1%s, y2%s;' % (q, q))
            self.emit_spline(s['val'][st], 'y1' + q, L, counter)
            self.emit_spline(s['val'][st + 1], 'y2' + q, L, counter)
            L.append('float dx%s = x2%s - x1%s, dy%s = y2%s - y1%s;' % (q, q, q, q, q, q))
            L.append('float a%s = d1%s * dx%s - dy%s;' % (q, q, q, q))
            L.append('float b%s = -d2%s * dx%s + dy%s;' % (q, q, q, q))
            L.append('float l1%s = mcc::mth_lerp(t%s, y1%s, y2%s);' % (q, q, q, q))
            L.append('float w%s = t%s * (1.0f - t%s);' % (q, q, q))
            L.append('float l2%s = mcc::mth_lerp(t%s, a%s, b%s);' % (q, q, q, q))
            L.append('%s = l1%s + w%s * l2%s;' % (target, q, q, q))
            L.append('break;')
            L.append('}')
        L.append('default: {')
        L.append('float yl%s;' % q)
        self.emit_spline(s['val'][n - 1], 'yl' + q, L, counter)
        if der[n - 1] == 0.0:
            L.append('%s = yl%s;' % (target, q))
        else:
            L.append('float dx%s = in%s - %s; %s = yl%s + %s * dx%s;' % (q, q, flit_v(loc[n - 1]), target, q, flit_v(der[n - 1]), q))
        L.append('break;')
        L.append('}')
        L.append('}')
        L.append('}')

    # ---- material rules -------------------------------------------------------------
    def emit_rules(self):
        out_c, out_r = [], []
        self.rand_names = []
        counter = {'c': 0, 'r': 0}

        def cond(c):
            t = c['t']
            if t == 'ref':
                return cond(c['cond'])
            k = counter['c']
            counter['c'] += 1
            name = 'cond%d' % k
            if t == 'biome':
                mask = [0, 0]
                for b in c['biomes']:
                    mask[b >> 6] |= 1 << (b & 63)
                body = 'const int b = c.biome(); return ((b < 64 ? (0x%016Xull >> b) : (0x%016Xull >> (b - 64))) & 1ull) != 0;' % (mask[0], mask[1])
            elif t == 'noise_threshold':
                body = 'const double v = c.%s(%d); return v >= %s && v <= %s;' % ('noise3d' if c['is_3d'] else 'noise2d', self.noise_index[c['noise']],
                                                                                   dlit(c['min']), dlit(c['max']))
            elif t == 'vertical_gradient':
                if c['random_name'] not in self.rand_names:
                    self.rand_names.append(c['random_name'])
                body = 'return c.vertical_gradient(%d, %d, %d);' % (self.rand_names.index(c['random_name']), c['true_at_and_below'], c['false_at_and_above'])
            elif t == 'y_above':
                body = 'return c.y + %s >= %d + c.surface_depth * %d;' % ('c.stone_above' if c['add_stone_depth'] else '0', c['anchor'], c['surface_depth_multiplier'])
            elif t == 'water':
                body = 'return c.water_height == INT_MIN || c.y + %s >= c.water_height + %d + c.surface_depth * %d;' % (
                    'c.stone_above' if c['add_stone_depth'] else '0', c['offset'], c['surface_depth_multiplier'])
            elif t == 'not':
                body = 'return !%s(c);' % cond(c['cond'])
            elif t == 'stone_depth':
                body = 'return c.stone_depth(%d, %s, %d, %s);' % (c['offset'], 'true' if c['add_surface_depth'] else 'false', c['secondary_depth_range'],
                                                                 'true' if c['ceiling'] else 'false')
            elif t == 'temperature':
                body = 'return c.cold_enough_to_snow();'
            elif t == 'steep':
                body = 'return c.grad_x <= -4 || c.grad_z >= 4;'
            elif t == 'hole':
                body = 'return c.surface_depth <= 0;'
            elif t == 'above_preliminary_surface':
                body = 'return c.y >= c.min_surface_level();'
            else:
                raise SystemExit('unsupported condition ' + t)
            out_c.append((name, body))
            return name

        def rule(r):
            t = r['t']
            if t == 'ref':
                return rule(r['rule'])
            k = counter['r']
            counter['r'] += 1
            name = 'rule%d' % k
            if t == 'block':
                body = 'return %d;' % r['state']
            elif t == 'bandlands':
                body = 'return c.band();'
            elif t == 'ore_vein':
                body = 'return c.ore_vein(%d);' % self.ore_rules.index(r)
            elif t == 'condition':
                cn = cond(r['if'])
                th = rule(r['then'])
                body = 'return %s(c) ? %s(c) : -1;' % (cn, th)
            elif t == 'sequence':
                subs = [rule(x) for x in r['rules']]
                body = 'int s;\n' + '\n'.join('    s = %s(c); if (s >= 0) return s;' % x for x in subs) + '\n    return -1;'
            else:
                raise SystemExit('unsupported rule ' + t)
            out_r.append((name, body))
            return name

        root = rule(self.tm['material_rule'])
        o = []
        for name, _ in out_c:
            o.append('MCC_RULE_FN bool %s(SurfaceCtx& c);' % name)
        for name, _ in out_r:
            o.append('MCC_RULE_FN int %s(SurfaceCtx& c);' % name)
        for name, body in out_c:
            o.append('MCC_RULE_FN bool %s(SurfaceCtx& c) { %s }' % (name, body))
        for name, body in out_r:
            o.append('MCC_RULE_FN int %s(SurfaceCtx& c) {\n    %s\n}' % (name, body))
        o.append('MCC_RULE_FN int apply_material_rules(SurfaceCtx& c) { return %s(c); }' % root)
        return o

    # ---- output ---------------------------------------------------------------------
    def entry(self, o, sig, struct_init, outs):
        o.append('MCC_GEN_FN %s {' % sig)
        o.append('    %s' % struct_init)
        for line in outs:
            o.append('    %s' % line)
        o.append('}')

    def generate(self):
        N = self.ir.nodes
        o = ['// GENERATED by cuda/codegen/gen_router.py from the local IR and terrain model. Do not edit, do not commit.',
             '#pragma once', '#include <climits>', '#include "mccuda/noise_eval.h"', '#include "mccuda/df_ops.h"', '#include "mccuda_cuda/router_tables.h"',
             '// Force-inline in the device pass only; the host pass (gen-host checks) uses plain inline.',
             '#if defined(__CUDA_ARCH__)', '#define MCC_GEN_FN __host__ __device__ __forceinline__',
             '#elif defined(__CUDACC__)', '#define MCC_GEN_FN __host__ __device__ inline', '#else', '#define MCC_GEN_FN inline', '#endif',
             '// Splines are large and evaluated per column: keep them out of line on the device.',
             '#if defined(__CUDA_ARCH__)', '#define MCC_SPLINE_FN __host__ __device__ __noinline__',
             '#elif defined(__CUDACC__)', '#define MCC_SPLINE_FN __host__ __device__ inline', '#else', '#define MCC_SPLINE_FN inline', '#endif',
             'namespace mcc::gen {']
        o.append('constexpr int kNumNoises = %d;' % len(self.noise_keys))
        o.append('constexpr int kNumBlended = %d;' % len(self.blended_ids))
        o.append('constexpr int kNumChannels = %d;' % len(self.chans))
        o.append('constexpr int kNumOreChannels = %d;' % len(self.ore_chans))
        o.append('constexpr int kNumColumns = %d;' % len(self.col_a))
        o.append('constexpr int kNumClimateColumns = %d;' % len(self.col_b))
        o.append('constexpr int kNumOreRules = %d;' % len(self.ore_rules))
        o.append('constexpr int kFtsLowerBound = %d, kFtsCellHeight = %d;' % (self.fts['lower_bound'], self.fts['cell_height']))
        o.append('static const char* const kNoiseKeys[kNumNoises] = {%s};' % ', '.join('"%s"' % k for k in self.noise_keys))
        o.append('static constexpr int kChannelNodeIds[kNumChannels] = {%s};' % ', '.join(str(c) for c in self.chans))
        o.append('static constexpr int kColumnNodeIds[kNumColumns > 0 ? kNumColumns : 1] = {%s};' % (', '.join(str(c) for c in self.col_a) or '0'))
        o.append('static const int kBlendedNodeIds[kNumBlended > 0 ? kNumBlended : 1] = {%s};' % (', '.join(str(b) for b in self.blended_ids) or '0'))
        o.append('static constexpr int kOreRuleState[kNumOreRules > 0 ? kNumOreRules : 1][3] = {%s};' % (
            ', '.join('{%d, %d, %d}' % (r['ore'], r['raw_ore'], r['filler']) for r in self.ore_rules) or '{0,0,0}'))
        o.append('static constexpr uint32_t kOreRuleRawChanceBits[kNumOreRules > 0 ? kNumOreRules : 1] = {%s};' % (
            ', '.join('0x%08Xu' % r['raw_ore_chance'] for r in self.ore_rules) or '0u'))
        o.append('')
        main = []
        main += self.emit_struct(Ctx('ColumnA', 'vol', 'ident_y0', 'err'), [N[c]['in'] for c in self.col_a])
        main += self.emit_struct(Ctx('ColumnB', 'vol', 'ident_y0', 'err'), [N[c]['in'] for c in self.col_b])
        main += self.emit_struct(Ctx('CornerEval', 'vol', 'col', 'pass', col_nodes=self.col_a), [N[c]['in'] for c in self.chans])
        main += self.emit_struct(Ctx('OreCornerEval', 'vol', 'col', 'pass', col_nodes=self.col_a), [N[c]['in'] for c in self.ore_chans])
        axis_member = 'const mcc::AxisRec *ax, *ay, *az;'
        main += self.emit_struct(Ctx('CornerEvalTab', 'vol', 'col', 'pass', col_nodes=self.col_a, axis=True), [N[c]['in'] for c in self.chans],
                                 extra_member=axis_member)
        main += self.emit_struct(Ctx('OreCornerEvalTab', 'vol', 'col', 'pass', col_nodes=self.col_a, axis=True), [N[c]['in'] for c in self.ore_chans],
                                 extra_member=axis_member)
        main += self.emit_struct(Ctx('BlockEval', 'vol', 'subst', 'chan', chans=self.chans), [self.root])
        main += self.emit_struct(Ctx('OreBlockEval', 'vol', 'subst', 'chan', chans=self.chans + self.ore_chans), self.ore_roots)
        main += self.emit_struct(Ctx('QuartEval', 'vol', 'col', 'err', col_nodes=self.col_b), self.climate_roots)
        main += self.emit_struct(Ctx('VolY0Eval', 'vol', 'ident_y0', 'err'), [self.fts['upper_bound']])
        y0_roots = [N[s]['in'] for s in self.pt_slices] + [self.fts['upper_bound']]
        main += self.emit_struct(Ctx('PointY0', 'pt', 'ident_y0', 'err'), y0_roots)
        main += self.emit_struct(Ctx('PointEval', 'pt', 'sub', 'err'), self.point_roots, extra_member='PointY0* sub;')
        o += self.pre
        o += main
        o.append('')
        o.append('// Volume noise call sites of the *Tab contexts; each has one AxisRec per layer and coordinate.')
        o.append('constexpr int kNumAxisSites = %d;' % len(self.axis_sites))
        o.append('static const mcc::AxisSite kAxisSites[kNumAxisSites > 0 ? kNumAxisSites : 1] = {%s};' % (', '.join(
            '{%d, %d, %s, %s, %d}' % (kd, ix, dlit_v(float.fromhex(xz)), dlit_v(float.fromhex(y)), sm) for kd, ix, xz, y, sm in self.axis_sites) or '{}'))
        self.entry(o, 'void eval_columns(const RouterTables& T, int32_t x, int32_t z, float* out)', 'ColumnA e{T, x, 0, z};',
                   ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.col_a)])
        self.entry(o, 'void eval_climate_columns(const RouterTables& T, int32_t x, int32_t z, float* out)', 'ColumnB e{T, x, 0, z};',
                   ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.col_b)])
        self.entry(o, 'void eval_corners(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, float* out)', 'CornerEval e{T, x, y, z, col};',
                   ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.chans)])
        self.entry(o, 'void eval_ore_corners(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, float* out)',
                   'OreCornerEval e{T, x, y, z, col};', ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.ore_chans)])
        tab_args = 'const mcc::AxisRec* ax, const mcc::AxisRec* ay, const mcc::AxisRec* az'
        self.entry(o, 'void eval_corners_tab(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, %s, float* out)' % tab_args,
                   'CornerEvalTab e{T, x, y, z, col, ax, ay, az};', ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.chans)])
        self.entry(o, 'void eval_ore_corners_tab(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, %s, float* out)' % tab_args,
                   'OreCornerEvalTab e{T, x, y, z, col, ax, ay, az};', ['out[%d] = e.n%d();' % (k, N[c]['in']) for k, c in enumerate(self.ore_chans)])
        self.entry(o, 'float eval_block(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* ch)', 'BlockEval e{T, x, y, z, ch};',
                   ['return e.n%d();' % self.root])
        o.append('// ch: density channels followed by ore channels. which = 2 * rule + (0 density | 1 richness)')
        self.entry(o, 'float eval_ore(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* ch, int which)', 'OreBlockEval e{T, x, y, z, ch};',
                   ['switch (which) {'] + ['    case %d: return e.n%d();' % (k, r) for k, r in enumerate(self.ore_roots)] + ['}', 'return 0.0f;'])
        self.entry(o, 'void eval_climate(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, float* out)', 'QuartEval e{T, x, y, z, col};',
                   ['out[%d] = e.n%d();' % (k, r) for k, r in enumerate(self.climate_roots)])
        dens, ub = self.fts['density'], self.fts['upper_bound']
        self.entry(o, 'float fts_find(const RouterTables& T, int32_t x, int32_t z, float upper, PointY0& sub)', 'float q = upper / static_cast<float>(kFtsCellHeight);', [
            'int top = mcc::mth_floor(static_cast<double>(q)) * kFtsCellHeight;',
            'if (top <= kFtsLowerBound) return static_cast<float>(kFtsLowerBound);',
            'for (int probe = top; probe >= kFtsLowerBound; probe -= kFtsCellHeight) {',
            '    PointEval e{T, x, probe, z, &sub};',
            '    if (e.n%d() > 0.0f) return static_cast<float>(probe);' % dens,
            '}',
            'return static_cast<float>(kFtsLowerBound);'])
        self.entry(o, 'float fts_volume(const RouterTables& T, int32_t x, int32_t z)', 'VolY0Eval u{T, x, 0, z};',
                   ['float ub = u.n%d();' % ub, 'PointY0 sub{T, x, 0, z};', 'return fts_find(T, x, z, ub, sub);'])
        self.entry(o, 'float fts_point(const RouterTables& T, int32_t x, int32_t z)', 'PointY0 sub{T, x, 0, z};',
                   ['float ub = sub.n%d();' % ub, 'return fts_find(T, x, z, ub, sub);'])
        o.append('// which: 0 barrier, 1 floodedness, 2 spread, 3 lava, 4 exclusion, 5 + k: filler gap of ore rule k')
        self.entry(o, 'float eval_point(const RouterTables& T, int32_t x, int32_t y, int32_t z, int which)', 'PointY0 sub{T, x, 0, z};',
                   ['PointEval e{T, x, y, z, &sub};', 'switch (which) {'] +
                   ['    case %d: return e.n%d();' % (k, r) for k, r in enumerate(self.point_roots[:-1])] + ['}', 'return 0.0f;'])
        o.append('}  // namespace mcc::gen')
        rules = self.emit_rules()
        o.append('#ifdef MCC_WITH_RULES')
        o.append('namespace mcc::gen {')
        o.append('static constexpr int kNumRandomNames = %d;' % len(self.rand_names))
        o.append('static const char* const kRandomNames[kNumRandomNames > 0 ? kNumRandomNames : 1] = {%s};' % (
            ', '.join('"%s"' % n for n in self.rand_names) or '""'))
        o += rules
        o.append('}  // namespace mcc::gen')
        o.append('#endif')
        return '\n'.join(o) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ir')
    ap.add_argument('terrain')
    ap.add_argument('out')
    args = ap.parse_args()
    sys.setrecursionlimit(20000)
    with open(args.terrain, encoding='utf-8') as fh:
        tm = json.load(fh)
    gen = Gen(IR(args.ir), tm)
    code = gen.generate()
    with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(code)
    print('gen_router: %d+%d channels, %d+%d column nodes, %d noises, %d ore rules, %d point roots, %d axis sites -> %s' % (
        len(gen.chans), len(gen.ore_chans), len(gen.col_a), len(gen.col_b), len(gen.noise_keys), len(gen.ore_rules), len(gen.point_roots),
        len(gen.axis_sites), args.out))


if __name__ == '__main__':
    main()
