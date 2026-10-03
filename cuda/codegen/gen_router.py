"""Generate CUDA/C++ evaluation code for the overworld final density from the IR.

The generated header (written to the build tree, never committed, because it embeds
game-derived constants) defines three __host__ __device__ evaluators that reproduce
vanilla's *volume-path* semantics at single points:

  eval_columns(T, x, z, out)        values of every Y-slice node used by the
                                    interpolated inputs, at a cell-lattice column
  eval_corners(T, x, y, z, col, out) the interpolated channels' inputs at one cell
                                    corner (col = this column's eval_columns output)
  eval_block(ch, x, y, z)           final density at one block, given the already
                                    interpolated channel values `ch`

Per-element semantics are the shared helpers in cpu/include/mccuda/{noise_eval,df_ops}.h.
Branches (range_choice, lerp alpha 0/1, interval_select, splines, blended-noise
limits) are evaluated lazily: vanilla evaluates both sides over the whole volume,
but every node is a pure function of position, so skipping an unused side cannot
change a result.
"""
import argparse
import struct
import sys

import numpy as np

from ir import IR, f32, f64

CTX_COLUMN, CTX_CORNER, CTX_BLOCK = 'column', 'corner', 'block'


def flit(bits):
    """Exact C++ float literal for raw float32 bits."""
    v = f32(bits)
    if v != v or v in (float('inf'), float('-inf')):
        return 'mcc::bits2f(0x%08Xu)' % (bits & 0xFFFFFFFF)
    return '(%sf)' % v.hex()


def flit_v(v32):
    b = struct.unpack('<I', struct.pack('<f', float(v32)))[0]
    return flit(b)


def dlit(hexbits):
    v = f64(hexbits)
    if v != v or v in (float('inf'), float('-inf')):
        return 'mcc::bits2d(0x%016Xull)' % int(hexbits, 16)
    return '(%s)' % v.hex()


def dlit_v(v):
    return '(%s)' % float(v).hex()


class Gen:
    def __init__(self, ir):
        self.ir = ir
        self.noise_keys = sorted(ir.noises.keys())
        self.noise_index = {k: i for i, k in enumerate(self.noise_keys)}
        self.blended_ids = sorted(n['id'] for n in ir.nodes.values() if n['t'] == 'blended_noise')
        self.root = ir.roots['final_density']
        self.channels = []          # interpolated node ids (block context leaves)
        self.column_nodes = []      # slice-y node ids evaluated in the column context
        self.discover()

    # ---- graph discovery -----------------------------------------------------
    def discover(self):
        seen = set()

        def walk_block(i):
            if i in seen:
                return
            seen.add(i)
            n = self.ir.nodes[i]
            if n['t'] == 'interpolated':
                if i not in self.channels:
                    self.channels.append(i)
                return
            if n['t'] == 'slice':
                raise SystemExit('slice in block context is not supported')
            for c in self.ir.children(n):
                walk_block(c)

        walk_block(self.root)
        seen2 = set()

        def walk_corner(i):
            if i in seen2:
                return
            seen2.add(i)
            n = self.ir.nodes[i]
            if n['t'] == 'slice':
                if n['axis'] != 'y' or n['coord'] != 0:
                    raise SystemExit('only slice y=0 supported in the corner context')
                if i not in self.column_nodes:
                    self.column_nodes.append(i)
                return
            if n['t'] == 'interpolated':
                raise SystemExit('nested interpolation not supported')
            for c in self.ir.children(n):
                walk_corner(c)

        for ch in self.channels:
            n = self.ir.nodes[ch]
            if (n['cell_xz'], n['cell_y']) != (4, 8):
                raise SystemExit('only 4x8 interpolation cells are supported')
            walk_corner(n['in'])

    # ---- vanilla compile-time decisions ---------------------------------------
    def binary_plan(self, n):
        L, R = self.ir.nodes[n['l']], self.ir.nodes[n['r']]
        lc, rc = L['t'] == 'constant', R['t'] == 'constant'
        op = n['op']
        lmin, lmax = f32(L['range'][0]), f32(L['range'][1])
        rmin, rmax = f32(R['range'][0]), f32(R['range'][1])
        if op in ('add', 'sub', 'mul'):
            return (op, n['l'], n['r'], None)
        if op == 'div':
            if not lc and rc:
                inv = np.float32(1.0) / np.float32(f32(R['v']))
                return ('divc', n['l'], n['r'], inv)
            return ('div', n['l'], n['r'], None)
        if op == 'min':
            if lmax < rmin:
                return ('pass', n['l'], None, None)
            if rmax < lmin:
                return ('pass', n['r'], None, None)
            return ('min', n['r'], n['l'], None) if lc else ('min', n['l'], n['r'], None)
        if op == 'max':
            if lmin > rmax:
                return ('pass', n['l'], None, None)
            if rmin > lmax:
                return ('pass', n['r'], None, None)
            return ('max', n['r'], n['l'], None) if lc else ('max', n['l'], n['r'], None)
        raise SystemExit('bad binary op ' + op)

    def gradient_plan(self, n):
        rng = n['to'] - n['from']
        fv, tv = np.float32(f32(n['from_value'])), np.float32(f32(n['to_value']))
        factor = (tv - fv) / np.float32(rng)
        mode = {'clamp_to_edge': 0, 'repeat': 1, 'mirrored_repeat': 2}[n['tiling']]
        return mode, n['from'], min(n['from'], n['to']), max(n['from'], n['to']), rng, fv, factor

    # ---- code emission -------------------------------------------------------
    def emit_context(self, name, ctx, roots):
        """Emit a struct with one memoised method per node reachable from roots."""
        reach = []
        seen = set()

        def walk(i):
            if i in seen:
                return
            seen.add(i)
            n = self.ir.nodes[i]
            stop = (ctx == CTX_BLOCK and n['t'] == 'interpolated') or (ctx == CTX_CORNER and n['t'] == 'slice')
            if not stop:
                for c in self.ir.children(n):
                    walk(c)
            reach.append(i)

        for r in roots:
            walk(r)

        out = []
        out.append('struct %s {' % name)
        out.append('    const RouterTables& T;')
        out.append('    int32_t x, y, z;')
        if ctx == CTX_CORNER:
            out.append('    const float* col;')
        if ctx == CTX_BLOCK:
            out.append('    const float* ch;')
        for i in reach:
            out.append('    float m%d; bool h%d;' % (i, i))
        out.append('')
        for i in reach:
            out.append('    MCC_GEN_FN float n%d() {' % i)
            out.append('        if (h%d) return m%d;' % (i, i))
            body = self.node_body(i, ctx)
            for line in body:
                out.append('        ' + line)
            out.append('        m%d = v; h%d = true; return v;' % (i, i))
            out.append('    }')
        out.append('    MCC_HD void dump(float* v, bool* h) const {')
        for k, i in enumerate(reach):
            out.append('        v[%d] = m%d; h[%d] = h%d;' % (k, i, k, i))
        out.append('    }')
        out.append('};')
        out.append('static constexpr int kReach_%s[] = {%s};' % (name, ', '.join(str(i) for i in reach)))
        out.append('static constexpr int kReachCount_%s = %d;' % (name, len(reach)))
        return out, reach

    def stack(self, key):
        return 'T.stacks[%d]' % self.noise_index[key]

    def node_body(self, i, ctx):
        n = self.ir.nodes[i]
        t = n['t']
        L = []
        if ctx == CTX_BLOCK and t == 'interpolated':
            L.append('float v = ch[%d];' % self.channels.index(i))
            return L
        if ctx == CTX_CORNER and t == 'slice':
            L.append('float v = col[%d];' % self.column_nodes.index(i))
            return L
        if t == 'cache' or t == 'blend_density':
            L.append('float v = n%d();' % n['in'])
        elif t == 'interpolated':
            # Corner/column context: the cell volume is aligned with step == cell size,
            # so vanilla's sampler passes straight through to its input.
            L.append('float v = n%d();' % n['in'])
        elif t == 'slice':
            if ctx != CTX_COLUMN or n['axis'] != 'y' or n['coord'] != 0:
                raise SystemExit('unsupported slice in %s context' % ctx)
            L.append('float v = n%d();  // y is already 0 here' % n['in'])
        elif t == 'constant':
            L.append('float v = %s;' % flit(n['v']))
        elif t == 'simple':
            L.append('float v = %s;' % ('1.0f' if n['kind'] == 'blend_alpha' else '0.0f'))
        elif t == 'gradient':
            mode, frm, mn, mx, rng, fv, factor = self.gradient_plan(n)
            coord = {'x': 'x', 'y': 'y', 'z': 'z'}[n['axis']]
            L.append('const mcc::GradientPlan g{%d, %d, %d, %d, %d, %s, %s};' % (mode, frm, mn, mx, rng, flit_v(fv), flit_v(factor)))
            L.append('float v = mcc::gradient_compute(g, %s);' % coord)
        elif t == 'noise':
            s = self.stack(n['noise'])
            xz, ys = dlit(n['xz_scale']), dlit(n['y_scale'])
            if n['shift_x_zero'] and n['shift_y_zero'] and n['shift_z_zero']:
                L.append('const mcc::StackRef2& s = %s;' % s)
                L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, x, y, z, %s, %s, 1.0f);' % (xz, ys))
            else:
                L.append('const mcc::StackRef2& s = %s;' % s)
                L.append('double nx = static_cast<double>(x) * %s + static_cast<double>(n%d());' % (xz, n['shift_x']))
                if n['shift_y_zero']:
                    L.append('double ny = static_cast<double>(y) * %s;' % ys)
                else:
                    L.append('double ny = static_cast<double>(y) * %s + static_cast<double>(n%d());' % (ys, n['shift_y']))
                L.append('double nz = static_cast<double>(z) * %s + static_cast<double>(n%d());' % (xz, n['shift_z']))
                L.append('float v = mcc::stack_get(T.layers + s.first, s.count, T.octaves, nx, ny, nz);')
        elif t == 'shift_a' or t == 'shift':
            s = self.stack(n['noise'])
            yscale = '0.0' if t == 'shift_a' else '0.25'
            L.append('const mcc::StackRef2& s = %s;' % s)
            L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, x, y, z, 0.25, %s, 1.0f) * 4.0f;' % yscale)
        elif t == 'shift_b':
            s = self.stack(n['noise'])
            L.append('const mcc::StackRef2& s = %s;' % s)
            L.append('float v = mcc::stack_add_volume(0.0f, T.layers + s.first, s.count, T.octaves, false, z, x, 0, 0.25, 0.25, 4.0f);')
        elif t == 'blended_noise':
            bi = self.blended_ids.index(i)
            xzm = 684.412 * f64(n['xz_scale'])
            ym = 684.412 * f64(n['y_scale'])
            L.append('const mcc::BlendedRef& b = T.blended[%d];' % bi)
            L.append('const double xzm = %s, ym = %s;' % (dlit_v(xzm), dlit_v(ym)))
            L.append('float a = mcc::stack_add_volume(0.0f, T.layers + b.main.first, b.main.count, T.octaves, true, x, y, z, xzm / %s, ym / %s, 1.0f);'
                     % (dlit(n['xz_factor']), dlit(n['y_factor'])))
            L.append('a = mcc::mth_clamp(a + 0.5f, 0.0f, 1.0f);')
            L.append('float v;')
            L.append('if (a == 0.0f) v = mcc::stack_add_volume(0.0f, T.layers + b.min_limit.first, b.min_limit.count, T.octaves, true, x, y, z, xzm, ym, 1.0f);')
            L.append('else if (a == 1.0f) v = mcc::stack_add_volume(0.0f, T.layers + b.max_limit.first, b.max_limit.count, T.octaves, true, x, y, z, xzm, ym, 1.0f);')
            L.append('else {')
            L.append('    float lo = mcc::stack_add_volume(0.0f, T.layers + b.min_limit.first, b.min_limit.count, T.octaves, true, x, y, z, xzm, ym, 1.0f);')
            L.append('    float hi = mcc::stack_add_volume(0.0f, T.layers + b.max_limit.first, b.max_limit.count, T.octaves, true, x, y, z, xzm, ym, 1.0f);')
            L.append('    v = mcc::mth_lerp(a, lo, hi);')
            L.append('}')
        elif t == 'binary':
            kind, a, b, inv = self.binary_plan(n)
            if kind == 'pass':
                L.append('float v = n%d();' % a)
            else:
                L.append('float a = n%d();' % a)
                L.append('float b = n%d();' % b)
                expr = {
                    'add': 'a + b', 'sub': 'a - b', 'mul': 'a * b', 'div': 'a / b',
                    'min': 'mcc::vol_min(a, b)', 'max': 'mcc::vol_max(a, b)',
                }.get(kind)
                if kind == 'divc':
                    expr = 'a * %s' % flit_v(inv)
                L.append('float v = %s;' % expr)
        elif t == 'unary':
            op = n['op']
            f = {
                'abs': 'mcc::un_abs(u)', 'square': 'mcc::un_square(u)', 'cube': 'mcc::un_cube(u)', 'sqrt': 'mcc::un_sqrt(u)',
                'half_negative': 'mcc::un_leaky(u, 0.5f)', 'quarter_negative': 'mcc::un_leaky(u, 0.25f)',
                'reciprocal': 'mcc::un_reciprocal(u)', 'negate': 'mcc::un_negate(u)', 'squeeze': 'mcc::un_squeeze(u)',
                'sign': 'mcc::un_sign(u)',
            }.get(op)
            if f is None:
                raise SystemExit('unsupported unary ' + op)
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
            L.append('float v;')
            self.emit_spline(n['spline'], 'v', L, [0])
        else:
            raise SystemExit('unsupported node type in codegen: ' + t)
        return L

    def emit_spline(self, s, target, L, counter):
        """Emit CubicSpline.Multipoint.sample into `target`. Every local carries a
        unique suffix so nested splines never shadow their parent's target."""
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
        L.append('const float in%s = n%d();' % (q, s['coord']))
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
        # start < 0: linear extension from the first point
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

    def generate(self):
        o = []
        o.append('// GENERATED by cuda/codegen/gen_router.py from the local IR. Do not edit, do not commit.')
        o.append('#pragma once')
        o.append('#include "mccuda/noise_eval.h"')
        o.append('#include "mccuda/df_ops.h"')
        o.append('#include "mccuda_cuda/router_tables.h"')
        o.append('// Force-inline in the device pass only; the host pass (gen-host checks) uses plain inline.')
        o.append('#if defined(__CUDA_ARCH__)')
        o.append('#define MCC_GEN_FN __host__ __device__ __forceinline__')
        o.append('#elif defined(__CUDACC__)')
        o.append('#define MCC_GEN_FN __host__ __device__ inline')
        o.append('#else')
        o.append('#define MCC_GEN_FN inline')
        o.append('#endif')
        o.append('namespace mcc::gen {')
        o.append('constexpr int kNumNoises = %d;' % len(self.noise_keys))
        o.append('constexpr int kNumBlended = %d;' % len(self.blended_ids))
        o.append('constexpr int kNumChannels = %d;' % len(self.channels))
        o.append('constexpr int kNumColumns = %d;' % len(self.column_nodes))
        o.append('static const char* const kNoiseKeys[kNumNoises] = {%s};' % ', '.join('"%s"' % k for k in self.noise_keys))
        o.append('static constexpr int kChannelNodeIds[kNumChannels] = {%s};' % ', '.join(str(c) for c in self.channels))
        o.append('static constexpr int kColumnNodeIds[kNumColumns > 0 ? kNumColumns : 1] = {%s};' % (', '.join(str(c) for c in self.column_nodes) or '0'))
        o.append('static const int kBlendedNodeIds[kNumBlended > 0 ? kNumBlended : 1] = {%s};' % (', '.join(str(b) for b in self.blended_ids) or '0'))
        o.append('')
        lines, _ = self.emit_context('ColumnEval', CTX_COLUMN, [self.ir.nodes[c]['in'] for c in self.column_nodes])
        o += lines
        lines, _ = self.emit_context('CornerEval', CTX_CORNER, [self.ir.nodes[c]['in'] for c in self.channels])
        o += lines
        lines, _ = self.emit_context('BlockEval', CTX_BLOCK, [self.root])
        o += lines
        o.append('')
        # Entry points. Memo flags are zero-initialised by value-initialising the struct.
        o.append('MCC_GEN_FN void eval_columns(const RouterTables& T, int32_t x, int32_t z, float* out) {')
        o.append('    ColumnEval e{T, x, 0, z};')
        for k, c in enumerate(self.column_nodes):
            o.append('    out[%d] = e.n%d();' % (k, self.ir.nodes[c]['in']))
        o.append('}')
        o.append('MCC_GEN_FN void eval_corners(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, float* out) {')
        o.append('    CornerEval e{T, x, y, z, col};')
        for k, c in enumerate(self.channels):
            o.append('    out[%d] = e.n%d();' % (k, self.ir.nodes[c]['in']))
        o.append('}')
        o.append('MCC_GEN_FN float eval_block(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* ch) {')
        o.append('    BlockEval e{T, x, y, z, ch};')
        o.append('    return e.n%d();' % self.root)
        o.append('}')
        # Debug entry points: same evaluation, plus a dump of every memoised node value.
        o.append('MCC_GEN_FN void eval_columns_dbg(const RouterTables& T, int32_t x, int32_t z, float* out, float* mv, bool* mh) {')
        o.append('    ColumnEval e{T, x, 0, z};')
        for k, c in enumerate(self.column_nodes):
            o.append('    out[%d] = e.n%d();' % (k, self.ir.nodes[c]['in']))
        o.append('    e.dump(mv, mh);')
        o.append('}')
        o.append('MCC_GEN_FN void eval_corners_dbg(const RouterTables& T, int32_t x, int32_t y, int32_t z, const float* col, float* out, float* mv, bool* mh) {')
        o.append('    CornerEval e{T, x, y, z, col};')
        for k, c in enumerate(self.channels):
            o.append('    out[%d] = e.n%d();' % (k, self.ir.nodes[c]['in']))
        o.append('    e.dump(mv, mh);')
        o.append('}')
        o.append('}  // namespace mcc::gen')
        return '\n'.join(o) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ir')
    ap.add_argument('out')
    args = ap.parse_args()
    sys.setrecursionlimit(20000)
    gen = Gen(IR(args.ir))
    code = gen.generate()
    with open(args.out, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(code)
    print('gen_router: %d channels, %d column nodes, %d noises -> %s' % (len(gen.channels), len(gen.column_nodes), len(gen.noise_keys), args.out))


if __name__ == '__main__':
    main()
