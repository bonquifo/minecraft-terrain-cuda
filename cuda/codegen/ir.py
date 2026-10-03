"""Loader and helpers for the mccuda density-function IR (data/gamedata/<ver>/overworld_ir.json).

The IR is produced locally by the oracle (`export-ir`) from the game jar and is never
committed. Floats are stored as raw IEEE-754 bits (int), doubles as raw bits (hex string).
"""
import json
import struct


def f32(bits):
    return struct.unpack('<f', struct.pack('<I', bits & 0xFFFFFFFF))[0]


def f64(hexbits):
    return struct.unpack('<d', struct.pack('<Q', int(hexbits, 16)))[0]


class IR:
    def __init__(self, path):
        with open(path, encoding='utf-8') as fh:
            self.doc = json.load(fh)
        self.nodes = {n['id']: n for n in self.doc['nodes']}
        self.roots = self.doc['roots']
        self.noises = self.doc['noises']

    def children(self, n):
        """Child node ids in a stable order (duplicates removed)."""
        t = n['t']
        out = []
        for k in ('in', 'l', 'r', 'alpha', 'first', 'second', 'when_in', 'when_out'):
            if k in n:
                out.append(n[k])
        if t == 'noise':
            for axis in 'xyz':
                if not n['shift_%s_zero' % axis]:
                    out.append(n['shift_' + axis])
        if t == 'interval_select':
            out += n['functions']
        if t == 'spline':
            out += spline_coords(n['spline'])
        return list(dict.fromkeys(out))

    def describe(self, n):
        t = n['t']
        d = t
        if t in ('binary', 'unary'):
            d += ':' + n['op']
        elif t == 'constant':
            d += ' %r' % f32(n['v'])
        elif t == 'noise':
            shift = 'none'
            if not (n['shift_x_zero'] and n['shift_z_zero'] and n['shift_y_zero']):
                shift = 'xz' if n['shift_y_zero'] else 'xyz'
            d += ' %s xz=%r y=%r shift=%s' % (n['noise'], f64(n['xz_scale']), f64(n['y_scale']), shift)
        elif t == 'slice':
            d += ' %s=%d' % (n['axis'], n['coord'])
        elif t == 'interpolated':
            d += ' %dx%d' % (n['cell_xz'], n['cell_y'])
        elif t == 'cache':
            d += ' #%d' % n['cache_id']
        elif t == 'gradient':
            d += ' %s %s %d..%d %r..%r' % (n['axis'], n['tiling'], n['from'], n['to'], f32(n['from_value']), f32(n['to_value']))
        elif t == 'clamp':
            d += ' [%r, %r]' % (f32(n['min']), f32(n['max']))
        elif t == 'range_choice':
            d += ' [%r, %r)' % (f32(n['min_inclusive']), f32(n['max_exclusive']))
        elif t == 'interval_select':
            d += ' thresholds=%s' % [f32(b) for b in n['thresholds']]
        elif t == 'simple':
            d += ' ' + n['kind']
        elif t in ('shift_a', 'shift_b', 'shift'):
            d += ' ' + n['noise']
        elif t == 'blended_noise':
            d += ' xz=%r y=%r xzf=%r yf=%r smear=%r' % tuple(f64(n[k]) for k in ('xz_scale', 'y_scale', 'xz_factor', 'y_factor', 'smear_scale_multiplier'))
        d += '  axes=%d range=[%.6g, %.6g]' % (n['axes'], f32(n['range'][0]), f32(n['range'][1]))
        return d


def spline_coords(s):
    out = []
    if 'coord' in s:
        out.append(s['coord'])
        for v in s['val']:
            out += spline_coords(v)
    return out
