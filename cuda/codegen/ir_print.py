"""Pretty-print an IR root as an indented tree (shared nodes printed once)."""
import argparse
import sys

from ir import IR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ir')
    ap.add_argument('--root', default='final_density')
    args = ap.parse_args()
    ir = IR(args.ir)
    seen = set()

    def show(i, depth):
        n = ir.nodes[i]
        pad = '  ' * depth
        if i in seen:
            print('%s@%d %s (shared)' % (pad, i, n['t']))
            return
        seen.add(i)
        print('%s@%d %s' % (pad, i, ir.describe(n)))
        for c in ir.children(n):
            show(c, depth + 1)

    show(ir.roots[args.root], 0)


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
