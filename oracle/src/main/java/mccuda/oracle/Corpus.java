package mccuda.oracle;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The fixed test corpus: seeds and chunk positions. Everything is derived from
 * constants, so every oracle run produces the same corpus.
 */
final class Corpus {
    record Seed(int index, long seed, boolean quick) {}

    record Chunk(int cx, int cz, String tag) {}

    /** World border is at +-29,999,984 blocks; these chunks contain or touch it. */
    static final int BORDER_CHUNK = 29_999_984 >> 4;

    private static final long[] FIXED = {0L, 1L, -1L, Long.MIN_VALUE, Long.MAX_VALUE, 2147483647L, -2147483648L, -4172144997902289642L};
    private static final int RANDOM_SEEDS = 24;

    static List<Seed> seeds() {
        List<Seed> out = new ArrayList<>();
        for (long s : FIXED) out.add(new Seed(out.size(), s, out.size() == 0 || out.size() == 1 || out.size() == 7));
        SplittableRandom r = new SplittableRandom(0x6D63637564614C4CL);
        for (int i = 0; i < RANDOM_SEEDS; i++) out.add(new Seed(out.size(), r.nextLong(), false));
        return out;
    }

    /**
     * Chunk set for one seed: the 3x3 around the origin (negative coordinates and
     * the -1/0 boundaries), random near and far chunks, world-border chunks, and
     * chunks straddling the coordinate where vanilla's wrap() starts folding the
     * highest-frequency BlendedNoise octave (|x * 171.103| >= 2^24).
     */
    static List<Chunk> chunks(Seed seed, boolean quick) {
        List<Chunk> out = new ArrayList<>();
        for (int dz = -1; dz <= 1; dz++) for (int dx = -1; dx <= 1; dx++) out.add(new Chunk(dx, dz, "origin"));

        int[][] border = {
            {BORDER_CHUNK, BORDER_CHUNK}, {-BORDER_CHUNK - 1, -BORDER_CHUNK - 1}, {BORDER_CHUNK, -BORDER_CHUNK - 1}, {-BORDER_CHUNK - 1, BORDER_CHUNK},
            {BORDER_CHUNK, 0}, {0, -BORDER_CHUNK - 1}, {-BORDER_CHUNK - 1, 7}, {3, BORDER_CHUNK},
        };
        // 2^24 / (684.412 * 0.25) ~= 98053.9 blocks: the first wrap fold of the top limit octave.
        int[][] wrap = {{6128, 0}, {-6129, 5}, {11, 6128}, {-3, -6129}};

        SplittableRandom r = new SplittableRandom(seed.seed() ^ 0x5DEECE66DL);
        if (quick) {
            out.add(new Chunk(border[seed.index() % border.length][0], border[seed.index() % border.length][1], "border"));
            out.add(new Chunk(wrap[seed.index() % wrap.length][0], wrap[seed.index() % wrap.length][1], "wrap"));
            out.add(new Chunk(r.nextInt(-6250, 6250), r.nextInt(-6250, 6250), "near"));
            return out;
        }
        for (int i = 0; i < 3; i++) out.add(new Chunk(r.nextInt(-6250, 6250), r.nextInt(-6250, 6250), "near"));
        for (int i = 0; i < 2; i++) out.add(new Chunk(r.nextInt(-625_000, 625_000), r.nextInt(-625_000, 625_000), "far"));
        for (int i = 0; i < 2; i++) {
            int[] b = border[(seed.index() * 2 + i) % border.length];
            out.add(new Chunk(b[0], b[1], "border"));
        }
        for (int i = 0; i < 2; i++) {
            int[] w = wrap[(seed.index() * 2 + i) % wrap.length];
            out.add(new Chunk(w[0], w[1], "wrap"));
        }
        return out;
    }
}
