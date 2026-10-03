package mccuda.oracle;

import java.io.IOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.stream.JsonWriter;

import net.minecraft.util.RandomSource;
import net.minecraft.world.level.levelgen.LegacyRandomSource;
import net.minecraft.world.level.levelgen.PositionalRandomFactory;
import net.minecraft.world.level.levelgen.RandomSupport;
import net.minecraft.world.level.levelgen.XoroshiroRandomSource;

/**
 * Reference streams for the RNGs: vanilla's Xoroshiro128++ source (including
 * its bounded nextInt, nextDouble and nextFloat), positional forks, the
 * MD5-based seedFromHashOf, and the legacy java.util.Random LCG source.
 * Longs are written as signed decimal strings; doubles/floats as raw bits.
 */
final class RngDump {
    static final long[] SEEDS = {0L, 1L, -1L, Long.MIN_VALUE, Long.MAX_VALUE, 12345L, -4172144997902289642L, 0x123456789ABCDEFL};
    static final int[] BOUNDS = {1, 2, 3, 7, 100, 255, 256, 1000, 65536, 1 << 30, (1 << 30) + 1, Integer.MAX_VALUE};
    static final String[] HASH_INPUTS = {
        "", "a", "minecraft:terrain", "minecraft:offset", "minecraft:continentalness", "octave_0", "octave_-7", "octave_-15",
        "minecraft:aquifer", "minecraft:ore", "The quick brown fox jumps over the lazy dog", "éè unicode ☃",
    };
    static final int N = 24;

    static void dump(Path out) throws IOException {
        Files.createDirectories(out.getParent());
        try (Writer w = Files.newBufferedWriter(out, StandardCharsets.UTF_8); JsonWriter json = new JsonWriter(w)) {
            json.setIndent(" ");
            json.beginObject();
            json.name("hash").beginArray();
            for (String s : HASH_INPUTS) {
                RandomSupport.Seed128bit seed = RandomSupport.seedFromHashOf(s);
                json.beginObject().name("input").value(s).name("lo").value(Long.toString(seed.seedLo())).name("hi").value(Long.toString(seed.seedHi())).endObject();
            }
            json.endArray();

            json.name("xoroshiro").beginArray();
            for (long seed : SEEDS) {
                json.beginObject().name("seed").value(Long.toString(seed));
                RandomSupport.Seed128bit up = RandomSupport.upgradeSeedTo128bit(seed);
                json.name("upgraded_lo").value(Long.toString(up.seedLo())).name("upgraded_hi").value(Long.toString(up.seedHi()));
                json.name("stafford13").value(Long.toString(RandomSupport.mixStafford13(seed)));
                RandomSource r = new XoroshiroRandomSource(seed);
                json.name("next_long").beginArray();
                for (int i = 0; i < N; i++) json.value(Long.toString(r.nextLong()));
                json.endArray();
                r = new XoroshiroRandomSource(seed);
                json.name("next_int_bounded").beginArray();
                for (int bound : BOUNDS) for (int i = 0; i < 4; i++) json.value(r.nextInt(bound));
                json.endArray();
                r = new XoroshiroRandomSource(seed);
                json.name("next_double").beginArray();
                for (int i = 0; i < N; i++) json.value(IrExporter.dbits(r.nextDouble()));
                json.endArray();
                r = new XoroshiroRandomSource(seed);
                json.name("next_float").beginArray();
                for (int i = 0; i < N; i++) json.value(IrExporter.fbits(r.nextFloat()));
                json.endArray();
                PositionalRandomFactory pos = new XoroshiroRandomSource(seed).forkPositional();
                json.name("positional_hash").beginArray();
                for (String s : HASH_INPUTS) json.value(Long.toString(pos.fromHashOf(s).nextLong()));
                json.endArray();
                json.name("positional_at").beginArray();
                int[][] at = {{0, 0, 0}, {1, -64, -1}, {-29999984, 319, 29999984}, {123456, 7, -654321}};
                for (int[] p : at) json.value(Long.toString(pos.at(p[0], p[1], p[2]).nextLong()));
                json.endArray();
                json.endObject();
            }
            json.endArray();

            json.name("legacy").beginArray();
            for (long seed : SEEDS) {
                json.beginObject().name("seed").value(Long.toString(seed));
                RandomSource r = new LegacyRandomSource(seed);
                json.name("next_long").beginArray();
                for (int i = 0; i < N; i++) json.value(Long.toString(r.nextLong()));
                json.endArray();
                r = new LegacyRandomSource(seed);
                json.name("next_int_bounded").beginArray();
                for (int bound : BOUNDS) for (int i = 0; i < 4; i++) json.value(r.nextInt(bound));
                json.endArray();
                r = new LegacyRandomSource(seed);
                json.name("next_double").beginArray();
                for (int i = 0; i < N; i++) json.value(IrExporter.dbits(r.nextDouble()));
                json.endArray();
                r = new LegacyRandomSource(seed);
                json.name("next_float").beginArray();
                for (int i = 0; i < N; i++) json.value(IrExporter.fbits(r.nextFloat()));
                json.endArray();
                r = new LegacyRandomSource(seed);
                r.consumeCount(262);
                json.name("after_consume_262").value(Long.toString(r.nextLong()));
                json.endObject();
            }
            json.endArray();
            json.name("bounds").beginArray();
            for (int b : BOUNDS) json.value(b);
            json.endArray();
            json.endObject();
        }
        System.out.println("dump-rng: " + out);
    }
}
