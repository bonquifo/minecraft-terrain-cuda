package mccuda.client.mixin;

import java.util.Set;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import mccuda.client.GpuDump;
import net.minecraft.core.Holder;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeManager;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseChunk;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.material.rule.MaterialRule;

/**
 * Chunks inside the dumped region take their blocks from the GPU: the noise fill writes
 * the GPU's finished TERRAIN-step blocks (noise, aquifers and surface rules) and the
 * surface pass is skipped, since its result is already in place.
 */
@Mixin(NoiseBasedChunkGenerator.class)
public abstract class NoiseGeneratorMixin {
    @Inject(method = "doFill", at = @At("HEAD"), cancellable = true)
    private void mccuda$fill(NoiseChunk noiseChunk, ChunkAccess chunk, CallbackInfo ci) {
        GpuDump dump = GpuDump.get();
        if (dump == null) return;
        if (dump.serves(chunk.getPos())) {
            dump.fillBlocks(chunk);
            ci.cancel();
        } else {
            dump.vanillaChunks.incrementAndGet();
        }
    }

    @Inject(method = "buildSurface(Lnet/minecraft/world/level/chunk/ChunkAccess;Lnet/minecraft/world/level/levelgen/NoiseChunk;"
        + "Lnet/minecraft/world/level/levelgen/RandomState;Lnet/minecraft/world/level/biome/BiomeManager;Ljava/util/Set;"
        + "Lnet/minecraft/world/level/levelgen/material/rule/MaterialRule;)V", at = @At("HEAD"), cancellable = true)
    private void mccuda$surface(ChunkAccess chunk, NoiseChunk noiseChunk, RandomState random, BiomeManager biomes, Set<Holder<Biome>> set, MaterialRule rule,
                                CallbackInfo ci) {
        GpuDump dump = GpuDump.get();
        if (dump != null && dump.serves(chunk.getPos())) ci.cancel();
    }

    @Inject(method = "buildSurface(Lnet/minecraft/world/level/chunk/ChunkAccess;Lnet/minecraft/world/level/levelgen/NoiseChunk;"
        + "Lnet/minecraft/world/level/levelgen/RandomState;Lnet/minecraft/world/level/biome/BiomeManager;Ljava/util/Set;"
        + "Lnet/minecraft/world/level/levelgen/material/rule/MaterialRule;)V", at = @At("TAIL"))
    private void mccuda$verifySurface(ChunkAccess chunk, NoiseChunk noiseChunk, RandomState random, BiomeManager biomes, Set<Holder<Biome>> set, MaterialRule rule,
                                      CallbackInfo ci) {
        GpuDump dump = GpuDump.get();
        if (dump != null && !dump.serve && dump.covers(chunk.getPos())) dump.verifyBlocks(chunk);
    }
}
