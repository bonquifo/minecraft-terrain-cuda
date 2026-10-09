package mccuda.client.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import mccuda.client.GpuDump;
import net.minecraft.world.level.biome.BiomeSource;
import net.minecraft.world.level.chunk.ChunkAccess;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.blending.Blender;

/** Chunks inside the dumped region take their quart biomes from the GPU. */
@Mixin(ChunkGenerator.class)
public abstract class ChunkGeneratorMixin {
    @Shadow
    public abstract BiomeSource getBiomeSource();

    @Inject(method = "doCreateBiomes", at = @At("HEAD"), cancellable = true)
    private void mccuda$biomes(Blender blender, RandomState random, ChunkAccess chunk, CallbackInfo ci) {
        GpuDump dump = GpuDump.get();
        if (dump != null && dump.serves(chunk.getPos())) {
            dump.fillBiomes(chunk, getBiomeSource());
            ci.cancel();
        }
    }

    @Inject(method = "doCreateBiomes", at = @At("TAIL"))
    private void mccuda$verifyBiomes(Blender blender, RandomState random, ChunkAccess chunk, CallbackInfo ci) {
        GpuDump dump = GpuDump.get();
        if (dump != null && !dump.serve && dump.covers(chunk.getPos())) dump.verifyBiomes(chunk, getBiomeSource());
    }
}
