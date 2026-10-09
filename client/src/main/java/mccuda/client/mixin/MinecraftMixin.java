package mccuda.client.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import mccuda.client.Recorder;
import net.minecraft.client.Minecraft;

/** Drives the recorder once per client tick. */
@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
    @Inject(method = "tick", at = @At("TAIL"))
    private void mccuda$tick(CallbackInfo ci) {
        Recorder.tick((Minecraft) (Object) this);
    }
}
