package mccuda.client.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import mccuda.client.Recorder;
import net.minecraft.client.Camera;

/** While recording, the camera follows the scripted path instead of the player (before the view matrix and frustum are built). */
@Mixin(Camera.class)
public abstract class CameraMixin {
    @Shadow
    protected abstract void setPosition(double x, double y, double z);

    @Shadow
    protected abstract void setRotation(float yRot, float xRot);

    @Inject(method = "alignWithEntity", at = @At("TAIL"))
    private void mccuda$pose(float partialTicks, CallbackInfo ci) {
        double[] p = Recorder.cameraPose();
        if (p == null) return;
        setPosition(p[0], p[1], p[2]);
        setRotation((float) p[3], (float) p[4]);
    }
}
