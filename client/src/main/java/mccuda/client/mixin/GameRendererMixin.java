package mccuda.client.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import mccuda.client.Recorder;
import net.minecraft.client.renderer.GameRenderer;

/** Captures each recorded frame once the world is drawn, before the GUI layer (toasts, chat) goes on top. */
@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {
    @Inject(method = "render", at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/GameRenderer;applyPostEffects()V", shift = At.Shift.AFTER))
    private void mccuda$capture(CallbackInfo ci) {
        Recorder.onWorldRendered((GameRenderer) (Object) this);
    }
}
