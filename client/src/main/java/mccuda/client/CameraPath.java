package mccuda.client;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The flythrough camera path (data/videos/flythrough_path.txt), sampled exactly like
 * viz/fly.cu: Catmull-Rom through the keyframes, height either above the ground ('g')
 * or absolute ('a'), blended across a change of mode, and the same smoothing of height
 * and heading per video frame.
 */
public final class CameraPath {
    record Key(double t, double x, double z, char mode, double y, double pitch) {}

    record Sample(double x, double z, double y, double pitch, char mode, double ya, double yb, double s, char ma, char mb) {}

    private final List<Key> keys = new ArrayList<>();
    private double camY, yaw;
    private boolean first = true;

    public CameraPath(Path file) throws IOException {
        for (String line : Files.readAllLines(file)) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            String[] p = line.split("\\s+");
            keys.add(new Key(Double.parseDouble(p[0]), Double.parseDouble(p[1]), Double.parseDouble(p[2]), p[3].charAt(0), Double.parseDouble(p[4]),
                Double.parseDouble(p[5])));
        }
        if (keys.size() < 2) throw new IOException("path needs at least 2 keyframes");
    }

    public double duration() {
        return keys.get(keys.size() - 1).t();
    }

    public void reset() {
        first = true;
    }

    private static double catmull(double p0, double p1, double p2, double p3, double s) {
        return 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s * s * s);
    }

    Sample sample(double t) {
        int i = 0;
        while (i + 2 < keys.size() && keys.get(i + 1).t() <= t) i++;
        Key a = keys.get(i), b = keys.get(Math.min(i + 1, keys.size() - 1));
        Key p = keys.get(i == 0 ? 0 : i - 1), q = keys.get(Math.min(i + 2, keys.size() - 1));
        double s = b.t() > a.t() ? Math.min(1.0, Math.max(0.0, (t - a.t()) / (b.t() - a.t()))) : 0.0;
        double x = catmull(p.x(), a.x(), b.x(), q.x(), s), z = catmull(p.z(), a.z(), b.z(), q.z(), s);
        double pitch = a.pitch() + (b.pitch() - a.pitch()) * (s * s * (3 - 2 * s));
        if (a.mode() == b.mode()) {
            double y = catmull(p.mode() == a.mode() ? p.y() : a.y(), a.y(), b.y(), q.mode() == b.mode() ? q.y() : b.y(), s);
            return new Sample(x, z, y, pitch, a.mode(), 0, 0, 0, a.mode(), a.mode());
        }
        return new Sample(x, z, 0, pitch, 'm', a.y(), b.y(), s * s * (3 - 2 * s), a.mode(), b.mode());
    }

    /** Camera pose at time t; `ground` is the surface height (top block + 1) at the camera; `dt` the frame step. */
    public double[] pose(double t, java.util.function.DoubleBinaryOperator ground, double dt) {
        Sample ps = sample(t);
        double gy = ground.applyAsDouble(ps.x(), ps.z());
        double target = ps.mode() == 'm' ? resolve(ps.ma(), ps.ya(), gy) * (1 - ps.s()) + resolve(ps.mb(), ps.yb(), gy) * ps.s() : resolve(ps.mode(), ps.y(), gy);
        if (first) camY = target;
        else camY += (target - camY) * Math.min(1.0, 6.0 * dt);
        camY = Math.max(camY, gy + 1.62);
        Sample ahead = sample(Math.min(duration(), t + 0.35)), behind = sample(Math.max(0.0, t - 0.05));
        double vx = ahead.x() - behind.x(), vz = ahead.z() - behind.z();
        if (vx * vx + vz * vz > 1e-6) {
            double ty = Math.atan2(vx, vz);
            double d = ty - yaw;
            while (d > Math.PI) d -= 2 * Math.PI;
            while (d < -Math.PI) d += 2 * Math.PI;
            yaw = first ? ty : yaw + d * Math.min(1.0, 4.0 * dt);
        }
        first = false;
        // Minecraft: yRot 0 faces +z and grows clockwise (forward = (-sin, cos)); xRot is positive looking down.
        return new double[] {ps.x(), camY, ps.z(), -Math.toDegrees(yaw), -ps.pitch(), ps.mode() == 'g' ? 1 : 0};
    }

    private static double resolve(char mode, double y, double gy) {
        return mode == 'g' ? gy + y : y;
    }
}
