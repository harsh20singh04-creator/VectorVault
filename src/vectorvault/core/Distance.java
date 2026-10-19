package vectorvault.core;

import java.util.List;
import java.util.Locale;

public final class Distance {
    private Distance() {}

    @FunctionalInterface
    public interface DistFn {
        float distance(List<Float> a, List<Float> b);
    }

    public static float euclidean(List<Float> a, List<Float> b) {
        requireSameDimensions(a, b);
        double sumSquares = 0.0;
        for (int i = 0; i < a.size(); i++) {
            double delta = (double) a.get(i) - b.get(i);
            sumSquares += delta * delta;
        }
        return finiteDistance(Math.sqrt(sumSquares));
    }

    public static float cosine(List<Float> a, List<Float> b) {
        requireSameDimensions(a, b);
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.size(); i++) {
            double x = a.get(i), y = b.get(i);
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if (na < 1e-18 || nb < 1e-18) return 1.0f;
        // Double intermediates avoid overflow for large but finite float inputs.
        double similarity = dot / (Math.sqrt(na) * Math.sqrt(nb));
        similarity = Math.max(-1.0, Math.min(1.0, similarity));
        return (float) (1.0 - similarity);
    }

    public static float manhattan(List<Float> a, List<Float> b) {
        requireSameDimensions(a, b);
        double sum = 0.0;
        for (int i = 0; i < a.size(); i++) sum += Math.abs((double) a.get(i) - b.get(i));
        return finiteDistance(sum);
    }

    public static DistFn getDistFn(String m) {
        String metric = m == null ? "euclidean" : m.trim().toLowerCase(Locale.ROOT);
        return switch (metric) {
            case "cosine" -> Distance::cosine;
            case "manhattan" -> Distance::manhattan;
            case "euclidean" -> Distance::euclidean;
            default -> throw new IllegalArgumentException(
                    "Unknown distance metric: " + m + ". Use cosine, euclidean, or manhattan.");
        };
    }

    private static float finiteDistance(double value) {
        return (float) Math.min(value, Float.MAX_VALUE);
    }

    private static void requireSameDimensions(List<Float> a, List<Float> b) {
        if (a == null || b == null || a.size() != b.size()) {
            throw new IllegalArgumentException("Vectors must be non-null and have matching dimensions");
        }
    }
}
