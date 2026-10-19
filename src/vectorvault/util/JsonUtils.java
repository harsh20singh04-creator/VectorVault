package vectorvault.util;

import java.util.*;

public final class JsonUtils {
    private JsonUtils() {}

    public static String jS(String s) {
        if (s == null) return "null";
        StringBuilder o = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> o.append("\\\"");
                case '\\' -> o.append("\\\\");
                case '\n' -> o.append("\\n");
                case '\r' -> o.append("\\r");
                case '\t' -> o.append("\\t");
                case '\b' -> o.append("\\b");
                case '\f' -> o.append("\\f");
                default -> {
                    if (c < 0x20) {
                        o.append("\\u");
                        String hex = Integer.toHexString(c);
                        for (int i = hex.length(); i < 4; i++) o.append('0');
                        o.append(hex);
                    } else o.append(c);
                }
            }
        }
        return o.append('"').toString();
    }

    public static String jVec(List<Float> v) {
    StringBuilder sb = new StringBuilder("[");

    for (int i = 0; i < v.size(); i++) {
        if (i > 0) sb.append(',');

        sb.append(Float.toString(v.get(i)));
    }

    return sb.append(']').toString();
}

    public static List<Float> parseVec(String s) {
        List<Float> v = new ArrayList<>();
        if (s == null || s.trim().isEmpty()) return v;
        for (String t : s.split(",", -1)) {
            try {
                float value = Float.parseFloat(t.trim());
                if (!Float.isFinite(value)) return new ArrayList<>();
                v.add(value);
            } catch (Exception ignored) {
                // A malformed component invalidates the vector; do not silently
                // drop it and accidentally accept a different-dimensional input.
                return new ArrayList<>();
            }
        }
        return v;
    }

    public static String extractStr(String body, String key) {
        if (body == null) return "";
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return "";
        p = body.indexOf(':', p);
        if (p < 0) return "";
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                c = body.charAt(p);
                switch (c) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    case 'b' -> result.append('\b');
                    case 'f' -> result.append('\f');
                    case 'u' -> {
                        if (p + 4 < body.length()) {
                            try {
                                result.append((char) Integer.parseInt(body.substring(p + 1, p + 5), 16));
                                p += 4;
                            } catch (Exception e) { result.append('u'); }
                        } else result.append('u');
                    }
                    default -> result.append(c);
                }
            } else result.append(c);
            p++;
        }
        return result.toString();
    }

    public static int extractInt(String body, String key, int def) {
        try {
            int p = body.indexOf('"' + key + '"');
            if (p < 0) return def;
            p = body.indexOf(':', p);
            if (p < 0) return def;
            p++;
            while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
            int e = p;
            while (e < body.length() && (body.charAt(e) == '-' || Character.isDigit(body.charAt(e)))) e++;
            return Integer.parseInt(body.substring(p, e));
        } catch (Exception e) { return def; }
    }

    public static List<Float> extractArray(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p < 0) return new ArrayList<>();
        p = body.indexOf('[', p);
        if (p < 0) return new ArrayList<>();
        int e = body.indexOf(']', p);
        if (e < 0) return new ArrayList<>();
        return parseVec(body.substring(p + 1, e));
    }

    public static boolean parseBody(String body, StringHolder meta, StringHolder cat, ListHolder emb) {
        meta.value = extractStr(body, "metadata");
        cat.value = extractStr(body, "category");
        emb.value = extractArray(body, "embedding");
        return !meta.value.isEmpty() && !emb.value.isEmpty();
    }

    public static class StringHolder { public String value = ""; }
    public static class ListHolder { public List<Float> value = new ArrayList<>(); }
}
