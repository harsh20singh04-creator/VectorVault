package vectorvault.util;

import com.sun.net.httpserver.HttpExchange;
import java.io.*;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public final class HttpUtils {
    private static final int MAX_BODY_BYTES = 8 * 1024 * 1024;
    private HttpUtils() {}

    public static void cors(HttpExchange ex) {
        String origin = ex.getRequestHeaders().getFirst("Origin");
        String configured = System.getenv().getOrDefault("VECTORVAULT_CORS_ORIGIN", "").trim();
        if (origin != null && !configured.isEmpty()) {
            boolean allowed = Arrays.stream(configured.split(","))
                    .map(String::trim)
                    .anyMatch(value -> value.equals("*") || value.equals(origin));
            if (allowed) {
                ex.getResponseHeaders().set("Access-Control-Allow-Origin", configured.equals("*") ? "*" : origin);
                ex.getResponseHeaders().set("Vary", "Origin");
                ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
                ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type, Authorization");
            }
        }
    }

    public static String body(HttpExchange ex) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int total = 0;
        try (InputStream in = ex.getRequestBody()) {
            int n;
            while ((n = in.read(buffer)) != -1) {
                total += n;
                if (total > MAX_BODY_BYTES) throw new BodyTooLargeException("Request body exceeds the 8 MB limit");
                out.write(buffer, 0, n);
            }
        }
        return out.toString(StandardCharsets.UTF_8);
    }

    public static void send(HttpExchange ex, int status, String content, String type) throws IOException {
        cors(ex);
        ex.getResponseHeaders().set("Content-Type", type + "; charset=utf-8");
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.getResponseHeaders().set("Referrer-Policy", "no-referrer");
        if (type.contains("json")) ex.getResponseHeaders().set("Cache-Control", "no-store");
        byte[] bytes = content.getBytes(StandardCharsets.UTF_8);
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) { os.write(bytes); }
    }

    public static void json(HttpExchange ex, String value) throws IOException {
        send(ex, 200, value, "application/json");
    }

    public static void error(HttpExchange ex, String message) throws IOException {
        error(ex, 400, message);
    }

    public static void error(HttpExchange ex, int status, String message) throws IOException {
        send(ex, status, "{\"error\":" + JsonUtils.jS(message) + "}", "application/json");
    }

    public static Map<String, String> queryParams(URI uri) {
        Map<String, String> map = new HashMap<>();
        String q = uri.getRawQuery();
        if (q == null || q.isEmpty()) return map;
        for (String part : q.split("&")) {
            String[] kv = part.split("=", 2);
            try {
                String k = URLDecoder.decode(kv[0], StandardCharsets.UTF_8);
                String v = kv.length > 1 ? URLDecoder.decode(kv[1], StandardCharsets.UTF_8) : "";
                map.put(k, v);
            } catch (IllegalArgumentException ignored) { }
        }
        return map;
    }

    public static int pathId(String path, String prefix) {
        try { return Integer.parseInt(path.substring(prefix.length())); }
        catch (Exception e) { return -1; }
    }

    public static final class BodyTooLargeException extends IOException {
        public BodyTooLargeException(String message) { super(message); }
    }
}
