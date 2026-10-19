package vectorvault.ollama;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import vectorvault.util.JsonUtils;

/** Small dependency-free Ollama client. Uses the current /api/embed endpoint first. */
public class OllamaClient {
    public final String host;
    public final int port;
    public final HttpClient client;
    public String embedModel;
    public String genModel;
    private final String baseUrl;

    public OllamaClient() {
        this(System.getenv().getOrDefault("OLLAMA_URL", "http://127.0.0.1:11434"));
    }

    public OllamaClient(String host, int port) {
        this("http://" + host + ":" + port);
    }

    public OllamaClient(String baseUrl) {
        String normalized = baseUrl == null || baseUrl.isBlank() ? "http://127.0.0.1:11434" : baseUrl.trim();
        while (normalized.endsWith("/")) normalized = normalized.substring(0, normalized.length() - 1);
        this.baseUrl = normalized;
        this.host = normalized;
        this.port = -1;
        this.embedModel = System.getenv().getOrDefault("OLLAMA_EMBED_MODEL", "nomic-embed-text");
        this.genModel = System.getenv().getOrDefault("OLLAMA_GEN_MODEL", "llama3.2");
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(4)).build();
    }

    String base() { return baseUrl; }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/tags"))
                    .timeout(Duration.ofSeconds(4)).GET().build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (Exception e) { return false; }
    }

    /** Returns model tags currently installed on the configured Ollama endpoint. */
    public List<String> availableModels() {
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/tags"))
                    .timeout(Duration.ofSeconds(5)).GET().build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return new ArrayList<>();
            Set<String> names = new LinkedHashSet<>();
            Matcher matcher = Pattern.compile("\\\"name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").matcher(res.body());
            while (matcher.find()) names.add(matcher.group(1));
            return new ArrayList<>(names);
        } catch (Exception e) { return new ArrayList<>(); }
    }

    public boolean modelAvailable(String model) {
        if (model == null || model.isBlank()) return false;
        String wanted = model.trim();
        for (String installed : availableModels()) {
            if (installed.equals(wanted) || installed.equals(wanted + ":latest")) return true;
        }
        return false;
    }

    public List<Float> embed(String text) {
        if (text == null || text.isBlank()) return new ArrayList<>();
        // Current Ollama API: input accepts one string and returns embeddings: [[...]].
        String currentBody = "{\"model\":" + JsonUtils.jS(embedModel)
                + ",\"input\":" + JsonUtils.jS(text) + ",\"truncate\":true}";
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/embed"))
                    .timeout(Duration.ofSeconds(60)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(currentBody, StandardCharsets.UTF_8)).build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() == 200) {
                List<Float> v = extractFirstEmbedding(res.body());
                if (!v.isEmpty()) return v;
            }
        } catch (Exception ignored) { }

        // Compatibility with older Ollama installations, where /api/embeddings
        // returned a single array in the "embedding" property.
        String legacyBody = "{\"model\":" + JsonUtils.jS(embedModel) + ",\"prompt\":" + JsonUtils.jS(text) + "}";
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(60)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(legacyBody, StandardCharsets.UTF_8)).build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return new ArrayList<>();
            return JsonUtils.extractArray(res.body(), "embedding");
        } catch (Exception e) { return new ArrayList<>(); }
    }


    /** Parses the first numeric vector from Ollama's nested {"embeddings":[[...]]} response. */
    private static List<Float> extractFirstEmbedding(String json) {
        if (json == null) return new ArrayList<>();
        int key = json.indexOf("\"embeddings\"");
        if (key < 0) return new ArrayList<>();
        int outer = json.indexOf('[', key);
        if (outer < 0) return new ArrayList<>();
        int vectorStart = json.indexOf('[', outer + 1);
        if (vectorStart < 0) return new ArrayList<>();
        int vectorEnd = json.indexOf(']', vectorStart + 1);
        if (vectorEnd < 0) return new ArrayList<>();
        return JsonUtils.parseVec(json.substring(vectorStart + 1, vectorEnd));
    }

    /**
     * Embeds multiple chunks in one request when supported by the configured
     * Ollama endpoint. Older installations gracefully fall back to single-text
     * embedding requests; an incomplete batch is never returned to the caller.
     */
    public List<List<Float>> embedBatch(List<String> texts) {
        if (texts == null || texts.isEmpty()) return new ArrayList<>();
        for (String text : texts) {
            if (text == null || text.isBlank()) throw new IllegalArgumentException("Embedding inputs cannot be blank");
        }

        StringBuilder inputs = new StringBuilder("[");
        for (int i = 0; i < texts.size(); i++) {
            if (i > 0) inputs.append(',');
            inputs.append(JsonUtils.jS(texts.get(i)));
        }
        inputs.append(']');
        String body = "{\"model\":" + JsonUtils.jS(embedModel)
                + ",\"input\":" + inputs + ",\"truncate\":true}";
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/embed"))
                    .timeout(Duration.ofSeconds(180)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8)).build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() == 200) {
                List<List<Float>> vectors = extractEmbeddingBatch(res.body());
                if (vectors.size() == texts.size() && validEmbeddingBatch(vectors)) return vectors;
            }
        } catch (Exception ignored) { }

        // Compatibility path: the single-input method itself tries /api/embed
        // first and then Ollama's older /api/embeddings endpoint.
        List<List<Float>> fallback = new ArrayList<>();
        int dimensions = -1;
        for (String text : texts) {
            List<Float> vector = embed(text);
            if (vector.isEmpty() || (dimensions >= 0 && vector.size() != dimensions)) return new ArrayList<>();
            dimensions = vector.size();
            fallback.add(vector);
        }
        return fallback;
    }

    private static boolean validEmbeddingBatch(List<List<Float>> vectors) {
        int dimensions = -1;
        for (List<Float> vector : vectors) {
            if (vector == null || vector.isEmpty()) return false;
            if (dimensions >= 0 && vector.size() != dimensions) return false;
            if (vector.stream().anyMatch(v -> v == null || !Float.isFinite(v))) return false;
            dimensions = vector.size();
        }
        return true;
    }

    /** Parses {"embeddings":[[...],[...]]} without requiring a third-party JSON library. */
    private static List<List<Float>> extractEmbeddingBatch(String json) {
        List<List<Float>> result = new ArrayList<>();
        if (json == null) return result;
        int key = json.indexOf("\"embeddings\"");
        if (key < 0) return result;
        int outer = json.indexOf('[', key);
        if (outer < 0) return result;
        int pos = outer + 1;
        while (pos < json.length()) {
            while (pos < json.length() && (Character.isWhitespace(json.charAt(pos)) || json.charAt(pos) == ',')) pos++;
            if (pos >= json.length() || json.charAt(pos) == ']') break;
            if (json.charAt(pos) != '[') return new ArrayList<>();
            int end = json.indexOf(']', pos + 1);
            if (end < 0) return new ArrayList<>();
            List<Float> vector = JsonUtils.parseVec(json.substring(pos + 1, end));
            if (vector.isEmpty()) return new ArrayList<>();
            result.add(vector);
            pos = end + 1;
        }
        return result;
    }

    public String generate(String prompt) {
        String body = "{\"model\":" + JsonUtils.jS(genModel) + ",\"prompt\":" + JsonUtils.jS(prompt)
                + ",\"stream\":false,\"options\":{\"temperature\":0.2}}";
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(base() + "/api/generate"))
                    .timeout(Duration.ofSeconds(180)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8)).build();
            HttpResponse<String> res = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return "ERROR: Ollama generation failed (HTTP " + res.statusCode() + "). Check that model '" + genModel + "' is installed.";
            String answer = JsonUtils.extractStr(res.body(), "response");
            return answer.isBlank() ? "ERROR: Ollama returned an empty response." : answer;
        } catch (Exception e) {
            return "ERROR: Could not reach Ollama. Start Ollama and confirm model '" + genModel + "' is installed.";
        }
    }
}
