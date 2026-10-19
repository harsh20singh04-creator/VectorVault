package vectorvault.db;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

import vectorvault.core.Distance;
import vectorvault.core.HitPair;
import vectorvault.core.VectorItem;
import vectorvault.index.BruteForce;
import vectorvault.index.HNSW;
import vectorvault.index.KDTree;
import vectorvault.util.JsonUtils;

/** Persistent chunk store and vector indexes used by the real-embedding RAG pipeline. */
public class DocumentDB {
    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW hnswCosine = new HNSW(16, 200);
    private final HNSW hnswEuclidean = new HNSW(16, 200);
    private final HNSW hnswManhattan = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private KDTree kdt;
    private final ReentrantLock mu = new ReentrantLock();

    private int nextId = 1;
    private int dims = 0;
    private static final Path DATA_FILE = Path.of("data", "documents.json");

    public DocumentDB() { load(); }

    public int insert(String title, String text, List<Float> emb) {
        return insertBatch(List.of(title), List.of(text), List.of(emb)).get(0);
    }

    /** Inserts a document's fully embedded chunks in one lock/persistence transaction. */
    public List<Integer> insertBatch(List<String> titles, List<String> texts, List<List<Float>> embeddings) {
        if (titles == null || texts == null || embeddings == null
                || titles.size() != texts.size() || titles.size() != embeddings.size()) {
            throw new IllegalArgumentException("Titles, texts, and embeddings must have matching lengths");
        }
        if (titles.isEmpty()) return new ArrayList<>();

        mu.lock();
        try {
            int proposedDims = dims == 0 ? embeddings.get(0).size() : dims;
            if (proposedDims <= 0) throw new IllegalArgumentException("Embedding dimensions must be positive");
            for (int i = 0; i < embeddings.size(); i++) {
                String title = titles.get(i);
                String text = texts.get(i);
                List<Float> emb = embeddings.get(i);
                if (title == null || title.isBlank() || text == null || text.isBlank()) {
                    throw new IllegalArgumentException("Chunk titles and text cannot be empty");
                }
                if (emb == null || emb.size() != proposedDims
                        || emb.stream().anyMatch(v -> v == null || !Float.isFinite(v))) {
                    throw new IllegalArgumentException("All embeddings must contain exactly " + proposedDims + " finite values");
                }
            }

            int previousNextId = nextId;
            int previousDims = dims;
            List<Integer> ids = new ArrayList<>();
            try {
                dims = proposedDims;
                if (kdt == null) kdt = new KDTree(dims);
                for (int i = 0; i < titles.size(); i++) {
                    int id = nextId++;
                    DocItem item = new DocItem(id, titles.get(i), texts.get(i), embeddings.get(i));
                    store.put(id, item);
                    ids.add(id);
                    VectorItem vi = new VectorItem(id, item.title, "doc", item.emb);
                    bf.insert(vi);
                    kdt.insert(vi);
                    hnswCosine.insert(vi, Distance::cosine);
                    hnswEuclidean.insert(vi, Distance::euclidean);
                    hnswManhattan.insert(vi, Distance::manhattan);
                }
                if (store.size() % 128 < titles.size()) kdt.rebuild(toVectorItems());
                save();
                return ids;
            } catch (RuntimeException failure) {
                for (int id : ids) {
                    store.remove(id);
                    bf.remove(id);
                }
                hnswCosine.rebuild(toVectorItems(), Distance::cosine);
                hnswEuclidean.rebuild(toVectorItems(), Distance::euclidean);
                hnswManhattan.rebuild(toVectorItems(), Distance::manhattan);
                nextId = previousNextId;
                dims = previousDims;
                if (dims == 0) {
                    kdt = null;
                } else {
                    kdt = new KDTree(dims);
                    kdt.rebuild(toVectorItems());
                }
                throw failure;
            }
        } finally {
            mu.unlock();
        }
    }

    /**
     * Searches document chunks with a chosen index/metric. The no-filter default
     * deliberately has no arbitrary cosine-distance cutoff; top-k always means
     * the nearest k chunks, including when their distances are above 0.7.
     */
    public List<Map.Entry<Float, DocItem>> search(List<Float> q, int k, float maxDist,
                                                    String metric, String algorithm) {
        mu.lock();
        try {
            if (store.isEmpty() || k <= 0) return new ArrayList<>();
            if (q == null || dims == 0 || q.size() != dims
                    || q.stream().anyMatch(v -> v == null || !Float.isFinite(v))) {
                throw new IllegalArgumentException("Query embedding must contain exactly " + dims + " finite values");
            }
            if (k > 1000) throw new IllegalArgumentException("k must not exceed 1000");
            String normalizedMetric = normalizeMetric(metric);
            String normalizedAlgo = algorithm == null ? "hnsw" : algorithm.trim().toLowerCase(Locale.ROOT);
            Distance.DistFn fn = Distance.getDistFn(normalizedMetric);
            boolean applyThreshold = Float.isFinite(maxDist);
            int candidateK = applyThreshold ? store.size() : Math.min(k, store.size());
            List<HitPair> raw;

            switch (normalizedAlgo) {
                case "bruteforce" -> raw = bf.knn(q, candidateK, fn);
                case "kdtree" -> {
                    // KD pruning has a valid coordinate lower-bound for L1/L2, but not cosine.
                    raw = "cosine".equals(normalizedMetric)
                            ? bf.knn(q, candidateK, fn)
                            : kdt.knn(q, candidateK, fn);
                }
                case "hnsw" -> raw = hnswForMetric(normalizedMetric)
                        .knn(q, candidateK, Math.max(50, candidateK * 4), fn);
                default -> throw new IllegalArgumentException("Unknown algorithm: " + algorithm);
            }

            List<Map.Entry<Float, DocItem>> out = new ArrayList<>();
            for (HitPair hit : raw) {
                DocItem item = store.get(hit.id);
                if (item != null && hit.dist <= maxDist) {
                    out.add(new AbstractMap.SimpleEntry<>(hit.dist, item));
                    if (out.size() >= k) break;
                }
            }
            return out;
        } finally {
            mu.unlock();
        }
    }

    public List<Map.Entry<Float, DocItem>> search(List<Float> q, int k, float maxDist) {
        return search(q, k, maxDist, "cosine", "hnsw");
    }

    public List<Map.Entry<Float, DocItem>> search(List<Float> q, int k) {
        return search(q, k, Float.POSITIVE_INFINITY, "cosine", "hnsw");
    }

    public boolean remove(int id) {
        mu.lock();
        try {
            DocItem removed = store.remove(id);
            if (removed == null) return false;
            bf.remove(id);
            hnswCosine.remove(id);
            hnswEuclidean.remove(id);
            hnswManhattan.remove(id);
            if (kdt != null) kdt.rebuild(toVectorItems());
            try {
                save();
                return true;
            } catch (RuntimeException failure) {
                store.put(id, removed);
                VectorItem vi = new VectorItem(id, removed.title, "doc", removed.emb);
                bf.insert(vi);
                hnswCosine.rebuild(toVectorItems(), Distance::cosine);
                hnswEuclidean.rebuild(toVectorItems(), Distance::euclidean);
                hnswManhattan.rebuild(toVectorItems(), Distance::manhattan);
                if (kdt == null) kdt = new KDTree(dims);
                kdt.rebuild(toVectorItems());
                throw failure;
            }
        } finally {
            mu.unlock();
        }
    }

    public List<DocItem> all() {
        mu.lock();
        try { return new ArrayList<>(store.values()); }
        finally { mu.unlock(); }
    }

    public int size() {
        mu.lock();
        try { return store.size(); }
        finally { mu.unlock(); }
    }

    public int getDims() {
        mu.lock();
        try { return dims; }
        finally { mu.unlock(); }
    }

    private HNSW hnswForMetric(String metric) {
        return switch (metric) {
            case "cosine" -> hnswCosine;
            case "euclidean" -> hnswEuclidean;
            case "manhattan" -> hnswManhattan;
            default -> throw new IllegalArgumentException("Unsupported distance metric: " + metric);
        };
    }

    private static String normalizeMetric(String metric) {
        String normalized = metric == null || metric.isBlank() ? "cosine" : metric.trim().toLowerCase(Locale.ROOT);
        Distance.getDistFn(normalized); // Validate instead of silently substituting Euclidean.
        return normalized;
    }

    private List<VectorItem> toVectorItems() {
        List<VectorItem> values = new ArrayList<>();
        for (DocItem d : store.values()) values.add(new VectorItem(d.id, d.title, "doc", d.emb));
        return values;
    }

    private void save() {
        try {
            Files.createDirectories(DATA_FILE.getParent());
            StringBuilder json = new StringBuilder("{\"nextId\":").append(nextId)
                    .append(",\"dims\":").append(dims).append(",\"items\":[\n");
            int count = 0;
            for (DocItem item : store.values()) {
                if (count++ > 0) json.append(",\n");
                json.append("  {\"id\":").append(item.id)
                    .append(",\"title\":").append(JsonUtils.jS(item.title))
                    .append(",\"text\":").append(JsonUtils.jS(item.text))
                    .append(",\"embedding\":").append(JsonUtils.jVec(item.emb)).append("}");
            }
            json.append("\n]}\n");
            Path temp = DATA_FILE.resolveSibling(DATA_FILE.getFileName() + ".tmp");
            Files.writeString(temp, json.toString());
            try {
                Files.move(temp, DATA_FILE, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temp, DATA_FILE, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Could not persist document index: " + e.getMessage(), e);
        }
    }

    private void load() {
        if (!Files.exists(DATA_FILE)) return;
        try {
            String json = Files.readString(DATA_FILE).trim();
            if (json.isEmpty()) return;
            List<String> objects;
            if (json.startsWith("[")) {
                // Backward-compatible reader for the original array-only format.
                if (!json.endsWith("]")) throw new IOException("Invalid legacy document JSON");
                objects = splitObjects(json.substring(1, json.length() - 1));
            } else if (json.startsWith("{")) {
                if (!json.endsWith("}")) throw new IOException("Invalid document JSON");
                nextId = Math.max(1, JsonUtils.extractInt(json, "nextId", 1));
                dims = Math.max(0, JsonUtils.extractInt(json, "dims", 0));
                int key = json.indexOf("\"items\"");
                int open = key < 0 ? -1 : json.indexOf('[', key);
                int close = json.lastIndexOf(']');
                if (open < 0 || close < open) throw new IOException("Missing items array in document JSON");
                objects = splitObjects(json.substring(open + 1, close));
                // Empty persisted indexes still retain their dimension/model contract.
                if (dims > 0) kdt = new KDTree(dims);
            } else {
                throw new IOException("Invalid document JSON");
            }

            for (String obj : objects) {
                int id = JsonUtils.extractInt(obj, "id", -1);
                String title = JsonUtils.extractStr(obj, "title");
                String text = JsonUtils.extractStr(obj, "text");
                List<Float> emb = JsonUtils.extractArray(obj, "embedding");
                if (id < 1 || title.isBlank() || text.isBlank() || emb.isEmpty()
                        || emb.stream().anyMatch(v -> !Float.isFinite(v))) continue;
                if (dims == 0) {
                    dims = emb.size();
                    kdt = new KDTree(dims);
                }
                if (emb.size() != dims) {
                    System.err.println("Skipping document chunk with inconsistent embedding dimensions (id=" + id + ")");
                    continue;
                }
                DocItem item = new DocItem(id, title, text, emb);
                store.put(id, item);
                VectorItem vi = new VectorItem(id, title, "doc", emb);
                bf.insert(vi);
                kdt.insert(vi);
                hnswCosine.insert(vi, Distance::cosine);
                hnswEuclidean.insert(vi, Distance::euclidean);
                hnswManhattan.insert(vi, Distance::manhattan);
                nextId = Math.max(nextId, id + 1);
            }
            if (kdt != null) kdt.rebuild(toVectorItems());
            if (!store.isEmpty()) System.out.println("DocumentDB loaded: " + store.size() + " chunks · " + dims + " dimensions");
        } catch (Exception e) {
            System.err.println("Failed to load DocumentDB: " + e.getMessage());
        }
    }

    private static List<String> splitObjects(String json) {
        List<String> result = new ArrayList<>();
        int depth = 0, start = -1;
        boolean inString = false, escaped = false;
        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (escaped) { escaped = false; continue; }
            if (c == '\\' && inString) { escaped = true; continue; }
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;
            if (c == '{') { if (depth == 0) start = i; depth++; }
            else if (c == '}') {
                depth--;
                if (depth == 0 && start >= 0) { result.add(json.substring(start, i + 1)); start = -1; }
            }
        }
        return result;
    }
}
