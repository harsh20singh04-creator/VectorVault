package vectorvault.db;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import vectorvault.core.*;
import vectorvault.index.*;
import vectorvault.util.JsonUtils;

public class VectorDB {

    private final int dims;

    private final Map<Integer, VectorItem> store =
            new LinkedHashMap<>();

    private final BruteForce bf;
    private final KDTree kdt;
    // An HNSW graph is metric-specific. Keep one graph per supported metric so
    // Manhattan/Euclidean searches do not traverse a graph built for cosine.
    private final HNSW hnswCosine;
    private final HNSW hnswEuclidean;
    private final HNSW hnswManhattan;

    private final ReentrantReadWriteLock mu =
            new ReentrantReadWriteLock();

    private int nextId = 1;

    private static final String DATA_FILE =
            "data/vectors.json";

    // =========================
    // Constructor
    // =========================

    public VectorDB(int dims) {

        if (dims <= 0) throw new IllegalArgumentException("Dimensions must be positive");

        this.dims = dims;

        bf = new BruteForce();
        kdt = new KDTree(dims);
        hnswCosine = new HNSW(16, 200);
        hnswEuclidean = new HNSW(16, 200);
        hnswManhattan = new HNSW(16, 200);

        load();
    }

    // =========================
    // Insert
    // =========================

    public int insert(
            String metadata,
            String category,
            List<Float> emb,
            Distance.DistFn fn
    ) {

        if (metadata == null || metadata.isBlank()) {
            throw new IllegalArgumentException("Metadata cannot be empty");
        }
        if (category == null || category.isBlank()) category = "general";
        if (emb == null || emb.size() != dims || emb.stream().anyMatch(v -> v == null || !Float.isFinite(v))) {
            throw new IllegalArgumentException(
                    "Embedding must contain exactly " + dims + " finite numeric values"
            );
        }

        mu.writeLock().lock();

        try {

            int id = nextId++;
            VectorItem item = new VectorItem(id, metadata, category, emb);
            try {
                store.put(id, item);
                bf.insert(item);
                kdt.insert(item);
                hnswCosine.insert(item, Distance::cosine);
                hnswEuclidean.insert(item, Distance::euclidean);
                hnswManhattan.insert(item, Distance::manhattan);

                // Keep the incremental tree from growing into a long chain after
                // repeated ordered inserts, without rebuilding it on every write.
                if (store.size() % 128 == 0) kdt.rebuild(store.values());
                save();
                return id;
            } catch (RuntimeException failure) {
                store.remove(id);
                bf.remove(id);
                kdt.rebuild(store.values());
                hnswCosine.rebuild(store.values(), Distance::cosine);
                hnswEuclidean.rebuild(store.values(), Distance::euclidean);
                hnswManhattan.rebuild(store.values(), Distance::manhattan);
                nextId = id;
                throw failure;
            }

        } finally {
            mu.writeLock().unlock();
        }
    }

    // =========================
    // Remove
    // =========================

    public boolean remove(int id) {

        mu.writeLock().lock();

        try {

            VectorItem removed =
                    store.remove(id);

            if (removed == null) {
                return false;
            }

            bf.remove(id);
            hnswCosine.remove(id);
            hnswEuclidean.remove(id);
            hnswManhattan.remove(id);
            kdt.rebuild(store.values());

            try {
                save();
                return true;
            } catch (RuntimeException failure) {
                // A failed durable write must not report a successful delete or
                // leave the in-memory indexes different from the persisted state.
                store.put(id, removed);
                bf.insert(removed);
                kdt.rebuild(store.values());
                hnswCosine.rebuild(store.values(), Distance::cosine);
                hnswEuclidean.rebuild(store.values(), Distance::euclidean);
                hnswManhattan.rebuild(store.values(), Distance::manhattan);
                throw failure;
            }

        } finally {
            mu.writeLock().unlock();
        }
    }

    // =========================
    // Normal Search
    // =========================

    public SearchOut search(
            List<Float> q,
            int k,
            String metric,
            String algo
    ) {

        return search(
                q,
                k,
                metric,
                algo,
                ""
        );
    }

    // =========================
    // Category Filter Search
    // =========================

    public SearchOut search(
            List<Float> q,
            int k,
            String metric,
            String algo,
            String category
    ) {

        mu.readLock().lock();

        try {

            if (q == null || q.size() != dims || q.stream().anyMatch(v -> v == null || !Float.isFinite(v))) {
                throw new IllegalArgumentException(
                        "Query vector must contain exactly " + dims + " finite numeric values"
                );
            }
            if (k < 1 || k > 1000) throw new IllegalArgumentException("k must be between 1 and 1000");

            Distance.DistFn dfn =
                    Distance.getDistFn(metric);

            String normalizedMetric = normalizeMetric(metric);
            String normalizedAlgo = algo == null ? "" : algo.trim().toLowerCase(Locale.ROOT);
            boolean categoryFilter = category != null && !category.isBlank();
            // A category filter is applied after distance ordering, so collect
            // all candidates to avoid returning fewer than k matching items.
            int candidateK = Math.min(store.size(), categoryFilter ? store.size() : k);

            long t0 =
                    System.nanoTime();

            List<HitPair> raw;
            boolean exact = true;
            String note = "";

            // =========================
            // Brute Force
            // =========================

            if ("bruteforce".equals(normalizedAlgo)) {

                raw =
                        bf.knn(
                                q,
                                candidateK,
                                dfn
                        );

            }

            // =========================
            // KD Tree
            // =========================

            else if ("kdtree".equals(normalizedAlgo)) {
                if ("cosine".equals(normalizedMetric)) {
                    // Axis-distance pruning is not a valid lower bound for
                    // cosine distance. Use an exact scan rather than returning
                    // silently incorrect results.
                    raw = bf.knn(q, candidateK, dfn);
                    note = "Cosine distance is not axis-prunable; exact Brute Force fallback used.";
                } else {
                    raw = kdt.knn(q, candidateK, dfn);
                }

            }

            // =========================
            // HNSW
            // =========================

            else if ("hnsw".equals(normalizedAlgo)) {

                raw =
                        hnswForMetric(normalizedMetric).knn(
                                q,
                                candidateK,
                                Math.max(50, candidateK * 4),
                                dfn
                        );
                // HNSW is an approximate nearest-neighbor algorithm by contract,
                // even when a small dataset happens to produce exact top-k results.
                exact = false;
                note = "HNSW is approximate by design; compare recall@k against Brute Force for your workload.";

            }

            else {

                throw new IllegalArgumentException(
                        "Unknown algorithm: " + algo
                );
            }

            SearchOut out =
                    new SearchOut();

            out.us =
                    (System.nanoTime()
                            - t0) / 1000;

            out.algo = normalizedAlgo;
            out.metric = normalizedMetric;
            out.exact = exact;
            out.note = note;

            // =========================
            // Category Filtering
            // =========================

            for (HitPair hp : raw) {

                VectorItem v =
                        store.get(hp.id);

                if (v == null) {
                    continue;
                }

                /*
                 * Empty category = no filter.
                 *
                 * Otherwise only matching
                 * categories are returned.
                 */

                if (categoryFilter
                        && !category.equalsIgnoreCase(
                                v.category
                        )) {

                    continue;
                }

                out.hits.add(
                        new Hit(
                                v.id,
                                v.metadata,
                                v.category,
                                v.emb,
                                hp.dist
                        )
                );

                if (out.hits.size() >= Math.min(k, store.size())) {
                    break;
                }
            }

            return out;

        } finally {
            mu.readLock().unlock();
        }
    }

    // =========================
    // Benchmark
    // =========================

    public BenchOut benchmark(
            List<Float> q,
            int k,
            String metric
    ) {

        mu.readLock().lock();

        try {

            if (q == null || q.size() != dims || q.stream().anyMatch(v -> v == null || !Float.isFinite(v))) {
                throw new IllegalArgumentException("Query vector must contain exactly " + dims + " finite numeric values");
            }
            if (k < 1 || k > 1000) throw new IllegalArgumentException("k must be between 1 and 1000");

            Distance.DistFn dfn =
                    Distance.getDistFn(metric);
            String normalizedMetric = normalizeMetric(metric);

            int limit =
                    Math.min(
                            k,
                            store.size()
                    );

            long t0 =
                    System.nanoTime();

            List<HitPair> exactHits = bf.knn(
                    q,
                    limit,
                    dfn
            );

            long bfUs =
                    (System.nanoTime()
                            - t0) / 1000;

            t0 =
                    System.nanoTime();

            List<HitPair> kdHits = "cosine".equals(normalizedMetric)
                    ? bf.knn(q, limit, dfn)
                    : kdt.knn(q, limit, dfn);

            long kdUs =
                    (System.nanoTime()
                            - t0) / 1000;

            t0 =
                    System.nanoTime();

            List<HitPair> hnswHits = hnswForMetric(normalizedMetric).knn(
                    q,
                    limit,
                    Math.max(50, limit * 4),
                    dfn
            );

            long hnswUs =
                    (System.nanoTime()
                            - t0) / 1000;

            Set<Integer> exactIds = new HashSet<>();
            for (HitPair hit : exactHits) exactIds.add(hit.id);
            int matched = 0;
            for (HitPair hit : hnswHits) if (exactIds.contains(hit.id)) matched++;
            double recall = limit == 0 ? 1.0 : (double) matched / limit;

            return new BenchOut(
                    bfUs,
                    kdUs,
                    hnswUs,
                    store.size(),
                    "cosine".equals(normalizedMetric),
                    recall
            );

        } finally {
            mu.readLock().unlock();
        }
    }

    // =========================
    // Get All
    // =========================

    public List<VectorItem> all() {

        mu.readLock().lock();

        try {

            return new ArrayList<>(
                    store.values()
            );

        } finally {
            mu.readLock().unlock();
        }
    }

    // =========================
    // Size
    // =========================

    public int size() {

        mu.readLock().lock();

        try {
            return store.size();
        } finally {
            mu.readLock().unlock();
        }
    }

    // =========================
    // HNSW Info
    // =========================

    public HNSW.GraphInfo hnswInfo() {

        mu.readLock().lock();

        try {

            return hnswCosine.getInfo();

        } finally {
            mu.readLock().unlock();
        }
    }

    private HNSW hnswForMetric(String metric) {
        return switch (normalizeMetric(metric)) {
            case "cosine" -> hnswCosine;
            case "euclidean" -> hnswEuclidean;
            case "manhattan" -> hnswManhattan;
            default -> throw new IllegalArgumentException("Unsupported distance metric: " + metric);
        };
    }

    private static String normalizeMetric(String metric) {
        if (metric == null || metric.isBlank()) return "euclidean";
        String normalized = metric.trim().toLowerCase(Locale.ROOT);
        // getDistFn is the canonical validator.
        Distance.getDistFn(normalized);
        return normalized;
    }

    // =========================
    // Save
    // =========================

    private void save() {

        try {

            Path path =
                    Paths.get(DATA_FILE);

            Files.createDirectories(
                    path.getParent()
            );

            StringBuilder sb =
                    new StringBuilder();

            sb.append("{");

            sb.append("\"nextId\":")
                    .append(nextId)
                    .append(",");

            sb.append("\"dims\":")
                    .append(dims)
                    .append(",");

            sb.append("\"items\":[");

            int i = 0;

            for (VectorItem v :
                    store.values()) {

                if (i++ > 0) {
                    sb.append(",");
                }

                sb.append("{");

                sb.append("\"id\":")
                        .append(v.id)
                        .append(",");

                sb.append("\"metadata\":")
                        .append(
                                JsonUtils.jS(
                                        v.metadata
                                )
                        )
                        .append(",");

                sb.append("\"category\":")
                        .append(
                                JsonUtils.jS(
                                        v.category
                                )
                        )
                        .append(",");

                sb.append("\"embedding\":")
                        .append(
                                JsonUtils.jVec(
                                        v.emb
                                )
                        );

                sb.append("}");
            }

            sb.append("]}");

            Path temp = path.resolveSibling(path.getFileName() + ".tmp");
            Files.writeString(temp, sb.toString());
            try {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (AtomicMoveNotSupportedException ignored) {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
            }

        } catch (Exception e) {
            throw new IllegalStateException("Could not persist vector index: " + e.getMessage(), e);
        }
    }

    // =========================
    // Load
    // =========================

    private void load() {

        Path path =
                Paths.get(DATA_FILE);

        if (!Files.exists(path)) {
            return;
        }

        try {

            String json =
                    Files.readString(path);

            nextId =
                    JsonUtils.extractInt(
                            json,
                            "nextId",
                            1
                    );

            int p =
                    json.indexOf(
                            "\"items\":["
                    );

            if (p < 0) {
                return;
            }

            p =
                    json.indexOf(
                            '[',
                            p
                    );

            int e =
                    json.lastIndexOf(']');

            if (p < 0 || e < p) {
                return;
            }

            String itemsJson =
                    json.substring(
                            p + 1,
                            e
                    );

            List<String> objects =
                    splitObjects(itemsJson);

            for (String obj :
                    objects) {

                int id =
                        JsonUtils.extractInt(
                                obj,
                                "id",
                                -1
                        );

                String metadata =
                        JsonUtils.extractStr(
                                obj,
                                "metadata"
                        );

                String category =
                        JsonUtils.extractStr(
                                obj,
                                "category"
                        );

                List<Float> emb =
                        JsonUtils.extractArray(
                                obj,
                                "embedding"
                        );

                if (id < 1
                        || emb.size() != dims
                        || emb.stream().anyMatch(v -> !Float.isFinite(v))) {
                    continue;
                }

                VectorItem item =
                        new VectorItem(
                                id,
                                metadata,
                                category,
                                emb
                        );

                store.put(
                        id,
                        item
                );
                nextId = Math.max(nextId, id + 1);
            }

            // =========================
            // Rebuild Brute Force
            // =========================

            for (VectorItem v :
                    store.values()) {

                bf.insert(v);
            }

            // =========================
            // Rebuild KD Tree
            // =========================

            kdt.rebuild(
                    store.values()
            );

            // =========================
            // Rebuild HNSW
            // =========================

            for (VectorItem v :
                    store.values()) {

                hnswCosine.insert(v, Distance::cosine);
                hnswEuclidean.insert(v, Distance::euclidean);
                hnswManhattan.insert(v, Distance::manhattan);
            }

            // =========================
            // Fix next ID
            // =========================

            if (!store.isEmpty()) {

                int maxId =
                        Collections.max(
                                store.keySet()
                        );

                nextId =
                        Math.max(
                                nextId,
                                maxId + 1
                        );
            }

            System.out.println(
                    "VectorDB loaded: "
                            + store.size()
                            + " vectors"
            );

        } catch (Exception ex) {

            System.err.println(
                    "VectorDB load failed: "
                            + ex.getMessage()
            );
        }
    }

    // =========================
    // Split JSON Objects
    // =========================

    private static List<String> splitObjects(
            String s
    ) {

        List<String> result =
                new ArrayList<>();

        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        int start = -1;

        for (int i = 0;
             i < s.length();
             i++) {

            char c = s.charAt(i);

            if (escaped) {
                escaped = false;
                continue;
            }

            if (c == '\\'
                    && inString) {

                escaped = true;
                continue;
            }

            if (c == '"') {
                inString = !inString;
                continue;
            }

            if (inString) {
                continue;
            }

            if (c == '{') {

                if (depth == 0) {
                    start = i;
                }

                depth++;

            } else if (c == '}') {

                depth--;

                if (depth == 0
                        && start >= 0) {

                    result.add(
                            s.substring(
                                    start,
                                    i + 1
                            )
                    );

                    start = -1;
                }
            }
        }

        return result;
    }

    // =========================
    // Hit
    // =========================

    public static class Hit {

        public int id;
        public String meta;
        public String cat;
        public List<Float> emb;
        public float dist;

        public Hit(
                int id,
                String meta,
                String cat,
                List<Float> emb,
                float dist
        ) {

            this.id = id;
            this.meta = meta;
            this.cat = cat;
            this.emb =
                    new ArrayList<>(emb);
            this.dist = dist;
        }
    }

    // =========================
    // Search Output
    // =========================

    public static class SearchOut {

        public List<Hit> hits =
                new ArrayList<>();

        public long us;
        public String algo;
        public String metric;
        public boolean exact;
        public String note;
    }

    // =========================
    // Benchmark Output
    // =========================

    public static class BenchOut {

        public long bfUs;
        public long kdUs;
        public long hnswUs;
        public int n;
        public boolean kdCosineFallback;
        public double hnswRecallAtK;

        public BenchOut(
                long bfUs,
                long kdUs,
                long hnswUs,
                int n,
                boolean kdCosineFallback,
                double hnswRecallAtK
        ) {

            this.bfUs = bfUs;
            this.kdUs = kdUs;
            this.hnswUs = hnswUs;
            this.n = n;
            this.kdCosineFallback = kdCosineFallback;
            this.hnswRecallAtK = hnswRecallAtK;
        }
    }
}