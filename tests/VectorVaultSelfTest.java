import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;

import vectorvault.core.Distance;
import vectorvault.core.VectorItem;
import vectorvault.db.DocItem;
import vectorvault.db.DocumentDB;
import vectorvault.db.VectorDB;
import vectorvault.document.TextChunker;
import vectorvault.util.JsonUtils;

/** Dependency-free regression tests. Run with test.sh or test.bat. */
public final class VectorVaultSelfTest {
    private static int checks;

    public static void main(String[] args) throws Exception {
        testDistanceMetrics();
        testVectorIndexesAndPersistence();
        testDocumentIndexesAndPersistence();
        testPersistenceFailureRollback();
        testChunkingAndJson();
        System.out.println("VectorVault self-test passed: " + checks + " assertions.");
    }

    private static void testDistanceMetrics() {
        eq(5.0f, Distance.euclidean(List.of(0f, 0f), List.of(3f, 4f)), 0.0001f, "Euclidean distance");
        eq(7.0f, Distance.manhattan(List.of(0f, 0f), List.of(3f, 4f)), 0.0001f, "Manhattan distance");
        eq(1.0f, Distance.cosine(List.of(1f, 0f), List.of(0f, 1f)), 0.0001f, "Orthogonal cosine distance");
        eq(0.0f, Distance.cosine(List.of(1f, 2f), List.of(2f, 4f)), 0.0001f, "Parallel cosine distance");
        eq(1.0f, Distance.cosine(List.of(0f, 0f), List.of(1f, 2f)), 0.0001f, "Zero-vector cosine convention");
        ok(Float.isFinite(Distance.euclidean(List.of(Float.MAX_VALUE), List.of(-Float.MAX_VALUE))), "Euclidean distance stays finite for finite float inputs");
        ok(Float.isFinite(Distance.manhattan(List.of(Float.MAX_VALUE), List.of(-Float.MAX_VALUE))), "Manhattan distance stays finite for finite float inputs");
        eq(0.0f, Distance.cosine(List.of(Float.MAX_VALUE), List.of(Float.MAX_VALUE)), 0.0001f, "Cosine uses stable intermediate precision");
        boolean rejected = false;
        try { Distance.getDistFn("cosine-ish"); } catch (IllegalArgumentException expected) { rejected = true; }
        ok(rejected, "Unknown metrics must not silently become Euclidean");
    }

    private static void testVectorIndexesAndPersistence() {
        VectorDB db = new VectorDB(16);
        Random rng = new Random(1227);
        List<List<Float>> vectors = new ArrayList<>();
        for (int i = 0; i < 24; i++) {
            List<Float> emb = new ArrayList<>();
            for (int j = 0; j < 16; j++) emb.add((float) (rng.nextDouble() * 2.0 - 1.0 + i * 0.003));
            vectors.add(emb);
            db.insert("vector-" + i, i == 23 ? "rare" : "sample", emb, Distance::cosine);
        }
        List<Float> q = vectors.get(8);
        for (String metric : List.of("cosine", "euclidean", "manhattan")) {
            List<Integer> exact = ids(db.search(q, 5, metric, "bruteforce"));
            eq(exact, ids(db.search(q, 5, metric, "kdtree")), "KD-tree top-k must match exact results for " + metric);
            eq(exact, ids(db.search(q, 5, metric, "hnsw")), "HNSW must match exact top-k when ef covers this small dataset for " + metric);
        }
        VectorDB.SearchOut fallback = db.search(q, 5, "cosine", "kdtree");
        ok(fallback.note.contains("fallback"), "KD-tree cosine must disclose exact fallback");
        ok(fallback.exact, "KD-tree cosine fallback should be exact");
        VectorDB.SearchOut filtered = db.search(q, 1, "euclidean", "kdtree", "rare");
        eq(1, filtered.hits.size(), "Category filter should search all candidates before filtering");
        eq("rare", filtered.hits.get(0).cat, "Category filter returned wrong category");

        int sizeBeforeDelete = db.size();
        ok(db.remove(1), "Delete existing demo vector");
        eq(sizeBeforeDelete - 1, db.size(), "Vector delete count");
        eq(false, db.remove(99999), "Deleting a missing vector should report false");

        VectorDB loaded = new VectorDB(16);
        eq(db.size(), loaded.size(), "Vector persistence reload count");
        int newId = loaded.insert("persisted-after-restart", "sample", vectors.get(3), Distance::cosine);
        ok(newId > 24, "Vector IDs should remain monotonic after restart");
        ok(loaded.all().stream().map(v -> v.id).noneMatch(id -> id == -1), "Vector items should be valid");
        VectorDB.BenchOut bench = loaded.benchmark(q, 5, "cosine");
        ok(bench.bfUs >= 0 && bench.kdUs >= 0 && bench.hnswUs >= 0, "Benchmark timings should be non-negative");
        ok(bench.hnswRecallAtK >= 0 && bench.hnswRecallAtK <= 1.0, "HNSW recall must be in [0, 1]");
    }

    private static void testDocumentIndexesAndPersistence() {
        DocumentDB db = new DocumentDB();
        List<String> titles = new ArrayList<>();
        List<String> texts = new ArrayList<>();
        List<List<Float>> embeddings = new ArrayList<>();
        Random rng = new Random(50);
        for (int i = 0; i < 18; i++) {
            titles.add("doc-" + i);
            texts.add("Text body for document chunk " + i);
            List<Float> emb = new ArrayList<>();
            for (int j = 0; j < 8; j++) emb.add((float) (rng.nextDouble() * 2.0 - 1.0 + i * 0.002));
            embeddings.add(emb);
        }
        List<Integer> inserted = db.insertBatch(titles, texts, embeddings);
        eq(18, inserted.size(), "Document batch insert count");
        eq(8, db.getDims(), "Document embedding dimensions");
        List<Float> q = embeddings.get(5);
        for (String metric : List.of("cosine", "euclidean", "manhattan")) {
            List<Integer> exact = docIds(db.search(q, 5, Float.POSITIVE_INFINITY, metric, "bruteforce"));
            eq(exact, docIds(db.search(q, 5, Float.POSITIVE_INFINITY, metric, "kdtree")), "Document KD-tree top-k for " + metric);
            eq(exact, docIds(db.search(q, 5, Float.POSITIVE_INFINITY, metric, "hnsw")), "Document HNSW top-k for " + metric);
        }
        ok(db.remove(inserted.get(0)), "Delete a document chunk");
        DocumentDB loaded = new DocumentDB();
        eq(17, loaded.size(), "Document persistence reload count");
        int next = loaded.insert("new after restart", "persisted text", embeddings.get(1));
        ok(next > inserted.get(inserted.size() - 1), "Document chunk IDs remain monotonic after restart");
        ok(loaded.remove(next), "Delete newest document id before restart");
        DocumentDB loadedAfterNewestDelete = new DocumentDB();
        int afterDelete = loadedAfterNewestDelete.insert("id must not be reused", "persisted text", embeddings.get(2));
        ok(afterDelete > next, "Document IDs must remain monotonic even after deleting the newest id and restarting");

        boolean rejected = false;
        try { loaded.insert("wrong dimensions", "must fail", List.of(0.1f, 0.2f)); }
        catch (IllegalArgumentException expected) { rejected = true; }
        ok(rejected, "Mixed embedding dimensions must be rejected");
    }

    /** Simulate an unwritable data directory and ensure mutations do not linger in memory. */
    private static void testPersistenceFailureRollback() throws IOException {
        Path data = Path.of("data");
        Path backup = Path.of("data-vectorvault-test-backup");
        if (Files.exists(backup)) throw new IOException("Unexpected test backup path already exists");
        if (Files.exists(data)) Files.move(data, backup);
        Files.writeString(data, "block persistence writes");
        try {
            VectorDB vectors = new VectorDB(16);
            boolean vectorWriteFailed = false;
            try { vectors.insert("must roll back", "test", new ArrayList<>(java.util.Collections.nCopies(16, 0.25f)), Distance::cosine); }
            catch (IllegalStateException expected) { vectorWriteFailed = true; }
            ok(vectorWriteFailed, "Vector insert should surface a persistence failure");
            eq(0, vectors.size(), "Failed vector insert must roll back its in-memory item");
            eq(0, vectors.hnswInfo().nodeCount, "Failed vector insert must roll back HNSW nodes");

            DocumentDB docs = new DocumentDB();
            boolean documentWriteFailed = false;
            try { docs.insert("must roll back", "test text", List.of(0.1f, 0.2f)); }
            catch (IllegalStateException expected) { documentWriteFailed = true; }
            ok(documentWriteFailed, "Document insert should surface a persistence failure");
            eq(0, docs.size(), "Failed document insert must roll back its in-memory item");
            eq(0, docs.getDims(), "Failed first document insert must restore unset dimensions");
        } finally {
            Files.deleteIfExists(data);
            if (Files.exists(backup)) Files.move(backup, data);
        }
    }

    private static void testChunkingAndJson() {
        String words = java.util.stream.IntStream.range(0, 600).mapToObj(i -> "w" + i).collect(Collectors.joining(" "));
        List<String> chunks = TextChunker.chunkText(words, 250, 30);
        eq(3, chunks.size(), "Long input should be split into overlapping chunks");
        ok(chunks.get(0).endsWith("w249") && chunks.get(1).startsWith("w220"), "Chunk overlap boundaries");
        eq(List.of(1.0f, -2.5f, 0.25f), JsonUtils.parseVec("1,-2.5,0.25"), "Vector JSON number parsing");
        eq(0, JsonUtils.parseVec("1,nope,2").size(), "Malformed vectors should be rejected atomically");
        String escaped = JsonUtils.jS("line\nwith \"quotes\" and \\ slash\t");
        eq("line\nwith \"quotes\" and \\ slash\t", JsonUtils.extractStr("{\"value\":" + escaped + "}", "value"), "JSON escaping round-trip");
    }

    private static List<Integer> ids(VectorDB.SearchOut out) {
        return out.hits.stream().map(h -> h.id).collect(Collectors.toList());
    }

    private static List<Integer> docIds(List<Map.Entry<Float, DocItem>> out) {
        return out.stream().map(e -> e.getValue().id).collect(Collectors.toList());
    }

    private static void ok(boolean value, String message) {
        checks++;
        if (!value) throw new AssertionError(message);
    }

    private static void eq(Object expected, Object actual, String message) {
        checks++;
        if (expected == null ? actual != null : !expected.equals(actual)) {
            throw new AssertionError(message + " — expected " + expected + ", got " + actual);
        }
    }

    private static void eq(float expected, float actual, float epsilon, String message) {
        checks++;
        if (Math.abs(expected - actual) > epsilon) throw new AssertionError(message + " — expected " + expected + ", got " + actual);
    }
}
