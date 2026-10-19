package vectorvault.index;

import java.util.*;
import vectorvault.core.*;

public class BruteForce {
        public BruteForce() {}
        final List<VectorItem> items = new ArrayList<>();

        public void insert(VectorItem v) { items.add(v); }

        public List<HitPair> knn(List<Float> q, int k, Distance.DistFn dist) {
            if (q == null || k <= 0) return new ArrayList<>();
            // Keep only k candidates: O(n log k) time and O(k) extra memory.
            PriorityQueue<HitPair> heap = new PriorityQueue<>(
                    (a, b) -> {
                        int c = Float.compare(b.dist, a.dist);
                        return c != 0 ? c : Integer.compare(b.id, a.id);
                    });
            for (VectorItem v : items) {
                HitPair hit = new HitPair(dist.distance(q, v.emb), v.id);
                if (heap.size() < k || compareAscending(hit, heap.peek()) < 0) {
                    heap.offer(hit);
                    if (heap.size() > k) heap.poll();
                }
            }
            List<HitPair> r = new ArrayList<>(heap);
            r.sort(BruteForce::compareAscending);
            return r;
        }

        private static int compareAscending(HitPair a, HitPair b) {
            int c = Float.compare(a.dist, b.dist);
            return c != 0 ? c : Integer.compare(a.id, b.id);
        }

        public void remove(int id) {
            items.removeIf(v -> v.id == id);
        }
    }
