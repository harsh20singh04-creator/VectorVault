package vectorvault.index;

import java.util.*;
import vectorvault.core.*;

public class KDTree {
        static class KDNode {
            VectorItem item;
            KDNode left, right;
            KDNode(VectorItem item) { this.item = item; }
        }

        KDNode root;
        final int dims;

        public KDTree(int dims) { this.dims = dims; }

        KDNode insertNode(KDNode n, VectorItem v, int d) {
            if (n == null) return new KDNode(v);
            int ax = d % dims;
            if (v.emb.get(ax) < n.item.emb.get(ax)) n.left = insertNode(n.left, v, d + 1);
            else n.right = insertNode(n.right, v, d + 1);
            return n;
        }

        public void insert(VectorItem v) { root = insertNode(root, v, 0); }

        void knn(KDNode n, List<Float> q, int k, int d, Distance.DistFn dist,
                PriorityQueue<HitPair> heap) {
            if (n == null || k <= 0) return;

            float dn = dist.distance(q, n.item.emb);
            if (heap.size() < k || dn < heap.peek().dist
                    || (Float.compare(dn, heap.peek().dist) == 0 && n.item.id < heap.peek().id)) {
                heap.offer(new HitPair(dn, n.item.id));
                if (heap.size() > k) heap.poll();
            }

            int ax = d % dims;
            float diff = q.get(ax) - n.item.emb.get(ax);
            KDNode closer = diff < 0 ? n.left : n.right;
            KDNode farther = diff < 0 ? n.right : n.left;

            knn(closer, q, k, d + 1, dist, heap);
            if (heap.size() < k || Math.abs(diff) <= heap.peek().dist)
                knn(farther, q, k, d + 1, dist, heap);
        }

        public List<HitPair> knn(List<Float> q, int k, Distance.DistFn dist) {
            if (k <= 0) return new ArrayList<>();
            // Java PriorityQueue is min-first, so reverse it to make a max-heap.
            PriorityQueue<HitPair> heap = new PriorityQueue<>(
                    (a, b) -> {
                        int c = Float.compare(b.dist, a.dist);
                        return c != 0 ? c : Integer.compare(b.id, a.id);
                    });
            knn(root, q, k, 0, dist, heap);
            List<HitPair> r = new ArrayList<>();
            while (!heap.isEmpty()) r.add(heap.poll());
            r.sort(Comparator.comparingDouble((HitPair x) -> x.dist).thenComparingInt(x -> x.id));
            return r;
        }

        public void rebuild(Collection<VectorItem> items) {
            List<VectorItem> values = new ArrayList<>(items);
            root = buildBalanced(values, 0);
        }

        private KDNode buildBalanced(List<VectorItem> values, int depth) {
            if (values.isEmpty()) return null;
            int axis = depth % dims;
            values.sort(Comparator.comparingDouble((VectorItem v) -> v.emb.get(axis))
                    .thenComparingInt((VectorItem v) -> v.id));
            int mid = values.size() / 2;
            VectorItem pivot = values.get(mid);
            KDNode node = new KDNode(pivot);
            node.left = buildBalanced(new ArrayList<>(values.subList(0, mid)), depth + 1);
            node.right = buildBalanced(new ArrayList<>(values.subList(mid + 1, values.size())), depth + 1);
            return node;
        }
    }
