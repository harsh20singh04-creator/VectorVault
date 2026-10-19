package vectorvault.index;

import java.util.*;
import vectorvault.core.*;

public class HNSW {
        static class Node {
            VectorItem item;
            int maxLyr;
            List<List<Integer>> nbrs;
            Node(VectorItem item, int maxLyr) {
                this.item = item;
                this.maxLyr = maxLyr;
                this.nbrs = new ArrayList<>();
                for (int i = 0; i <= maxLyr; i++) this.nbrs.add(new ArrayList<>());
            }
        }

        final Map<Integer, Node> G = new HashMap<>();
        final int M, M0, efBuild;
        final float mL;
        int topLayer = -1;
        int entryPt = -1;
        final Random rng = new Random(42);
        private Distance.DistFn distanceFunction;

        public HNSW() { this(16, 200); }

        public HNSW(int m, int efBuild) {
            if (m < 2) throw new IllegalArgumentException("HNSW M must be at least 2");
            if (efBuild < 1) throw new IllegalArgumentException("HNSW efBuild must be positive");
            this.M = m;
            this.M0 = 2 * m;
            this.efBuild = efBuild;
            this.mL = 1.0f / (float) Math.log(m);
        }

        int randLevel() {
            double u = rng.nextDouble();
            while (u == 0.0) u = rng.nextDouble();
            return (int) Math.floor(-Math.log(u) * mL);
        }

        List<HitPair> searchLayer(List<Float> q, int ep, int ef, int lyr, Distance.DistFn dist) {
            if (!G.containsKey(ep)) return new ArrayList<>();

            Set<Integer> vis = new HashSet<>();
            PriorityQueue<HitPair> cands = new PriorityQueue<>(
                    Comparator.comparingDouble((HitPair x) -> x.dist).thenComparingInt(x -> x.id));
            PriorityQueue<HitPair> found = new PriorityQueue<>(
                    (a, b) -> {
                        int c = Float.compare(b.dist, a.dist);
                        return c != 0 ? c : Integer.compare(b.id, a.id);
                    });

            float d0 = dist.distance(q, G.get(ep).item.emb);
            vis.add(ep);
            cands.offer(new HitPair(d0, ep));
            found.offer(new HitPair(d0, ep));

            while (!cands.isEmpty()) {
                HitPair cur = cands.poll();
                if (found.size() >= ef && cur.dist > found.peek().dist) break;
                Node node = G.get(cur.id);
                if (node == null || lyr >= node.nbrs.size()) continue;

                for (int nid : node.nbrs.get(lyr)) {
                    if (vis.contains(nid) || !G.containsKey(nid)) continue;
                    vis.add(nid);
                    float nd = dist.distance(q, G.get(nid).item.emb);
                    if (found.size() < ef || nd < found.peek().dist
                            || (Float.compare(nd, found.peek().dist) == 0 && nid < found.peek().id)) {
                        cands.offer(new HitPair(nd, nid));
                        found.offer(new HitPair(nd, nid));
                        if (found.size() > ef) found.poll();
                    }
                }
            }

            List<HitPair> res = new ArrayList<>();
            while (!found.isEmpty()) res.add(found.poll());
            res.sort(Comparator.comparingDouble((HitPair x) -> x.dist).thenComparingInt(x -> x.id));
            return res;
        }

        List<Integer> selectNbrs(List<HitPair> cands, int maxM) {
            List<Integer> r = new ArrayList<>();
            for (int i = 0; i < Math.min(cands.size(), maxM); i++) r.add(cands.get(i).id);
            return r;
        }

        public void insert(VectorItem item, Distance.DistFn dist) {
            if (dist == null) throw new IllegalArgumentException("HNSW distance function cannot be null");
            distanceFunction = dist;
            int id = item.id;
            int lvl = randLevel();
            G.put(id, new Node(item, lvl));

            if (entryPt == -1) {
                entryPt = id;
                topLayer = lvl;
                return;
            }

            int ep = entryPt;
            for (int lc = topLayer; lc > lvl; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<HitPair> W = searchLayer(item.emb, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }

            for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
                List<HitPair> W = searchLayer(item.emb, ep, efBuild, lc, dist);
                int maxM = lc == 0 ? M0 : M;
                List<Integer> sel = selectNbrs(W, maxM);
                G.get(id).nbrs.set(lc, sel);

                for (int nid : sel) {
                    Node neighbor = G.get(nid);
                    if (neighbor == null) continue;
                    while (neighbor.nbrs.size() <= lc) neighbor.nbrs.add(new ArrayList<>());
                    List<Integer> conn = neighbor.nbrs.get(lc);
                    conn.add(id);

                    if (conn.size() > maxM) {
                        List<HitPair> ds = new ArrayList<>();
                        for (int c : conn) {
                            if (G.containsKey(c))
                                ds.add(new HitPair(dist.distance(neighbor.item.emb, G.get(c).item.emb), c));
                        }
                        ds.sort(Comparator.comparingDouble((HitPair x) -> x.dist).thenComparingInt(x -> x.id));
                        conn.clear();
                        for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id);
                    }
                }
                if (!W.isEmpty()) ep = W.get(0).id;
            }

            if (lvl > topLayer) {
                topLayer = lvl;
                entryPt = id;
            }
        }

        public List<HitPair> knn(List<Float> q, int k, int ef, Distance.DistFn dist) {
            if (entryPt == -1 || k <= 0) return new ArrayList<>();
            int ep = entryPt;
            for (int lc = topLayer; lc > 0; lc--) {
                if (lc < G.get(ep).nbrs.size()) {
                    List<HitPair> W = searchLayer(q, ep, 1, lc, dist);
                    if (!W.isEmpty()) ep = W.get(0).id;
                }
            }
            List<HitPair> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
            if (W.size() > k) return new ArrayList<>(W.subList(0, k));
            return W;
        }

        /**
         * Rebuilds the graph in a deterministic order. This is useful after deletes:
         * simply unlinking a node can fragment an HNSW graph and lower recall.
         */
        public void rebuild(Collection<VectorItem> items, Distance.DistFn dist) {
            if (dist == null) throw new IllegalArgumentException("HNSW distance function cannot be null");
            G.clear();
            topLayer = -1;
            entryPt = -1;
            rng.setSeed(42L);
            distanceFunction = dist;
            List<VectorItem> ordered = new ArrayList<>(items);
            ordered.sort(Comparator.comparingInt(v -> v.id));
            for (VectorItem item : ordered) insert(item, dist);
        }

        public void remove(int id) {
            if (!G.containsKey(id)) return;
            if (distanceFunction == null) {
                G.remove(id);
                for (Node node : G.values()) for (List<Integer> layer : node.nbrs) layer.removeIf(x -> x == id);
                if (G.isEmpty()) { entryPt = -1; topLayer = -1; }
                return;
            }
            List<VectorItem> remaining = new ArrayList<>();
            for (Node node : G.values()) if (node.item.id != id) remaining.add(node.item);
            rebuild(remaining, distanceFunction);
        }

        public static class GraphInfo {
            public int topLayer, nodeCount;
            public List<Integer> nodesPerLayer = new ArrayList<>();
            public List<Integer> edgesPerLayer = new ArrayList<>();
            public static class NV {
                public int id, maxLyr; public String metadata, category;
                public NV(int id, String metadata, String category, int maxLyr) {
                    this.id=id; this.metadata=metadata; this.category=category; this.maxLyr=maxLyr;
                }
            }
            public static class EV {
                public int src, dst, lyr;
                public EV(int src, int dst, int lyr) { this.src=src; this.dst=dst; this.lyr=lyr; }
            }
            public List<NV> nodes = new ArrayList<>();
            public List<EV> edges = new ArrayList<>();
        }

        public GraphInfo getInfo() {
            GraphInfo gi = new GraphInfo();
            gi.topLayer = topLayer;
            gi.nodeCount = G.size();
            int maxL = Math.max(topLayer + 1, 1);
            for (int i = 0; i < maxL; i++) {
                gi.nodesPerLayer.add(0);
                gi.edgesPerLayer.add(0);
            }

            for (Map.Entry<Integer, Node> e : G.entrySet()) {
                int id = e.getKey(); Node nd = e.getValue();
                gi.nodes.add(new GraphInfo.NV(id, nd.item.metadata, nd.item.category, nd.maxLyr));
                for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                    gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                    if (lc < nd.nbrs.size()) {
                        for (int nid : nd.nbrs.get(lc)) {
                            if (id < nid) {
                                gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                                gi.edges.add(new GraphInfo.EV(id, nid, lc));
                            }
                        }
                    }
                }
            }
            return gi;
        }

        public int size() { return G.size(); }
    }
