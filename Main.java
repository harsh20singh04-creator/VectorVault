import vectorvault.db.DocumentDB;
import vectorvault.db.VectorDB;
import vectorvault.demo.DemoData;
import vectorvault.ollama.OllamaClient;
import vectorvault.server.ApiServer;

public class Main {

    public static final int DIMS = 16;

    public static void main(String[] args) throws Exception {

        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        // Load demo data only on first run
        if (db.size() == 0) {
            System.out.println("Loading demo data...");
            DemoData.loadDemo(db);
        } else {
            System.out.println("Using persisted VectorDB data...");
        }

        boolean ollamaUp = ollama.isAvailable();

        System.out.println("=== VectorVault Engine ===");
        System.out.println("Starting dashboard and API server...");
        System.out.println(
                db.size() + " vectors | " +
                DIMS + " dims | HNSW+KD-Tree+BruteForce"
        );

        System.out.println(
                "Ollama: " +
                (ollamaUp
                        ? "ONLINE"
                        : "OFFLINE (install from ollama.com)")
        );

        if (ollamaUp) {
            System.out.println(
                    "  embed model: " + ollama.embedModel +
                    "  gen model: " + ollama.genModel
            );
        }

        ApiServer.start(db, docDB, ollama, DIMS);
    }
}