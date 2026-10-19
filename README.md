# VectorVault — Vector Search Lab & RAG Workbench

VectorVault is a portfolio-ready vector retrieval and Retrieval-Augmented Generation (RAG) application built in Java. It gives you a visual search lab for comparing Brute Force, KD-tree, and HNSW retrieval, plus a document workspace that indexes real embeddings from Ollama and uses retrieved passages to ground LLM answers.

> **Important distinction:** the interactive 16-dimensional Vector Lab uses deterministic, hand-crafted feature vectors to demonstrate and visualize index behavior. Documents in the RAG workspace use actual vectors returned by the configured Ollama embedding model. They are separate stores because demo feature vectors are not semantic model embeddings.

![Java](https://img.shields.io/badge/Java-17%2B-orange)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black)
![Tests](https://img.shields.io/badge/tests-self--test%20%2B%20integration-success)

## Highlights

- **Three retrieval algorithms:** Brute Force exact search, KD-tree search, and HNSW approximate nearest-neighbor search.
- **Three distance metrics:** cosine distance (`1 − cosine similarity`), Euclidean (L2), and Manhattan (L1).
- **Metric-specific HNSW graphs:** the demo and document stores maintain separate graphs for each supported metric rather than querying a graph built with a different distance function.
- **Correct KD-tree behavior:** KD-tree axis pruning is used for Euclidean and Manhattan distances. Cosine distance does not have the required axis lower-bound, so the KD-tree choice reports and uses an exact Brute Force fallback instead of returning incorrectly pruned results.
- **RAG ingestion and Q&A:** source text is split into overlapping chunks, embedded by Ollama, persisted, retrieved, and passed to a grounded generation prompt with inline source markers.
- **Search diagnostics:** latency, HNSW recall@k against the exact baseline, cosine similarity scores (when cosine is selected), result distances for L1/L2, HNSW graph visualization, and index/metric selection.
- **Usable dashboard:** responsive browser UI, document management, error states, context display, and safe text rendering.
- **Dependency-light Java backend:** Java's built-in HTTP server, no Maven/Gradle build requirement, JSON REST endpoints, local-file persistence, request-size limits, security headers, configurable CORS, and optional API-key protection.
- **Regression coverage:** Java self-tests and an end-to-end HTTP test using a local Ollama-protocol stub. No downloaded model is required to run the tests.

## Architecture

```text
Browser dashboard (index.html)
           │ same-origin HTTP/JSON
           ▼
    Java HTTP API (JDK)
      ┌────┴───────────────┐
      ▼                    ▼
 VectorDB               DocumentDB
 16-D demo vectors      real Ollama embeddings
      │                    │
 ┌────┼────┐          ┌────┼────┐
 BF   KD   HNSW       BF   KD   HNSW
      │                    │
      └───────┐    ┌───────┘
              ▼    ▼
          OllamaClient
          /api/embed + /api/generate
```

### RAG request flow

1. The Documents tab accepts a title and source text.
2. `TextChunker` splits the text into chunks of 250 words with 30-word overlap.
3. `OllamaClient` obtains one embedding for each chunk using Ollama's `/api/embed` endpoint (with a legacy `/api/embeddings` compatibility fallback).
4. `DocumentDB` persists the chunks and vectors and indexes them in Brute Force, KD-tree, and metric-specific HNSW structures.
5. A question is embedded with the same configured embedding model and the selected algorithm/metric retrieves the top-k passages.
6. For `/doc/ask`, the retrieved passages are included in the generation prompt. The prompt tells the model to treat passage contents as untrusted data, avoid unsupported claims, and cite passages as `[1]`, `[2]`, etc.

## Requirements

- **Java JDK 17 or newer** (`java` and `javac` on `PATH`).
- **Ollama** for real document embeddings and generated answers. You can run and explore the Vector Lab without Ollama.
- **Python 3.10+** is optional and only used for the end-to-end integration smoke test; the Java self-tests need no Python packages.
- **Docker + Docker Compose** are optional for container deployment.

## Run locally

### Windows

1. Install Java JDK 17+ and Ollama.
2. Pull the default models in PowerShell or Command Prompt:

   ```powershell
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. Open this project folder and run `run.bat`.
4. Open **http://localhost:8080** in Chrome or another browser.

### macOS / Linux

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
chmod +x run.sh test.sh
./run.sh
```

Then open **http://localhost:8080**. The backend binds to `127.0.0.1` by default for local development. Keep the terminal open while using it; the app is served by the Java process, so opening `index.html` directly as a `file://` URL is not the recommended launch method.

If Ollama is offline, VectorVault still serves the demo search lab. Document ingestion and RAG Q&A require Ollama and the configured models.

## Run tests

On macOS/Linux:

```bash
./test.sh
```

On Windows:

```bat
test.bat
```

The scripts compile against Java 17, run the dependency-free Java regression tests, and run `tests/integration_smoke.py` when Python is available. The integration test starts an isolated VectorVault instance and a deterministic local stub implementing the small subset of the Ollama HTTP protocol used by the app. This validates HTTP behavior, API-key protection, chunk ingestion, all nine RAG algorithm/metric combinations, answer generation flow, deletion, and status/stat endpoints without downloading models.

The stub is only a protocol test fixture; it is **not** a semantic embedding model. For a final real-world acceptance check, run the app with Ollama and test at least one document containing known facts, verify that returned passages contain the answer, and check the model/error indicators in the UI.

## Configuration

All configuration is via environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | HTTP listening port (also accepts `VECTORVAULT_PORT` if `PORT` is unset) |
| `VECTORVAULT_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` inside a container or behind a secured reverse proxy. |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL, without a trailing slash |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Model used for both document and question embeddings |
| `OLLAMA_GEN_MODEL` | `llama3.2` | Model used to generate the RAG response |
| `VECTORVAULT_API_KEY` | *(unset)* | When set, requires `Authorization: Bearer <key>` for application API routes. The browser prompts for the key and keeps it in page memory only. |
| `VECTORVAULT_CORS_ORIGIN` | *(unset)* | Optional comma-separated allowlist for cross-origin browser requests. Same-origin use does not require CORS configuration. |

Example (PowerShell):

```powershell
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
$env:OLLAMA_GEN_MODEL = "llama3.2"
$env:VECTORVAULT_API_KEY = "replace-with-a-long-random-secret"
.\run.bat
```

For local development, the API key may be unset. **Do not expose an unauthenticated instance directly to the public internet.** For a public deployment, configure a long random API key, terminate HTTPS at a trusted reverse proxy/platform, configure CORS only for a known frontend origin if a separate frontend is used, and apply platform-level rate limiting/monitoring. The application does not implement TLS itself.

### Embedding model consistency

Use the same embedding model for indexing and querying. If you switch `OLLAMA_EMBED_MODEL`, rebuild/re-ingest your stored document chunks; vectors produced by different model families can be incompatible even if their dimensions happen to match. The document index fixes its vector dimensionality on the first successful insertion and rejects mismatched dimensions.

## Docker Compose

Build and start the Java app plus Ollama service:

```bash
export VECTORVAULT_API_KEY="replace-with-a-long-random-secret"
docker compose up --build -d
```

On PowerShell:

```powershell
$env:VECTORVAULT_API_KEY = [guid]::NewGuid().ToString("N")
docker compose up --build -d
```

Pull the models into the Ollama container (first download can be large):

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull llama3.2
```

Open **http://localhost:8080** and enter the API key when the dashboard prompts. The compose setup persists Ollama models and VectorVault's local index files in named Docker volumes. The local HTTP endpoint is for development; use HTTPS through a reverse proxy/platform before making a deployment public.

## API quick reference

Application routes require the bearer API key only when `VECTORVAULT_API_KEY` is configured. `/` and `/health` are public.

| Method | Route | Description |
|---|---|---|
| `GET` | `/health` | Lightweight liveness probe |
| `GET` | `/status` | Ollama/model availability and index sizes |
| `GET` | `/stats` | Vector/chunk counts and supported algorithms/metrics |
| `GET` | `/items` | Demo vector items |
| `GET` | `/search?v=<comma-separated-vector>&k=5&metric=cosine&algo=hnsw` | Search demo vectors |
| `GET` | `/benchmark?v=<vector>&k=5&metric=cosine` | Brute Force/KD-tree/HNSW latency and HNSW recall@k |
| `GET` | `/hnsw-info` | Cosine HNSW graph structure for the demo lab |
| `POST` | `/insert` | Insert a demo vector (`metadata`, `category`, `embedding`) |
| `DELETE` | `/delete/{id}` | Delete a demo vector |
| `POST` | `/doc/insert` | Chunk, embed, and persist a source (`title`, `text`) |
| `GET` | `/doc/list` | List stored document chunks |
| `POST` | `/doc/search` | Retrieve source contexts (`question`, optional `k`, `algo`, `metric`) |
| `POST` | `/doc/ask` | Retrieve contexts and generate a grounded answer |
| `DELETE` | `/doc/delete/{id}` | Delete a stored document chunk |

Supported values are `algo=bruteforce|kdtree|hnsw` and `metric=cosine|euclidean|manhattan`. `hnsw` is approximate by design; recall depends on graph/search parameters and data. The benchmark reports recall@k against Brute Force rather than claiming HNSW is always faster or always exact. KD-tree + cosine is explicitly labeled as an exact Brute Force fallback.

Example using `curl` (when the key is enabled):

```bash
curl -H "Authorization: Bearer $VECTORVAULT_API_KEY" http://localhost:8080/status
curl -X POST http://localhost:8080/doc/insert \
  -H "Authorization: Bearer $VECTORVAULT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"title":"Vector search notes","text":"Vector search retrieves the nearest embeddings according to a distance function."}'
```

## Persistence and project layout

```text
Main.java
index.html                         responsive dashboard
src/vectorvault/core/              distance functions and vector models
src/vectorvault/index/             Brute Force, KD-tree, HNSW
src/vectorvault/db/                demo vector store and real document store
src/vectorvault/document/          text chunking
src/vectorvault/ollama/             Ollama HTTP client
src/vectorvault/server/             API endpoints/static frontend
src/vectorvault/util/               small JSON and HTTP helpers
tests/VectorVaultSelfTest.java      dependency-free Java regression tests
tests/integration_smoke.py          Ollama-protocol integration test
run.sh, run.bat                     local launchers
test.sh, test.bat                   build and test
Dockerfile, docker-compose.yml      container workflow
.github/workflows/ci.yml            continuous integration
```

Data is stored locally in `data/vectors.json` and `data/documents.json`. Do not commit private or sensitive source documents. The UI's 16-D examples are intentionally illustrative and should not be presented as production-semantic embeddings.

## Current limitations / honest scope

- HNSW is a compact educational implementation, not a drop-in replacement for a highly optimized native vector database. Deletes rebuild its graph to avoid leaving a fragmented index; benchmark write-heavy workloads and your own data before making throughput or scale claims.
- KD-tree effectiveness decreases in high dimensions; RAG model embeddings are usually high-dimensional, so its wall-clock performance may be worse than a linear scan even when the results are exact for L1/L2.
- The API-key option is lightweight app-level protection, not a full identity/access-control system. There is no built-in TLS, multi-tenant isolation, per-user quotas, distributed storage, or backup scheduler.
- The automated integration test verifies the application and Ollama protocol with a fake local model endpoint. Real generation quality, model download, and hardware-specific latency must be validated on your machine with the models installed.
