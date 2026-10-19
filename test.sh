#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
command -v javac >/dev/null 2>&1 || { echo "Java JDK 17+ is required." >&2; exit 1; }
mkdir -p out out-test
find src -name '*.java' -print > .sources.list
printf '%s\n' Main.java >> .sources.list
javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -d out @.sources.list
rm -f .sources.list
javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -cp out -d out-test tests/VectorVaultSelfTest.java
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
(cd "$WORK" && java --add-modules jdk.httpserver -cp "$ROOT/out:$ROOT/out-test" VectorVaultSelfTest)
if command -v python3 >/dev/null 2>&1; then
  python3 tests/integration_smoke.py
elif command -v python >/dev/null 2>&1; then
  python tests/integration_smoke.py
else
  echo "Python not found; skipped optional HTTP/Ollama-protocol integration smoke test."
fi
