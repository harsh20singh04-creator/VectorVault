#!/usr/bin/env sh
set -eu
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
if ! command -v javac >/dev/null 2>&1; then
  echo "Java JDK 17+ is required (javac was not found)." >&2
  exit 1
fi
mkdir -p out
find src -name '*.java' -print > .sources.list
printf '%s\n' Main.java >> .sources.list
javac --release 17 --add-modules jdk.httpserver -encoding UTF-8 -d out @.sources.list
rm -f .sources.list
exec java --add-modules jdk.httpserver -Dvectorvault.root="$ROOT" -cp out Main "$@"
