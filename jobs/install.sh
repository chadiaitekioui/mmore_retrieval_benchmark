#!/usr/bin/env bash
# Deprecated path — use repo-root install.sh (works from any cwd).
exec "$(cd "$(dirname "$0")/.." && pwd)/install.sh"
