"""
Annotate chunks.json rows with coerced_decision when MMORE server logs are available.

MMORE logs lines like:
  Judge chose disallowed action ADD_QUESTIONS, falling back to RE_RETRIEVE

Usage:
    python scripts/infer_coerced_decisions.py \\
        --chunks results/run_C/chunks.json \\
        --log /path/to/mmore/shared_log_file.log
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_COERCE_RE = re.compile(
    r"Judge chose disallowed action (\w+), falling back to (\w+)"
)
_QUERY_RE = re.compile(r"Judge step \d+ \| query='([^']{0,200})")


def _load_coercions_by_query_prefix(log_path: Path) -> dict[str, list[dict]]:
    """Map query prefix -> list of {raw, coerced} from log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_query: dict[str, list[dict]] = {}
    current_prefix = ""

    for line in text.splitlines():
        m_q = _QUERY_RE.search(line)
        if m_q:
            current_prefix = m_q.group(1)[:120]
        m_c = _COERCE_RE.search(line)
        if m_c and current_prefix:
            entry = {"llm_decision": m_c.group(1), "coerced_to": m_c.group(2)}
            by_query.setdefault(current_prefix, []).append(entry)
    return by_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default="", help="Default: overwrite --chunks")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else chunks_path

    with chunks_path.open(encoding="utf-8") as f:
        rows = json.load(f)

    by_query = _load_coercions_by_query_prefix(log_path)
    n_tagged = 0
    for row in rows:
        q = (row.get("query") or row.get("input") or "")[:120]
        coercions = by_query.get(q, [])
        if coercions:
            row["coerced_decision"] = True
            row["coercion_trace"] = coercions
            n_tagged += 1
        else:
            row.setdefault("coerced_decision", False)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"✓ {out_path}  coerced_decision tagged on {n_tagged}/{len(rows)} rows")


if __name__ == "__main__":
    main()
