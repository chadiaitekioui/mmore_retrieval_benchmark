"""Generate run_C RAG configs for judge sufficiency threshold sweep (0.3–0.9)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SUFFICIENCY_LEVELS = (0.3, 0.5, 0.7, 0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="config/rag/run_C_api.yaml")
    parser.add_argument("--out-dir", default="config/rag/calib")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    template = (root / args.template).read_text()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for suff in SUFFICIENCY_LEVELS:
        min_ctx = suff * 10.0
        text = re.sub(
            r"min_context_relevance:\s*[\d.]+",
            f"min_context_relevance: {min_ctx:.1f}",
            template,
            count=1,
        )
        tag = f"{int(suff * 100):03d}"
        header = (
            f"# Judge calibration: min_context_relevance={min_ctx:.1f} "
            f"(sufficiency={suff} on 1–10 scale)\n"
        )
        out_path = out_dir / f"run_C_suff_{tag}.yaml"
        out_path.write_text(header + text)
        print(f"✓ {out_path.name}  min_context_relevance={min_ctx:.1f}")


if __name__ == "__main__":
    main()
