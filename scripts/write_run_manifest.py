"""
Write results/<run>/manifest.json for reproducibility (git commits, config hash, frozen judge prompts).

Usage:
    python scripts/write_run_manifest.py --run run_C
    python scripts/write_run_manifest.py --run run_C --config config/rag/run_C_api.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").is_dir():
        return {"path": str(repo), "available": False}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        dirty = (
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"],
                text=True,
            ).strip()
            != ""
        )
        return {
            "path": str(repo),
            "available": True,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": str(repo), "available": False}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_text(path.read_text(encoding="utf-8"))


def _extract_judge_block_raw(config_path: Path) -> dict[str, Any] | None:
    """Regex fallback when PyYAML is not installed."""
    text = config_path.read_text(encoding="utf-8")
    if "judge:" not in text:
        return None

    def _bool(key: str, default: bool | None = None) -> bool | None:
        m = re.search(rf"^\s*{key}:\s*(true|false)\s*$", text, re.MULTILINE)
        if not m:
            return default
        return m.group(1) == "true"

    def _int(key: str, default: int | None = None) -> int | None:
        m = re.search(rf"^\s*{key}:\s*(\d+)\s*$", text, re.MULTILINE)
        return int(m.group(1)) if m else default

    def _str(key: str) -> str | None:
        m = re.search(rf'^\s*{key}:\s*"?([A-Z_]+)"?\s*$', text, re.MULTILINE)
        return m.group(1) if m else None

    out: dict[str, Any] = {
        "force_corrective_action": _str("force_corrective_action"),
        "max_corrective_steps": _int("max_corrective_steps"),
        "allow_re_retrieve": _bool("allow_re_retrieve"),
        "allow_add_questions": _bool("allow_add_questions"),
        "allow_add_context": _bool("allow_add_context"),
    }
    for key in ("system_prompt", "user_prompt"):
        m = re.search(rf"^\s*{key}:\s*\|\s*\n((?:\s+.+\n)+)", text, re.MULTILINE)
        if m:
            block = textwrap.dedent("\n".join(line[4:] if len(line) > 4 else line for line in m.group(1).splitlines()))
            out[key] = block.strip()
            out[f"{key}_sha256"] = _sha256_text(block.strip())
        else:
            m1 = re.search(rf'^\s*{key}:\s*"([^"]*)"', text, re.MULTILINE)
            if m1:
                out[key] = m1.group(1)
                out[f"{key}_sha256"] = _sha256_text(m1.group(1))
    return out


def _extract_judge_prompts(config_path: Path) -> dict[str, Any] | None:
    if yaml is not None:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        judge = (data or {}).get("rag", {}).get("judge") or (data or {}).get("judge")
        if judge:
            out: dict[str, Any] = {}
            for key in (
                "force_corrective_action",
                "max_corrective_steps",
                "allow_re_retrieve",
                "allow_add_questions",
                "allow_add_context",
                "metric_thresholds",
            ):
                if key in judge:
                    out[key] = judge[key]
            llm = judge.get("llm") or {}
            if llm:
                out["llm"] = {
                    k: llm[k]
                    for k in ("llm_name", "temperature", "max_new_tokens")
                    if k in llm
                }
            for key in ("system_prompt", "user_prompt"):
                if key in judge and judge[key]:
                    text = str(judge[key]).strip()
                    out[key] = text
                    out[f"{key}_sha256"] = _sha256_text(text)
            return out
    return _extract_judge_block_raw(config_path)


def default_config_for_run(root: Path, run: str) -> Path:
    if run == "run_C":
        return root / "config/rag/run_C_api.yaml"
    if run == "run_C_ctrl":
        return root / "config/rag/run_C_ctrl_api.yaml"
    if run.startswith(("run_steps_", "run_force_")) or run == "run_judge_scout":
        return root / "config/rag/study" / f"{run}.yaml"
    return root / "config/retrieve" / f"{run}.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--config", default="", help="Override config path")
    parser.add_argument("--mmore-root", default="", help="Sibling MMORE checkout")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    run = args.run
    config_path = Path(args.config) if args.config else default_config_for_run(root, run)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    mmore_root = Path(args.mmore_root) if args.mmore_root else root.parent / "mmore"
    workdir = root.parent

    try:
        rel_config = str(config_path.relative_to(root))
    except ValueError:
        rel_config = str(config_path)
    manifest: dict[str, Any] = {
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": _git_info(root),
        "mmore": _git_info(mmore_root),
        "workdir": str(workdir),
        "config": {
            "path": rel_config,
            "sha256": _sha256_file(config_path),
        },
        "judge": _extract_judge_prompts(config_path),
        "protocol_notes": {
            "do_not_use": "mmore/examples/rag/config_judge.yaml for benchmark comparison",
            "run_C": "allow_re_retrieve only; max_corrective_steps=1; legacy frozen prompts",
            "run_C_ctrl": "force_corrective_action: PROCEED + max_corrective_steps=0",
            "retrieve_runs": "no judge block; mmore retrieve only",
        },
    }

    out_path = root / "results" / run / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✓ {out_path}")


if __name__ == "__main__":
    main()
