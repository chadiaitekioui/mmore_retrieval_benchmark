# Judge study configs

Generated for axes 2 & 3. Index thresholds only (no `min_context_relevance` — unused by MMORE).

| Run | Purpose |
|-----|---------|
| `run_steps_0` … `run_steps_3` | Axis 2 — sweep `max_corrective_steps` |
| `run_judge_scout` | Axis 3 — free judge; defines trigger subset T |
| `run_force_rr` | Axis 3 — RE_RETRIEVE only |
| `run_force_aq` | Axis 3 — ADD_QUESTIONS only |
| `run_force_ac` | Axis 3 — ADD_CONTEXT only (needs `mmore[websearch]`) |

Regenerate:

```bash
python scripts/generate_judge_study_configs.py
```

Collect + analyse: see README § Judge study.

MMORE changes required for correct metrics: [`../../docs/MMORE_JUDGE_STUDY_CHANGES.md`](../../docs/MMORE_JUDGE_STUDY_CHANGES.md).
