# Optional wandb hooks. skipped if WANDB_DISABLED or import fails.

import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional


def _wandb_disabled() -> bool:
    return os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes")


def _try_import_wandb():
    # `try_import_wandb` lives here so benchmark steps read top-down.
    if _wandb_disabled():
        return None
    # wandb is optional in student environments — ignore import failures
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def _system_metrics_settings(wb):
    # Shared `system_metrics_settings` logic reused by multiple benchmark paths.
    try:
        return wb.Settings(_stats_sample_rate_seconds=2)
    except Exception:
        try:
            return wb.Settings()
        except Exception:
            return None


SCENARIO_COLUMNS = (
    "scenario_id",
    "category",
    "tool",
    "e2e_ms",
    "raw_response",
    "error",
)


def log_variant_run(
    variant: Any,
    scenarios: list[dict],
    rows: list[dict],
    vllm_metrics: dict[str, Any],
    project: Optional[str] = None,
    extra_config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    wb = _try_import_wandb()

    # wandb client missing — skip logging side-effects quietly
    if wb is None:
        return None

    project = project or os.environ.get("WANDB_PROJECT", "hpml-assetopsbench-vlm")
    started_at = int(time.time())
    run_name = f"{variant.name}-{started_at}"
    model_family = "qwen" if "qwen" in (variant.model_id or "").lower() else (
        "llama" if "llama" in (variant.model_id or "").lower() else "other"
    )
    config = {
        "variant": variant.name,
        "variant_short": variant.short,
        "family": variant.family,
        "model_id": variant.model_id,
        "image_max_side": variant.image_max_side,
        "vllm_extra": " ".join(variant.vllm_extra) if variant.vllm_extra else "",
        "n_scenarios": len(scenarios),
        "model_family": model_family,
        **(extra_config or {}),
    }
    init_kwargs = dict(
        project=project,
        name=run_name,
        job_type="vlm_benchmark",
        config=config,
        tags=[variant.family, variant.short, model_family, variant.name],
        reinit=True,
    )
    settings = _system_metrics_settings(wb)

    # merge optional teammate overrides before logging charts
    if settings is not None:
        init_kwargs["settings"] = settings
    run = wb.init(**init_kwargs)

    # latency table here. accuracy comes from llm_judge later
    table_rows = []
    # process records in deterministic order
    for sc, row in zip(scenarios, rows):
        e2e = row.get("e2e_ms", 0) or 0
        table_rows.append([
            row.get("scenario_id", sc.get("id")),
            row.get("category", sc.get("category", "")),
            row.get("tool", ""),
            float(e2e) if e2e else 0.0,
            (row.get("raw_response", "") or "")[:1500],
            row.get("error", "") or "",
        ])
        run.log({
            "scenario/e2e_ms": float(e2e) if e2e and e2e > 0 else None,
        })

    # rollups only at bench time
    valid_e2e = [float(r["e2e_ms"]) for r in rows if r.get("e2e_ms") and float(r["e2e_ms"]) > 0]
    e2e_mean = sum(valid_e2e) / len(valid_e2e) if valid_e2e else 0.0
    e2e_p50 = sorted(valid_e2e)[len(valid_e2e) // 2] if valid_e2e else 0.0
    e2e_p95 = sorted(valid_e2e)[max(0, int(0.95 * (len(valid_e2e) - 1)))] if valid_e2e else 0.0
    e2e_max = max(valid_e2e) if valid_e2e else 0.0

    summary: dict[str, Any] = {
        "e2e_mean_ms": float(e2e_mean),
        "e2e_p50_ms": float(e2e_p50),
        "e2e_p95_ms": float(e2e_p95),
        "e2e_max_ms": float(e2e_max),
        "n_scenarios": len(rows),
    }
    # each pass handles the next item in the sequence
    for k, v in (vllm_metrics or {}).items():
        if isinstance(v, (int, float)):
            summary[f"vllm/{k}"] = v

    for k, v in summary.items():
        run.summary[k] = v

    # chart helpers can fail on minimal CI images — skip instead of crashing
    try:
        table = wb.Table(columns=list(SCENARIO_COLUMNS), data=table_rows)
        run.log({"scenarios_table": table, **summary})
    except Exception:
        run.log(summary)

    url = getattr(run, "url", None)
    run.finish()
    return url


# Cross-variant summary helpers
@contextmanager
def comparison_run(
    project: Optional[str] = None,
    name: str = "all-variants-summary",
    config: Optional[dict[str, Any]] = None,
):
    wb = _try_import_wandb()

    # wandb client missing — skip logging side-effects quietly
    if wb is None:
        yield None
        return
    project = project or os.environ.get("WANDB_PROJECT", "hpml-assetopsbench-vlm")
    init_kwargs = dict(
        project=project,
        name=name,
        job_type="cross_variant_summary",
        config=config or {},
        tags=["summary", "cross-variant"],
        reinit=True,
    )
    settings = _system_metrics_settings(wb)

    # merge optional teammate overrides before logging charts
    if settings is not None:
        init_kwargs["settings"] = settings
    run = wb.init(**init_kwargs)
    # isolate errors so the rest of the call can bail cleanly
    try:
        yield run
    finally:
        try:
            run.finish()
        except Exception:
            pass


def log_comparison_table(
    run,
    columns: list[str],
    rows: Iterable[Iterable[Any]],
    name: str = "variants_summary",
) -> None:
    # Shared `log_comparison_table` logic reused by multiple benchmark paths.
    if run is None:
        return
    # chart helpers can fail on minimal CI images — skip instead of crashing
    try:
        import wandb
        run.log({name: wandb.Table(columns=columns, data=list(rows))})
    except Exception:
        pass


def log_comparison_image(run, key: str, path: str) -> None:
    # CLI/helper entry for `log_comparison_image`.
    if run is None:
        return
    # wandb is optional in student environments — ignore import failures
    try:
        import wandb
        run.log({key: wandb.Image(path)})
    except Exception:
        pass


# Artifact logging for quantized checkpoints
def log_checkpoint_artifact(
    name: str,
    out_dir: str,
    metadata: Optional[dict[str, Any]] = None,
    project: Optional[str] = None,
    description: str = "",
    include_weights: bool = False,
) -> Optional[str]:
    wb = _try_import_wandb()

    # wandb client missing — skip logging side-effects quietly
    if wb is None:
        return None
    from pathlib import Path
    p = Path(out_dir)

    # nothing to artifact if checkpoint path vanished
    if not p.exists():
        return None

    project = project or os.environ.get("WANDB_PROJECT", "hpml-assetopsbench-vlm")
    init_kwargs = dict(
        project=project,
        name=f"{name}-quantize",
        job_type="quantize",
        config={"out_dir": str(p), **(metadata or {})},
        tags=[name, "quantize", "artifact"],
        reinit=True,
    )
    settings = _system_metrics_settings(wb)

    # merge optional teammate overrides before logging charts
    if settings is not None:
        init_kwargs["settings"] = settings
    run = wb.init(**init_kwargs)

    art = wb.Artifact(
        name=name,
        type="quantized-checkpoint",
        description=description or f"Quantized checkpoint at {out_dir}",
        metadata=metadata or {},
    )

    # Always log small metadata files.
    for fname in ("config.json", "generation_config.json", "tokenizer_config.json",
                  "preprocessor_config.json", "processor_config.json"):
        fp = p / fname
        if fp.exists():
            art.add_file(str(fp), name=fname)

    # File-size manifest as a tiny JSON in the artifact.
    sizes = {}
    # process records in deterministic order
    for child in p.glob("*"):
        if child.is_file():
            sizes[child.name] = child.stat().st_size
    import json as _json
    manifest_path = p / "_artifact_manifest.json"
    manifest_path.write_text(_json.dumps({
        "out_dir": str(p),
        "files": sizes,
        "total_bytes": sum(sizes.values()),
    }, indent=2))
    art.add_file(str(manifest_path), name="_artifact_manifest.json")

    # upload heavyweight checkpoint blobs only when enabled
    if include_weights:
        art.add_dir(str(p))

    run.log_artifact(art)
    summary = {
        "checkpoint/total_bytes": sum(sizes.values()),
        "checkpoint/n_files": len(sizes),
    }
    for k, v in summary.items():
        run.summary[k] = v

    # attach teammate tags/extra descriptors when wandb accepts them
    if metadata:
        # process records in deterministic order
        for k, v in metadata.items():
            if isinstance(v, (int, float)):
                run.summary[f"checkpoint/{k}"] = v

    url = getattr(run, "url", None)
    run.finish()
    return url

