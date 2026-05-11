# Figures from results/summary.csv + hpml_metrics.csv -> results/plots/.

import argparse
import csv
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SUMMARY_CSV = RESULTS / "summary.csv"
HPML_CSV = RESULTS / "hpml_metrics.csv"
PLOTS_DIR_DEFAULT = RESULTS / "plots"


def _apply_style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIX"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": ":",
        "grid.alpha": 0.4,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
    })


# Colorblind-safe Tableau 10
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # rose
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#999999",  # grey
]


def _color_for(family: str, idx: int = 0) -> str:
    table = {
        "L0": _PALETTE[7],
        "L1": _PALETTE[0],
        "L2": _PALETTE[2],
        "L3": _PALETTE[1],
    }
    return table.get(family, _PALETTE[idx % len(_PALETTE)])


def _figsize(kind: str = "single") -> tuple[float, float]:
    # Shared `figsize` logic reused by multiple benchmark paths.
    return {
        "single":      (3.5, 2.4),
        "single_tall": (3.5, 3.0),
        "double":      (7.16, 3.0),
        "double_tall": (7.16, 4.5),
    }[kind]


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(x, default: float | None = None) -> float | None:
    try:
        if x in ("", None):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _by_variant(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    # process records in deterministic order
    for r in rows:
        out.setdefault(r.get("variant", "?"), []).append(r)
    return out


def _load_llm_judge() -> dict[tuple[str, str], dict]:
    # Shared `load_llm_judge` logic reused by multiple benchmark paths.
    path = REPO / "results" / "llm_judge.csv"

    # skip whenever path.exists() is missing/false
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict] = {}

    with path.open(encoding="utf-8") as f:
        # process records in deterministic order
        for r in csv.DictReader(f):
            try:
                score = int(r.get("llm_judge_score") or 0)
            except (ValueError, TypeError):
                continue
            if score == 0:
                continue
            out[(r["variant"], str(r["scenario_id"]))] = {
                "score": score,
                "pass": int(r.get("llm_judge_pass") or 0),
            }
    return out


def _accuracy_for(variant: str, scenario_rows: list[dict],
                  judge: dict[tuple[str, str], dict]) -> float | None:
    passes = [
        judge[(variant, str(r["scenario_id"]))]["pass"]
        for r in scenario_rows
        if (variant, str(r["scenario_id"])) in judge
    ]
    return (sum(passes) / len(passes)) if passes else None


def _no_judge_warning(fig, ax, msg: str = (
    "No LLM-judge data found at results/llm_judge.csv.\n"
    "Run: uv run python -m benchmark.llm_judge"
)) -> None:
    fig.text(
        0.5, 0.99, msg, ha="center", va="top",
        fontsize=7, color="darkred",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "darkred"},
    )
    fig.subplots_adjust(top=0.84)


def _variant_metadata(name: str) -> dict[str, str]:
    # `variant_metadata` lives here so benchmark steps read top-down.
    n = name.lower()

    # runs when 'l0' in n
    if "l0" in n:
        family = "L0"
    elif "l1" in n:
        family = "L1"
    elif "l2" in n:
        family = "L2"
    elif "l3" in n:
        family = "L3"
    else:
        family = "?"

    # runs when 'l0' in n and 'llama' in n
    if "l0" in n and "llama" in n:
        short = "L0-llama"
    elif "l0" in n:
        short = "L0"
    elif "domain" in n and "w4a16" in n and "llama" in n:
        short = "L1d-llama"
    elif "domain" in n and "w4a16" in n:
        short = "L1d"
    elif "generic" in n and "w4a16" in n and "llama" in n:
        short = "L1g-llama"
    elif "generic" in n and "w4a16" in n:
        short = "L1g"
    elif "full_bundle" in n and "llama" in n:
        short = "L2-llama"
    elif "full_bundle" in n:
        short = "L2"
    elif "image_512" in n and "llama" in n:
        short = "L3-512-llama"
    elif "image_512" in n:
        short = "L3-512"
    else:
        short = name

    model_family = "llama" if "llama" in n else "qwen"
    return {"family": family, "short": short, "model_family": model_family}


# Plot 1: Pareto
def plot_pareto(out_path: Path, summary: list[dict]) -> bool:
    # `plot_pareto` lives here so benchmark steps read top-down.
    import matplotlib.pyplot as plt

    judge = _load_llm_judge()
    by_variant = _by_variant(summary)
    points = []
    # Build one scatter point per variant once judge scores exist.
    for v_name, scenario_rows in by_variant.items():
        e2e_vals = [_to_float(r["e2e_ms"], 0) or 0 for r in scenario_rows]
        valid_e2e = [v for v in e2e_vals if v > 0]

        # skip whenever valid_e2e is missing/false
        if not valid_e2e:
            continue
        accuracy = _accuracy_for(v_name, scenario_rows, judge)

        # runs when accuracy is None
        if accuracy is None:
            continue  # skip variants without LLM-judge data
        mean_e2e = sum(valid_e2e) / len(valid_e2e)
        meta = _variant_metadata(v_name)
        points.append({
            **meta,
            "variant": v_name,
            "accuracy": accuracy,
            "mean_e2e_ms": mean_e2e,
        })

    # skip whenever points is missing/false
    if not points:
        return False

    fig, ax = plt.subplots(figsize=(7.16, 3.45))
    seen_families: set[str] = set()
    seen_model_families: set[str] = set()

    label_offsets = [(8, 8), (8, -13), (-42, 8), (-42, -13)]
    manual_offsets = {
        "L0": (-42, -16),
        "L0-llama": (-54, 12),
        "L2": (8, 10),
        "L2-llama": (10, -18),
        "L3-512": (8, 14),
        "L3-512-llama": (8, -16),
    }
    used_offsets: dict[tuple[float, float], int] = {}

    def _bucket(p):
        # CLI/helper entry for `bucket`.
        return (round(p["mean_e2e_ms"] / 500) * 500, round(p["accuracy"] * 20) / 20)

    for p in sorted(points, key=lambda x: (x["mean_e2e_ms"], x["accuracy"])):
        family = p["family"]
        marker = "o" if p["model_family"] == "qwen" else "s"
        color = _color_for(family)
        # Stagger labels so dense clusters stay readable.
        seen_families.add(family)
        seen_model_families.add(p["model_family"])
        ax.scatter(
            p["mean_e2e_ms"], p["accuracy"],
            s=46, color=color, marker=marker,
            edgecolor="black", linewidth=0.6, zorder=3,
        )
        b = _bucket(p)
        idx = used_offsets.get(b, 0)
        used_offsets[b] = idx + 1
        offset = manual_offsets.get(p["short"], label_offsets[idx % len(label_offsets)])
        ax.annotate(
            p["short"], (p["mean_e2e_ms"], p["accuracy"]),
            xytext=offset, textcoords="offset points", fontsize=7,
            arrowprops={
                "arrowstyle": "-", "color": "gray", "lw": 0.4, "alpha": 0.6,
            } if abs(offset[0]) > 10 else None,
        )

    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_ylabel("LLM-judge accuracy (score >= 4)")
    ax.set_title("Latency-Accuracy Pareto across optimization variants")
    ax.grid(True, axis="both")
    accs = [p["accuracy"] for p in points]
    ymin = -0.04 if min(accs) <= 0.02 else max(0, min(accs) - 0.05)
    ymax = min(1.02, max(accs) + 0.10)
    ax.set_ylim(ymin, ymax)
    xs = [p["mean_e2e_ms"] for p in points]
    if xs:
        span = max(xs) - min(xs)
        pad = max(350, span * 0.06)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)

    handles = []
    labels = []
    # process records in deterministic order
    for fam in sorted(seen_families):
        handles.append(plt.Line2D([], [], marker="o", linestyle="",
                                  color=_color_for(fam), markersize=6,
                                  markeredgecolor="black", markeredgewidth=0.6))
        labels.append(fam)
    if len(seen_model_families) > 1:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color="white",
                                  markeredgecolor="black", markersize=6))
        labels.append("Qwen")
        handles.append(plt.Line2D([], [], marker="s", linestyle="", color="white",
                                  markeredgecolor="black", markersize=6))
        labels.append("Llama")
    ax.legend(handles, labels, loc="upper right", framealpha=0.9, frameon=True, ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 2: Latency distribution
def plot_latency_box(out_path: Path, summary: list[dict]) -> bool:
    # Shared `plot_latency_box` logic reused by multiple benchmark paths.
    import matplotlib.pyplot as plt

    by_variant = _by_variant(summary)

    # skip whenever by_variant is missing/false
    if not by_variant:
        return False

    ordered = sorted(by_variant.items(), key=lambda x: (_variant_metadata(x[0])["family"], x[0]))
    data, labels, colors = [], [], []
    # each pass handles the next item in the sequence
    for name, rows in ordered:
        valid = [v for v in (_to_float(r["e2e_ms"], 0) or 0 for r in rows) if v > 0]

        # skip whenever valid is missing/false
        if not valid:
            continue
        data.append(valid)
        meta = _variant_metadata(name)
        labels.append(meta["short"])
        colors.append(_color_for(meta["family"]))

    # skip whenever data is missing/false
    if not data:
        return False

    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    bp = ax.boxplot(
        data, tick_labels=labels, widths=0.55, showfliers=True, patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.6},
    )
    # process records in deterministic order
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)

    import random
    random.seed(0)
    for i, vals in enumerate(data):
        xs = [i + 1 + random.uniform(-0.18, 0.18) for _ in vals]
        ax.scatter(xs, vals, s=8, color="black", alpha=0.5, zorder=3)
    ax.set_ylabel("End-to-end latency per scenario (ms)")
    ax.set_title("Per-scenario latency distribution by variant")
    ax.grid(True, axis="y")
    ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 3: VRAM breakdown
WEIGHT_GIB = {
    "L0_baseline":                       15.0,
    "L0_llama_baseline":                 15.5,
    "L1_awq_w4a16_domain":               5.9,
    "L1_awq_w4a16_generic":              5.9,
    "L1_llama_awq_w4a16_domain":         5.9,
    "L1_llama_awq_w4a16_generic":        5.9,
    "L2_full_bundle":                    15.0,
    "L2_llama_full_bundle":              15.5,
    "L3_image_512":                      15.0,
    "L3_llama_image_512":                15.5,
}


def plot_vram_breakdown(out_path: Path, hpml: list[dict]) -> bool:
    # Helper for `plot_vram_breakdown`.
    import matplotlib.pyplot as plt

    # skip whenever hpml is missing/false
    if not hpml:
        return False

    latest_per_variant: dict[str, dict] = {}
    # process records in deterministic order
    for r in hpml:
        latest_per_variant[r.get("variant", "?")] = r

    ordered = sorted(latest_per_variant.items(), key=lambda x: (_variant_metadata(x[0])["family"], x[0]))
    labels, weight, kv = [], [], []
    # process records in deterministic order
    for name, r in ordered:
        used = (_to_float(r.get("vram_used_mib"), 0) or 0) / 1024.0

        # runs when used <= 0
        if used <= 0:
            continue
        w = WEIGHT_GIB.get(name, 0)
        labels.append(_variant_metadata(name)["short"])
        weight.append(w)
        kv.append(max(0, used - w) if w > 0 else used * 0.15)

    # skip whenever labels is missing/false
    if not labels:
        return False

    fig, ax = plt.subplots(figsize=_figsize("double"))
    x = list(range(len(labels)))
    ax.bar(x, weight, label="Model weights (GiB)", color=_PALETTE[0], edgecolor="black", linewidth=0.6)
    ax.bar(x, kv, bottom=weight, label="KV cache + activations (GiB)", color=_PALETTE[1], edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("VRAM (GiB)")
    ax.set_title("VRAM allocation: weights vs KV cache by variant")
    ax.grid(True, axis="y")
    for xi, (w, k) in enumerate(zip(weight, kv)):
        total = w + k
        if total > 0:
            ax.text(xi, total + 0.3, f"{total:.1f}", ha="center", fontsize=7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.33),
        ncol=2,
        framealpha=0.9,
        frameon=True,
    )
    ymax = max((w + k) for w, k in zip(weight, kv)) if weight else 1
    ax.set_ylim(0, ymax * 1.18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.72)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 4: TTFT + ITL
def plot_ttft_itl(out_path: Path, hpml: list[dict]) -> bool:
    # CLI/helper entry for `plot_ttft_itl`.
    import matplotlib.pyplot as plt

    # skip whenever hpml is missing/false
    if not hpml:
        return False

    latest_per_variant: dict[str, dict] = {}
    # process records in deterministic order
    for r in hpml:
        latest_per_variant[r.get("variant", "?")] = r
    ordered = sorted(latest_per_variant.items(), key=lambda x: (_variant_metadata(x[0])["family"], x[0]))

    labels, ttft50, ttft95, itl50 = [], [], [], []
    # process records in deterministic order
    for name, r in ordered:
        labels.append(_variant_metadata(name)["short"])
        ttft50.append(_to_float(r.get("ttft_ms_p50"), 0) or 0)
        ttft95.append(_to_float(r.get("ttft_ms_p95"), 0) or 0)
        itl50.append(_to_float(r.get("itl_ms_p50"), 0) or 0)

    # skip whenever labels is missing/false
    if not labels:
        return False

    itl_unavailable = all(itl == 0 for itl in itl50)

    fig, ax_ttft = plt.subplots(figsize=_figsize("double"))
    x = list(range(len(labels)))
    width = 0.35
    ax_ttft.bar([xi - width / 2 for xi in x], ttft50, width=width, label="TTFT p50 (ms)",
                color=_PALETTE[0], edgecolor="black", linewidth=0.5)
    ax_ttft.bar([xi + width / 2 for xi in x], ttft95, width=width, label="TTFT p95 (ms)",
                color=_PALETTE[1], edgecolor="black", linewidth=0.5)
    ax_ttft.set_xticks(x)
    ax_ttft.set_xticklabels(labels, rotation=30, ha="right")
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.set_title("Time-to-first-token and inter-token latency")
    ax_ttft.grid(True, axis="y")
    h1, l1 = ax_ttft.get_legend_handles_labels()
    if itl_unavailable:
        ax_ttft.legend(h1, l1, loc="upper right", framealpha=0.9, frameon=True)
        fig.text(
            0.5, 0.02,
            "ITL unavailable: vLLM histogram was empty during the metric scrape.",
            ha="center", va="bottom",
            fontsize=7, color="darkred",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "darkred"},
        )
    else:
        ax_itl = ax_ttft.twinx()
        ax_itl.spines["top"].set_visible(False)
        ax_itl.plot(x, itl50, marker="o", color=_PALETTE[2], label="ITL p50 (ms/tok)", linewidth=1.2)
        ax_itl.set_ylabel("ITL (ms/token)")
        h2, l2 = ax_itl.get_legend_handles_labels()
        ax_ttft.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9, frameon=True)

    fig.tight_layout(rect=(0, 0.08, 1, 1) if itl_unavailable else (0, 0, 1, 1))
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 5: Per-category accuracy heatmap
def plot_accuracy_per_category(out_path: Path, summary: list[dict]) -> bool:
    # CLI/helper entry for `plot_accuracy_per_category`.
    import matplotlib.pyplot as plt
    import numpy as np

    judge = _load_llm_judge()

    # skip whenever judge is missing/false
    if not judge:
        return False  # no LLM-judge data -> can't compute accuracy

    by_variant = _by_variant(summary)

    # skip whenever by_variant is missing/false
    if not by_variant:
        return False

    categories: list[str] = []
    seen = set()
    for rows in by_variant.values():
        for r in rows:
            c = r.get("category") or "Unknown"
            if c not in seen:
                seen.add(c)
                categories.append(c)
    variants = sorted(by_variant.keys(), key=lambda v: (_variant_metadata(v)["family"], v))

    matrix = np.zeros((len(variants), len(categories)), dtype=float)
    counts = np.zeros_like(matrix)
    for i, v in enumerate(variants):
        for r in by_variant[v]:
            sid = str(r.get("scenario_id", ""))
            if (v, sid) not in judge:
                continue
            j = categories.index(r.get("category") or "Unknown")
            counts[i, j] += 1
            matrix[i, j] += int(judge[(v, sid)]["pass"])
    accuracy = np.divide(matrix, counts, out=np.full_like(matrix, np.nan), where=counts > 0)

    fig, ax = plt.subplots(figsize=_figsize("double_tall"))
    im = ax.imshow(accuracy, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([_variant_metadata(v)["short"] for v in variants])
    for i in range(len(variants)):
        for j in range(len(categories)):
            v = accuracy[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}\nn={int(counts[i,j])}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v < 0.4 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("LLM-judge accuracy (score >= 4)")
    ax.set_title("Per-category LLM-judge accuracy (variant x category)")
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 6: Calibration ablation
def plot_calibration_compare(out_path: Path, summary: list[dict]) -> bool:
    # `plot_calibration_compare` lives here so benchmark steps read top-down.
    import matplotlib.pyplot as plt

    by_variant = _by_variant(summary)
    candidate_triplets = (
        ("Qwen2.5-VL-7B", (
            "L0_baseline",
            "L1_awq_w4a16_domain",
            "L1_awq_w4a16_generic",
        )),
        ("Llama-3-LLaVA-NeXT-8B", (
            "L0_llama_baseline",
            "L1_llama_awq_w4a16_domain",
            "L1_llama_awq_w4a16_generic",
        )),
    )

    chosen_label: str | None = None
    targets: tuple[str, ...] | None = None
    best_present = 0
    # each pass handles the next item in the sequence
    for label, triplet in candidate_triplets:
        present = sum(1 for t in triplet if t in by_variant)
        if present > best_present and present >= 2:
            chosen_label = label
            targets = triplet
            best_present = present
            if present == 3:
                break

    # runs when targets is None
    if targets is None:
        return False

    judge = _load_llm_judge()

    # skip whenever judge is missing/false
    if not judge:
        return False  # accuracy panel can't render without LLM-judge data
    labels = ["FP16", "AWQ INT4 domain", "AWQ INT4 generic"]
    accuracy: list[float | None] = []
    mean_e2e: list[float] = []
    p50_e2e: list[float] = []
    runaway: list[int] = []
    for t in targets:
        rows = by_variant.get(t, [])
        # only enter this block when the guard passes
        if not rows:
            accuracy.append(None); mean_e2e.append(0); p50_e2e.append(0); runaway.append(0)
            continue
        e2e_vals = [_to_float(r["e2e_ms"], 0) or 0 for r in rows]
        valid = [v for v in e2e_vals if v > 0]
        accuracy.append(_accuracy_for(t, rows, judge))
        mean_e2e.append(sum(valid) / len(valid) if valid else 0)
        p50_e2e.append(sorted(valid)[len(valid) // 2] if valid else 0)
        runaway.append(sum(1 for v in valid if v > 60_000))

    fig, axes = plt.subplots(1, 2, figsize=_figsize("double"))
    x = list(range(len(targets)))
    colors = [_color_for("L0"), _color_for("L1"), _PALETTE[3]]

    ax_a = axes[0]
    accuracy_for_plot = [a if a is not None else 0 for a in accuracy]
    ax_a.bar(x, accuracy_for_plot, color=colors, edgecolor="black", linewidth=0.6)
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, rotation=20, ha="right")
    ax_a.set_ylabel("LLM-judge accuracy (score >= 4)")
    ax_a.set_ylim(0, 1)
    ax_a.set_title("Accuracy")
    # each pass handles the next item in the sequence
    for xi, a in zip(x, accuracy):
        if a is None:
            ax_a.text(xi, 0.04, "no\njudge", ha="center", va="bottom",
                      fontsize=7, color="gray")
        else:
            ax_a.text(xi, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax_a.grid(True, axis="y")

    ax_l = axes[1]
    width = 0.35
    ax_l.bar([xi - width / 2 for xi in x], mean_e2e, width=width, label="mean E2E (ms)",
             color=_PALETTE[0], edgecolor="black", linewidth=0.5)
    ax_l.bar([xi + width / 2 for xi in x], p50_e2e, width=width, label="p50 E2E (ms)",
             color=_PALETTE[1], edgecolor="black", linewidth=0.5)
    ax_l.set_xticks(x); ax_l.set_xticklabels(labels, rotation=20, ha="right")
    ax_l.set_ylabel("Latency (ms)")
    ax_l.set_title("Latency")
    ax_l.grid(True, axis="y")
    for xi, n in zip(x, runaway):
        if n > 0:
            ax_l.text(xi, max(mean_e2e[xi], p50_e2e[xi]) * 1.05,
                      f"{n} runaway", ha="center", fontsize=7, color="red")
    ax_l.legend(loc="upper right", framealpha=0.9, frameon=True)
    fig.suptitle(f"Domain vs generic AWQ calibration - {chosen_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Plot 7: Family comparison
def plot_family_compare(out_path: Path, summary: list[dict]) -> bool:
    # `plot_family_compare` lives here so benchmark steps read top-down.
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    by_variant = _by_variant(summary)
    targets: list[tuple[str, str, str]] = []

    # runs when 'L0_baseline' in by_variant
    if "L0_baseline" in by_variant:
        targets.append(("L0_baseline",         "Qwen FP16",        "qwen"))

    # runs when 'L1_awq_w4a16_domain' in by_variant
    if "L1_awq_w4a16_domain" in by_variant:
        targets.append(("L1_awq_w4a16_domain", "Qwen INT4 domain", "qwen"))

    # runs when 'L0_llama_baseline' in by_variant
    if "L0_llama_baseline" in by_variant:
        targets.append(("L0_llama_baseline",         "Llama FP16",        "llama"))

    # runs when 'L1_llama_awq_w4a16_domain' in by_variant
    if "L1_llama_awq_w4a16_domain" in by_variant:
        targets.append(("L1_llama_awq_w4a16_domain", "Llama INT4 domain", "llama"))

    avail = [(name, label, fam) for name, label, fam in targets if name in by_variant]

    # runs when len(avail) < 2
    if len(avail) < 2:
        return False

    judge = _load_llm_judge()
    labels: list[str] = []
    accuracy: list[float] = []
    mean_e2e: list[float] = []
    families: list[str] = []
    # each pass handles the next item in the sequence
    for name, label, fam in avail:
        rows = by_variant.get(name, [])
        e2e_vals = [_to_float(r["e2e_ms"], 0) or 0 for r in rows]
        valid = [v for v in e2e_vals if v > 0]
        acc = _accuracy_for(name, rows, judge)
        accuracy.append(acc if acc is not None else 0)
        mean_e2e.append(sum(valid) / len(valid) if valid else 0)
        labels.append(label)
        families.append(fam)

    # skip whenever any((_accuracy_for(name, by_variant[name], judge... is missing/false
    if not any(_accuracy_for(name, by_variant[name], judge) is not None
               for name, _, _ in avail):
        return False

    fig, ax = plt.subplots(figsize=(7.16, 3.25))
    colors = [_PALETTE[0] if f == "qwen" else _PALETTE[3] for f in families]
    ax.scatter(mean_e2e, accuracy, s=80, color=colors, edgecolor="black", linewidth=0.7, zorder=3)
    for xi, yi, lab in zip(mean_e2e, accuracy, labels):
        offsets = {
            "Qwen FP16": (12, 10),
            "Llama FP16": (-72, 10),
            "Qwen INT4 domain": (12, 10),
            "Llama INT4 domain": (14, 8),
        }
        offset = offsets.get(lab, (8, 6))
        ax.annotate(
            lab, (xi, yi), xytext=offset, textcoords="offset points", fontsize=8,
            arrowprops={
                "arrowstyle": "-", "color": "gray", "lw": 0.4, "alpha": 0.6,
            } if offset[0] < 0 else None,
        )
    ax.set_xlabel("Mean end-to-end latency (ms)")
    ax.set_ylabel("LLM-judge accuracy (score >= 4)")
    ax.set_title("Cross-family comparison: Qwen vs Llama (FP16 vs AWQ INT4)")
    ax.grid(True, axis="both")
    ax.set_ylim(-0.02, max(accuracy) + 0.1 if accuracy else 1.0)
    if mean_e2e:
        span = max(mean_e2e) - min(mean_e2e)
        pad = max(350, span * 0.08)
        ax.set_xlim(min(mean_e2e) - pad, max(mean_e2e) + pad)
    handles = [
        mpatches.Patch(color=_PALETTE[0], label="Qwen2.5-VL-7B"),
        mpatches.Patch(color=_PALETTE[3], label="Llama-3-LLaVA-NeXT-8B"),
    ]
    ax.legend(handles=handles, loc="lower right", framealpha=0.9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


# Top-level
def plot_accuracy_compare(out_path: Path, summary: list[dict]) -> bool:
    import matplotlib.pyplot as plt

    by_variant = _by_variant(summary)

    # skip whenever by_variant is missing/false
    if not by_variant:
        return False

    judge = _load_llm_judge()
    # only enter this block when the guard passes
    if not judge:
        # empty placeholder until llm_judge has been run
        fig, ax = plt.subplots(figsize=_figsize("double"))
        ax.text(0.5, 0.5, "No LLM-judge data yet.\n"
                          "Run: uv run python -m benchmark.llm_judge",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="darkred")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("Per-variant LLM-judge accuracy")
        fig.tight_layout()
        fig.savefig(out_path)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
        return True

    ordered = sorted(by_variant.items(), key=lambda x: (_variant_metadata(x[0])["family"], x[0]))
    labels: list[str] = []
    accuracy: list[float | None] = []
    mean_score: list[float | None] = []
    # process records in deterministic order
    for v_name, rows in ordered:
        meta = _variant_metadata(v_name)
        labels.append(meta["short"])
        passes = [judge[(v_name, str(r["scenario_id"]))]["pass"] for r in rows
                  if (v_name, str(r["scenario_id"])) in judge]
        scores = [judge[(v_name, str(r["scenario_id"]))]["score"] for r in rows
                  if (v_name, str(r["scenario_id"])) in judge]
        accuracy.append(sum(passes) / len(passes) if passes else None)
        mean_score.append(sum(scores) / len(scores) if scores else None)

    fig, ax = plt.subplots(figsize=_figsize("double"))
    x = list(range(len(labels)))
    accuracy_plot = [a if a is not None else 0 for a in accuracy]
    bars = ax.bar(x, accuracy_plot, color=_PALETTE[2], edgecolor="black",
                  linewidth=0.6, label="LLM-judge accuracy (score >= 4)")
    for xi, a in zip(x, accuracy):
        if a is None:
            ax.text(xi, 0.02, "no\njudge", ha="center", va="bottom",
                    fontsize=7, color="gray")
        else:
            ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("LLM-judge accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-variant LLM-judge accuracy (gpt-4o-mini grading)")
    ax.grid(True, axis="y")

    # Mean 1-5 score on secondary axis.
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(x, [s if s is not None else 0 for s in mean_score],
             marker="o", color=_PALETTE[3], linewidth=1.4,
             label="mean score (1-5)")
    ax2.set_ylabel("Mean LLM-judge score (1-5)")
    ax2.set_ylim(0, 5.2)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


PLOTS: dict[str, dict] = {
    "pareto":                {"requires": ("summary",), "fn": plot_pareto},
    "latency_box":           {"requires": ("summary",), "fn": plot_latency_box},
    "vram_breakdown":        {"requires": ("hpml",),    "fn": plot_vram_breakdown},
    "ttft_itl":              {"requires": ("hpml",),    "fn": plot_ttft_itl},
    "accuracy_per_category": {"requires": ("summary",), "fn": plot_accuracy_per_category},
    "calibration_compare":   {"requires": ("summary",), "fn": plot_calibration_compare},
    "family_compare":        {"requires": ("summary",), "fn": plot_family_compare},
    "accuracy_compare":      {"requires": ("summary",), "fn": plot_accuracy_compare},
}


def make_all(
    out_dir: Path = PLOTS_DIR_DEFAULT,
    summary_csv: Path = SUMMARY_CSV,
    hpml_csv: Path = HPML_CSV,
) -> list[Path]:
    # Helper for `make_all`.
    _apply_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_csv(summary_csv)
    hpml = _load_csv(hpml_csv)

    written: list[Path] = []
    # process records in deterministic order
    for name, spec in PLOTS.items():
        if "summary" in spec["requires"] and not summary:
            continue

        # runs when 'hpml' in spec['requires'] and (not hpml)
        if "hpml" in spec["requires"] and not hpml:
            continue
        out_path = out_dir / f"{name}.png"

        # push risky ops here so failures stay easy to reshape
        try:
            sig_args = []
            if "summary" in spec["requires"]:
                sig_args.append(summary)
            if "hpml" in spec["requires"]:
                sig_args.append(hpml)
            ok = spec["fn"](out_path, *sig_args)
        except Exception as exc:
            print(f"   [skip] {name}: {exc}", file=sys.stderr)
            continue

        # runs when ok is False or not out_path.exists()
        if ok is False or not out_path.exists():
            print(f"   [skip] {name}: insufficient data in CSVs")
            continue
        written.append(out_path)
        print(f"   wrote {out_path}")
    return written


def main() -> int:
    # CLI/helper entry for `main`.
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=False)
    p.add_argument("--out-dir", default=str(PLOTS_DIR_DEFAULT))
    p.add_argument("--summary-csv", default=str(SUMMARY_CSV))
    p.add_argument("--hpml-csv", default=str(HPML_CSV))

    sub.add_parser("all", help="Generate all available plots")
    sub.add_parser("list", help="List plot names")
    # each pass handles the next item in the sequence
    for name in PLOTS:
        sub.add_parser(name, help=f"Generate {name}")

    args = p.parse_args()

    # runs when args.cmd == 'list'
    if args.cmd == "list":
        print("Available plots:")
        for name in PLOTS:
            print(f"  {name}")
        return 0

    cmd = args.cmd or "all"

    # runs when cmd == 'all'
    if cmd == "all":
        n = make_all(
            out_dir=Path(args.out_dir),
            summary_csv=Path(args.summary_csv),
            hpml_csv=Path(args.hpml_csv),
        )
        print(f"==> generated {len(n)} plots in {args.out_dir}")
        return 0

    # runs when cmd not in PLOTS
    if cmd not in PLOTS:
        print(f"Unknown plot {cmd!r}. Try 'list'.", file=sys.stderr)
        return 2
    _apply_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _load_csv(Path(args.summary_csv))
    hpml = _load_csv(Path(args.hpml_csv))
    out_path = out_dir / f"{cmd}.png"
    sig_args = []

    # runs when 'summary' in PLOTS[cmd]['requires']
    if "summary" in PLOTS[cmd]["requires"]:
        sig_args.append(summary)

    # runs when 'hpml' in PLOTS[cmd]['requires']
    if "hpml" in PLOTS[cmd]["requires"]:
        sig_args.append(hpml)
    ok = PLOTS[cmd]["fn"](out_path, *sig_args)

    # runs when ok is False or not out_path.exists()
    if ok is False or not out_path.exists():
        print(f"==> {cmd}: insufficient data - no plot written.", file=sys.stderr)
        return 1
    print(f"==> wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

