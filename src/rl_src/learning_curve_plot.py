"""TensorBoard learning-curve plotting helpers.

The training scripts write TensorBoard scalar events under:

    <group>/<algo>/tb/<run>/events.out.tfevents...

This module turns those events into a Notion/report-ready PNG and overlays the
existing heuristic_best TensorBoard run as a horizontal baseline when present.
Plotting is deliberately best-effort: callers should never fail training because
matplotlib or a TensorBoard event file has a problem.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DEFAULT_TAG = "rollout/ep_rew_mean"
DEFAULT_OUTPUT_NAME = "learning_curve_ep_rew_mean.png"
REGIONAL_SUBPLOTS_OUTPUT_NAME = "learning_curves_by_region_ep_rew_mean.png"

_EXCLUDE_DIRS = {
    "checkpoints",
    "eval_logs",
    "randeval_logs",
    "run_logs",
    "tb",
}

_ALGO_ORDER = [
    "ppo",
    "ppo_reward",
    "ppo_enriched",
    "ppo_bc",
    "recurrentppo",
    "dqn",
    "qrdqn",
    "reinforce",
    "a2c",
    "trpo",
]

_COLORS = {
    "ppo": "#2563eb",
    "ppo_reward": "#4f46e5",
    "ppo_enriched": "#0f766e",
    "ppo_bc": "#7c3aed",
    "recurrentppo": "#0891b2",
    "dqn": "#dc2626",
    "qrdqn": "#ea580c",
    "reinforce": "#16a34a",
    "a2c": "#9333ea",
    "trpo": "#ca8a04",
    "heuristic_best": "#111827",
}

_REGION_EN = {
    "서울": "Seoul",
    "부산": "Busan",
    "대구": "Daegu",
    "인천": "Incheon",
    "광주": "Gwangju",
    "대전": "Daejeon",
    "울산": "Ulsan",
    "세종": "Sejong",
    "경기": "Gyeonggi",
    "강원": "Gangwon",
    "충북": "Chungbuk",
    "충남": "Chungnam",
    "전북": "Jeonbuk",
    "전남": "Jeonnam",
    "경북": "Gyeongbuk",
    "경남": "Gyeongnam",
    "제주": "Jeju",
}

_REGION_ORDER = [
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
]


def _sort_key(path: Path):
    name = path.name
    if name == "heuristic_best":
        return (1, 999, name)
    try:
        return (0, _ALGO_ORDER.index(name), name)
    except ValueError:
        return (0, 998, name)


def _read_scalar_series(run_dir: Path, tag: str):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    rows = []
    for event_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        try:
            acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:
            continue
        if tag not in acc.Tags().get("scalars", []):
            continue
        rows.extend((int(s.step), float(s.value), str(event_path)) for s in acc.Scalars(tag))

    if not rows:
        return []

    # Keep the last value for a duplicated step. This handles resumed runs.
    latest_by_step = {}
    for step, value, event_path in sorted(rows, key=lambda r: (r[0], r[2])):
        latest_by_step[step] = value
    return sorted(latest_by_step.items())


def _smooth(values: list[float]) -> list[float]:
    n = len(values)
    if n < 8:
        return values
    span = max(5, min(80, n // 12))
    alpha = 2.0 / (span + 1.0)
    out = []
    cur = values[0]
    for v in values:
        cur = alpha * v + (1.0 - alpha) * cur
        out.append(cur)
    return out


def _iter_run_dirs(group_dir: Path) -> Iterable[Path]:
    if not group_dir.exists():
        return []
    return sorted(
        (p for p in group_dir.iterdir() if p.is_dir() and p.name not in _EXCLUDE_DIRS),
        key=_sort_key,
    )


def _ascii_piece(text: str) -> str:
    if text in _REGION_EN:
        return _REGION_EN[text]
    if text.isascii():
        return text
    return "region"


def _default_title(group_dir: Path) -> str:
    name = _ascii_piece(group_dir.name)
    parent = _ascii_piece(group_dir.parent.name)
    if group_dir.name == "national":
        return f"Training curve - {parent} / national"
    if group_dir.parent.name == "plan1":
        return f"Training curve - plan1 / {name}"
    return f"Training curve - {name}"


def _collect_group_series(group_dir: Path, tag: str, include_heuristic: bool):
    series = []
    heuristic = None
    for run_dir in _iter_run_dirs(group_dir):
        values = _read_scalar_series(run_dir / "tb", tag)
        if not values:
            continue
        if run_dir.name == "heuristic_best":
            if include_heuristic:
                heuristic = float(values[-1][1])
            continue
        series.append((run_dir.name, values))
    return series, heuristic


def _plot_series_on_axis(ax, series, heuristic, *, max_step: int | None = None,
                         show_raw: bool = True):
    for name, values in series:
        xs = [s for s, _ in values]
        ys = [v for _, v in values]
        color = _COLORS.get(name)
        if show_raw and len(values) > 25:
            ax.plot(xs, ys, color=color, alpha=0.14, linewidth=0.8, label="_nolegend_")
        ax.plot(xs, _smooth(ys), label=name, color=color, linewidth=2.2)

    if heuristic is not None:
        if max_step is None:
            max_step = max((values[-1][0] for _, values in series), default=1)
        max_step = max(max_step, 1)
        ax.hlines(
            heuristic,
            xmin=0,
            xmax=max_step,
            label="heuristic_best",
            color=_COLORS["heuristic_best"],
            linestyle=(0, (5, 3)),
            linewidth=2.0,
            alpha=0.9,
        )


def _dedup_legend(handles, labels):
    seen = set()
    out_h, out_l = [], []
    for handle, label in zip(handles, labels):
        if label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        out_h.append(handle)
        out_l.append(label)
    return out_h, out_l


def plot_learning_curve(
    group_dir: str | os.PathLike,
    *,
    output_path: str | os.PathLike | None = None,
    tag: str = DEFAULT_TAG,
    title: str | None = None,
    include_heuristic: bool = True,
) -> Path | None:
    """Create a learning-curve PNG for one experiment group.

    ``group_dir`` is the parent folder that contains algorithm subdirectories,
    for example ``results/rl/plan1/서울`` or ``results/rl/plan1nat/national``.
    The heuristic baseline is read from ``group_dir/heuristic_best/tb`` and
    plotted as a horizontal dashed line when available.
    """
    group_dir = Path(group_dir)
    output_path = Path(output_path) if output_path else group_dir / DEFAULT_OUTPUT_NAME

    series, heuristic = _collect_group_series(group_dir, tag, include_heuristic)

    if not series and heuristic is None:
        return None

    max_step = max((values[-1][0] for _, values in series), default=0)
    if heuristic is not None:
        max_step = max(max_step, 1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5.8), dpi=150)
    ax = plt.gca()

    _plot_series_on_axis(ax, series, heuristic, max_step=max_step)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel(tag)
    ax.set_title(title or _default_title(group_dir))
    ax.grid(True, alpha=0.25, linewidth=0.8)
    handles, labels = _dedup_legend(*ax.get_legend_handles_labels())
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=min(4, max(1, len(labels))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=0.0,
    )
    ax.margins(x=0.01)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    os.chmod(output_path, 0o644)
    plt.close()
    return output_path


def plot_regional_learning_curves(
    plan1_dir: str | os.PathLike,
    *,
    output_path: str | os.PathLike | None = None,
    tag: str = DEFAULT_TAG,
    include_heuristic: bool = True,
) -> Path | None:
    """Create one large multi-panel PNG for all plan1 regional runs."""
    plan1_dir = Path(plan1_dir)
    output_path = Path(output_path) if output_path else plan1_dir / REGIONAL_SUBPLOTS_OUTPUT_NAME

    regions = [r for r in _REGION_ORDER if (plan1_dir / r).is_dir()]
    regions.extend(
        sorted(
            p.name for p in plan1_dir.iterdir()
            if p.is_dir() and p.name not in set(regions) | {"eval_logs", "run_logs"}
        )
    )
    if not regions:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = (len(regions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.0 * nrows), dpi=150)
    axes_flat = list(axes.flat if hasattr(axes, "flat") else [axes])

    any_data = False
    legend_handles, legend_labels = [], []
    for ax, region in zip(axes_flat, regions):
        group_dir = plan1_dir / region
        series, heuristic = _collect_group_series(group_dir, tag, include_heuristic)
        if not series and heuristic is None:
            ax.axis("off")
            continue
        any_data = True
        max_step = max((values[-1][0] for _, values in series), default=1)
        _plot_series_on_axis(ax, series, heuristic, max_step=max_step, show_raw=False)
        ax.set_title(_REGION_EN.get(region, region), fontsize=12)
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.margins(x=0.01)
        handles, labels = _dedup_legend(*ax.get_legend_handles_labels())
        for handle, label in zip(handles, labels):
            if label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(label)

    for ax in axes_flat[len(regions):]:
        ax.axis("off")

    if not any_data:
        plt.close(fig)
        return None

    fig.suptitle("Training curves by region - plan1", fontsize=18, y=0.995)
    fig.supxlabel("Timesteps", y=0.055)
    fig.supylabel(tag, x=0.01)
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=min(4, max(1, len(legend_labels))),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=(0.025, 0.075, 1, 0.975), h_pad=2.0, w_pad=1.6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    os.chmod(output_path, 0o644)
    plt.close(fig)
    return output_path


def plot_learning_curve_for_training(log_dir: str | os.PathLike) -> Path | None:
    """Best-effort plot helper for train_*.py scripts.

    Training scripts receive an algorithm-specific ``--log_dir``. The report
    plot should live one level above it so multiple algorithms and
    ``heuristic_best`` share the same figure.
    """
    log_dir = Path(log_dir)
    group_dir = log_dir.parent if log_dir.parent != Path("") else log_dir
    return plot_learning_curve(group_dir)


def try_plot_learning_curve(log_dir: str | os.PathLike) -> Path | None:
    """Plot and print a warning instead of raising on optional plotting errors."""
    try:
        out = plot_learning_curve_for_training(log_dir)
        log_dir_path = Path(log_dir)
        group_dir = log_dir_path.parent if log_dir_path.parent != Path("") else log_dir_path
        if group_dir.parent.name == "plan1":
            subplot_out = plot_regional_learning_curves(group_dir.parent)
        else:
            subplot_out = None
    except Exception as exc:  # pragma: no cover - best-effort reporting helper
        print(f"[plot] skipped learning curve: {exc}")
        return None
    if out is not None:
        print(f"[plot] learning curve: {out}")
    else:
        print("[plot] skipped learning curve: no TensorBoard scalar data")
    if subplot_out is not None:
        print(f"[plot] regional subplots: {subplot_out}")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("group_dir", help="Folder containing algorithm subdirectories")
    parser.add_argument("--out", default=None)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--no_heuristic", action="store_true")
    parser.add_argument("--regional_subplots", action="store_true",
                        help="Treat group_dir as results/rl/plan1 and draw all regions")
    ns = parser.parse_args()

    if ns.regional_subplots:
        result = plot_regional_learning_curves(
            ns.group_dir,
            output_path=ns.out,
            tag=ns.tag,
            include_heuristic=not ns.no_heuristic,
        )
    else:
        result = plot_learning_curve(
            ns.group_dir,
            output_path=ns.out,
            tag=ns.tag,
            include_heuristic=not ns.no_heuristic,
        )
    if result is None:
        raise SystemExit(f"No scalar data found under {ns.group_dir}")
    print(result)
