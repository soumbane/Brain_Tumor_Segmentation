"""Local reimplementation of the BraTS ranking scheme.

Why this exists. Ferreira et al. (arXiv 2402.17317, GLI 1st place) reimplemented the
challenge ranking locally to choose their submission, because *"selecting the
solution with the best DSC and/or HD95 is not the best approach."* Their
best-mean-Dice post-processing threshold (WT 1450) scored highest on their own
validation, and they judged it too risky and shipped WT 250 instead.

The ranking is rank-based, not mean-based, which changes what wins:

1. For each case, each metric, each region: rank all competing configurations.
2. Average those ranks per configuration to get a "rank score".
3. Lower is better.

Because it ranks per case, a configuration that is slightly better on most cases
beats one that is much better on a few and worse on the rest. Mean Dice rewards the
opposite. Tuning post-processing thresholds to mean Dice therefore selects
configurations that the challenge metric would have rejected -- which is exactly the
trap Ferreira documented.

Used here to select among post-processing configurations on the **validation** split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from brats.constants import REGIONS

#: Metrics entering the rank, with the direction that counts as better.
RANK_METRICS: dict[str, str] = {"lw_dice": "higher", "lw_hd95": "lower"}


def rank_configurations(
    scores: pd.DataFrame,
    config_col: str = "config",
    case_col: str = "case_id",
    region_col: str = "region",
    metrics: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Rank configurations the way the BraTS leaderboard does.

    Args:
        scores: Long-form frame with one row per (config, case, region) and one
            column per metric.
        metrics: Metric name -> ``"higher"`` or ``"lower"`` is better.

    Returns:
        One row per configuration with ``rank_score`` (lower is better) plus the
        per-region mean ranks, so it is visible *where* a configuration wins.
    """
    metrics = metrics or RANK_METRICS
    missing = [m for m in metrics if m not in scores.columns]
    if missing:
        raise ValueError(f"scores is missing metric columns: {missing}")

    frames: list[pd.DataFrame] = []
    for metric, direction in metrics.items():
        ascending = direction == "lower"
        # Rank configurations within each (case, region) cell.
        ranked = (
            scores.groupby([case_col, region_col])[metric]
            .rank(ascending=ascending, method="average")
            .rename("rank")
        )
        block = scores[[config_col, case_col, region_col]].copy()
        block["rank"] = ranked.to_numpy()
        block["metric"] = metric
        frames.append(block)

    allranks = pd.concat(frames, ignore_index=True)

    overall = (
        allranks.groupby(config_col)["rank"].mean().rename("rank_score").reset_index()
    )
    per_region = (
        allranks.pivot_table(
            index=config_col, columns=region_col, values="rank", aggfunc="mean"
        )
        .rename(columns={r: f"rank_{r}" for r in REGIONS})
        .reset_index()
    )
    per_metric = (
        allranks.pivot_table(
            index=config_col, columns="metric", values="rank", aggfunc="mean"
        )
        .rename(columns=lambda c: f"rank_{c}")
        .reset_index()
    )

    out = overall.merge(per_region, on=config_col).merge(per_metric, on=config_col)
    return out.sort_values("rank_score").reset_index(drop=True)


def select_best_config(
    scores: pd.DataFrame,
    config_col: str = "config",
    risk_margin: float = 0.02,
    **kwargs,
) -> tuple[str, pd.DataFrame]:
    """Pick a configuration by rank, preferring conservative ties.

    Ferreira's judgement call, encoded: when several configurations sit within
    ``risk_margin`` of the best rank score, prefer the one whose *worst-case* behavior
    is mildest rather than the one with the best average. An aggressive
    small-component threshold can win on validation and then fail on unseen data --
    validation consistently needs larger thresholds than training folds.

    Returns:
        ``(chosen_config, ranking_table)``. The table always carries every candidate,
        so the choice is auditable.
    """
    table = rank_configurations(scores, config_col=config_col, **kwargs)
    best = float(table.loc[0, "rank_score"])
    contenders = table[table.rank_score <= best + risk_margin][config_col].tolist()

    if len(contenders) == 1:
        return contenders[0], table

    # Tie-break on the 5th-percentile Dice: how bad the bad cases are.
    worst_case = (
        scores[scores[config_col].isin(contenders)]
        .groupby(config_col)["lw_dice"]
        .quantile(0.05)
        .sort_values(ascending=False)
    )
    table = table.assign(
        tiebreak_p05_dice=table[config_col].map(worst_case),
        in_tie=table[config_col].isin(contenders),
    )
    return str(worst_case.index[0]), table


def ablation_table(scores: pd.DataFrame, stage_col: str = "stage") -> pd.DataFrame:
    """Reproduce BiomedMBZ's post-processing ablation on our own predictions.

    Shows lesion-wise and legacy Dice side by side per stage. The point of the table
    is the *contrast*: in the published version legacy Dice moved +0.34 across the
    whole pipeline while lesion-wise Dice moved +9.97. If our table does not show a
    similar divergence, the post-processing is not doing its job.
    """
    rows = []
    for stage, sub in scores.groupby(stage_col, sort=False):
        row: dict[str, float | str | int] = {stage_col: stage, "n_cases": sub.case_id.nunique()}
        for metric in ("lw_dice", "legacy_dice", "lw_hd95"):
            if metric in sub.columns:
                row[f"{metric}_mean"] = float(np.nanmean(sub[metric]))
                row[f"{metric}_median"] = float(np.nanmedian(sub[metric]))
        rows.append(row)
    out = pd.DataFrame(rows)
    if "lw_dice_mean" in out.columns and len(out) > 1:
        out["lw_dice_delta"] = out["lw_dice_mean"] - out["lw_dice_mean"].iloc[0]
    if "legacy_dice_mean" in out.columns and len(out) > 1:
        out["legacy_dice_delta"] = (
            out["legacy_dice_mean"] - out["legacy_dice_mean"].iloc[0]
        )
    return out


__all__ = [
    "rank_configurations",
    "select_best_config",
    "ablation_table",
    "RANK_METRICS",
]
