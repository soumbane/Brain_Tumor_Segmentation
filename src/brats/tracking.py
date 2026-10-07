"""Experiment tracking: MLflow (Databricks) and Weights & Biases.

Wrapped rather than called inline for three reasons:

1. **Rank discipline.** Only rank 0 may log. Four ranks logging to one run produces
   interleaved, unusable curves; four separate runs is worse.
2. **Never kill a training run.** A network blip, an expired key, or a missing
   integration must degrade to a no-op, not crash a job that has been
   burning GPU credits for hours. Every call here is failure-tolerant.
3. **Platform flexibility.** On Databricks, MLflow is the native tracker and is
   pre-configured. On Snowflake/SPCS, W&B in offline mode is used instead.

Enable in ``configs/segresnet_base.yaml``::

    train:
      use_mlflow: true          # Databricks native tracking
      wandb: true               # W&B tracking (can coexist with mlflow)
      wandb_project: brats2023
      wandb_mode: offline       # online | offline | disabled
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("brats.tracking")


class WandbLogger:
    """Thin, failure-tolerant W&B wrapper. A disabled instance is a silent no-op."""

    def __init__(
        self,
        enabled: bool = False,
        project: str = "brats2023",
        run_name: str | None = None,
        mode: str = "offline",
        entity: str | None = None,
        config: Any = None,
        group: str | None = None,
        tags: list[str] | None = None,
        dir: str | Path | None = None,
        is_rank_zero: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and is_rank_zero and mode != "disabled"
        self.run = None
        self._warned = False

        if not self.enabled:
            if enabled and not is_rank_zero:
                log.debug("wandb disabled on non-zero rank")
            return

        try:
            import wandb
        except ImportError:
            log.warning(
                "wandb requested but not installed; continuing without tracking. "
                "pip install wandb"
            )
            self.enabled = False
            return

        # Set before init: wandb reads these at init time.
        os.environ.setdefault("WANDB_MODE", mode)
        if mode == "offline":
            # Silence the login prompt that would otherwise block a headless container.
            os.environ.setdefault("WANDB_SILENT", "true")

        run_dir = Path(dir) if dir else Path("runs") / "wandb"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg_dict = asdict(config) if is_dataclass(config) else (config or {})
        # patch_size is a tuple; wandb prefers plain JSON-able values.
        cfg_dict = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg_dict.items()
        }

        try:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                mode=mode,
                config=cfg_dict,
                group=group,
                tags=tags or [],
                dir=str(run_dir),
                resume="allow",
                # Deterministic id so a resumed job continues the same run rather than
                # starting a second one -- essential given SPCS can cancel and we
                # restart from a stage checkpoint.
                id=run_name,
            )
            log.info("wandb run %r (mode=%s, project=%s)", run_name, mode, project)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb init failed (%s); continuing without tracking", exc)
            self.enabled = False
            self.run = None

    # -- logging ---------------------------------------------------------

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a metric dict. Silently degrades on any failure."""
        if not self.enabled or self.run is None:
            return
        try:
            clean = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float, bool)) and v == v  # drop NaN
            }
            if clean:
                self.run.log(clean, step=step)
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                log.warning("wandb logging failed (%s); further errors suppressed", exc)
                self._warned = True

    def log_config_update(self, updates: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.config.update(updates, allow_val_change=True)
        except Exception:  # noqa: BLE001
            pass

    def watch(self, model, log_freq: int = 500) -> None:
        """Track gradient/parameter histograms.

        Off by default at call sites: for a 20M-parameter 3D CNN this adds real
        overhead and large uploads for information that rarely changes a decision.
        """
        if not self.enabled or self.run is None:
            return
        try:
            import wandb

            wandb.watch(model, log_freq=log_freq, log="gradients")
        except Exception:  # noqa: BLE001
            pass

    def log_artifact(self, path: str | Path, name: str, type_: str = "model") -> None:
        """Upload an artifact (e.g. a best checkpoint).

        Note checkpoints are already mirrored to a Snowflake stage, which is the
        resumability guarantee. This is for convenience/versioning, not durability, so
        it is called only for `best_*` checkpoints rather than every epoch.
        """
        if not self.enabled or self.run is None:
            return
        try:
            import wandb

            art = wandb.Artifact(name=name, type=type_)
            art.add_file(str(path))
            self.run.log_artifact(art)
        except Exception as exc:  # noqa: BLE001
            log.warning("wandb artifact upload failed (%s)", exc)

    def summary(self, key: str, value: Any) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.summary[key] = value
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self.run = None

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()


class MLflowLogger:
    """MLflow tracker for Databricks. Same interface as WandbLogger."""

    def __init__(
        self,
        enabled: bool = True,
        run_name: str | None = None,
        config: Any = None,
        experiment_name: str | None = None,
        is_rank_zero: bool = True,
    ) -> None:
        self.enabled = bool(enabled) and is_rank_zero
        self._run = None
        self._warned = False

        if not self.enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            log.warning("mlflow not installed; continuing without tracking.")
            self.enabled = False
            return

        try:
            # Resolve experiment: explicit arg > env var > default path.
            exp = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
            if exp:
                mlflow.set_experiment(exp)

            cfg_dict = asdict(config) if is_dataclass(config) else (config or {})
            # MLflow params must be strings; flatten complex types.
            params = {}
            for k, v in cfg_dict.items():
                if isinstance(v, (dict, list, tuple)):
                    params[k] = str(v)
                else:
                    params[k] = v

            self._run = mlflow.start_run(run_name=run_name)
            mlflow.log_params(params)
            log.info("mlflow run %r started (run_id=%s)", run_name, self._run.info.run_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow init failed (%s); continuing without tracking", exc)
            self.enabled = False
            self._run = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            clean = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and v == v  # drop NaN
            }
            if clean:
                self._mlflow.log_metrics(clean, step=step)
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                log.warning("mlflow logging failed (%s); further errors suppressed", exc)
                self._warned = True

    def log_config_update(self, updates: dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            params = {k: str(v) if isinstance(v, (dict, list, tuple)) else v
                      for k, v in updates.items()}
            self._mlflow.log_params(params)
        except Exception:  # noqa: BLE001
            pass

    def watch(self, model, log_freq: int = 500) -> None:
        # MLflow does not have gradient watching; no-op.
        pass

    def log_artifact(self, path: str | Path, name: str, type_: str = "model") -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("mlflow artifact upload failed (%s)", exc)

    def summary(self, key: str, value: Any) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._mlflow.log_metrics({key: value})
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._run = None

    def __enter__(self) -> "MLflowLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()


def build_logger(cfg, is_rank_zero: bool = True) -> WandbLogger | MLflowLogger:
    """Construct the appropriate tracker from a ``TrainConfig``.

    Returns an :class:`MLflowLogger` when ``use_mlflow`` is set (Databricks),
    otherwise falls back to :class:`WandbLogger`.
    """
    if getattr(cfg, "use_mlflow", False):
        return MLflowLogger(
            enabled=True,
            run_name=cfg.run_name,
            config=cfg,
            experiment_name=getattr(cfg, "mlflow_experiment", None) or None,
            is_rank_zero=is_rank_zero,
        )
    return WandbLogger(
        enabled=getattr(cfg, "wandb", False),
        project=getattr(cfg, "wandb_project", "brats2023"),
        entity=getattr(cfg, "wandb_entity", None) or None,
        run_name=cfg.run_name,
        mode=getattr(cfg, "wandb_mode", "offline"),
        config=cfg,
        group=getattr(cfg, "wandb_group", None) or None,
        tags=_auto_tags(cfg),
        is_rank_zero=is_rank_zero,
    )


def _auto_tags(cfg) -> list[str]:
    """Tags that make the ablation arms findable without manual bookkeeping."""
    tags = [
        f"pool={getattr(cfg, 'pooling', '?')}",
        f"loss={getattr(cfg, 'seg_loss', '?')}",
        f"lambda={getattr(cfg, 'lambda_cls', '?')}",
    ]
    if getattr(cfg, "lambda_cls", None) == 0.0:
        tags.append("ablation-seg-only")
    if getattr(cfg, "pooling", None) == "gap":
        tags.append("ablation-gap-confound")
    if getattr(cfg, "loss_weighting", None) == "uncertainty":
        tags.append("uncertainty-weighting")
    return tags


__all__ = ["WandbLogger", "MLflowLogger", "build_logger"]
