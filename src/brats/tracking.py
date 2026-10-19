"""Rank-zero experiment tracking.

MLflow retains the Databricks behavior. The SPCS profile requires an online
numeric-only W&B preflight, then records visible nonfatal failures locally.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("brats.tracking")


# Closed vocabularies: accepting any numeric key would leak case IDs in key names.
_CONFIG_NUMBERS = frozenset("batch_size_per_gpu epochs warmup_epochs lr weight_decay momentum grad_clip max_nonfinite_steps lambda_cls cls_warmup_epochs num_workers val_every val_max_cases seed expected_world_size prefetch_factor val_num_workers val_sw_batch_size world_size effective_batch params_total_m steps_per_epoch train_cases val_cases".split())
_CONFIG_BOOLS = frozenset({"batch_dice", "cohort_balanced_sampling"})
_CONFIG_ENUMS = {
    "optimizer": {"adamw", "sgd"}, "seg_loss": {"dice_ce", "dice_focal"},
    "loss_weighting": {"fixed", "uncertainty"},
    "pooling": {"tumor_attention", "gap", "tafe"},
    "amp_dtype": {"bfloat16", "float16", "float32"},
    "execution_profile": {"spcs", "legacy"},
}
_METRICS = {"epoch", "train/optimizer_step", "tracking/preflight", "tracking/flush_marker"}
for _prefix in ("train", "train_rank0", "epoch"):
    _METRICS.update(f"{_prefix}/{key}" for key in ("loss", "loss_seg", "loss_cls", "lr", "warmup_alpha", "seconds"))
_METRICS.update(f"runtime/{key}" for key in "attempted_updates successful_updates skipped_steps effective_batch data_seconds_mean_rank step_seconds_mean_rank train_seconds_mean_rank samples_per_second validation_seconds epoch_seconds checkpoint_seconds disk_free_bytes_rank0 rss_bytes_rank0 cpu_percent_rank0 host_available_ram_bytes_rank0 host_load1_rank0 grad_norm_rank0_last".split())
_METRICS.update(f"runtime/peak_vram_bytes_rank_{rank}" for rank in range(4))
_METRICS.update({"loss_weight/seg", "loss_weight/cls", "val/cls_acc", "val/cls_balanced_acc"})
for _cohort in ("", "GLI_", "MEN_", "PED_"):
    for _region in ("ET", "TC", "WT", "avg", "mean"):
        _METRICS.update(f"val/{kind}{_cohort}{_region}" for kind in ("dice_", "legacy_dice_"))
_SUMMARIES = {f"best_dice_{region}{suffix}" for region in ("ET", "TC", "WT", "avg", "mean") for suffix in ("", "_epoch")}


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _private_config(config: dict) -> dict:
    clean = {}
    for key, value in config.items():
        if key in _CONFIG_NUMBERS and _number(value):
            clean[key] = value
        elif key in _CONFIG_BOOLS and type(value) is bool:
            clean[key] = value
        elif key in _CONFIG_ENUMS and isinstance(value, str) and value in _CONFIG_ENUMS[key]:
            clean[key] = value
        elif key == "patch_size" and isinstance(value, (list, tuple)) and len(value) == 3 and all(type(v) is int and 0 < v <= 1024 for v in value):
            clean[key] = list(value)
    return clean


class WandbLogger:
    """Required online preflight, then visible, nonfatal runtime tracking errors.

    Numeric-only mode never forwards user names, groups, tags, paths or arbitrary
    keys. SDK event steps are independent of replayed optimizer steps. Local
    health/metric files are authoritative and must be published by the caller.
    """

    def __init__(self, enabled: bool = False, project: str = "brats2023",
                 run_name: str | None = None, mode: str = "offline",
                 entity: str | None = None, config: Any = None,
                 group: str | None = None, tags: list[str] | None = None,
                 dir: str | Path | None = None, is_rank_zero: bool = True,
                 required_online: bool = False, numeric_only: bool = False,
                 run_id: str | None = None, resume: bool = False,
                 attempt_id: str | None = None) -> None:
        self.enabled = bool(enabled) and is_rank_zero and mode != "disabled"
        self.run = None
        self.numeric_only = numeric_only
        self.run_id = run_id or uuid.uuid4().hex
        self.logging_sequence = 0
        self.tracking_ok = True
        self.flush_ok = False
        self.remote_history_verified = False
        self.errors = 0
        self.finish_timeout_seconds = 30.0
        self.attempt_id = attempt_id  # local provenance only
        self.health_path = self.metrics_path = None
        self._last_metrics = {}
        if not is_rank_zero:
            return
        if required_online and (not self.enabled or mode != "online" or not numeric_only):
            raise ValueError("required online tracking needs enabled online numeric-only mode")
        if not self.enabled:
            return
        if numeric_only:
            if project != "brats2023":
                raise ValueError("numeric-only tracking requires project brats2023")
            if not re.fullmatch(r"[a-f0-9]{32}", self.run_id):
                raise ValueError("numeric-only W&B ID must be a generated 32-character hex UUID")
        if required_online and (not entity or not os.environ.get("WANDB_API_KEY", "").strip()):
            raise RuntimeError("W&B preflight requires an explicit entity and injected API key")
        run_dir = Path(dir) if dir else Path("runs") / "wandb"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.health_path = run_dir / "tracking-health.json"
        self.metrics_path = run_dir / "tracking-metrics.jsonl"
        cfg = asdict(config) if is_dataclass(config) else (config or {})
        try:
            import wandb
            self._wandb = wandb
            settings = None
            if numeric_only:
                # Override environment defaults, including inherited developer settings.
                os.environ.update(WANDB_CONSOLE="off", WANDB_DISABLE_CODE="true",
                                  WANDB_DISABLE_GIT="true", WANDB_SAVE_CODE="false",
                                  WANDB_SILENT="true", WANDB_MODE=mode)
                # Do not inherit free-form metadata or sweep configuration.
                for variable in ("WANDB_NOTES", "WANDB_RUN_GROUP", "WANDB_JOB_TYPE",
                                 "WANDB_SWEEP_ID", "WANDB_CONFIG_PATHS", "WANDB_RUN_NAME",
                                 "WANDB_TAGS", "WANDB_PROGRAM", "WANDB_PROGRAM_ARGS",
                                 "WANDB_GIT_REMOTE_URL", "WANDB_GIT_COMMIT", "WANDB_HOST"):
                    os.environ.pop(variable, None)
                settings = wandb.Settings(
                    console="off", disable_code=True, disable_git=True, save_code=False,
                    x_disable_stats=True, x_disable_meta=True,
                    x_disable_machine_info=True, x_disable_viewer=True,
                    x_save_requirements=False, disable_job_creation=True,
                    host="spcs", program="brats.train", git_remote_url="", git_commit="",
                    x_files_dir=str(run_dir / "private-sdk-files"),
                    ignore_globs=["*", "**/*"], init_timeout=60,
                    x_service_wait=30, x_file_stream_timeout_seconds=30,
                    x_file_stream_retry_max=2, x_file_transfer_retry_max=2,
                    x_file_transfer_timeout_seconds=30,
                )
            self.run = wandb.init(
                project=project, entity=entity,
                name=f"brats-{self.run_id[:12]}" if numeric_only else run_name,
                mode=mode, config=_private_config(cfg) if numeric_only else cfg,
                group=None if numeric_only else group,
                tags=["numeric-only"] if numeric_only else (tags or []),
                notes="" if numeric_only else None, job_type="train" if numeric_only else None,
                dir=str(run_dir), resume="must" if resume else "never",
                id=self.run_id, settings=settings,
            )
            if self.run is None or (required_online and self.run.settings.mode != "online"):
                raise RuntimeError("online W&B run not established")
            # The SDK reconciles the server high-water on resume. Never pass a
            # restored optimizer counter as W&B's event step.
            self.logging_sequence = int(self.run.step)
            self.run.log({"tracking/preflight": 1}, commit=True)
            self.logging_sequence = int(self.run.step)
            self._write_health()
        except Exception as exc:
            self.enabled = False
            self._error("preflight", exc)
            if required_online:
                raise RuntimeError("W&B required-online preflight failed; see local health") from None
            self.run = None

    def state_dict(self) -> dict:
        return {"wandb_run_id": self.run_id, "logging_sequence": self.logging_sequence,
                "tracking_ok": self.tracking_ok, "flush_ok": self.flush_ok,
                "remote_history_verified": self.remote_history_verified, "errors": self.errors}

    def _write_health(self) -> None:
        if self.health_path is not None:
            # Local disk only: stage snapshots are closed files, never this append log.
            self.health_path.write_text(json.dumps(self.state_dict(), sort_keys=True) + "\n")

    def _error(self, operation: str, exc: Exception) -> None:
        self.tracking_ok = False
        self.errors += 1
        # Exception messages can contain credentials/URLs/data; report type only.
        log.warning("W&B %s failed (%s); local tracking health is authoritative", operation, type(exc).__name__)
        try:
            self._write_health()
        except OSError:
            log.error("W&B local health snapshot write failed")

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        clean = {key: value for key, value in metrics.items()
                 if _number(value) and (not self.numeric_only or key in _METRICS)}
        if step is not None and _number(step):
            clean["train/optimizer_step"] = step
        if not clean:
            return
        try:
            if self.metrics_path is not None:
                with self.metrics_path.open("a") as stream:
                    stream.write(json.dumps({"event": self.logging_sequence, "metrics": clean}) + "\n")
            if self.run is not None:
                self.run.log(clean, commit=True)
                self.logging_sequence = int(self.run.step)
                self._last_metrics = clean
            self._write_health()
        except Exception as exc:
            self._error("log", exc)

    def log_config_update(self, updates: dict[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.config.update(_private_config(updates) if self.numeric_only else updates,
                                   allow_val_change=True)
        except Exception as exc:
            self._error("config", exc)

    def watch(self, model, log_freq: int = 500) -> None:
        if self.numeric_only or not self.enabled or self.run is None:
            return
        try:
            self._wandb.watch(model, log_freq=log_freq, log="gradients")
        except Exception as exc:
            self._error("watch", exc)

    def log_artifact(self, path: str | Path, name: str, type_: str = "model") -> None:
        if self.numeric_only or not self.enabled or self.run is None:
            return
        try:
            artifact = self._wandb.Artifact(name=name, type=type_)
            artifact.add_file(str(path))
            self.run.log_artifact(artifact)
        except Exception as exc:
            self._error("artifact", exc)

    def summary(self, key: str, value: Any) -> None:
        if not self.enabled or self.run is None:
            return
        if self.numeric_only and (key not in _SUMMARIES or not _number(value)):
            return
        try:
            self.run.summary[key] = value
        except Exception as exc:
            self._error("summary", exc)

    def finish(self) -> bool:
        if not self.enabled or self.run is None:
            return self.flush_ok
        run = self.run
        errors = []
        def flush():
            try:
                run.finish()
            except Exception as exc:
                errors.append(exc)
        try:
            worker = threading.Thread(target=flush, daemon=True, name="brats-wandb-flush")
            worker.start()
            worker.join(self.finish_timeout_seconds)
            if worker.is_alive():
                raise TimeoutError("W&B flush deadline exceeded")
            if errors:
                raise errors[0]
            self.flush_ok = True
            # SDK completion is not independent remote history/inventory verification.
            # A timed-out SDK child is eventually cleaned up by the job supervisor.
        except Exception as exc:
            self._error("finish", exc)
        finally:
            self.run = None
            try:
                self._write_health()
            except OSError as exc:
                self._error("health", exc)
        return self.flush_ok and self.tracking_ok

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *exc_info) -> None:
        self.finish()


class MLflowLogger:
    """MLflow tracker for Databricks. Same interface as WandbLogger.

    When the env var ``MLFLOW_RUN_ID`` is set the logger attaches to that
    existing run instead of creating a new one. This lets the notebook cell
    own the run lifecycle (create before training, end after evaluation) while
    the training subprocess logs metrics into the same run. The logger will
    *not* call ``end_run()`` for externally-managed runs.

    System-metrics monitoring is disabled by default
    (``MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false``) because its trace data
    causes the Databricks experiment page to render the GenAI observability
    view instead of the classic Runs / Charts view.
    """

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
        self._externally_managed = False

        if not self.enabled:
            return

        try:
            import mlflow
            self._mlflow = mlflow
        except ImportError:
            log.warning("mlflow not installed; continuing without tracking.")
            self.enabled = False
            return

        # Disable system-metrics monitor to keep the classic MLflow UI.
        os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

        try:
            cfg_dict = asdict(config) if is_dataclass(config) else (config or {})
            params = {}
            for k, v in cfg_dict.items():
                if isinstance(v, (dict, list, tuple)):
                    params[k] = str(v)
                else:
                    params[k] = v

            # If a run was already started by the notebook cell, attach to it.
            run_id = os.environ.get("MLFLOW_RUN_ID")
            if run_id:
                self._run = mlflow.start_run(run_id=run_id)
                self._externally_managed = True
                mlflow.log_params(params)
                log.info("mlflow attached to existing run (run_id=%s)", run_id)
            else:
                exp = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                if exp:
                    mlflow.set_experiment(exp)
                self._run = mlflow.start_run(run_name=run_name)
                mlflow.log_params(params)
                log.info("mlflow run %r started (run_id=%s)",
                         run_name, self._run.info.run_id)
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

    def log_figure(self, fig: Any, artifact_file: str) -> None:
        """Log a matplotlib figure directly as an image artifact."""
        if not self.enabled or self._run is None:
            return
        try:
            self._mlflow.log_figure(fig, artifact_file)
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                log.warning("mlflow figure logging failed (%s)", exc)
                self._warned = True

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
        if self._externally_managed:
            # Run lifecycle is owned by the notebook cell; don't end it here.
            self._run = None
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
        required_online=getattr(cfg, "require_online_tracking", False),
        numeric_only=getattr(cfg, "numeric_only_tracking", False),
        run_id=getattr(cfg, "wandb_run_id", None) or None,
        resume=getattr(cfg, "wandb_resume", False),
        attempt_id=getattr(cfg, "attempt_id", None),
        dir=(Path(cfg.ckpt_dir) / cfg.run_id / cfg.attempt_id / "tracking")
        if getattr(cfg, "execution_profile", "legacy") == "spcs" else None,
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
