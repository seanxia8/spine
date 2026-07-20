"""Structured scalar logging manager."""

import os
from collections.abc import Mapping
from typing import Any

import numpy as np
import psutil

from spine.io.write.csv import CSVWriter
from spine.utils.logger import logger
from spine.utils.torch import runtime

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

__all__ = ["LogManager"]


class LogManager:
    """Manage structured scalar logs for a driver-like processing loop.

    The manager writes one flat scalar row to CSV on every call and can mirror
    numeric entries to TensorBoard. It also owns the human-readable progress
    table printed periodically during training or inference.
    """

    def __init__(
        self,
        file_name: str,
        overwrite: bool = False,
        buffer_size: int = 1,
        tensorboard: bool | Mapping[str, Any] | None = None,
        tensorboard_dir: str | None = None,
        wandb: bool | Mapping[str, Any] | None = None,
        wandb_dir: str | None = None,
        cfg: dict[str, Any] | None = None,
        main_process: bool = True,
    ) -> None:
        """Initialize scalar logging backends.

        Parameters
        ----------
        file_name : str
            CSV log file path.
        overwrite : bool, default False
            If ``True``, overwrite an existing CSV log file.
        buffer_size : int, default 1
            CSV writer buffer size.
        tensorboard : bool | Mapping[str, Any] | None, optional
            TensorBoard logging configuration. ``False`` or ``None`` disable
            TensorBoard logging, ``True`` uses default settings, and a mapping
            forwards keyword arguments to the TensorBoard writer.
        tensorboard_dir : str | None, optional
            Default TensorBoard event-file directory. If ``tensorboard`` is a
            mapping with a ``log_dir`` key, that value takes precedence.
        wandb : bool | Mapping[str, Any] | None, optional
            Weights & Biases logging configuration. ``False`` or ``None``
            disable W&B, ``True`` uses default settings, and a mapping
            forwards keyword arguments to ``wandb.init`` (e.g. ``project``,
            ``entity``, ``name``, ``tags``, ``run_dir``).
        wandb_dir : str | None, optional
            Default local directory for W&B run files. If ``wandb`` is a
            mapping with a ``run_dir`` key, that value takes precedence.
            Falls back to the directory of ``file_name``.
        cfg : dict[str, Any] | None, optional
            Full SPINE configuration dictionary logged as W&B run config.
        main_process : bool, default True
            If ``False``, W&B and TensorBoard are suppressed (only the CSV
            backend is active). Set to ``False`` on non-main distributed ranks.
        """
        self.csv_logger = CSVWriter(
            file_name, overwrite=overwrite, buffer_size=buffer_size
        )
        if tensorboard and wandb:
            raise ValueError(
                "Only one metric-logging backend can be active at a time. "
                "Set either 'tensorboard' or 'wandb' in the config, not both."
            )

        self.main_process = main_process
        self.tb_logger = self.initialize_tensorboard_logger(
            tensorboard if main_process else None, tensorboard_dir
        )
        self.wandb_run = self.initialize_wandb_logger(
            wandb if main_process else None,
            wandb_dir or os.path.dirname(file_name),
            cfg,
        )

    @staticmethod
    def initialize_tensorboard_logger(
        tensorboard: bool | Mapping[str, Any] | None,
        tensorboard_dir: str | None = None,
    ) -> Any | None:
        """Initialize an optional TensorBoard summary writer.

        Parameters
        ----------
        tensorboard : bool | Mapping[str, Any] | None
            TensorBoard logging configuration.
        tensorboard_dir : str | None, optional
            Default TensorBoard event-file directory.

        Returns
        -------
        Any | None
            TensorBoard summary writer instance when enabled, otherwise
            ``None``.
        """
        if not tensorboard:
            return None

        tb_cfg = {} if tensorboard is True else dict(tensorboard)
        tb_dir = tb_cfg.pop("log_dir", None)
        if tb_dir is None:
            tb_dir = tensorboard_dir
        elif not os.path.isabs(tb_dir) and tensorboard_dir is not None:
            tb_dir = os.path.join(os.path.dirname(tensorboard_dir), tb_dir)

        if tb_dir is None:
            raise ValueError(
                "A TensorBoard log directory is required when TensorBoard "
                "logging is enabled."
            )

        return runtime.create_summary_writer(tb_dir, **tb_cfg)

    @staticmethod
    def initialize_wandb_logger(
        wandb: bool | Mapping[str, Any] | None,
        wandb_dir: str | None = None,
        cfg: dict[str, Any] | None = None,
    ) -> Any | None:
        """Initialize an optional Weights & Biases run.

        Parameters
        ----------
        wandb : bool | Mapping[str, Any] | None
            W&B logging configuration.
        wandb_dir : str | None, optional
            Default local directory for W&B run files.
        cfg : dict[str, Any] | None, optional
            Full SPINE config logged as the W&B run config.

        Returns
        -------
        Any | None
            Active ``wandb.Run`` when enabled, otherwise ``None``.
        """
        if not wandb:
            return None

        if not _WANDB_AVAILABLE:
            logger.warning(
                "wandb block found in config but the `wandb` package is not "
                "installed. Run `pip install wandb` to enable W&B logging."
            )
            return None

        wb_cfg = {} if wandb is True else dict(wandb)
        run_dir = wb_cfg.pop("run_dir", wandb_dir)
        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)

        run = _wandb.init(
            config=cfg,
            dir=run_dir,
            settings=_wandb.Settings(start_method="fork"),
            **wb_cfg,
        )
        logger.info("Weights & Biases run initialized: %s", run.url)
        return run

    def append(
        self,
        data: Mapping[str, Any],
        watch: Any,
        iteration: int,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Collect and write one scalar log row.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data products returned by the processing loop.
        watch : object
            Stopwatch manager with ``items`` and ``time`` methods.
        iteration : int
            Iteration counter.
        epoch : float | None, optional
            Progress in the training loop measured in epochs.

        Returns
        -------
        dict[str, Any]
            Flat log row written to all enabled structured backends.
        """
        log_row = self.collect(data, watch, iteration, epoch)
        self.csv_logger.append(log_row)
        if self.tb_logger is not None:
            self.append_tensorboard(log_row, iteration)
        elif self.wandb_run is not None:
            self.append_wandb(log_row, iteration)
        return log_row

    def collect(
        self,
        data: Mapping[str, Any],
        watch: Any,
        iteration: int,
        epoch: float | None = None,
    ) -> dict[str, Any]:
        """Collect scalar iteration metrics into one flat log row.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data products returned by the processing loop.
        watch : object
            Stopwatch manager with ``items`` and ``time`` methods.
        iteration : int
            Iteration counter.
        epoch : float | None, optional
            Progress in the training loop measured in epochs.

        Returns
        -------
        dict[str, Any]
            Flat row of scalar values ready to be written to logging backends.
        """
        first_entry = get_first_entry(data["index"])
        log_row = {"iter": iteration, "epoch": epoch, "first_entry": first_entry}
        log_row.update(self.get_memory_metrics())
        log_row.update(self.get_watch_metrics(watch))

        for key, value in data.items():
            if np.isscalar(value):
                log_row[key] = value
            elif runtime.is_tensor(value) and value.dim() == 0:
                log_row[key] = value.item()

        return log_row

    @staticmethod
    def get_memory_metrics() -> dict[str, float]:
        """Collect CPU and GPU memory metrics for the current process."""
        metrics = {
            "cpu_mem": psutil.virtual_memory().used / 1.0e9,
            "cpu_mem_perc": psutil.virtual_memory().percent,
            "gpu_mem": 0.0,
            "gpu_mem_perc": 0.0,
        }
        if runtime.cuda_is_available():
            gpu_total = runtime.cuda_mem_info()[-1] / 1.0e9
            metrics["gpu_mem"] = runtime.cuda_max_memory_allocated() / 1.0e9
            metrics["gpu_mem_perc"] = 100 * metrics["gpu_mem"] / gpu_total

        return metrics

    @staticmethod
    def get_watch_metrics(watch: Any) -> dict[str, float]:
        """Flatten stopwatch timings into loggable scalar metrics."""
        metrics: dict[str, float] = {}
        suffix = "_time"
        for key, timer in watch.items():
            time_iter, time_sum = timer.time, timer.time_sum
            metrics[f"{key}{suffix}"] = time_iter.wall
            metrics[f"{key}{suffix}_cpu"] = time_iter.cpu
            metrics[f"{key}{suffix}_sum"] = time_sum.wall
            metrics[f"{key}{suffix}_sum_cpu"] = time_sum.cpu

        return metrics

    def append_tensorboard(self, log_row: Mapping[str, Any], iteration: int) -> None:
        """Write collected scalar metrics to TensorBoard."""
        for key, value in log_row.items():
            if key == "iter":
                continue
            if isinstance(value, bool):
                self.tb_logger.add_scalar(key, int(value), iteration)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                self.tb_logger.add_scalar(key, float(value), iteration)

    def append_wandb(self, log_row: Mapping[str, Any], iteration: int) -> None:
        """Write collected scalar metrics to Weights & Biases."""
        wandb_dict = {
            k: v for k, v in log_row.items()
            if isinstance(v, (bool, int, float, np.integer, np.floating))
            and not (isinstance(v, float) and np.isnan(v))
        }
        self.wandb_run.log(wandb_dict, step=iteration)

    @staticmethod
    def log_stdout_summary(
        log_row: Mapping[str, Any],
        data: Mapping[str, Any],
        watch: Any,
        tstamp: str,
        iteration: int,
        epoch: float | None,
        model_train: bool,
        rank: int | None,
        distributed: bool,
        main_process: bool,
    ) -> None:
        """Emit the human-readable iteration summary to stdout.

        Parameters
        ----------
        log_row : Mapping[str, Any]
            Flat scalar row produced by :meth:`collect`.
        data : Mapping[str, Any]
            Original data products used to fetch common display metrics.
        watch : object
            Stopwatch manager with a ``time`` method.
        tstamp : str
            Timestamp string associated with the iteration.
        iteration : int
            Iteration counter.
        epoch : float | None
            Progress in the training loop measured in epochs.
        model_train : bool
            Whether the current model, if any, is in training mode.
        rank : int | None
            Current process rank. ``None`` indicates CPU/single-process mode.
        distributed : bool
            Whether distributed synchronization is active.
        main_process : bool
            Whether this process should print shared headers and blank lines.
        """
        proc = "train" if model_train else "inference"
        device = "GPU" if rank is not None else "CPU"
        keys = [f"Time ({proc})", f"{device} memory", "Loss", "Accuracy"]
        widths = [20, 20, 9, 9]
        if distributed:
            keys = ["Rank"] + keys
            widths = [5] + widths
        if main_process:
            epoch_value = -1.0 if epoch is None else epoch
            header = "  | " + "| ".join(
                [f"{keys[i]:<{widths[i]}}" for i in range(len(keys))]
            )
            separator = "  |" + "+".join(["-" * (w + 1) for w in widths])
            msg = f"Iter. {iteration} (epoch {epoch_value:.3f}) @ {tstamp}\n"
            msg += header + "|\n"
            msg += separator + "|"
            logger.info(msg)
        if distributed:
            runtime.distributed_barrier()

        t_iter = watch.time("iteration").wall
        t_net = 0.0
        if "model_time" in log_row:
            t_net = watch.time("model").wall
        net_fraction = 0.0 if t_iter == 0.0 else 100 * t_net / t_iter

        if rank is not None:
            mem, mem_perc = log_row["gpu_mem"], log_row["gpu_mem_perc"]
        else:
            mem, mem_perc = log_row["cpu_mem"], log_row["cpu_mem_perc"]

        acc = data.get("accuracy", -1.0)
        loss = data.get("loss", -1.0)
        values = [
            f"{t_iter:0.2f} s ({net_fraction:0.2f} %)",
            f"{mem:0.2f} GB ({mem_perc:0.2f} %)",
            f"{loss:0.3f}",
            f"{acc:0.3f}",
        ]
        if distributed:
            values = [f"{rank}"] + values

        msg = "  | " + "| ".join(
            [f"{values[i]:<{widths[i]}}" for i in range(len(keys))]
        )
        msg += "|"
        if distributed:
            rows = runtime.distributed_all_gather_object((rank, msg))
            if main_process:
                for _, row_msg in sorted(
                    rows, key=lambda item: -1 if item[0] is None else item[0]
                ):
                    logger.info(row_msg)
                logger.info("")
            return

        logger.info(msg)
        if main_process:
            logger.info("")

    def close(self) -> None:
        """Flush and close all owned logging backends."""
        self.csv_logger.close()
        if self.tb_logger is not None:
            self.tb_logger.flush()
            self.tb_logger.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def get_first_entry(index: Any) -> Any:
    """Return the first entry identifier from a scalar or sequence index."""
    if isinstance(index, (list, tuple)):
        return index[0]
    if isinstance(index, np.ndarray) and index.ndim > 0:
        return index[0]
    return index
