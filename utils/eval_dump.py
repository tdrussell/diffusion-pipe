from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EvalDumpState:
    step: Optional[int] = None
    dataset: Optional[str] = None
    quantile: Optional[float] = None
    samples_written: int = 0
    active: bool = False


class EvalDumpManager:
    """Simple runtime holder for eval-dump configuration/state."""

    def __init__(self):
        self.enabled = False
        self.caption_max_bytes = 160
        self.source_max_bytes = 256
        self.max_per_eval = 0
        self.output_dir: Optional[Path] = None
        self.video_format = "mp4"
        self.video_fps = 15
        self.write_metadata_json = True
        self._config: dict = {}
        self._state = EvalDumpState()
        self.group_by = 'eval'

    def apply_config(self, cfg: Optional[dict]):
        cfg = cfg or {}
        self._config = cfg
        self.enabled = bool(cfg.get('enabled', False))
        self.caption_max_bytes = int(cfg.get('caption_max_bytes', self.caption_max_bytes))
        self.source_max_bytes = int(cfg.get('source_max_bytes', self.source_max_bytes))
        self.max_per_eval = int(cfg.get('max_per_eval', cfg.get('max_samples', 0)))
        self.video_format = cfg.get('format', self.video_format)
        self.video_fps = int(cfg.get('video_fps', self.video_fps))
        self.write_metadata_json = bool(cfg.get('write_metadata_json', True))
        self.group_by = cfg.get('group_by', 'eval')

    def set_run_dir(self, run_dir: Path | str):
        if not self.enabled:
            return
        base = self._config.get('output_dir')
        path = Path(base) if base else Path(run_dir) / 'eval_samples'
        if not path.is_absolute():
            path = Path(run_dir) / path
        path.mkdir(parents=True, exist_ok=True)
        self.output_dir = path

    def requires_metadata(self) -> bool:
        return self.enabled

    def begin_eval(self, step: int):
        if not self.enabled:
            return
        self._state = EvalDumpState(step=step, samples_written=0, active=True)

    def end_eval(self):
        self._state = EvalDumpState()

    def set_dataset(self, name: str):
        if not self.enabled:
            return
        self._state.dataset = name

    def set_quantile(self, quantile: float):
        if not self.enabled:
            return
        self._state.quantile = quantile

    def remaining_slots(self) -> int:
        if not self.enabled or self.max_per_eval <= 0:
            return 1_000_000_000
        return max(self.max_per_eval - self._state.samples_written, 0)

    def should_record(self) -> bool:
        return (
            self.enabled
            and self.output_dir is not None
            and self._state.active
            and self.remaining_slots() > 0
            and self._state.dataset is not None
            and self._state.quantile is not None
        )

    def increment(self, count: int):
        if not self.enabled:
            return
        self._state.samples_written += count

    def current_context(self) -> EvalDumpState:
        return self._state

    def group_by_sample(self) -> bool:
        return self.group_by == 'sample'

    def group_flat(self) -> bool:
        return self.group_by == 'flat'

    def build_output_dir(self, sample_slug: Optional[str] = None) -> Optional[Path]:
        if not self.should_record():
            return None
        dataset = self._state.dataset or "dataset"
        quantile = self._state.quantile if self._state.quantile is not None else 0.0
        step = self._state.step or 0
        if self.group_by_sample() and sample_slug:
            path = self.output_dir / dataset / f"quantile_{quantile:.2f}" / sample_slug
        elif self.group_flat():
            path = self.output_dir
        else:
            path = self.output_dir / dataset / f"step_{step}" / f"quantile_{quantile:.2f}"
        path.mkdir(parents=True, exist_ok=True)
        return path


eval_dump_manager = EvalDumpManager()
