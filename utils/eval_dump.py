from __future__ import annotations

import random
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


@dataclass
class SnapshotJob:
    prompt: str
    seed: int
    frames: int
    resolution: tuple[int, int]
    steps: Optional[int] = None
    guidance: Optional[float] = None


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
        self.mode = 'teacher_forced'
        self.eval_prompts: list[str] = []
        self.snapshot_seed: int | str | None = None
        self.snapshot_frames: int = 49
        self.snapshot_resolution: tuple[int, int] | None = None
        self.snapshot_steps: int | None = None
        self.snapshot_guidance: float | None = None
        self.snapshot_init_seed: int = -1
        self._eval_seed_base: Optional[int] = None

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
        self.mode = cfg.get('mode', 'teacher_forced')
        self.eval_prompts = list(cfg.get('eval_prompts', []))
        seed_cfg = cfg.get('snapshot_seed')
        if isinstance(seed_cfg, str) and seed_cfg.isdigit():
            self.snapshot_seed = int(seed_cfg)
        else:
            self.snapshot_seed = seed_cfg
        self.snapshot_frames = int(cfg.get('snapshot_frames', self.snapshot_frames))
        resolution = cfg.get('snapshot_resolution')
        if resolution and len(resolution) == 2:
            self.snapshot_resolution = (int(resolution[0]), int(resolution[1]))
        else:
            self.snapshot_resolution = None
        steps_cfg = cfg.get('snapshot_steps')
        self.snapshot_steps = int(steps_cfg) if steps_cfg is not None else None
        guidance_cfg = cfg.get('snapshot_guidance')
        self.snapshot_guidance = float(guidance_cfg) if guidance_cfg is not None else None
        self.snapshot_init_seed = int(cfg.get('snapshot_init_seed', self.snapshot_init_seed))

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
        return self.wants_teacher_forced()

    def wants_teacher_forced(self) -> bool:
        return self.enabled and self.mode in {'teacher_forced', 'both'}

    def wants_inference_snapshots(self) -> bool:
        return self.enabled and self.mode in {'inference', 'both'} and len(self.eval_prompts) > 0

    def begin_eval(self, step: int):
        if not self.enabled:
            return
        self._state = EvalDumpState(step=step, samples_written=0, active=True)
        self._eval_seed_base = self._compute_eval_seed_base()

    def end_eval(self):
        self._state = EvalDumpState()
        self._eval_seed_base = None

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

    def build_snapshot_jobs(self, step: int) -> list[SnapshotJob]:
        if not self.wants_inference_snapshots():
            return []
        jobs: list[SnapshotJob] = []
        width, height = (self.snapshot_resolution or (832, 480))
        for idx, prompt in enumerate(self.eval_prompts):
            seed = self._resolve_snapshot_seed(idx, step)
            jobs.append(
                SnapshotJob(
                    prompt=prompt,
                    seed=seed,
                    frames=self.snapshot_frames,
                    resolution=(width, height),
                    steps=self.snapshot_steps,
                    guidance=self.snapshot_guidance,
                )
            )
        return jobs

    def _resolve_snapshot_seed(self, prompt_idx: int, step: int) -> int:
        seed_cfg = self.snapshot_seed
        if isinstance(seed_cfg, int):
            base = seed_cfg
        elif isinstance(seed_cfg, str):
            lowered = seed_cfg.lower()
            if lowered == 'constant':
                base = self._eval_seed_base if self._eval_seed_base is not None else self._compute_eval_seed_base()
            elif lowered == 'step':
                base = step
            else:
                base = self._random_seed()
        else:
            base = self._eval_seed_base if self._eval_seed_base is not None else self._compute_eval_seed_base()
        return int(base) + prompt_idx

    def _compute_eval_seed_base(self) -> int:
        if self.snapshot_init_seed is not None and self.snapshot_init_seed >= 0:
            return int(self.snapshot_init_seed)
        return self._random_seed()

    def _random_seed(self) -> int:
        return random.randint(0, 2**31 - 1)

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
