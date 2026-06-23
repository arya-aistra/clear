"""Data-Free Knowledge Distillation trainer.

Pipeline (no external labeled data):
  1. SyntheticAudioGenerator produces a refreshable POOL of clips.
  2. SileroTeacher labels the pool (per-chunk soft probs) via audio_forward.
  3. ClearVADModel.forward_sequence learns to match the teacher (DFKDLoss).

Two stages (configs/train/dfkd_stage{1,2}.yaml): stage 1 = soft-label alignment only;
stage 2 = + boundary + smoothness, more transition-heavy data.

Teacher runs on CPU; student trains on GPU. The pool is labeled in mini-batches and
refreshed every `refresh_every` steps for diversity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from clearvad import CHUNK_SAMPLES
from clearvad.distill.losses import DFKDLoss, agreement_rate
from clearvad.distill.synthetic_gen import SyntheticAudioGenerator
from clearvad.distill.teacher import SileroTeacher
from clearvad.utils.logging_utils import get_logger, write_csv, write_json

LOG = get_logger("dfkd")


class DataPool:
    """Refreshable pool of synthetic clips + teacher soft labels (kept on CPU)."""

    def __init__(self, teacher: SileroTeacher, generator: SyntheticAudioGenerator,
                 pool_size: int, clip_chunks: int, label_batch: int = 256) -> None:
        self.teacher = teacher
        self.gen = generator
        self.pool_size = pool_size
        self.clip_chunks = clip_chunks
        self.clip_samples = clip_chunks * CHUNK_SAMPLES
        self.label_batch = label_batch
        self.audio: Optional[torch.Tensor] = None      # [P, L] cpu
        self.probs: Optional[torch.Tensor] = None       # [P, K] cpu
        self.categories: List[str] = []

    def refresh(self, seed: int, category_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        torch_ = self.teacher.torch
        audio_np, cats = self.gen.generate_batch(
            self.pool_size, self.clip_samples, seed=seed, category_weights=category_weights)
        audio = torch_.from_numpy(audio_np)
        probs_chunks = []
        for i in range(0, self.pool_size, self.label_batch):
            probs_chunks.append(self.teacher.label(audio[i:i + self.label_batch]))
        self.audio = audio
        self.probs = torch_.cat(probs_chunks, dim=0)
        self.categories = cats
        # report teacher activation (fraction of frames labeled speech) — the DFKD health check
        speech_frac = float((self.probs > 0.5).float().mean())
        return {"pool_seed": seed, "teacher_speech_frac": round(speech_frac, 4)}

    def sample(self, batch_size: int, rng: np.random.Generator, device):
        idx = rng.integers(0, self.pool_size, size=batch_size)
        idx_t = torch.from_numpy(idx)
        audio_b = self.audio.index_select(0, idx_t)
        windows = self.teacher.build_student_windows(audio_b).to(device)
        probs_b = self.probs.index_select(0, idx_t).to(device)
        return windows, probs_b


class DFKDTrainer:
    def __init__(self, model, teacher: SileroTeacher, generator: SyntheticAudioGenerator,
                 device: str = "cuda", out_dir: str = "checkpoints") -> None:
        self.model = model
        self.teacher = teacher
        self.gen = generator
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model.to(self.device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self._holdout = None  # (windows, teacher_probs) fixed eval batch

    # ------------------------------------------------------------ eval helpers
    def _ensure_holdout(self, n: int = 64, clip_chunks: int = 64):
        if self._holdout is not None:
            return
        audio_np, _ = self.gen.generate_batch(n, clip_chunks * CHUNK_SAMPLES, seed=999999)
        audio = self.teacher.torch.from_numpy(audio_np)
        probs = self.teacher.label(audio)
        windows = self.teacher.build_student_windows(audio)
        self._holdout = (windows, probs)

    @torch.no_grad()
    def evaluate_agreement(self, windows=None, teacher_probs=None, threshold: float = 0.5) -> float:
        """Teacher-student agreement at threshold on a (held-out) batch."""
        self.model.eval()
        if windows is None:
            self._ensure_holdout()
            windows, teacher_probs = self._holdout
        windows = windows.to(self.device)
        logits = self.model.forward_sequence(windows, return_logit=True)
        student_probs = torch.sigmoid(logits).cpu()
        self.model.train()
        return agreement_rate(student_probs, teacher_probs.cpu(), threshold)

    @torch.no_grad()
    def evaluate_transfer_real(self, real_audio, threshold: float = 0.5) -> Dict[str, float]:
        """HONEST-REFRAME hook: teacher-student agreement on REAL audio.

        real_audio: [B, L] tensor/ndarray of real speech @16k. Verifies the student
        reproduces the teacher on real audio (not just the synthetic training distribution).
        Returns agreement + teacher speech fraction.
        """
        torch_ = self.teacher.torch
        if isinstance(real_audio, np.ndarray):
            real_audio = torch_.from_numpy(np.ascontiguousarray(real_audio, dtype=np.float32))
        if real_audio.dim() == 1:
            real_audio = real_audio.unsqueeze(0)
        # trim to a whole number of chunks
        K = real_audio.shape[1] // CHUNK_SAMPLES
        real_audio = real_audio[:, : K * CHUNK_SAMPLES]
        teacher_probs = self.teacher.label(real_audio)
        windows = self.teacher.build_student_windows(real_audio)
        agree = self.evaluate_agreement(windows, teacher_probs, threshold)
        return {"transfer_agreement": round(agree, 4),
                "teacher_speech_frac": round(float((teacher_probs > 0.5).float().mean()), 4)}

    # ------------------------------------------------------------ training
    def run_stage(self, cfg: Dict[str, Any], stage_name: str = "stage1",
                  init_from: Optional[str] = None) -> Dict[str, Any]:
        torch_ = torch
        if init_from:
            self.model.load_state_dict(torch_.load(init_from, map_location=self.device))
            LOG.info("Loaded init weights from %s", init_from)

        steps = int(cfg["steps"])
        batch_size = int(cfg["batch_size"])
        clip_chunks = int(cfg.get("chunks_per_sample", 64))
        lr = float(cfg["lr"])
        wd = float(cfg.get("weight_decay", 1e-4))
        pool_size = int(cfg.get("pool_size", 2048))
        refresh_every = int(cfg.get("refresh_every", 2000))
        log_every = int(cfg.get("log_every", 50))
        agree_every = int(cfg.get("agree_every", 1000))
        ckpt_every = int(cfg.get("ckpt_every", 5000))
        cat_weights = cfg.get("category_weights")

        criterion = DFKDLoss(
            lambda_soft=cfg.get("lambda_soft", 1.0),
            lambda_boundary=cfg.get("lambda_boundary", 0.0),
            lambda_smooth=cfg.get("lambda_smooth", 0.0),
            temperature=cfg.get("temperature", 2.0),
            boundary_width=cfg.get("boundary_width", 3),
            boundary_weight=cfg.get("boundary_weight", 5.0),
        )
        opt = torch_.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        sched = torch_.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

        pool = DataPool(self.teacher, self.gen, pool_size, clip_chunks)
        rng = np.random.default_rng(int(cfg.get("seed", 1234)))

        self.model.train()
        LOG.info("[%s] steps=%d batch=%d pool=%d device=%s", stage_name, steps,
                 batch_size, pool_size, self.device)
        for step in range(steps):
            if step % refresh_every == 0:
                info = pool.refresh(seed=int(cfg.get("seed", 1234)) + step,
                                    category_weights=cat_weights)
                LOG.info("[%s] pool refresh @%d  teacher_speech_frac=%.3f",
                         stage_name, step, info["teacher_speech_frac"])

            windows, teacher_probs = pool.sample(batch_size, rng, self.device)
            student_logits = self.model.forward_sequence(windows, return_logit=True)
            loss, parts = criterion(student_logits, teacher_probs)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch_.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            opt.step()
            sched.step()

            if step % log_every == 0:
                row = {"stage": stage_name, "step": step, "lr": sched.get_last_lr()[0], **parts}
                if step % agree_every == 0:
                    row["agreement"] = round(self.evaluate_agreement(), 4)
                self.history.append(row)
                LOG.info("[%s] step %d  total=%.4f soft=%.4f bnd=%.4f sm=%.4f%s",
                         stage_name, step, parts["total"], parts["soft"], parts["boundary"],
                         parts["smooth"],
                         f"  agree={row['agreement']:.3f}" if "agreement" in row else "")

            if ckpt_every and step > 0 and step % ckpt_every == 0:
                self.save(f"{stage_name}_step{step}.pt")

        final_agree = self.evaluate_agreement()
        self.save(f"{stage_name}_final.pt")
        LOG.info("[%s] DONE final agreement=%.4f", stage_name, final_agree)
        return {"stage": stage_name, "final_agreement": final_agree, "steps": steps}

    # ------------------------------------------------------------ io
    def save(self, name: str) -> Path:
        p = self.out_dir / name
        torch.save(self.model.state_dict(), p)
        return p

    def dump_history(self, csv_path: str, json_path: Optional[str] = None) -> None:
        if self.history:
            write_csv(self.history, csv_path)
        if json_path:
            write_json(self.history, json_path)
