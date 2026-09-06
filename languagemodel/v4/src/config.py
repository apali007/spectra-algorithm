"""
Central configuration for the mass spectrometry proteomics reasoning-corpus fine-tuning pipeline.

Keeping every hyperparameter and path in one dataclass (instead of scattered
module-level constants) makes it possible to:
  - log/save the exact config used for a training run (see run_config.json below)
  - override individual fields from the CLI without touching this file
  - reuse the same config object across train.py, inference.py, and eval scripts
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # --- Model ---
    model_id: str = "unsloth/qwen2.5-14b-instruct-unsloth-bnb-4bit"
    max_seq_length: int = 31210
    max_new_tokens: int = 6144
    temperature: float = 0.1
    inference_batch_size: int = 4

    # --- LoRA ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_rslora: bool = True

    # --- Training ---
    epochs: int = 3
    per_device_batch_size: int = 1
    grad_accum_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    seed: int = 42

    # --- Evaluation / checkpointing
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    eval_steps: int = 400
    save_steps: int = 400
    eval_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    report_to: str = "wandb"
    wandb_project: str = "proteomics-reasoning-corpus"
    save_total_limit: int = 2
    load_best_model_at_end: bool = False

    # --- Multi-GPU (see next step) ---
    # Effective batch size = per_device_batch_size * grad_accum_steps * num_gpus.
    # No field needed here yet — num_gpus is determined by the launcher (torchrun),
    # not something we set in config. Left as a comment marker for now.

    # --- Data paths ---
    train_set_dir: str = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/40k_mass_analyzer_aware_train.json"
    val_set_dir: str = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/40k_mass_analyzer_aware_val.json"
    test_set_dir: str = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/data/processed/40k_mass_analyzer_aware_test.json"

    # --- Output ---
    output_dir: str = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/default_run"

    # --- Misc / flags ---
    full_finetune: bool = False
    push_to_hub: Optional[str] = None

    def save(self, path: str):
        """Write this config to JSON, e.g. into a checkpoint dir as run_config.json."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Reconstruct a Config from a saved run_config.json."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)