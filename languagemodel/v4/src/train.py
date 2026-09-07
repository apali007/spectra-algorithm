import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import os
import torch

from config import Config
from constants import SYSTEM_PROMPT, SAMPLE_INPUT

def preprocess_logits_for_metrics(logits, labels):
    """
    Runs once per eval batch, immediately reducing the huge
    (batch, seq_len, vocab_size) logits down to just the model's
    top guess per position (batch, seq_len). Without this, the trainer
    would try to hold full-vocabulary logits for the whole eval set in memory
    before computing accuracy - a real risk of runninf out of memory
    given Qwen's large vocabulary
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    """
    Cheap, token-evel accuracy: for each position in the assistant's
    response, did the model's top guess token match the true token?
    This is NOT the same as full peptide-sequence accuracy (that requires
    actually generating text, which is inference.py's job, rum separately
    on the heldout test set after training). This is a fast proxy signal
    computed for free from the same forward pass eval_loss already uses.
    """
    predictions, labels = eval_pred
    predictions = predictions[:, :-1]
    labels = labels[:, 1:]

    mask = labels != -100
    correct = ((predictions == labels) & mask).sum()
    total = mask.sum()
    accuracy = correct / total if total > 0 else 0.0
    return {"token_accuracy": float(accuracy)}




def load_and_format(cfg: Config, tokenizer):
    """Loads train/val/test JSON into chat-formatted text the model expects."""

    def _format(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        }

    train_raw = load_dataset("json", data_files = cfg.train_set_dir, split="train")
    val_raw = load_dataset("json", data_files=cfg.val_set_dir, split="train")

    train_formatted = train_raw.map(_format, remove_columns=train_raw.column_names)
    val_formatted = val_raw.map(_format, remove_columns=val_raw.column_names)

    return train_formatted, val_formatted


def build_argparser(cfg: Config) -> argparse.ArgumentParser:
    """CLI overrides layered on top of the Config's defaults. Only exposes the
    knobs you actually tend to change run-to-run; everything else lives in
    config.py and can be edited there directly."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", default=cfg.full_finetune)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.learning_rate)
    parser.add_argument("--output_dir", default=cfg.output_dir)
    parser.add_argument("--train_set_dir", default=cfg.train_set_dir)
    parser.add_argument("--val_set_dir", default=cfg.val_set_dir)
    parser.add_argument("--push_to_hub", default=cfg.push_to_hub)

    return parser

def main():

    os.environ["WANDB_MODE"] = "offline"

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    cfg = Config()

    parser = build_argparser(cfg)
    args = parser.parse_args()

    cfg.full_finetune = args.full
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.output_dir = args.output_dir
    cfg.train_set_dir = args.train_set_dir
    cfg.val_set_dir = args.val_set_dir
    cfg.push_to_hub = args.push_to_hub

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = cfg.model_id,
        max_seq_length = cfg.max_seq_length,
        dtype = None,
        load_in_4bit = not cfg.full_finetune,
        device_map = {"": local_rank},
    )

    model.config.use_cache = False

    tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.eos_token_id

    if not cfg.full_finetune:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules = cfg.lora_target_modules,
            lora_alpha = cfg.lora_alpha,
            lora_dropout = cfg.lora_dropout,
            bias = "none",
            use_gradient_checkpointing = False,
            random_state = cfg.seed,
            use_rslora = cfg.use_rslora,
        )

    train_ds, eval_ds = load_and_format(cfg, tokenizer)

    training_args = SFTConfig(
        output_dir = cfg.output_dir,
        num_train_epochs = cfg.epochs,
        per_device_train_batch_size = cfg.per_device_batch_size,
        per_device_eval_batch_size = cfg.per_device_batch_size,
        gradient_accumulation_steps = cfg.grad_accum_steps,
        learning_rate = cfg.learning_rate,
        lr_scheduler_type = cfg.lr_scheduler_type,
        warmup_ratio = cfg.warmup_ratio,
        weight_decay = cfg.weight_decay,
        optim = cfg.optim,
        logging_steps = cfg.logging_steps,

        # Evaluation + checkpointing
        eval_strategy = cfg.eval_strategy,
        eval_steps = cfg.eval_steps,
        save_strategy = cfg.save_strategy,
        save_steps = cfg.save_steps,
        save_total_limit = cfg.save_total_limit,
        load_best_model_at_end = cfg.load_best_model_at_end,
        metric_for_best_model = "eval_loss",
        eval_accumulation_steps = cfg.eval_accumulation_steps,
        max_grad_norm = cfg.max_grad_norm,
        report_to = cfg.report_to,
        run_name = os.path.basename(cfg.output_dir.rstrip("/")),

        fp16= not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        seed = cfg.seed,
        dataset_text_field = "text",
        max_length = cfg.max_seq_length,
        padding_free = False,

        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = eval_ds,
        processing_class = tokenizer,
        compute_metrics = compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )

    # --- Loss masking fix ---
    """Without this (and therefore every gradient update) is computed over
    the ENTIRE formatted text - system prompt + full spectrum peak list + 
    reasoning trace - even though the system prompt and input are always
    given at inference, never generated. That dilutes the training signal
    on the part that actually matters: the reasoning trace + final sequence.
    train_on_responses_only wraps the trainer's data collator so loss is
    only computed on tokens after "<|im_start|>assistant\n"""

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

    try:

        stats = trainer.train()

    except Exception as e:
        """
        load_best_model_at_end = True triggers a real bug in this peft/transformers
        version combo at the very end of training(model.load_adapter -> broken
        EmbeddingParallel import). Training itself completes successfully before
        this point - only the final automatic reload crashes - so we catch it
        here and continue using whatever checkpoint HuggingFace already tracked
        as best throughout training (trainer.state.best_model_checkpoint),
        rather than losing the run's progress entirely.
        """
        print(f"Trainer.train() raised during final best-model reload (known bug): {type(e).__name__}: {e}")
        print("Training itself completed; continuing using the tracked best checkpoints.")


    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint:
        print(f"Best checkpoint: {best_checkpoint} (eval_loss ={trainer.state.best_metric:.5f})")
    else:
        print("No best checkpoint was tracked; using the final training state")


    # --- Multi-GPU (DDP) note ---
    """Under 'torchrun --nproc_per_node N src/train.py', this script runs once
    PER GPU process. Saving/pushing/inference below would otherwise happen
    N times redundantly (and for push_to_hub), could race). Only the main
    process (rank 0) should do these steps.
    """

    if trainer.is_world_process_zero():

        final_dir = cfg.output_dir + "/final"

        if best_checkpoint and not cfg.full_finetune:
            '''
            The live 'model' object may hold the LAST checkpoint's weights, not the
            BEST one --  the automatic in-place reaload (model.load_adapter) crashed
            due to a known peft/transformers bug before it could swap weights.
            A fresh load from disk (same pattern inference.py already uses) avoids
            that broken code path entirely.
            '''
            print(f"Reloading best checkpoint fresh from disk: {best_checkpoint}")
            del model
            torch.cuda.empty_cache()
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = best_checkpoint,
                max_seq_length = cfg.max_seq_length,
                dtype = None,
                load_in_4bit = True,
            )
    
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
    
        if not cfg.full_finetune:
            merged_dir = cfg.output_dir + "/merged_16bit"
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method = "merged_16bit"
            )

        """
        Save the exact config used for this run, so the checkpoint dir is
        self-describing later (which dataset, which prompt-adjacent settings,
        which hyperparamters produced this model).
        """
        cfg.save(f"{cfg.output_dir}/run_config.json")

        if cfg.push_to_hub:
            if cfg.full_finetune:
                model.push_to_hub(cfg.push_to_hub)
                tokenizer.push_to_hub(cfg.push_to_hub)
            else:
                model.push_to_hub_merged(
                    cfg.push_to_hub,
                    tokenizer,
                    save_method = "merged_16bit",
                )

        # Inference sanity check
        FastLanguageModel.for_inference(model)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SAMPLE_INPUT},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        out = model.generate(
            input_ids = inputs,
            max_new_tokens = cfg.max_new_tokens,
            temperature = cfg.temperature,
            do_sample = True,
            pad_token_id = tokenizer.eos_token_id,
        )

        print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()
