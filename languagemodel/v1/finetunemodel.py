import argparse
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MAX_SEQ_LENGTH = 3072

LORA_R = 16
LORA_ALPHA = 32

LORA_TARGET_MODULES = [
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj",
]

BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05

OUTPUT_DIR = "./spectra-chat-checkpoints"


def load_and_format(data_path, tokenizer):

    raw = load_dataset("json", data_files=data_path, split="train")

    def _format(example):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are SpectraChatModel, an expert mass spectrometry assistant. "
                    "Analyze MS/MS spectra step-by-step and conclude with the peptide."

                    "Fragmentation context.\n"
                "When a peptide is fragmented inside a mass spectrometer via collision-induced "
                "dissociation (CID), the peptide backbone cleaves preferentially at amide bonds. "
                "This produces two complementary ion series: b-ions (carrying the N-terminus) "
                "and y-ions (carrying the C-terminus, plus H2O). Taken together, the mass "
                "differences between consecutive ions in each series reveal individual amino-acid "
                "residue masses. I would infer the closest b-ions and y-ions and reason to infer a closest peptide to this spectrum."
                ),
            },
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]

        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    formatted = raw.map(_format, remove_columns=raw.column_names)

    split = formatted.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--push_to_hub", default=None)

    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=not args.full,
    )

    model.config.use_cache = False

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not args.full:

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=LORA_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=True,
        )

    train_ds, eval_ds = load_and_format(args.data, tokenizer)

    training_args = SFTConfig(

        output_dir=args.output_dir,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(100 * WARMUP_RATIO),

        weight_decay=0.01,
        optim="adamw_8bit",

        logging_steps=10,

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        report_to="none",
        seed=42,

        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(

        model=model,
        args=training_args,

        train_dataset=train_ds,
        eval_dataset=eval_ds,

        processing_class=tokenizer,
    )

    stats = trainer.train()

    final_dir = args.output_dir + "/final"

    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if not args.full:

        merged_dir = args.output_dir + "/merged_16bit"

        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )

    if args.push_to_hub:

        if args.full:

            model.push_to_hub(args.push_to_hub)
            tokenizer.push_to_hub(args.push_to_hub)

        else:

            model.push_to_hub_merged(
                args.push_to_hub,
                tokenizer,
                save_method="merged_16bit",
            )

    FastLanguageModel.for_inference(model)

    sample = """
Experimental MS/MS spectrum (m/z intensity):

88.0393 1100
175.1190 2200
290.1169 1960
417.2456 1600
645.3389 2400
716.3760 1100

Identify the peptide sequence.
"""

    messages = [
        {"role": "system", "content": "You are SpectraChatModel."},
        {"role": "user", "content": sample},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    out = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))


if __name__ == "__main__":
    main()