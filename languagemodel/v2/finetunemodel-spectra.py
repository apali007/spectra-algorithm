# pass arguments from the command line
import argparse
# loads the training file into structured dataset object that the trainer understands
from datasets import load_dataset
# Unsloth is a an optimized LLM training library. FastLanguageModel loads the model efficiently with quantization. is_bfloat16_supported checks if your GPU supprots bfloat16 precision
from unsloth import FastLanguageModel, is_bfloat16_supported
# SFT trainer does supervised fine tuning. SFTConfig holds all the training hyperparameters
from trl import SFTTrainer, SFTConfig
# This applies the correct chat format for the respective language model
from unsloth.chat_templates import get_chat_template

# Model to download from Hugging face
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Maximum token length per training example
# MAX_SEQ_LENGTH = 3072
MAX_SEQ_LENGTH = 4096

# Generate up to 2048 tokens
MAX_NEW_TOKENS = 2048

TEMPERATURE = 0.1

# LoRA hyperparameters. r is the rank, controls how many parameters get updated. 
LORA_R =16
# alpha controls the learning scale. alpha/r = 2 is a standard ratio.
LORA_ALPHA = 32

# Which layers inside the transformer get the LoRA adapters. These cover both attention layers (q/k/v/o) and feed-forward layers (gate/up/down)
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

#Training settings
# Effective batch size is BATCH_SIZE * GRAD_ACCUM = 8
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
# Warmup gradually increases the learning rate at the start of training to stabalize it.
WARMUP_RATIO = 0.05

# During training, the model weights change every step. A checkpoint is a snapshot of the model at a specific training, saved to disk so you dont lose progress.
OUTPUT_DIR = "./checkpoints/spectra-chat-checkpoints-v3"

# Takes the path from JSONL/input file and the tokenizer. Returns formatted train and test datasets
def load_and_format(data_path, tokenizer):

    # loading raw data
    raw = load_dataset("json", data_files=data_path, split="train")

    # Splits 90% for training, 10% for evaluation. Seed=42 makes it reproducible
    raw_split = raw.train_test_split(test_size=0.1, seed=42)

    # save the test split with other information like ground truth, scan_number
    raw_split["test"].to_json("/lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/input_data/test_split.json")
    raw_split["train"].to_json("/lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/input_data/train_split.json")


    def _format(example):

        # structures each example as a conversation with three turns, system prompt, user input (the spectrum), and assistant output (the reasoning trace + peptide). This is the chat format for model expects.
        messages = [
            {
                "role": "system",
                  "content": (
                    "You are SpectraChatModel, an expert mass spectrometry assistant. "
                    "Given a MS/MS spectrum with precursor mass and charge, identify the peptide sequence "
                    "by reasoning step-by-step through the y-ion ladder. "
                    "Start by identifying y1 candidates, then extend the y-ion ladder residue by residue of amino acids. "
                    "If the ladder becomes ambiguous or no candidates are found, state clearly that the "
                    "peptide sequence cannot be determined. "
                    "Conclude with the complete predicted peptide sequence if derivable."
                ),
            },
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]

        # converts the messages into a single formatted string with the correct special tokens for the model. 
        # tokenize = False, returns a string, not token IDs.
        # add_generation_prompt = False because this is training data, not inference.
        return {
            "text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        }


    train_formatted = raw_split["train"].map(_format, remove_columns=raw_split["train"].column_names)
    test_formatted = raw_split["test"].map(_format, remove_columns=raw_split["test"].column_names)

    return train_formatted, test_formatted
    
    # # Applies _format to every example
    # # remove_column_names = raw.column_names drops the original input and output columns (of input data), keeping only the new text column.
    # formatted = raw.map(_format, remove_columns=raw.column_names)

    # # Splits 90% for training, 10% for evaluation. Seed=42 makes it reproducible
    # split = formatted.train_test_split(test_size=0.1, seed=42)



    # return split["train"], split["test"]


def main():

    parser = argparse.ArgumentParser()

    # defines what command line arguments the script accepts. --full is a flag meaning full fine-tune instead of LoRA
    parser.add_argument("--data", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--push_to_hub", default=None)

    args = parser.parse_args()

    # downloads and loads Qwen model.
    # dtype = None lets unsloth auto-detect the best precision
    # load_in_4bit means use 4-bit quantization unless --full flag is passed
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_ID,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = not args.full,
    )

    # disables KV cache during training, necessary for gradient checkpointing to work correctly.
    model.config.use_cache = False

    # Applies model chat format to the tokenizer. This teaches it special tokens like <|im_start|> and <|im_end|> that model uses to separate system, user and assistant turns.
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    # sets padding to be the same as the end-of-sequence token. Needed because the models doesnt have a dedicated padding token by defualt
    tokenizer.pad_token = tokenizer.eos_token
    # When paddind sequences to the same length, add padding tokens on the right side. Important for training stability
    tokenizer.padding_side = "right"
    # Tells the model config which token ID represents padding, so it knows to ignore those positions.
    model.config.pad_token_id = tokenizer.eos_token_id
    # model.config.padding_side = tokenizer.eos_token_id

    #LoRA setup
    # wraps the model with LoRA adapters. Only runs if not doing full fine tuning.
    if not args.full:

        model = FastLanguageModel.get_peft_model(
            model,
            # r is LoRA ranl. Think of it as the size of the adapter. Higher rank more parameters updated, more expensive but more memory
            r=LORA_R,
            # which layers get LoRA adapters added. We're targeting all attention and feed-forward layers
            target_modules = LORA_TARGET_MODULES,
            # alpha controls the scaling of LoRA updates. The actual scaling factor applied is alpha/r =2
            lora_alpha = LORA_ALPHA,
            # No dropout in LoRA layers. Dropout randomly zeros out values during training to prevent overfitting. setting to 0 disables it, fine for small datasets.
            lora_dropout = 0,
            # Dont add LoRA adapters to bias terms. Bias parameters are tiny and not worth adapting
            bias = "none",
            # Uses Unsloth's optimized gradient checkpointing implementation. Saves GPU memory at the ocst of some extra computation during backward pass.
            use_gradient_checkpointing="unsloth",
            # Seeds the random initialization of LoRA weights. Makes results reproducible.
            random_state=42,
            # Uses rank-stabalized LoRA. Normalizes the LoRA updates by sqrt(r) instead of r. Makes training more stable especially at higher ranks.
            use_rslora=True,
        )   

    train_ds, eval_ds = load_and_format(args.data, tokenizer)

    # Training configuration
    training_args = SFTConfig(
        # where to save checkpoints during training. Every epoch a checkpoint gets saved here
        output_dir = args.output_dir,

        # How many times to go through the entire dataset
        num_train_epochs = args.epochs,

        # How many examples to process at once per GPU
        per_device_train_batch_size = BATCH_SIZE,
        
        # Instead of updating weights every step, accumulate gradients over 8 steps then update. So, effective batch size = 1*8 = 8. This stimulates larger batch without needing more GPU memory
        gradient_accumulation_steps = GRAD_ACCUM,

        # How big each weight update step is 2e-4.
        learning_rate = args.lr,

        # How learning rate changes over time. Cosine means it starts at 2e-4, gradually decreases following a cosine curve, ending near zero. Prevents overshooting at the end of the training.
        lr_scheduler_type = "cosine",

        # For the first 5 steps, learning rate gradually increases from 0 to 2e-4. Prevents unstable updates at the very start.
        warmup_steps = int(100 * WARMUP_RATIO),

        # Regularization: Penalizes large weights slightly to prevent overfitting
        weight_decay = 0.01,

        # Which optimizer to use. AdamW in 8-bit precision, saves GPU memory compared to regualar AdamW.
        optim = "adamw_8bit",

        # Print loss to terminal every 10 training steps
        logging_steps = 10,


        # evaluate on the test set and save a check point after every epoch
        eval_strategy = "epoch",
        #checkpoint log
        
        # training at a step
        # {'loss': 0.9992, 'grad_norm': 0.35, 'learning_rate': 7.77e-05, 'epoch': 0.89}
        # loss - how wrong the model. Lower the better
        # grad_norm - how big the weight updates were
        # learning_rate - the learning rate at this point
        # epoch - how much percentage through epoch 1

        # Evaluation results of that step
        # {'eval_loss': 0.778, 'eval_runtime': 7.8, 'eval_samples_per_second': 1.28, 'epoch': 1.0}
        # eval_loss - how wrong the model was on the batch of examples it never trained on. Lower than train loss it a good sign
        # eval_samples_per_second - processed how many examples per second during evaluation
        # epoch - end of epoch 1

        # Final training summary
        # {'eval_loss': 0.778, 'eval_runtime': 7.8, 'eval_samples_per_second': 1.28, 'epoch': 1.0}
        # training_runtime - entire training took in seconds
        # train_samples_per_second - average loss across all training steps
        # epoch - completed 1 fill epoch

        # This means save a checkpoint after every epoch
        save_strategy = "epoch",

        # only keep the 2 most recent checkpoints. Older ones get deleted to save disk space.
        save_total_limit = 2,

        # After training finishes, automatically load the checkpoint with the best evaluation loss
        load_best_model_at_end = True,

        # Use bfloat16 if the GPU supports it, otherwise fallback to float16. Both are half precision formats that save memory. 
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),

        # Dont send training logs to Weights and Biases or any other tracking service
        report_to="none",

        # Makes training reproducible, same results every run.
        seed=42,

        # tells trainer which column in the datast contains the training text. Remember we named it "text" in load_and_format function.
        dataset_text_field="text",

        # maximum token length per example. Longer sequences get truncated.
        max_length = MAX_SEQ_LENGTH,
    )


    # creates the trainer and starts training. 
    trainer = SFTTrainer(
        model = model,
        args = training_args,

        train_dataset = train_ds,
        eval_dataset = eval_ds,

        processing_class = tokenizer,
    )

    # stats tolds the final loss and training metrics
    stats = trainer.train()

    # /final -> Lora weights as safe tensors (small size): parameters changed during training
    # safetensor -> trained weigts of the neural network stored on a disk
    final_dir = args.output_dir + "/final"

    # Save the LoRA adapter weights and tokenizer
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if not args.full:

        # All 14 billion paramterts, both the original Qwen weights and your LoRA weights combined together into one complete model
        merged_dir = args.output_dir + "/merged_16bit"

        # merges the LoRA weights back into the base model and saves as full 16-bit weights. Useful for deployment.
        model.save_pretrained_merged(
            merged_dir,
            tokenizer,
            save_method="merged_16bit",
        )

    # Optionally uploads the model to HuggingFace Hub. Only runs if --push_to_hub argument is provided
    if args.push_to_hub:

        if args.full:

            model.push_to_hub(args.push_to_hub)
            tokenizer.push_to_hub(args.push_to_hub)

        else:

            model.push_to_hub_merged(
                args.push_to_hub,
                tokenizer,
                save_method = "merged_16bit",
            )
    
    # Inference test
    # switches the model from training mode to inference mode, renables KV cache and optimizes for generation speed
    FastLanguageModel.for_inference(model)

    sample = "Precursor mz: 667.85479 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z):\nm/z 147.112804\nm/z 185.055683\nm/z 195.040033\nm/z 212.066582\nm/z 213.050597\nm/z 218.149918\nm/z 230.077146\nm/z 278.113532" \
    "\nm/z 330.108447\nm/z 331.233982\nm/z 348.119012\nm/z 364.724313\nm/z 364.724313\nm/z 375.129911\nm/z 376.113926\nm/z 393.140475\nm/z 393.140475\nm/z 405.763438\nm/z 406.261063\nm/z 414.276713\nm/z 414.276713\nm/z 419.156126" \
    "\nm/z 448.769252\nm/z 470.308544\nm/z 471.790184\nm/z 473.130305\nm/z 485.224874\nm/z 487.335093\nm/z 490.156854\nm/z 491.140869\nm/z 493.738149\nm/z 494.230156\nm/z 494.722164\nm/z 502.743431\nm/z 502.743431\nm/z 508.167418" \
    "\nm/z 544.316566\nm/z 544.795998\nm/z 544.808574\nm/z 544.808574\nm/z 544.814191\nm/z 545.282389\nm/z 545.288006\nm/z 545.288006\nm/z 550.772188\nm/z 550.772189\nm/z 550.777805\nm/z 553.321849\nm/z 553.813856\nm/z 558.314024" \
    "\nm/z 559.285463\nm/z 559.285463\nm/z 569.365725\nm/z 569.376958\nm/z 586.290745\nm/z 586.392274\nm/z 586.403507\nm/z 601.188883\nm/z 601.830038\nm/z 610.343312\nm/z 618.215432\nm/z 619.199447\nm/z 636.225996\nm/z 636.225996" \
    "\nm/z 682.461022\nm/z 687.262048\nm/z 699.487571\nm/z 704.288597\nm/z 715.256962\nm/z 731.299496\nm/z 732.283511\nm/z 732.294745\nm/z 749.310060\nm/z 796.456331\nm/z 810.519600\nm/z 811.514849\nm/z 827.546149\nm/z 827.546149" \
    "\nm/z 831.351925\nm/z 848.378474\nm/z 924.562527\nm/z 925.546543\nm/z 942.573092\nm/z 959.519660\nm/z 976.484671\nm/z 976.509824\nm/z 986.469021\nm/z 987.453036\nm/z 1004.479585\nm/z 1004.479585\nm/z 1073.562587\nm/z 1073.573820" \
    "\nm/z 1073.573820\nm/z 1087.625856\nm/z 1088.584719\nm/z 1088.609872\nm/z 1088.609872\nm/z 1088.621105\nm/z 1089.557502\nm/z 1089.568735\nm/z 1089.568735\nm/z 1090.552751\nm/z 1100.537100\nm/z 1100.537101\nm/z 1105.636421" \
    "\nm/z 1106.620436\nm/z 1117.563649\nm/z 1117.563650\nm/z 1118.547665\nm/z 1118.558899\nm/z 1118.558899\nm/z 1118.584051\nm/z 1160.605849\nm/z 1160.605849\nm/z 1171.574214\nm/z 1188.600763" \
    "\nAnalyze this spectrum and identify the most likely peptide sequence."

    messages = [
        {"role": "system", 
        "content": (
                    "You are SpectraChatModel, an expert mass spectrometry assistant. "
                    "Given a MS/MS spectrum with precursor mass and charge, identify the peptide sequence "
                    "by reasoning step-by-step through the y-ion ladder. "
                    "Start by identifying y1 candidates, then extend the y-ion ladder residue by residue of amino acids. "
                    "If the ladder becomes ambiguous or no candidates are found, state clearly that the "
                    "peptide sequence cannot be determined. "
                    "Conclude with the complete predicted peptide sequence if derivable."
                ),
         },
        {"role": "user", "content": sample},
    ]

    # language model cannot read text directly. It only understands numbers. So, the tokenizer's jobs is to convert the text into numbers and numbers back to text.
    # "peptide sequence" → [79, 485, 9867, 1234]

    # <|im_start|>: Start of a message turn
    # <|im_end|>: End of a message turn
    # <eos>: End of sequence, model should stop generating
    # <pad>: Padding, ignore this position

    # When you train, you process multiple examples at once. But each example has a different length.
    # Example 1: [1, 5, 8, 3] (4 tokens) 
    # Example 2: [2, 7, 9, 1, 4, 6, 8]  (7 tokens)
    
    # The model needs them all to be the same length. So you pad the shorter ones with a special padding token.
    # Example 1: [1, 5, 8, 3, PAD, PAD, PAD]
    # Example 2: [2, 7, 9, 1, 4,   6,   8  ]


    # converts message to token IDS. 
    inputs = tokenizer.apply_chat_template(
        messages,
        # returns numbers not strings this time
        tokenize=True,
        # adds the assistant start token so the model knows to start generating
        add_generation_prompt=True,
        # returns a pyTorch tensor
        return_tensors="pt",
        # moves to GPU
    ).to("cuda")
    
    out = model.generate(
        input_ids = inputs,
        # Generate up to 2048 tokens
        max_new_tokens = MAX_NEW_TOKENS,
        # controls randomness, 0.1 is very low, almost deterministic. Close to zero means the model almost always picks the highest probability token, Higher values like 1.0 make output more random and creative.
        temperature = TEMPERATURE,
        # enables sampling, necessary for temperature to have nay effect. If false, the model just always picks the top token
        do_sample = True,
        # Tells the generator what token to use for padding during generation
        pad_token_id = tokenizer.eos_token_id,
    )

    # Decodes the generated tokens back to text. out[0] get the first sequence.
    # [inputs.shape[1]:] slices off the input tokens, we only want the newly generated part, not the prompt.
    # skip_special_tokens=True: removes tokens like <|im_end|> from the output.
    print(tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True))

if __name__ == "__main__":
    main()