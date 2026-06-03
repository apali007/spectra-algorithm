# pass arguments from the command line
import argparse
# loads the training file into structured dataset object that the trainer understands
from datasets import load_dataset
# Unsloth is a an optimized LLM training library. FastLanguageModel loads the model efficiently with quantization. is_bfloat16_supported checks if your GPR supprots bfloat16 precision
from unsloth import FastLanguageModel, is_bfloat16_supported
# SFT trainer does supervised fine tuning. SFTConfig holds all the training hyperparameters
from trl import SFTTrainer, SFTConfig
# This applies the correct chat format for the respective language model
from unsloth.chat_templates import get_chat_template

# Model to download from Hugging face
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# Maximum token length per training example
MAX_SEQ_LENGTH = 3072

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

OUTPUT_DIR = "./spectra-chat-checkpoints"

# Takes the path from JSONL/input file and the tokenizer. Returns formatted train and test datasets
def load_and_format(data_path, tokenizer):

    # loading raw data
    raw = load_dataset("json", data_files=data_path, split="train")

    def _format(example):

        # structures each example as a conversation with three turns, system prompt, user input (the spectrum), and assistant output (the reasoning trace + peptide). This is the chat format for model expects.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are SpectraChatModel, an expert mass spectrometry assistant. "
                    "Analyze MS/MS Spectra step-by-step and conclude with the peptide."

                    "Fragmentation context.\n"
                "When a peptide is fragmented inside a mass spectrometer via collision-induced "
                "dissociation (CID), the peptide backbone cleaves preferentially at amide bonds. "
                "This produces two complementary ion series: b-ions (carrying the N-terminus) "
                "and y-ions (carrying the C-terminus, plus H20). Taken together, the mass "
                "differences between consecutive ions in each series reveal individual amino-acid "
                "residue masses. I would infer the closest b-ions and y-ions and reason to infer a closest peptide to this spectrum."
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
    
    # Applies _format to every example
    # remove_column_names = raw.column_names drops the original input and output columns (of input data), keeping only the new text column.
    formatted = raw.map(_format, remove_columns=raw.column_names)

    # Splits 90% for training, 10% for evaluation. Seed=42 makes it reproducible
    split = formatted.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]

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

    final_dir = args.output_dir + "/final"

    # Save the LoRA adapter weights and tokenizer
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if not args.full:

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
        # Generate up to 512 tokens
        max_new_tokens = 512,
        # controls randomness, 0.1 is very low, almost deterministic. Close to zero means the model almost always picks the highest probability token, Higher values like 1.0 make output more random and creative.
        temperature = 0.1,
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