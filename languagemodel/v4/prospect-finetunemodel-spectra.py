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
MAX_SEQ_LENGTH = 20000

# Generate up to 6144  tokens
MAX_NEW_TOKENS = 6144 

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

# PROSPECT reasoning dataset
# TRAIN_SET_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_train.json"
# TEST_SET_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/prospect_reasoning_dataset_test.json"

# OUTPUT_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-prospect"

# PROSPECT all y-ion clean case reasoning dataset
TRAIN_SET_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_clean_y_ion_case_prospect_reasoning_dataset_train.json"
TEST_SET_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/input_data/all_clean_y_ion_case_prospect_reasoning_dataset_test.json"

OUTPUT_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/all_clean_y_ion_case_spectra-chat-checkpoints-prospect"

AA_MASSES = {
    'A': 71.03711, 'R': 156.10111, 'N': 114.04293, 'D': 115.02694,
    'C': 103.00919, 'E': 129.04259, 'Q': 128.05858, 'G': 57.02146,
    'H': 137.05891, 'I': 113.08406, 'L': 113.08406, 'K': 128.09496,
    'M': 131.04049, 'F': 147.06841, 'P': 97.05276,  'S': 87.03203,
    'T': 101.04768, 'W': 186.07931, 'Y': 163.06333, 'V': 99.06841,
}

NEUTRAL_LOSS_MASSES = {
    'H2O': 18.01056,
    'NH3': 17.02655
}

MASS_PROTON =  1.00728
TOLERANCE = 0.2
PRECURSOR_TOLERANCE = 0.1

AA_MASS_TABLE_TEXT = "\n".join(
    f"{aa}: {mass:.5f} Da" for aa, mass in AA_MASSES.items()
)
SYSTEM_PROMPT = (
    "You are an expert in de novo peptide sequencing from tandem mass spectrometry (MS/MS) "
    "proteomics data. Given a spectrum's precursor mass, precursor charge, and a list of fragment "
    "ion peaks (experimental mass and relative intensity), determine the most likely peptide sequence.\n\n"
    "Use the following procedure:\n"
    "1. First, attempt to derive the sequence using the y-ion ladder (C-terminus to N-terminus). "
    "For each position, identify the fragment peak, verify its mass against the theoretical fragment mass "
    f"within a tolerance of {TOLERANCE} Da, and compute the amino acid residue from "
    "the mass difference between consecutive fragments. If the y-ion ladder resolves the complete "
    "sequence, validate it against the precursor mass "
    "and stop. No need to derive a b-ion ladder if the sequence is already confirmed.\n"
    "2. If gaps remain in the y-ion ladder, attempt the b-ion ladder (N-terminus to C-terminus) the "
    "same way. If the b-ion ladder alone resolves the complete sequence, validate it against the "
    "precursor mass and stop.\n"
    "3. If both ladders have gaps, merge them: for each unresolved b-ion position, check whether the "
    "y-ion ladder resolved that same position. If merging resolves the complete "
    "sequence, validate it against the precursor mass.\n"
    "4. If exactly one residue remains unresolved after merging, recover it by subtracting the sum "
    "of the known residues and one water molecule from the precursor mass. Check whether the "
    f"resulting mass matches any residue mass within the fragment tolerance ({TOLERANCE} Da). If it does, "
    "predict the full sequence and validate it against the precursor mass.\n"
    "5. When a terminal residue (the first or last position) has no fragment peak, recover it from "
    "the adjacent known fragment and the precursor mass rather than reporting it as missing.\n"
    "6. If two or more residues remain unresolved after merging, or the mass recovered in step 4 "
    f"does not match any amino acid residue within tolerance of {TOLERANCE} Da, state clearly that the complete "
    "sequence cannot be determined, report the partial sequence with unresolved positions marked, "
    "and show the arithmetic that led to that conclusion.\n\n"
    "Amino acid residue masses (monoisotopic, Da):\n"
    f"{AA_MASS_TABLE_TEXT}\n\n"
    f"Proton mass: {MASS_PROTON} Da. Water (H2O) mass: {NEUTRAL_LOSS_MASSES['H2O']} Da. "
    f"Ammonia (NH3) mass: {NEUTRAL_LOSS_MASSES['NH3']} Da.\n\n"
    "When a fragment ion has charge greater than 1, convert it to a singly-charged mass before using "
    "it in any calculation: singly_charged_mass = (m/z x charge) - (charge - 1) x proton_mass.\n\n"
    "When a fragment ion involves a neutral loss, add the corresponding neutral loss mass (H2O or NH3) "
    "to the singly-charged mass: mass_after_correction = singly_charged_mass + neutral_loss_mass.\n\n"
    f"Always validate any complete candidate sequence against the precursor mass "
    f"within a precursor tolerance of {PRECURSOR_TOLERANCE} Da before reporting it "
    "as final. Show your arithmetic explicitly at every step."
)

# Takes the path from JSONL/input file and the tokenizer. Returns formatted train and test datasets
def load_and_format(tokenizer):

    def _format(example):

        # structures each example as a conversation with three turns, system prompt, user input (the spectrum), and assistant output (the reasoning trace + peptide). This is the chat format for model expects.
        messages = [
            {
                "role": "system",
                  "content": SYSTEM_PROMPT,
            },
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

    train_raw = load_dataset("json", data_files = TRAIN_SET_DIR, split="train")
    test_raw = load_dataset("json", data_files = TEST_SET_DIR, split="train")

    train_formatted = train_raw.map(_format, remove_columns=train_raw.column_names)
    test_formatted = test_raw.map(_format, remove_columns=test_raw.column_names)

    return train_formatted, test_formatted

def main():

    parser = argparse.ArgumentParser()

    # defines what command line arguments the script accepts. --full is a flag meaning full fine-tune instead of LoRA
    parser.add_argument("--data", required=False)
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

    # train_ds, eval_ds = load_and_format(tokenizer)
    train_ds, _ = load_and_format(tokenizer)

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

        warmup_ratio = WARMUP_RATIO,
 
        # Regularization: Penalizes large weights slightly to prevent overfitting
        weight_decay = 0.01,

        # Which optimizer to use. AdamW in 8-bit precision, saves GPU memory compared to regualar AdamW.
        optim = "adamw_8bit",

        # Print loss to terminal every 10 training steps
        logging_steps = 10,


        # evaluate on the test set and save a check point after every epoch
        # eval_strategy = "epoch",
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
        # save_strategy = "epoch",
        save_strategy = "no",

        # only keep the 2 most recent checkpoints. Older ones get deleted to save disk space.
        # save_total_limit = 2,

        # After training finishes, automatically load the checkpoint with the best evaluation loss
        # load_best_model_at_end = True,

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
        # eval_dataset = eval_ds,

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

    sample = "Precursor mz: 667.85479 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z, intensity):" \
    "\nm/z 147.11246, intensity 0.31000\nm/z 185.05533, intensity 0.10000\nm/z 195.03891, intensity 0.03000\nm/z 212.06584, intensity 0.14000" \
    "\nm/z 213.05009, intensity 0.12000\nm/z 218.14929, intensity 0.26000\nm/z 230.07646, intensity 1.00000\nm/z 278.11417, intensity 0.02000" \
    "\nm/z 330.10748, intensity 0.02000\nm/z 331.23291, intensity 0.10000\nm/z 348.11713, intensity 0.13000\nm/z 364.72824, intensity 0.01000" \
    "\nm/z 375.13028, intensity 0.02000\nm/z 376.11121, intensity 0.08000\nm/z 393.13873, intensity 0.15000\nm/z 405.76367, intensity 0.09000" \
    "\nm/z 406.26431, intensity 0.02000\nm/z 414.27802, intensity 0.11000\nm/z 419.15717, intensity 0.03000\nm/z 448.76730, intensity 0.02000" \
    "\nm/z 470.30945, intensity 0.02000\nm/z 471.78757, intensity 0.08000\nm/z 473.13153, intensity 0.04000\nm/z 485.22443, intensity 0.03000" \
    "\nm/z 487.33368, intensity 0.49000\nm/z 490.15421, intensity 0.04000\nm/z 491.13788, intensity 0.09000\nm/z 493.73444, intensity 0.02000" \
    "\nm/z 494.22699, intensity 0.11000\nm/z 494.73117, intensity 0.02000\nm/z 502.74188, intensity 0.24000\nm/z 508.16580, intensity 0.18000" \
    "\nm/z 544.31299, intensity 0.02000\nm/z 544.80511, intensity 0.05000\nm/z 545.29236, intensity 0.06000\nm/z 550.77295, intensity 0.02000" \
    "\nm/z 553.32050, intensity 0.44000\nm/z 553.82349, intensity 0.14000\nm/z 558.31573, intensity 0.02000\nm/z 559.28375, intensity 0.04000" \
    "\nm/z 569.37494, intensity 0.03000\nm/z 586.29449, intensity 0.03000\nm/z 586.40192, intensity 0.47000\nm/z 601.17957, intensity 0.02000" \
    "\nm/z 601.82806, intensity 0.06000\nm/z 610.34058, intensity 0.40000\nm/z 618.21326, intensity 0.02000\nm/z 619.19519, intensity 0.05000" \
    "\nm/z 636.22620, intensity 0.12000\nm/z 682.45514, intensity 0.04000\nm/z 687.25146, intensity 0.02000\nm/z 699.48553, intensity 0.43000" \
    "\nm/z 704.29138, intensity 0.02000\nm/z 715.25244, intensity 0.03000\nm/z 731.29834, intensity 0.02000\nm/z 732.28296, intensity 0.06000" \
    "\nm/z 749.30914, intensity 0.12000\nm/z 796.46094, intensity 0.05000\nm/z 810.51990, intensity 0.16000\nm/z 811.52142, intensity 0.04000" \
    "\nm/z 827.54382, intensity 0.55000\nm/z 831.35925, intensity 0.03000\nm/z 848.36932, intensity 0.03000\nm/z 924.56531, intensity 0.05000" \
    "\nm/z 925.55377, intensity 0.08000\nm/z 942.57092, intensity 0.82000\nm/z 959.51770, intensity 0.05000\nm/z 976.49286, intensity 0.01000" \
    "\nm/z 986.47412, intensity 0.02000\nm/z 987.46826, intensity 0.03000\nm/z 1004.47638, intensity 0.24000\nm/z 1073.55994, intensity 0.02000" \
    "\nm/z 1087.62561, intensity 0.03000\nm/z 1088.60034, intensity 0.03000\nm/z 1089.56531, intensity 0.11000\nm/z 1090.55603, intensity 0.02000" \
    "\nm/z 1100.52588, intensity 0.03000\nm/z 1105.63049, intensity 0.29000\nm/z 1106.63342, intensity 0.08000\nm/z 1117.57019, intensity 0.16000" \
    "\nm/z 1118.56238, intensity 0.04000\nm/z 1160.59924, intensity 0.06000\nm/z 1171.57361, intensity 0.03000\nm/z 1188.59583, intensity 0.32000" \
    "\nAnalyze this spectrum and identify the most likely peptide sequence."

    messages = [
        {"role": "system", 
                "content": SYSTEM_PROMPT,
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