from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MAX_SEQ_LENGTH = 4096

# #LORA PATH FOR 10 percent peptide overlap in train and test
# MODEL_ID = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-10-OVERLAP/final"
# OUTPUT_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-10-OVERLAP"

#LORA PATH FOR 20 percent peptide overlap in train and test
MODEL_ID = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-20-OVERLAP/final"
OUTPUT_DIR = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v4/checkpoints/spectra-chat-checkpoints-v4-20-OVERLAP"

# All 14 billion paramterts, both the original Qwen weights and your LoRA weights combined together into one complete model
merged_dir = OUTPUT_DIR + "/merged_16bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_ID,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# merges the LoRA weights back into the base model and saves as full 16-bit weights. Useful for deployment.
model.save_pretrained_merged(
    merged_dir,
    tokenizer,
    save_method="merged_16bit")

print("Done.")
