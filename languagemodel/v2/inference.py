import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# checkpoint_path: path to your saved checkpoint folder
# use_merged: whether to load the merged 16-bit model or LoRA adapter version
# load_in_4bit: whether to quantize to 4-bit to save memory
def load_model(checkpoint_path, use_merged=False, load_in_4bit=True):
    print(f"Loading model from: {checkpoint_path}")

    # if your trained eith 3072, you must load 3072
    max_seq_length = 3072

    # loads the saved checkpoint instead of the original model. Notice model_name here points to yout local checkpoint folder, not hugging face
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = load_in_4bit,
    )

    # Tokenizer setup
    # same setup as training
    # if your tokenizer setup differs between training and inference the model will produce garbage output
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Turns on KV cache and inference optimizations. Always call this before generating.
    FastLanguageModel.for_inference(model)


    print("Model loaded succesfully")

    return model, tokenizer

# spectrum_text: the raw spectrum string the user provides
# tokenizer: the loaded tokenizer
# device: either cuda or cpu
# system_prompt: optional custom system prompt, defaults to None
def prepare_input(spectrum_text, tokenizer, device, system_prompt=None):

# if no custom system prompt is provided, use this as default one. Keeps the function flexible, you can override it from outside
    if system_prompt is None:
        system_prompt = (
            "You are SpectraChatModel, an expert mass spectrometry assistant. "
            "Analyze MS/MS spectra step-by-step and conclude with the peptide. "
        )

# same chat format as before in training. Notice there's no assistant turn, because we want the model to generate it.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": spectrum_text},
    ]

# Two step tokenization
# step 1: format string.
# tokenize = False means just format the messages into a string with special tokens, dont convert the numbers yet.
# it looks something like this: <|im_start|>system You are SpectraChatModel... <|im_end|> <|im_start|>user 88.0393 1100... <|im_end|> <|im_start|>assistant

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# step 2: convert to tensor:
# Now convert the string into actual token ID numbers. 
# padding = True. Pads if needed.
# truncation = True. cuts off it longer than 3072 tokens
# return_tensors = "pt" returns a PyTorch tensor
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072
    )

# moving to device
# inputs is a dictionary with keys like "input_ids" and "attention_mask". We extract just the "input_ids" and move them to GPU. 
# The attention mask is a binary flag, 1 means real token, 0 means padding. This tells the model "pay attention to these positions, ignore these"
# In our setup, tokenizer.pad_token = tokenizer.eos_token, we set the padding token to be the same as the end-of-sequence token. so now:
# Real tokens:    [1, 5, 8, 3, EOS] 
# Padding tokens: [1, 5, 8, 3, EOS, EOS, EOS]
# The model cant tell the difference between a real EOS and a padding EOS. So, we need to pass attention mask explicitly.

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    return input_ids, attention_mask

# takes the model tokenizer, and input tensor. Has sensible defaults, generate up to 512 tokens, low temperature for near-deterministic output.
def generate_prediction(model, tokenizer, inputs, attention_mask, max_new_tokens=512, temperature=0.1, do_sample=True):

# saves how many tokens the input has. We'll use to slice the input portion from the ouput. shape[-1] gets the last dimension which is the sequence length.
    input_length = inputs.shape[-1]

# tells PyTorch not to track gradients during this operation
    with torch.no_grad():
        # generates new tokens. The model takes the input tokens and keeps predicting the next token until it hits max_new_tokens or generates an eos token.
        outputs = model.generate(
            input_ids = inputs,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            do_sample = do_sample,
            pad_token_id = tokenizer.eos_token_id
        )

# When you call model.generate(), you pass in the input tokens. The model then appends new tokens to the end of those input tokens. 
# So the output contains everything, input + generated. So you have to manually slice off the input portion to get just the generated response.

# decoding output
# ouputs[0] gets the first sequence, output contain the full sequence including input tokens
# [input_length:] slices off the input part, keeping only the newly generated tokens.
# skip_special_tokens = True removes special tokens like <|im_end|> from the final text
    generated_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

# returns the predicted peptide sequence and reasoning trace as a plain string
    return generated_text

# This is a wrapper that combines "prepare_input" and "generate_prediction" into one call.
# **generation_kwargs means you can pass any extra arguments and they get forwarded automatically
def predict_from_spectrum(model, tokenizer, spectrum_text, **generation_kwargs):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs, attention_mask = prepare_input(spectrum_text, tokenizer, device)
    prediction = generate_prediction(model, tokenizer, inputs, attention_mask, **generation_kwargs)
    return prediction

# takes the laoded model and tokenizer. This is the function that runs when you launch inference without any input file, like a chat interface in your terminal.
def interactive_mode(model, tokenizer):

    print("SpectraChatModel - Interactive Mode")
    print("Enter MS/MS spectrum data. Type 'quit' or 'exit' to stop.")
    print("Type 'example' to see a sample input format")

    example_spectrum = """Experimental MS/MS spectrum (m/z intensity):

88.0393 1100
175.1190 2200
290.1169 1960
417.2456 1600
645.3389 2400
716.3760 1100

Identify the peptide sequence."""

    while True:
        print("")
        user_input = input("Enter spectrum (or command): ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break

        if user_input.lower() == 'example':
            print("Example input:")
            print(example_spectrum)
            continue

        if not user_input:
            print("Please enter spectrum data.")
            continue

        print("(type END on a new line)")
        lines = [user_input]
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)

        spectrum_text = '\n'.join(lines)

        print("Analyzing spectrum...")
        prediction = predict_from_spectrum(
            model, tokenizer, spectrum_text,
            max_new_tokens = 512,
            temperature = 0.1,
            do_sample = True,
        )

        print("PREDICTION: ")
        print(prediction)

# Processes multiple spectra from a file all at once
def batch_inference(model, tokenizer, input_file, output_file=None):
    import json

    print(f"Processing batch from: {input_file}")

    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        spectra = [item.get('input', str(item)) for item in data]
        # try:
        #     data = json.load(f)
        #     if isinstance(data,list):
        #         spectra = [item.get('input', str(item)) for item in data]
        #     else:
        #         spectra = [data.get('input', str(data))]
        
        # except json.JSONDecodeError:
        #     f.seek(0)
        #     spectra = [line.strip() for line in f if line.strip()]

    results = []

    for i, spectrum in enumerate(spectra, 1):
        print(f"processing spectrum {i}/{len(spectra)}...")
        prediction = predict_from_spectrum(
            model, tokenizer, spectrum,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
        )

        results.append({
            "spectrum_id": i,
            "input": spectrum,
            "prediction": prediction,
        })

        print(f"Completed {i}/{len(spectra)}")

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")

    return results
    
# entry point of the inference script. Handles argument parsing and decides which mode to run.
def main():

    # argparse lets you control your script from the terminal without editing code.
    
    # # Training
    # python finetunemodel.py --data sample1_spectra_training_data.jsonl --epochs 1

    # # Single prediction
    # python inference.py --checkpoint ./spectra-chat-checkpoints/final --text "88.03 1100\n175.11 2200"

    # # Batch prediction
    # python inference.py --checkpoint ./spectra-chat-checkpoints/final --batch myspectra.jsonl --output results.json

    # # Interactive mode
    # python inference.py --checkpoint ./spectra-chat-checkpoints/final

    # creates the argument parser.
    # RawDescriptionHelpFormatter preserves formatting in the help text
    # epilog="" is where user had usage examples
    parser = argparse.ArgumentParser(
        description="SpectraChatModel Inference",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog=""
    )

    # path to the saved model checkpoint. Defaults to where finetunemodel.py saves it
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./spectra-chat-checkpoints/final",
        help="Path to model checkpoint directory"
    )

    # Flag to use merged 16-bit model instead of LoRA adapters
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged 16-bit model (for LoRA checkpoints)"
    )

    # 4-bit is disabled and the model laods in full precision, uses more GPU memory but more accurate
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more memory)"
    )

    # For single spectrum prediction directly from command line
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Spectrum text for single prediction"
    )

    # Path to input file for batch processing
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Input file for batch processing (JSON or text)"
    )

    # Path to save batch results
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch processing (JSON or text)"
    )

    # Maximum tokens to generate
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )

    # sampling temperature
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.0-1.0)"
    )

    args = parser.parse_args()

    # loading Model
    # not args.no_4bit means 4-bit is on by default, disabled if --no-4bit flag is passed
    global model, tokenizer
    model, tokenizer = load_model(
        args.checkpoint,
        use_merged=args.merged,
        load_in_4bit=not args.no_4bit
    )

    # mode selection
    # if "--text" argument is provided, run single prediction mode
    if args.text:
        print("Analyzing spectrum...")
        prediction = predict_from_spectrum(
            model, tokenizer, args.text,
            max_new_tokens = args.max_tokens,
            temperature = args.temperature
        )

        print("PREDICTION:")
        print(prediction)

    # if "--batch" argument is provided, run batch processing mode
    elif args.batch:
        batch_inference(model, tokenizer, args.batch, args.output)

    # if neither provided, launch interactive terminal mode.
    else:
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    main()
