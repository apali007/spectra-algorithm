import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# Maximum token length per training example
# MAX_SEQ_LENGTH = 3072
MAX_SEQ_LENGTH = 4096

# Generate up to 2048 tokens
MAX_NEW_TOKENS = 2048

SYSTEM_PROMPT = (
                "You are SpectraChatModel, an expert mass spectrometry assistant. "
                "Given a MS/MS spectrum with precursor mass and charge, identify the peptide sequence "
                "by reasoning step-by-step through the y-ion ladder. "
                "Start by identifying y1 candidates, then extend the y-ion ladder residue by residue of amino acids. "
                "If the ladder becomes ambiguous or no candidates are found, state clearly that the "
                "peptide sequence cannot be determined. "
                "Conclude with the complete predicted peptide sequence if derivable."
            )

sample_input = "Precursor mz: 505.27670 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z):\nm/z 147.112804\nm/z 159.076419\nm/z 159.094612\nm/z 169.078961\nm/z 170.056017\nm/z 170.584411\nm/z 225.584608\nm/z 225.620994\nm/z 230.077147\nm/z 230.092403\nm/z 238.100425\nm/z 238.118617\nm/z 238.118617\nm/z 239.600258\nm/z 246.108448\nm/z 255.108781\nm/z 255.139551\nm/z 255.631559\nm/z 255.631559\nm/z 256.108053\nm/z 258.108448\nm/z 266.113532\nm/z 267.110793\nm/z 267.110793\nm/z 267.121358\nm/z 268.147376\nm/z 273.121358\nm/z 273.155732\nm/z 274.118618\nm/z 274.129183\nm/z 274.139748\nm/z 275.207767\nm/z 276.660650\nm/z 283.134465\nm/z 283.140081\nm/z 283.140081\nm/z 284.098945\nm/z 284.102967\nm/z 284.142290\nm/z 285.155732\nm/z 288.116075\nm/z 288.134268\nm/z 288.624265\nm/z 300.642458\nm/z 301.129516\nm/z 301.129516\nm/z 301.134465\nm/z 302.639915\nm/z 304.157941\nm/z 304.626807\nm/z 304.655565\nm/z 354.177195\nm/z 354.189772\nm/z 354.687397\nm/z 354.687397\nm/z 355.140081\nm/z 355.176467\nm/z 355.215790\nm/z 356.135330\nm/z 358.150980\nm/z 358.150980\nm/z 358.174122\nm/z 358.244881\nm/z 359.171382\nm/z 359.663390\nm/z 360.161014\nm/z 360.189772\nm/z 360.705589\nm/z 360.705589\nm/z 361.168839\nm/z 361.197597\nm/z 368.656096\nm/z 368.674289\nm/z 368.703047\nm/z 369.192117\nm/z 376.174122\nm/z 376.255446\nm/z 377.181947\nm/z 386.674289\nm/z 387.166297\nm/z 387.682114\nm/z 387.700307\nm/z 387.700307\nm/z 391.197597\nm/z 391.695222\nm/z 394.174122\nm/z 395.156126\nm/z 395.187563\nm/z 395.679571\nm/z 395.679571\nm/z 396.687396\nm/z 396.705589\nm/z 396.705589\nm/z 397.171777\nm/z 397.187031\nm/z 397.197597\nm/z 397.695222\nm/z 397.695222\nm/z 398.145895\nm/z 398.211039\nm/z 400.197931\nm/z 400.708132\nm/z 401.697764\nm/z 401.697764\nm/z 402.189772\nm/z 402.195389\nm/z 402.687396\nm/z 402.687396\nm/z 402.705589\nm/z 402.705589\nm/z 403.713414\nm/z 404.721239\nm/z 407.156126\nm/z 412.161545\nm/z 412.197931\nm/z 412.718697\nm/z 414.213580\nm/z 414.705589\nm/z 415.172444\nm/z 417.729064\nm/z 418.226689\nm/z 418.718697\nm/z 420.224146\nm/z 421.171777\nm/z 423.187427\nm/z 423.218864\nm/z 423.710871\nm/z 425.203077\nm/z 426.213581\nm/z 426.242339\nm/z 426.734347\nm/z 427.231971\nm/z 432.187760\nm/z 432.224146\nm/z 436.219061\nm/z 437.203077\nm/z 439.250164\nm/z 444.187760\nm/z 446.203410\nm/z 447.162274\nm/z 448.219061\nm/z 448.255446\nm/z 450.161940\nm/z 450.234711\nm/z 452.177590\nm/z 452.250361\nm/z 453.234377\nm/z 454.193241\nm/z 454.208495\nm/z 458.203410\nm/z 458.203410\nm/z 461.177924\nm/z 461.177924\nm/z 462.234711\nm/z 464.177590\nm/z 466.193241\nm/z 466.229626\nm/z 468.245276\nm/z 471.235044\nm/z 472.219061\nm/z 478.193241\nm/z 478.266011\nm/z 479.224875\nm/z 479.224875\nm/z 480.208891\nm/z 480.245276\nm/z 481.240525\nm/z 486.198325\nm/z 493.240525\nm/z 494.224541\nm/z 496.276576\nm/z 539.318775\nm/z 559.251089\nm/z 559.251089\nm/z 563.246004\nm/z 563.261260\nm/z 565.261654\nm/z 578.235773\nm/z 578.329674\nm/z 579.277305\nm/z 582.288204\nm/z 586.261988\nm/z 587.246004\nm/z 589.276910\nm/z 594.288204\nm/z 594.288204\nm/z 594.324589\nm/z 595.308605\nm/z 596.303854\nm/z 596.340239\nm/z 606.324589\nm/z 607.308605\nm/z 608.246338\nm/z 608.303854\nm/z 609.335488\nm/z 617.271824\nm/z 618.288203\nm/z 618.324589\nm/z 634.261988\nm/z 634.261988\nm/z 634.298373\nm/z 635.257237\nm/z 635.351138\nm/z 636.335154\nm/z 646.298374\nm/z 646.298374\nm/z 650.350804\nm/z 665.398088\nm/z 691.319838\nm/z 691.340967\nm/z 691.413738\nm/z 705.356618\nm/z 705.356618\nm/z 707.372268\nm/z 709.424303\nm/z 717.335488\nm/z 718.319503\nm/z 719.314752\nm/z 719.372268\nm/z 721.330402\nm/z 721.387918\nm/z 722.383167\nm/z 722.383167\nm/z 734.289266\nm/z 734.419552\nm/z 735.346052\nm/z 735.403568\nm/z 736.304916\nm/z 736.341301\nm/z 736.398817\nm/z 749.325317\nm/z 749.361703\nm/z 764.336216\nm/z 764.430117\nm/z 765.356618\nm/z 776.372602\nm/z 776.372602\nm/z 782.383167\nm/z 786.356951\nm/z 786.393337\nm/z 790.351866\nm/z 790.351866\nm/z 792.367516\nm/z 792.403902\nm/z 792.403902\nm/z 806.419552\nm/z 835.446101\nm/z 845.430451\nm/z 846.414466\nm/z 863.441015\nm/z 895.503616\nAnalyze this spectrum and identify the most likely peptide sequence."


# checkpoint_path: path to your saved checkpoint folder
# use_merged: whether to load the merged 16-bit model or LoRA adapter version
# load_in_4bit: whether to quantize to 4-bit to save memory
def load_model(checkpoint_path, use_merged=False, load_in_4bit=True):
    print(f"Loading model from: {checkpoint_path}")

    # loads the saved checkpoint instead of the original model. Notice model_name here points to yout local checkpoint folder, not hugging face
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_path,
        # if your trained eith 4096, you must load 4096
        max_seq_length = MAX_SEQ_LENGTH,
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
        system_prompt = SYSTEM_PROMPT

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
# truncation = True. cuts off it longer than MAX_SEQ_LENGTH tokens
# return_tensors = "pt" returns a PyTorch tensor
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH
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

# takes the model tokenizer, and input tensor. Has sensible defaults, generate up to MAX_NEW_TOKENS tokens, low temperature for near-deterministic output.
def generate_prediction(model, tokenizer, inputs, attention_mask, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, do_sample=True):

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
            max_new_tokens=MAX_NEW_TOKENS,
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
        default=MAX_NEW_TOKENS,
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
  
if __name__ == "__main__":
    main()
