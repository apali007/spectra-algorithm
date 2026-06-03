import argparse
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_model(checkpoint_path, use_merged=False, load_in_4bit=True):
    """
    Load the trained SpectraChatModel.
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., './spectra-chat-checkpoints/final')
        use_merged: Whether to use merged 16bit model (for LoRA adapters)
        load_in_4bit: Whether to load in 4-bit quantization
    
    Returns:
        model, tokenizer
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Determine max sequence length
    max_seq_length = 3072
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    # Configure tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set model to inference mode
    FastLanguageModel.for_inference(model)
    
    print("Model loaded successfully")
    
    return model, tokenizer


def prepare_input(spectrum_text, tokenizer, device, system_prompt=None):
    """
    Prepare input for the model.
    
    Args:
        spectrum_text: MS/MS spectrum data as string
        tokenizer: Model tokenizer
        device: Device to place tensors on
        system_prompt: Custom system prompt (optional)
    
    Returns:
        Tokenized input tensor
    """
    if system_prompt is None:
        system_prompt = (
            "You are SpectraChatModel, an expert mass spectrometry assistant. "
            "Analyze MS/MS spectra step-by-step and conclude with the peptide."
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": spectrum_text},
    ]
    
    # Apply chat template and ensure we get proper tensor
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize and convert to tensor
    inputs = tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=3072,
    )
    
    # Move to device and extract input_ids
    input_ids = inputs["input_ids"].to(device)
    
    return input_ids


def generate_prediction(model, tokenizer, inputs, max_new_tokens=512, temperature=0.1, do_sample=True):
    """
    Generate peptide sequence prediction.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        inputs: Prepared input tensor
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        do_sample: Whether to use sampling
    
    Returns:
        Generated text (prediction)
    """
    # Store input length for later slicing
    input_length = inputs.shape[-1]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated portion (exclude input)
    generated_text = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )
    
    return generated_text


def predict_from_spectrum(model, tokenizer, spectrum_text, **generation_kwargs):
    """
    End-to-end prediction from spectrum text.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        spectrum_text: MS/MS spectrum data
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Predicted peptide sequence and analysis
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = prepare_input(spectrum_text, tokenizer, device)
    prediction = generate_prediction(model, tokenizer, inputs, **generation_kwargs)
    return prediction


def interactive_mode(model, tokenizer):
    """Run interactive inference session."""
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
        
        # Allow multi-line input
        print("(Press Enter twice when done, or type END on a new line)")
        lines = [user_input]
        while True:
            line = input()
            if line.strip().upper() == 'END' or line == '':
                break
            lines.append(line)
        
        spectrum_text = '\n'.join(lines)
        
        print("Analyzing spectrum...")
        prediction = predict_from_spectrum(
            model, tokenizer, spectrum_text,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        print("PREDICTION:")
        print(prediction)


def batch_inference(model, tokenizer, input_file, output_file=None):
    """
    Process multiple spectra from a file.
    
    Args:
        model: Loaded model
        tokenizer: Model tokenizer
        input_file: Path to input file (one spectrum per line or JSON)
        output_file: Path to save predictions (optional)
    """
    import json
    
    print(f"Processing batch from: {input_file}")
    
    with open(input_file, 'r') as f:
        # Try to load as JSON first
        try:
            data = json.load(f)
            if isinstance(data, list):
                spectra = [item.get('input', str(item)) for item in data]
            else:
                spectra = [data.get('input', str(data))]
        except json.JSONDecodeError:
            # Fall back to line-by-line
            f.seek(0)
            spectra = [line.strip() for line in f if line.strip()]
    
    results = []
    
    for i, spectrum in enumerate(spectra, 1):
        print(f"Processing spectrum {i}/{len(spectra)}...")
        prediction = predict_from_spectrum(
            model, tokenizer, spectrum,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True
        )
        
        results.append({
            "spectrum_id": i,
            "input": spectrum,
            "prediction": prediction
        })
        
        print(f"Completed {i}/{len(spectra)}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SpectraChatModel Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with final checkpoint
  python inference.py --checkpoint ./spectra-chat-checkpoints/final

  # Use merged 16-bit model
  python inference.py --checkpoint ./spectra-chat-checkpoints/merged_16bit --merged

  # Single prediction from command line
  python inference.py --checkpoint ./spectra-chat-checkpoints/final --text "88.0 100\\n175.1 200"

  # Batch processing
  python inference.py --checkpoint ./spectra-chat-checkpoints/final --batch input.json --output predictions.json
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./spectra-chat-checkpoints/final",
        help="Path to model checkpoint directory"
    )
    
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged 16-bit model (for LoRA checkpoints)"
    )
    
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (requires more memory)"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Spectrum text for single prediction"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Input file for batch processing (JSON or text)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch results"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # Load model
    global model, tokenizer
    model, tokenizer = load_model(
        args.checkpoint,
        use_merged=args.merged,
        load_in_4bit=not args.no_4bit
    )
    
    # Single prediction
    if args.text:
        print("Analyzing spectrum...")
        prediction = predict_from_spectrum(
            model, tokenizer, args.text,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print("PREDICTION:")
        print(prediction)
    
    # Batch processing
    elif args.batch:
        batch_inference(model, tokenizer, args.batch, args.output)
    
    # Interactive mode
    else:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
