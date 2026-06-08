from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_PATH = "/lclhome/apali007/projects/spectra_algorithm/languagemodel/v2/spectra-chat-checkpoints/merged_16bit/"
MAX_SEQ_LENGTH = 4096
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

# sample_input = "Precursor mz: 505.27670 Da, Precursor Charge: 2\nExperimental MS/MS spectrum (m/z):\nm/z 147.112804\nm/z 159.076419\nm/z 159.094612\nm/z 169.078961\nm/z 170.056017\nm/z 170.584411\nm/z 225.584608\nm/z 225.620994\nm/z 230.077147\nm/z 230.092403\nm/z 238.100425\nm/z 238.118617\nm/z 238.118617\nm/z 239.600258\nm/z 246.108448\nm/z 255.108781\nm/z 255.139551\nm/z 255.631559\nm/z 255.631559\nm/z 256.108053\nm/z 258.108448\nm/z 266.113532\nm/z 267.110793\nm/z 267.110793\nm/z 267.121358\nm/z 268.147376\nm/z 273.121358\nm/z 273.155732\nm/z 274.118618\nm/z 274.129183\nm/z 274.139748\nm/z 275.207767\nm/z 276.660650\nm/z 283.134465\nm/z 283.140081\nm/z 283.140081\nm/z 284.098945\nm/z 284.102967\nm/z 284.142290\nm/z 285.155732\nm/z 288.116075\nm/z 288.134268\nm/z 288.624265\nm/z 300.642458\nm/z 301.129516\nm/z 301.129516\nm/z 301.134465\nm/z 302.639915\nm/z 304.157941\nm/z 304.626807\nm/z 304.655565\nm/z 354.177195\nm/z 354.189772\nm/z 354.687397\nm/z 354.687397\nm/z 355.140081\nm/z 355.176467\nm/z 355.215790\nm/z 356.135330\nm/z 358.150980\nm/z 358.150980\nm/z 358.174122\nm/z 358.244881\nm/z 359.171382\nm/z 359.663390\nm/z 360.161014\nm/z 360.189772\nm/z 360.705589\nm/z 360.705589\nm/z 361.168839\nm/z 361.197597\nm/z 368.656096\nm/z 368.674289\nm/z 368.703047\nm/z 369.192117\nm/z 376.174122\nm/z 376.255446\nm/z 377.181947\nm/z 386.674289\nm/z 387.166297\nm/z 387.682114\nm/z 387.700307\nm/z 387.700307\nm/z 391.197597\nm/z 391.695222\nm/z 394.174122\nm/z 395.156126\nm/z 395.187563\nm/z 395.679571\nm/z 395.679571\nm/z 396.687396\nm/z 396.705589\nm/z 396.705589\nm/z 397.171777\nm/z 397.187031\nm/z 397.197597\nm/z 397.695222\nm/z 397.695222\nm/z 398.145895\nm/z 398.211039\nm/z 400.197931\nm/z 400.708132\nm/z 401.697764\nm/z 401.697764\nm/z 402.189772\nm/z 402.195389\nm/z 402.687396\nm/z 402.687396\nm/z 402.705589\nm/z 402.705589\nm/z 403.713414\nm/z 404.721239\nm/z 407.156126\nm/z 412.161545\nm/z 412.197931\nm/z 412.718697\nm/z 414.213580\nm/z 414.705589\nm/z 415.172444\nm/z 417.729064\nm/z 418.226689\nm/z 418.718697\nm/z 420.224146\nm/z 421.171777\nm/z 423.187427\nm/z 423.218864\nm/z 423.710871\nm/z 425.203077\nm/z 426.213581\nm/z 426.242339\nm/z 426.734347\nm/z 427.231971\nm/z 432.187760\nm/z 432.224146\nm/z 436.219061\nm/z 437.203077\nm/z 439.250164\nm/z 444.187760\nm/z 446.203410\nm/z 447.162274\nm/z 448.219061\nm/z 448.255446\nm/z 450.161940\nm/z 450.234711\nm/z 452.177590\nm/z 452.250361\nm/z 453.234377\nm/z 454.193241\nm/z 454.208495\nm/z 458.203410\nm/z 458.203410\nm/z 461.177924\nm/z 461.177924\nm/z 462.234711\nm/z 464.177590\nm/z 466.193241\nm/z 466.229626\nm/z 468.245276\nm/z 471.235044\nm/z 472.219061\nm/z 478.193241\nm/z 478.266011\nm/z 479.224875\nm/z 479.224875\nm/z 480.208891\nm/z 480.245276\nm/z 481.240525\nm/z 486.198325\nm/z 493.240525\nm/z 494.224541\nm/z 496.276576\nm/z 539.318775\nm/z 559.251089\nm/z 559.251089\nm/z 563.246004\nm/z 563.261260\nm/z 565.261654\nm/z 578.235773\nm/z 578.329674\nm/z 579.277305\nm/z 582.288204\nm/z 586.261988\nm/z 587.246004\nm/z 589.276910\nm/z 594.288204\nm/z 594.288204\nm/z 594.324589\nm/z 595.308605\nm/z 596.303854\nm/z 596.340239\nm/z 606.324589\nm/z 607.308605\nm/z 608.246338\nm/z 608.303854\nm/z 609.335488\nm/z 617.271824\nm/z 618.288203\nm/z 618.324589\nm/z 634.261988\nm/z 634.261988\nm/z 634.298373\nm/z 635.257237\nm/z 635.351138\nm/z 636.335154\nm/z 646.298374\nm/z 646.298374\nm/z 650.350804\nm/z 665.398088\nm/z 691.319838\nm/z 691.340967\nm/z 691.413738\nm/z 705.356618\nm/z 705.356618\nm/z 707.372268\nm/z 709.424303\nm/z 717.335488\nm/z 718.319503\nm/z 719.314752\nm/z 719.372268\nm/z 721.330402\nm/z 721.387918\nm/z 722.383167\nm/z 722.383167\nm/z 734.289266\nm/z 734.419552\nm/z 735.346052\nm/z 735.403568\nm/z 736.304916\nm/z 736.341301\nm/z 736.398817\nm/z 749.325317\nm/z 749.361703\nm/z 764.336216\nm/z 764.430117\nm/z 765.356618\nm/z 776.372602\nm/z 776.372602\nm/z 782.383167\nm/z 786.356951\nm/z 786.393337\nm/z 790.351866\nm/z 790.351866\nm/z 792.367516\nm/z 792.403902\nm/z 792.403902\nm/z 806.419552\nm/z 835.446101\nm/z 845.430451\nm/z 846.414466\nm/z 863.441015\nm/z 895.503616\nAnalyze this spectrum and identify the most likely peptide sequence."

print("Loading model ... ")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True
)

FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template = "qwen-2.5")
print("Model loaded.\n")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": sample_input},
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = MAX_NEW_TOKENS,
    temperature = 0.1,
    do_sample = True,
    pad_token_id= tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs.shape[1]:], skip_special_tokens=True
)

print(f"OUTPUT:\n{response}")