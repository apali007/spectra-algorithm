# Check actual length of your longest traces
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

with open('/lclhome/apali007/projects/spectra_algorithm/output/reasoning_dataset.json') as f:
    dataset = json.load(f)

lengths = []
for item in dataset:
    text = item['input'] + item['output']
    tokens = tokenizer.encode(text)
    lengths.append(len(tokens))

print(f"Max tokens: {max(lengths)}")
print(f"Mean tokens: {sum(lengths)/len(lengths):.0f}")
print(f"95th percentile: {sorted(lengths)[int(len(lengths)*0.95)]}")