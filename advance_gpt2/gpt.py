# advanced_task_gpt2.ipynb
# ShadowFox AIML Internship - Advanced Task: Language Model (GPT-2 Text Generation)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# -------------------------------
# 1. Load Pre-trained GPT-2 Model
# -------------------------------
model_name = "gpt2"  # can try "gpt2-medium" or "gpt2-large" if you want bigger
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# -------------------------------
# 2. Prepare Input Prompt
# -------------------------------
prompt = "Artificial Intelligence is transforming the world of healthcare by"
inputs = tokenizer.encode(prompt, return_tensors="pt")

# -------------------------------
# 3. Generate Text
# -------------------------------
output = model.generate(
    inputs,
    max_length=80,          # length of output text
    num_return_sequences=1, # how many outputs
    no_repeat_ngram_size=2, # prevent repeating
    top_p=0.95,             # nucleus sampling
    temperature=0.9,        # creativity control
    do_sample=True          # enable sampling
)

# -------------------------------
# 4. Decode & Print Result
# -------------------------------
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n Generated Text:\n")
print(generated_text)
