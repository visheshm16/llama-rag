import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()
import os
import time
import json


with open('test-prompts.json') as f:
    data = json.load(f).get('prompts', [])



# Load tokenizer and model separately
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained('huggingface_model', local_files_only=True)
model = AutoModelForCausalLM.from_pretrained('huggingface_model', torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
print("Tokenizer and model loaded.")

# Create the pipeline using the loaded tokenizer and model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
print("Pipeline created.")

print("Generating text...")
for prompt in data:
    sys_prompt = """You will will give very short responses to the user.
    You will only answer the question asked by the user.
    You will not give any additional information.
    Always format your reponses with html tags.
    Always use <p>, <ul>, <b> tags to format your responses."""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    st = time.time()
    output = pipe(prompt, max_length=512, do_sample=True, temperature=0.7)
    et = time.time() - st
    print("#" * 20)
    print("## Prompt:", prompt)
    print("#####     Time taken to generate the output:", et)
    print("Output:\n", output[0]['generated_text'])