import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os
import time
import json


with open('test-prompts.json') as f:
    data = json.load(f).get('prompts', [])

# --- Config ---
collection_name = "sentence_transformer_K_B"
EMBEDDING_DIM = 384  # Value for sentence-transformers/all-MiniLM-L6-v2
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}

# --- Load Embedding Model ---
embeddings = HuggingFaceEmbeddings(model_name='./huggingface_embedder')  # local folder path
print("Embedding model loaded!")
# --- Define Vector Store ---
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": 'http://localhost:19530'},
    collection_name=collection_name,
    index_params=index_params,
    text_field="text",
    vector_field="embedding",
    auto_id=True,
)
print("Milvus VectorStore is ready.")

# --- RAG test ---
test_query = "test sentence"
print("Here is the test query:", test_query)
# Perform similarity search
print("Performing similarity search test...")
results = vector_store.similarity_search_with_score(test_query, k=2)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

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
    Always format your reponses with html tags like <p>, <ul>, <b>."""

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