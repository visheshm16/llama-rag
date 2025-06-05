from dotenv import load_dotenv
load_dotenv()
import os
import time

from huggingface_hub import login, snapshot_download
login(token=os.getenv('HF_TOKEN'))


lm_model_id = "meta-llama/Llama-3.1-8B-Instruct"
embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"

# -----------------------------------------------------------------------------------------------

st = time.time()
model_path = snapshot_download(
    repo_id=lm_model_id,
    local_dir="huggingface_model"
)
et = time.time() - st
print("Time taken to download the model:", et)

print("Model downloaded to:", model_path)

# -----------------------------------------------------------------------------------------------

print(f"Starting download of {embedding_model_id}...")
st = time.time()
model_path = snapshot_download(
    repo_id=embedding_model_id,
    local_dir="huggingface_embedder",
    local_dir_use_symlinks=False  # Ensure actual files are downloaded
)
et = time.time() - st
print(f"Time taken to download the model: {et:.2f} seconds")
print(f"Model downloaded to: {model_path}")