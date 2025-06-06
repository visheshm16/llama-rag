from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from io import BytesIO
import time
from dotenv import load_dotenv
load_dotenv()
import os
import re
import torch
from collections import defaultdict
import markdown

# --- Config ---
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "default_collection")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
NUM_DOCS = int(os.getenv("NUM_DOCS", 8))
GENERATION_LENGTH = 512
# EMBEDDING_DIM = 384  # Value for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM = 768  # Value for sentence-transformers/all-mpnet-base-v2

# Initialize global variables
text_splitter = None
vector_store = None
tokenizer = None
model = None
embeddings = None
pipe = None
retriever = None

# if __name__ == '__main__' or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
def initialize_models():
    """Initialize all models and components"""
    global text_splitter, vector_store, tokenizer, model, embeddings, pipe, retriever
    
    if text_splitter is not None:  # Already initialized
        return
    
    print("Starting library imports...")
    # from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    # from langchain_core.documents import Document
    from langchain_milvus import Milvus
    from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
    from langchain_huggingface import HuggingFaceEmbeddings
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    

    print("Done ✅")
    print("Using CUDA" if torch.cuda.is_available() else "CUDA not found, using CPU")

    # Index parameters for Milvus
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }

    # --- Load Embedding Model ---
    embeddings = HuggingFaceEmbeddings(model_name='./huggingface_embedder')
    print("Embedding model loaded ✅")

    # --- Define Text Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, separators=["\n\n", "\n", ". "], length_function=len)

    # --- Connect to Milvus ---
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print("Connected to Milvus ✅")

    # --- Check and Create Collection ---
    existing_collections = utility.list_collections()
    if COLLECTION_NAME not in existing_collections:
        print(f"Creating new collection: {COLLECTION_NAME}")
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                # Add metadata fields
                FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=400),
                FieldSchema(name="page", dtype=DataType.INT64),
            ]
        )
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        print(f"Collection '{COLLECTION_NAME}' created and loaded ✅")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists, skipping creation.")

    # --- Define Vector Store ---
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": f'http://{MILVUS_HOST}:{MILVUS_PORT}'},
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        primary_field="id",
        text_field="text",
        vector_field="embedding",
        auto_id=True,
    )
    print("Milvus VectorStore is ready ✅")

    # Maximum Marginal Relevance (MMR) - reduces redundancy
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": NUM_DOCS, "fetch_k": 20, "lambda_mult": 0.65}
    )
    print("Retriever is ready ✅")

    # Load tokenizer and model separately
    tokenizer = AutoTokenizer.from_pretrained('huggingface_model', local_files_only=True)
    print("Loaded tokenizer ✅")
    model = AutoModelForCausalLM.from_pretrained('huggingface_model', torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    print("Model loaded ✅")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    print("Pipeline created ✅")

# Initialize models when module is imported (works with both Flask dev server and Gunicorn)
initialize_models()

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploaddocument')
def upload_document():
    return render_template('upload.html')

@app.route('/indexdoc', methods=['POST'])
def index_document():
    from PyPDF2 import PdfReader
    from langchain_core.documents import Document
    
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({"error": "No files provided"}), 400
    
    # Check if initialization completed
    if text_splitter is None or vector_store is None:
        return jsonify({"error": "System not fully initialized. Please try again."}), 503
    
    try:
        doc_list = []
        for file in uploaded_files:
            print("Processing file: ",file.filename)
            stream = BytesIO(file.read())
            reader = PdfReader(stream)
            pages = reader.pages

            for page_num in range(len(pages)):
                page = pages[page_num]
                text = page.extract_text()
                if text:
                    text = text.replace('\n', ' ').strip()
                    doc_list.append(Document(page_content=text, metadata={"filename": file.filename, "page": page_num + 1}))
        
        if len(doc_list) == 0:
            return jsonify({"error": "No text extracted from the provided documents"}), 400

        chunks = text_splitter.split_documents(doc_list)
        print(f"Created {len(chunks)} chunk(s).")

        st = time.time()
        vector_store.add_documents(chunks)
        et = time.time()
        print(f"Took {et - st} seconds to insert chunk(s) into Milvus!")
        
        return jsonify({"message": "Document indexed successfully", "time":str(et - st)}), 200
    except Exception as e:
        print(f"Error indexing document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/fetchreponse', methods=['POST'])
def fetch_response():
    data = request.json
    query = data.get('query', None)
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if not isinstance(query, str):
        return jsonify({"error": "Query must be a string"}), 400
    
    query = query.strip()

    print("Recieved query: ", query)

    # Check if initialization completed
    if vector_store is None or tokenizer is None or model is None:
        return jsonify({"error": "System not fully initialized. Please try again."}), 503
    
    r_st = time.time()
    # relevant_docs = vector_store.similarity_search(query, k=NUM_DOCS)
    relevant_docs = retriever.invoke(query)
    r_time = time.time() - r_st

    print(f"Retrieval time: {r_time}, {len(relevant_docs)} docs fetched.")

    retrieval_info = defaultdict(list)

    context = "### Relevant Context:\n"
    if not relevant_docs:
        context += "No relevant documents found.\n"

    for idx, doc in enumerate(relevant_docs):
        # print(doc)
        context += f"{idx}:\n{doc.page_content}\n"

        retrieval_info[doc.metadata.get('filename', 'unknown')].append(str(doc.metadata.get('page', 'unknown')))
    
    context += "### End of Context\n"

    sys_prompt = """You are a helpful question answering chatbot. You will use the given context to answer user queries in concise manner.
If the given context does not help in answering user's query then let the user know that you are not able to answer their query.
Interact with user in short responses unless asked to elaborate, use context when they have a query, but do not answer question if not present in given context.
ALways format your responses using html tags (<p>, <b>, <ul>, <li>). Do NOT use markdown formatting."""
 
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "system", "content": context},
        {"role": "user", "content": query},
    ]
    
    # import torch
    # print("Tokenizing input string")
    # if torch.cuda.is_available():
    #     inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    # else:
    #     inputs = tokenizer(prompt, return_tensors="pt")

    print("Starting generation...")
    g_st = time.time()
    # output = model.generate(
    #     **inputs,
    #     max_new_tokens=512,
    #     temperature=0.25,      # Lower for more consistency
    #     top_p=0.95,          # Slightly higher for quality
    #     do_sample=True,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    # response = tokenizer.decode(output[0], skip_special_tokens=True)
    output = pipe(messages, max_length=GENERATION_LENGTH, do_sample=True, temperature=0.2, top_p=0.95)
    response = output[0]['generated_text'][-1]['content']
    g_time = time.time() - g_st
    # breakpoint()

    print("Response created, took "+str(g_time)+" seconds.")
    
    print("\n\nFull reponse:\n",response)
    # Extract just the assistant's response
    response_parts = response.split("<|assistant|>")
    final_response = f"<p><b>Retrieval time {r_time} s.</b></p>"
    if len(response_parts) > 1:
        # final_response += markdown.markdown(response_parts[1].strip())
        final_response += response_parts[1].strip()
    else:
        # final_response += markdown.markdown(response)
        final_response += response

    final_response +=  f"<p><b>Generation time {g_time} s.</b></p>"

    final_response = re.sub(r'<\|[^|]*\|>', '', final_response)
    final_response = re.sub(r'<\|reserved_special_token_\d+\|>', '', final_response)

    final_response +=  "<p><b>Sources:</b></p>"
    for filename in retrieval_info.keys():
        final_response += f"<p>File: {filename}, Pages: {', '.join(retrieval_info[filename])}</p>"

    print("\n\nGenerated response:\n",final_response)
    
    return jsonify({"message": final_response}), 200

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8000)
