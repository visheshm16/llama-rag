from flask import Flask, request, jsonify, render_template, redirect, url_for, Response, session
from flask_cors import CORS
from io import BytesIO, StringIO
from transformers import TextIteratorStreamer
import uuid
import threading
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
NUM_DOCS = int(os.getenv("NUM_DOCS", 5))
GENERATION_LENGTH = 512
TEMPERATURE = 0.2
TOP_P = 0.95
CHUNK_CHARACTER_SIZE = 1500 # MAX 4000
CHUNK_OVERLAP = int(CHUNK_CHARACTER_SIZE * 0.1) # 10% of chunk size
# EMBEDDING_DIM = 384  # Value for sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM = 768  # Value for sentence-transformers/all-mpnet-base-v2

# Index parameters for Milvus
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}

# Initialize global variables
text_splitter = None
vector_store = None
tokenizer = None
model = None
embeddings = None
pipe = None
retriever = None

sys_prompt = """You are a helpful question answering chatbot. You will use the given context to answer user queries in concise manner.
If the given context does not help in answering user's query then let the user know that you are not able to answer their query.
Interact with user in short responses unless asked to elaborate, use context when they have a query, but do not answer question if not present in given context.
ALways format your responses using html tags (<p>, <b>, <ul>, <li>). Do NOT use markdown formatting."""

# if __name__ == '__main__' or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
def initialize_models():
    """Initialize all models and components"""
    global text_splitter, vector_store, tokenizer, model, embeddings, pipe, retriever
    
    if text_splitter is not None:  # Already initialized
        print("# INITIALIZATION ALREADY DONE #")
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

    # --- Load Embedding Model ---
    embeddings = HuggingFaceEmbeddings(model_name='./huggingface_embedder')
    print("Embedding model loaded ✅")

    # --- Define Text Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_CHARACTER_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ". "], length_function=len)

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
        search_kwargs={"k": NUM_DOCS, "fetch_k": 15, "lambda_mult": 0.65}
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
app.secret_key = "super_secret_123"
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

        # for ci in range(len(chunks)):
        #     modified_text = "passage: " + chunks[ci].page_content
        #     chunks[ci].page_content = modified_text

        st = time.time()
        vector_store.add_documents(chunks)
        et = time.time()
        print(f"Took {et - st} seconds to insert chunk(s) into Milvus!")
        
        return jsonify({"message": "Document indexed successfully", "time":str(et - st)}), 200
    except Exception as e:
        print(f"Error indexing document: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/fetchresponse', methods=['POST'])
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
    
    # Initialize session-specific conversation
    if 'conversation_id' not in session:
        session['conversation_id'] = str(uuid.uuid4())
        # session['messages'] = [{"role": "system", "content": sys_prompt}]
        session['assistant_response'] = ""
    
    # messages = session['messages'].copy()  # Copy messages to avoid modifying session directly
    assistant_response = session.get('assistant_response', "")

    # append assistant response to messages
    if len(assistant_response) > 1:
        print("Previous assistant response: ", assistant_response)
        # messages.append({"role": "assistant", "content": assistant_response})
        # Save updated state back to session
        # session['messages'] = messages
        session['assistant_response'] = assistant_response = """"""  # Reset assistant response in session
    

    print("Current conversation ID: ", session['conversation_id'])
    

    r_st = time.time()
    # relevant_docs = vector_store.similarity_search(query, k=NUM_DOCS)
    # relevant_docs = []
    relevant_docs = retriever.invoke(query)
    r_time = time.time() - r_st

    print(f"Retrieval time: {r_time}, {len(relevant_docs)} docs fetched.")

    retrieval_info = defaultdict(list)

    context = "### Relevant Context:\n"
    if not relevant_docs:
        context += "No relevant documents found.\n"

    for idx, doc in enumerate(relevant_docs):
        # print(doc)
        context += f"{idx}:\n{doc.page_content}\n(SOURCE: {doc.metadata.get('filename', 'unknown')}, {doc.metadata.get('page', 'unknown')})\n\n"

        retrieval_info[doc.metadata.get('filename', 'unknown')].append(str(doc.metadata.get('page', 'unknown')))
    
    context += "### End of Context\n"

    # print(context)
 
    msg_list = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": context},
        {"role": "user", "content": query},
    ]

    # messages.extend(msg_list)

    # Save updated state back to session
    # session['messages'] = messages

    # print("Messages prepared for model input. Length:", len(messages))
    print("Generating response...")

    def generate_stream():
        nonlocal assistant_response  # Allow modification of the outer variable
        # assistant_response += f"<p><b>Retrieval time: {r_time:.2f} s.</b></p>\n\n"
        yield f"data: <p><b>Retrieval time {r_time:.2f} s.</b></p><div>\n\n"
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run_model():
            _ = pipe(
                msg_list,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                streamer=streamer
            )

        thread = threading.Thread(target=run_model)
        thread.start()

        for chunk in streamer:
            assistant_response += chunk
            yield f"data: {chunk}\n\n"

        thread.join()  # Wait for the model generation to complete

        # Final sources section
        sources_text = "</div><p><b>Sources:</b></p>"
        for filename, pages in retrieval_info.items():
            sources_text += f"<p>File: {filename}, Pages: {', '.join(pages)}</p>"
        yield f"data: {sources_text}\n\n"
        yield "data: [DONE]\n\n"
    
    # Return the response as a stream
    # Create response with proper headers for streaming
    response = Response(
        generate_stream(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Access-Control-Allow-Origin': '*',
        }
    )
    return response

@app.route('/chatclear', methods=['POST'])
def chat_clear():
    session['conversation_id'] = str(uuid.uuid4())
    # session['messages'] = [{"role": "system", "content": sys_prompt}]
    session['assistant_response'] = ""
    print("Chat cleared for new session: ", session['conversation_id'])

    return jsonify({"message": "Chat cleared successfully"}), 200

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8000)
