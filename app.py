from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS

from io import BytesIO
import time

from dotenv import load_dotenv
load_dotenv()
import os
import re

import boto3

from collections import defaultdict
import uuid
import json

from langchain_core.documents import Document
from langchain.text_splitter import TokenTextSplitter

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document as DocxDocument


# --- Config ---
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
BEDROCK_LLM_MODEL_ID = os.getenv("BEDROCK_LLM_MODEL_ID")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")
NUM_DOCS = int(os.getenv("NUM_DOCS", 5))
GENERATION_LENGTH = 1024  # Max length of generated response
TEMPERATURE = 0.2
TOP_P = 0.95
CHUNK_TOKENS_SIZE = 2000 # Size of each text chunk in tokens
CHUNK_OVERLAP = int(CHUNK_TOKENS_SIZE * 0.1) # 10% of chunk size
# EMBEDDING_DIM = 384  # Value for sentence-transformers/all-MiniLM-L6-v2
# EMBEDDING_DIM = 768  # Value for sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIM = 1024  # Value for amazon.titan-embed-text-v2:0
ENCODING_NAME = "cl100k_base"  # Byte Pair Encoding (BPE)

# Initialize global variables
text_splitter = TokenTextSplitter(chunk_size=CHUNK_TOKENS_SIZE, chunk_overlap=CHUNK_OVERLAP, encoding_name=ENCODING_NAME)
print("Text splitter created ✅")

bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
print("Bedrock client created ✅")

s3vectors = boto3.client(
    "s3vectors",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
print("S3 Vectors client created ✅")

sys_prompt = """You are a helpful question-answering chatbot. Use the provided context to provide to-the-point answer to user queries. 

- If the context does not contain information relevant to the user's query, inform them that you are unable to answer.
- Respond with short answers unless the user explicitly asks for more detail. Only answer questions using the provided context — do not attempt to answer from general knowledge.
- If available, cite all the filenames and page numbers of all the sources used to answer the query at the end of your response.
- If no sources were used, no citation is needed.
- If no relevant documents were found then do not mention any sources.
- Format all responses using HTML tags (<p>, <b>, <ul>, <li>, etc.). Do NOT use Markdown formatting."""

# Embed one text chunk
def embed_text(text):
    response = bedrock.invoke_model(
        modelId=BEDROCK_EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text})
    )
    embedding = json.loads(response["body"].read())["embedding"]
    return embedding

# Query S3 vector store with user prompt
def query_prompt(prompt, top_k=5):
    """Query the vector store and return relevant chunks"""
    embedding = embed_text(prompt)
    response = s3vectors.query_vectors(
        vectorBucketName=VECTOR_BUCKET_NAME,
        indexName=VECTOR_INDEX_NAME,
        queryVector={"float32": embedding},
        topK=top_k,
        returnDistance=True,
        returnMetadata=True
    )
    
    retrieved_chunks = []
    
    for v in response["vectors"]:
        chunk_text = v["metadata"].get("chunk_text", "")
        source = v["metadata"].get("filename", "Unknown")
        page = v["metadata"].get("page", "1")
        distance = v.get("distance", 0)

        langchain_doc = Document(
            page_content=chunk_text,
            metadata={
                "filename": source,
                "page": page,
                "distance": 1 - distance,
            }
        )
        
        retrieved_chunks.append(langchain_doc)
    
    return retrieved_chunks

def batch_index_documents(documents, batch_size=10):
    """Helper function to index documents in batches.
    Convert langchain DOcuments to S3 Vectors format and index them in batches.
    }"""
    total_length = len(documents)
    number_of_batches = (total_length + batch_size - 1) // batch_size

    for i in range(number_of_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_length)
        batch = documents[start_index:end_index]

        # convert batch to list of dictionaries for S3 Vectors
        batch = [{
            "key": str(uuid.uuid4()),
            "data": {"float32": embed_text(doc.page_content)},
            "metadata": {
                "chunk_text": doc.page_content, # to be set as non filterable metadata when creating index in S3 Vectors
                "filename": doc.metadata.get("filename", "unknown"),
                "page": str(doc.metadata.get("page", "unknown"))
            }
        } for doc in batch]
        
        # Add the batch to the vector store
        response = s3vectors.put_vectors(
            vectorBucketName=VECTOR_BUCKET_NAME,
            indexName=VECTOR_INDEX_NAME,
            vectors=batch
        )
        
        print(f"Indexed batch {i + 1}/{number_of_batches} with {len(batch)} documents.")


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
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({"error": "No files provided"}), 400
    
    # Check if initialization completed
    if text_splitter is None or bedrock is None or s3vectors is None:
        return jsonify({"error": "System not fully initialized. Please try again."}), 503
    
    try:
        doc_list = []
        for file in uploaded_files:
            if file.filename.endswith('.pdf'):
                print("Processing PDF: ",file.filename)
                stream = BytesIO(file.read())
                reader = PdfReader(stream)
                pages = reader.pages

                for page_num in range(len(pages)):
                    page = pages[page_num]
                    text = page.extract_text()
                    if text:
                        text = text.replace('\n', ' ').strip()
                        doc_list.append(Document(page_content=text, metadata={"filename": file.filename, "page": page_num + 1}))
            elif file.filename.endswith('.docx'):
                print("Processing DOCX: ", file.filename)
                # Read the file content into a BytesIO stream
                stream = BytesIO(file.read())
                # Create a Document object from the stream
                docx_doc = DocxDocument(stream)

                # Extract text from all paragraphs
                for paragraph in docx_doc.paragraphs:
                    if paragraph.text.strip():
                        doc_list.append(Document(page_content=paragraph.text.strip(), metadata={"filename": file.filename, "page": 1}))
            
            elif file.filename.endswith('.txt'):
                print("Processing TXT: ", file.filename)
                text = file.read().decode('utf-8').strip()
                if text:
                    doc_list.append(Document(page_content=text, metadata={"filename": file.filename, "page": 1}))
            else:
                print("Unsupported file type, skipping: ", file.filename)
        
        if len(doc_list) == 0:
            return jsonify({"error": "No text extracted from the provided documents"}), 400

        chunks = text_splitter.split_documents(doc_list)
        for chunk in chunks:
            chunk.page_content = chunk.page_content + f"\n\n(SOURCE: {chunk.metadata.get('filename', 'unknown')}, PAGE: {chunk.metadata.get('page', 'unknown')})"
            encoded = chunk.page_content.encode("utf-8")
            print("Size in bytes: ", len(encoded))
        print(f"Created {len(chunks)} chunk(s).")
        # breakpoint()

        # for ci in range(len(chunks)):
        #     modified_text = "passage: " + chunks[ci].page_content
        #     chunks[ci].page_content = modified_text

        st = time.time()
        batch_index_documents(chunks, batch_size=10)
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

    print("Received query: ", query)

    # Check if initialization completed
    if bedrock is None or s3vectors is None:
        return jsonify({"error": "System not fully initialized. Please try again."}), 503
    
    r_st = time.time()
    # relevant_docs = vector_store.similarity_search(query, k=NUM_DOCS)
    relevant_docs = query_prompt(prompt=query, top_k=NUM_DOCS)
    r_time = time.time() - r_st

    print(f"Retrieval time: {r_time}, {len(relevant_docs)} docs fetched.")

    context = "### Relevant Context:\n"
    if not relevant_docs:
        context += "No relevant documents found.\n"

    for idx, doc in enumerate(relevant_docs):
        # print(doc)
        context += f"{idx}, RELEVANCE: {doc.metadata.get('distance', 'unknown')}\n{doc.page_content}\n-----\n"
    
    context += "### End of Context\n"

    print(context)

    messages = [
        {"role": "user", "content": [{"text":context},{"text":"### USER QUERY: "+query}]}
    ]

    print("Starting generation...")
    g_st = time.time()
    model_response = bedrock.converse(
        modelId=BEDROCK_LLM_MODEL_ID,
        messages=messages,
        system=[{"text": sys_prompt}],
        inferenceConfig={
            'maxTokens': GENERATION_LENGTH,
            'temperature': TEMPERATURE,
            'topP': TOP_P,
        },
    )
    response = model_response["output"]["message"]["content"][0]["text"]
    g_time = time.time() - g_st
    # breakpoint()

    print("Response created, took "+str(g_time)+" seconds.")
    
    print("\n\nModel reponse:\n",response)

    response = response.strip() + f"<div><pre>Retrieval time {r_time} s.</pre><pre>Generation time {g_time} s.</pre></div>"
    
    return jsonify({"message": response}), 200

if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8000)
