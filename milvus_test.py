from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

# Load environment variables
load_dotenv()

# --- Config ---
collection_name = "sentence_transformer_K_B"
EMBEDDING_DIM = 384  # Value for sentence-transformers/all-MiniLM-L6-v2

# --- Connect to Milvus ---
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)
print("Connected to Milvus!")

# --- Check and Create Collection ---
existing_collections = utility.list_collections()
if collection_name not in existing_collections:
    print(f"Creating new collection: {collection_name}")
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        ]
    )
    collection = Collection(name=collection_name, schema=schema)

    # Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    print(f"Collection '{collection_name}' created and loaded.")
else:
    print(f"Collection '{collection_name}' already exists, skipping creation.")

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

# --- Example Test Insert ---
test_text = ["This is a test sentence for embedding"]
documents = [Document(page_content=text) for text in test_text]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunk(s).")

vector_store.add_documents(chunks)
print("Inserted chunk(s) into Milvus!")





# --- Verify Inserted Data ---

collections = utility.list_collections()
print("Collections in Milvus:", collections)

collection = Collection(name=collection_name)
collection.load()
collection.flush() # forces Milvus to persist all in-memory data to disk and make it visible for queries
# Get entity count
print(f"Total entities in {collections[0]}: {collection.num_entities}")

# Fetch first 5 records
results = collection.query(
    expr="id >= 0",  # Query condition
    output_fields=["id", "text"],  # Fields to retrieve
    limit=5
)

# print("Sample Data from Milvus:", results)

index_info = collection.indexes
print("Index Info:", index_info)