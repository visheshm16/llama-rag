from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# --- Connect to Milvus ---
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)
print("Connected to Milvus!")


existing_collections = utility.list_collections()

print(existing_collections)



# for c in existing_collections:
#     utility.drop_collection(c)