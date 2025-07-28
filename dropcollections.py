import boto3
import os
from dotenv import load_dotenv
load_dotenv()

s3vectors = boto3.client(
    "s3vectors",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
print("S3 Vectors client created ✅\n\n")

response = s3vectors.list_vector_buckets(
    maxResults=123
)

print("BUKCETS....")
for i, bucket in enumerate(response['vectorBuckets']):
    print("*"*40)
    print(f"{i}: ",bucket["vectorBucketName"])
    print("-"*40)
    print("\tINDEXES...")
    response = s3vectors.list_indexes(
        vectorBucketName=bucket["vectorBucketName"],
        maxResults=123
    )
    for idx, index in enumerate(response['indexes']):
        print(f"\t{idx}: ", index["indexName"])
print("*"*40)
print("END OF BUCKETS\n\n")

names = input("Enter bukcet names to drop (comma separated with no space, enter 'skip' to skip deletion): ")

if not names.strip().lower() == "skip":
    names = names.split(",")

    for name in names:
        print(f"Dropping bucket {name}...")
        print("Deleting indexes first...")
        response = s3vectors.list_indexes(
            vectorBucketName=name,
            maxResults=123
        )
        index_list = [index["indexName"] for index in response['indexes']]
        for index in index_list:
            print(f"Deleting index {index}...")
            response = s3vectors.delete_index(
                vectorBucketName=name,
                indexName=index
            )
        
        print(f"Deleting bucket {name}...")
        response = s3vectors.delete_vector_bucket(
            vectorBucketName=name
        )
        print(f"Bucket {name} deleted successfully ✅\n")
    print("All specified buckets and their indexes have been deleted successfully.")
    print("*"*40)
else:
    print("Skipping deletion of buckets and indexes.")
ans = input("Create new bucket & index specified in environment variables? (y/n): ").strip().lower()

if ans == "y":
    print(f"Creating BUCKET: {os.getenv('VECTOR_BUCKET_NAME')}...")

    response = s3vectors.create_vector_bucket(
        vectorBucketName=os.getenv("VECTOR_BUCKET_NAME"),
        encryptionConfiguration={
            'sseType': 'AES256' # default encryption type
        }
    )
    print(response)
    print(f"New bucket {os.getenv('VECTOR_BUCKET_NAME')} created successfully ✅")

    print(f"Creating INDEX:{os.getenv('VECTOR_INDEX_NAME')}...")

    em_dim = int(input("Enter embedding dimension (e.g. 1024, 768, 512): ").strip())
    dis_metric = input("Enter distance metric (euclidean/cosine): ").strip().lower()

    response = s3vectors.create_index(
        vectorBucketName=os.getenv("VECTOR_BUCKET_NAME"),
        indexName=os.getenv("VECTOR_INDEX_NAME"),
        dataType='float32',
        dimension=em_dim,
        distanceMetric=dis_metric,
        metadataConfiguration={
            'nonFilterableMetadataKeys': [
                'chunk_text','text','content',
            ]
        }
    )
    print(response)
    print(f"New index {os.getenv('VECTOR_INDEX_NAME')} created successfully ✅")
else:
    print("Skipping bucket creation.")