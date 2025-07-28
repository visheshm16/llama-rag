# Install milvus-backup CLI
mkdir milvus-backup && cd milvus-backup
wget https://github.com/zilliztech/milvus-backup/releases/download/v0.5.7/milvus-backup_Linux_x86_64.tar.gz

# Extract the downloaded tarball
tar -xvzf milvus-backup_Linux_x86_64.tar.gz

chmod +x milvus-backup
touch backup.yaml
####################################################################################
# milvus:
#   address: localhost
#   port: 19530

# etcdEndpoints: localhost:2379
# minioAddress: localhost:9000
# minioAccessKeyID: minioadmin
# minioSecretAccessKey: minioadmin
# minioUseSSL: false
# bucketName: milvus-bucket
# backupDir: my_backup_folder

# log:
#   level: info
#   file:
#     rootpath: ./logs
#     maxsize: 300
#     maxage: 10
#     maxbackups: 20
#     compress: true
####################################################################################

# Create the Milvus Backup (Full)
mkdir -p logs
./milvus-backup delete --name my_backup
./milvus-backup create --config backup.yaml --name my_backup

# List available backups
./milvus-backup list --config backup.yaml

# Set up MinIO client
wget https://dl.min.io/client/mc/release/linux-amd64/mc
chmod +x mc
./mc --version
./mc alias set localminio http://localhost:9000 minioadmin minioadmin
./mc ls localminio
# Copy the backup to MinIO
./mc cp --recursive localminio/ ./localminio
# (Optional) Create a Bucket if it doesn’t exist
./mc rm --recursive --force localminio/milvus-bucket # Remove existing bucket contents
./mc rm --recursive --force localminio/a-bucket # Remove existing bucket contents
./mc rb localminio/milvus-bucket # Remove existing bucket itself
./mc rb localminio/a-bucket # Remove existing bucket itself
# Recreate bucket
./mc mb localminio/milvus-bucket
./mc mb localminio/a-bucket

cd localminio/a-bucket/
../../mc cp --recursive . localminio/
cd localminio/milvus-bucket/
../../mc cp --recursive . localminio/

# Create collection from backup
./milvus-backup restore --config backup.yaml --name my_backup --drop_exist_collection --restore_index