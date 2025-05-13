#!/bin/bash
set -e

mkdir -p ./milvus
cd ./milvus
sudo apt update

# Docker check and install
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo systemctl enable --now docker

    # Refresh shell to detect docker
    export PATH=$PATH:/usr/local/bin:/usr/bin
    hash -r
fi

# Docker Compose Plugin
sudo apt install -y docker-compose-plugin

# Download Milvus docker-compose file
wget -O docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml

# Run Milvus
docker compose up -d >> milvus_compose.log 2>&1

# Show running containers
docker ps
