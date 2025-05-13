#!/bin/bash
set -e  # Exit on error

# Step 1: Ensure Python 3.10 and venv are installed
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv


# Step 2: Create venv if it doesn't exist
echo "Creating virtual environment..."
python3.10 -m venv venv


# Step 3: Activate venv
source ./venv/bin/activate

# Step 4: Install PyTorch based on GPU presence
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Installing PyTorch with CUDA support."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "No GPU detected. Installing PyTorch without CUDA."
    pip install torch torchvision
fi

# Step 5: Install other dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

# Step 6: Clean up pip cache
echo "Purging pip cache to save space..."
pip cache purge

echo "✅ Setup complete."
echo "To activate the virtual environment, run: source ./venv/bin/activate"
echo "To deactivate the virtual environment, run: deactivate"