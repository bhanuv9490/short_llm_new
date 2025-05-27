#!/bin/bash

# Exit on error
set -e

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# Install system dependencies
echo "Installing system dependencies..."
brew install cmake pkg-config

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools
echo "Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch for Apple Silicon
echo "Installing PyTorch for Apple Silicon..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Install other requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install sentencepiece separately if needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "macOS detected, installing sentencepiece with system compiler..."
    CFLAGS="-mmacosx-version-min=11.0" pip install --no-binary :all: sentencepiece
fi

echo "
Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
