#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up LLM Playground..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8+ and try again."
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js 16+ and try again."
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed. Please install npm and try again."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Clean up old app directory if it exists
if [ -d "app" ]; then
    echo "🗑️  Removing old app directory..."
    rm -rf app
fi

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete"

# Set up frontend
echo "💻 Setting up frontend..."
cd frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "🔧 Creating .env file..."
    cp .env.example .env
fi

cd ..
echo "✅ Frontend setup complete"

echo ""
echo "✨ Setup complete! ✨"
echo ""
echo "To start the application, run the following commands in separate terminals:"
echo ""
echo "Terminal 1 (Backend):"
echo "  source venv/bin/activate  # On Windows: .\\venv\\Scripts\\activate"
echo "  uvicorn main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Terminal 2 (Frontend):"
echo "  cd frontend"
echo "  npm start"
echo ""
echo "Then open your browser and navigate to: http://localhost:3000"
echo ""

# Make the script executable
chmod +x setup.sh
