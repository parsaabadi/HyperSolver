#!/bin/bash

# HyperSolver Installation Script
# This script sets up the environment and installs all dependencies

set -e

echo "🚀 Setting up HyperSolver..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.7+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
echo "Choose installation mode:"
echo "1) Minimal dependencies (recommended for most users)"
echo "2) Complete environment (exact development versions)"
read -p "Enter choice (1 or 2): " choice

case $choice in
    2)
        echo "Installing complete environment..."
        pip install -r requirements_full.txt
        ;;
    *)
        echo "Installing minimal dependencies..."
        pip install -r requirements.txt
        ;;
esac

echo ""
echo "✅ Installation complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation:"
echo "  python run.py --problem set_cover"
echo ""
echo "🎉 HyperSolver is ready to use!"
