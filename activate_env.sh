#!/bin/bash
# Activation script for the virtual environment

echo "Activating virtual environment..."
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "Available commands:"
echo "  python main.py                              # Run generation and validation (gpt-3.5-turbo)"
echo "  python main.py --samples 50                 # Generate 50 samples"
echo "  python main.py --model gpt-4 --samples 10   # Use GPT-4 (lower rate limits)"
echo "  python main.py --generation-only            # Run only generation phase"
echo "  python main.py --validation-only            # Run only validation phase"
echo "  python main.py stats                        # Show quick stats"
echo ""
echo "To deactivate the environment, type: deactivate"
