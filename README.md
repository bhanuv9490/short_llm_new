# LLM Playground

A Python-based web interface for interacting with multiple Language Learning Models (LLMs), featuring Microsoft's Phi-2 and OpenAI's GPT-2 models.

## Features

- Interactive web interface with Material You design
- Support for multiple LLM models (Phi-2, GPT-2)
- Real-time chat interface with typing indicators
- Adjustable generation parameters (temperature, max length, top-p)
- Dark/light mode with system preference detection
- Responsive design for desktop and mobile
- GPU acceleration support (CUDA)

## Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/short_llm.git
   cd short_llm
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Running the Application

Start the development server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open your browser and navigate to:
```
http://localhost:8000
```

## Project Structure

```
short_llm/
├── app/                    # Web application
│   ├── static/             # Static files (CSS, JS, images)
│   ├── templates/          # HTML templates
│   └── main.py             # FastAPI application
├── models/                 # Model implementations
│   └── saved_models/       # Saved model configurations
├── src/                    # Source code
│   └── main.py             # Core application logic
├── tests/                  # Test files
├── .gitignore             # Git ignore rules
├── requirements.txt        # Project dependencies
└── setup.py               # Package configuration
```

## Available Models

1. **Phi-2** - Microsoft's 2.7B parameter language model
2. **GPT-2** - OpenAI's 1.5B parameter language model

## Development

### Adding New Models

1. Create a new model class in `models/saved_models/`
2. Implement the required methods (predict, generate, etc.)
3. Update the `LLMInterface` class in `src/main.py`
4. Update the web interface to include the new model

### Testing

Run tests using pytest:

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [Material You](https://m3.material.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
- [OpenAI GPT-2](https://huggingface.co/gpt2)

# Initialize the interface (automatically uses GPU if available)
interface = LLMInterface()

# Process text with Microsoft Phi-2
result = interface.process_text_with_llm1(
    "Explain quantum computing in simple terms",
    max_length=150,
    temperature=0.7
)
print("Phi-2 Response:", result)

# Generate text with GPT-2
generated = interface.generate_text_with_llm2(
    "Once upon a time in a land far away,",
    max_length=200,
    temperature=0.8,
    top_p=0.9
)
print("GPT-2 Response:", generated)
```

### Command Line Interface

You can also use the built-in CLI to test the models:

```bash
# Test both models with default prompt
python -m src.main

# Test only Phi-2 with a custom prompt
python -m src.main --mode phi2 --prompt "Explain the concept of artificial intelligence"

# Test only GPT-2 with a custom prompt and parameters
python -m src.main --mode gpt2 --prompt "Write a short story about" --temperature 0.8 --max_length 200
```

### Advanced Usage

Both models support the following parameters for text generation:

- `max_length`: Maximum number of tokens to generate (default: 100)
- `temperature`: Controls randomness (lower = more deterministic, default: 0.7)
- `top_p`: Nucleus sampling parameter (0.0-1.0, default: 0.9 for GPT-2)

Example with custom parameters:

```python
# More deterministic output
result = interface.process_text_with_llm1(
    "What is machine learning?",
    temperature=0.3,
    max_length=50
)

# More creative output
creative_text = interface.generate_text_with_llm2(
    "In a futuristic city,",
    temperature=0.9,
    top_p=0.95,
    max_length=150
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/
```

### Code Style

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.

## License

[Specify License]
