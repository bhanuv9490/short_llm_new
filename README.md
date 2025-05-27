# LLM Playground

A modern web application for interacting with multiple Language Learning Models (LLMs), featuring:
- **Frontend**: React-based UI with Material-UI components
- **Backend**: Python FastAPI service
- **Models**: Microsoft's Phi-2 and OpenAI's GPT-2

## Features

- Modern, responsive web interface
- Dark/light mode with system preference detection
- Real-time model responses
- Support for multiple LLM models (Phi-2, GPT-2)
- Adjustable generation parameters (temperature, max length, top-p)
- GPU acceleration support (CUDA)

## Project Structure

```
short_llm/
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React components and logic
│   ├── .env.example        # Example environment variables
│   └── package.json        # Frontend dependencies
├── models/                 # Model implementations
│   └── saved_models/       # Model classes
├── src/                    # Core Python code
│   └── main.py             # LLM interface
├── main.py                 # FastAPI application entry point
├── setup.sh                # Setup script
└── requirements.txt        # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- pip (Python package manager)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/short_llm.git
   cd short_llm
   ```

2. Run the setup script (Linux/macOS):
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   
   Or follow these steps manually:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Install frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

1. Start the backend server (in the project root directory):
   ```bash
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. In a new terminal, start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

### API Documentation

When the backend server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

### Adding New Models

1. Create a new model class in `models/saved_models/`
2. Implement the required methods (predict, generate, etc.)
3. Update the `LLMInterface` class in `src/main.py`
4. Add the new model to the frontend UI

### Testing

#### Backend Tests
```bash
pytest tests/
```

#### Frontend Tests
```bash
cd frontend
npm test
```

## Deployment

### Building for Production

1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. The production build will be in the `frontend/build` directory.

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Backend
PORT=8000

# Frontend (in frontend/.env)
REACT_APP_API_URL=http://localhost:8000
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [Material-UI](https://mui.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
- [OpenAI GPT-2](https://huggingface.co/gpt2)
