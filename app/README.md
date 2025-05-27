# LLM Playground Web Interface

A modern web interface for interacting with different language models, built with FastAPI and Material You design principles.

## Features

- **Modern UI**: Clean, responsive interface with Material You design
- **Multiple Models**: Switch between Microsoft Phi-2 and GPT-2
- **Customizable Generation**: Adjust parameters like temperature, max length, and top-p
- **Dark/Light Mode**: Automatic theme switching based on system preferences
- **Real-time Chat**: Interactive chat interface with typing indicators

## Project Structure

```
app/
├── static/                 # Static files (CSS, JS, images)
│   ├── css/               # Stylesheets
│   └── js/                 # JavaScript files
├── templates/              # HTML templates
│   ├── base.html          # Base template
│   └── index.html         # Main chat interface
└── main.py                # FastAPI application
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Node.js and npm (for frontend development, optional)

### Installation

1. Install the required Python packages:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Start the development server:
   ```bash
   cd app
   uvicorn main:app --reload
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Development

### Frontend Development

The frontend is built with vanilla JavaScript and CSS. To make changes:

1. Edit files in `static/css/` and `static/js/`
2. The changes will be automatically reloaded when you save (thanks to FastAPI's auto-reload)

### Backend Development

The backend is built with FastAPI. The main endpoints are:

- `GET /`: Serves the main chat interface
- `POST /api/generate`: Handles text generation requests

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
