# LLM Playground Frontend

A modern React-based frontend for interacting with the LLM Playground API.

## Features

- Clean, responsive UI with Material-UI components
- Dark/light mode support
- Real-time model responses
- Adjustable generation parameters
- Support for multiple LLM models (Phi-2, GPT-2)

## Prerequisites

- Node.js 16+ and npm/yarn
- Running LLM Playground API server (see backend README)

## Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. The application will open in your default browser at `http://localhost:3000`

## Available Scripts

- `npm start` - Start the development server
- `npm test` - Run tests
- `npm run build` - Build for production
- `npm run eject` - Eject from create-react-app (advanced)

## Environment Variables

Create a `.env` file in the frontend directory to override default settings:

```env
REACT_APP_API_URL=http://localhost:8000  # URL of the backend API
```

## Project Structure

```
frontend/
├── public/              # Static files
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Page components
│   ├── services/       # API services
│   ├── utils/          # Utility functions
│   ├── App.js          # Main application component
│   └── index.js        # Application entry point
├── .gitignore
└── package.json
```

## Deployment

To create a production build:

```bash
npm run build
```

This will create an optimized production build in the `build` directory that can be served using any static file server.
