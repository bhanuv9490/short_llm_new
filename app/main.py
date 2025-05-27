from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.main import LLMInterface

# Initialize FastAPI app
app = FastAPI(title="LLM Playground")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Mount static files
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static"
)

# Initialize templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Initialize LLM interface
llm_interface = LLMInterface()

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "phi2"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    try:
        if request.model == "phi2":
            response = llm_interface.process_text_with_llm1(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature
            )
        else:  # gpt2
            response = llm_interface.generate_text_with_llm2(
                request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            )
        
        return {"status": "success", "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
