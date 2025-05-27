from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.main import LLMInterface

# Initialize FastAPI app
app = FastAPI(
    title="LLM Playground API",
    description="API for interacting with various language models",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM interface
llm_interface = LLMInterface()

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "phi2"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

@app.post("/api/generate")
async def generate_text(request: GenerationRequest):
    """
    Generate text using the specified language model.
    
    Parameters:
    - prompt: The input text prompt
    - model: The model to use (phi2 or gpt2)
    - max_length: Maximum length of the generated text (default: 100)
    - temperature: Controls randomness (0.1-1.0, default: 0.7)
    - top_p: Nucleus sampling parameter (0.0-1.0, default: 0.9)
    
    Returns:
    - JSON response with status and generated text or error message
    """
    import asyncio
    from fastapi import HTTPException
    
    # Input validation
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
    if request.temperature <= 0 or request.temperature > 2.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 2.0")
    
    if request.max_length <= 0 or request.max_length > 1000:
        raise HTTPException(status_code=400, detail="Max length must be between 1 and 1000")
    
    try:
        # Generate response with timeout
        if request.model == "phi2":
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_interface.process_text_with_llm1,
                    request.prompt,
                    max_length=min(request.max_length, 500),  # Limit max tokens for demo
                    temperature=request.temperature
                ),
                timeout=120  # 2 minute timeout
            )
        elif request.model == "gpt2":
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_interface.generate_text_with_llm2,
                    request.prompt,
                    max_length=min(request.max_length, 500),  # Limit max tokens for demo
                    temperature=request.temperature,
                    top_p=request.top_p
                ),
                timeout=120  # 2 minute timeout
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {request.model}")
        
        return {
            "status": "success",
            "model": request.model,
            "response": response,
            "tokens_generated": len(response.split())  # Approximate
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The model is taking too long to respond."
        )
    except Exception as e:
        error_msg = str(e)
        print(f"Error in generate_text: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {error_msg}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
