
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from configs.default import APIConfig, ModelConfig
from src.model.inference import load_finetuned_model, generate_response


_model = None
_tokenizer = None


class ChatRequest(BaseModel):

    prompt: str = Field(..., min_length=1, max_length=2048)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ChatResponse(BaseModel):

    response: str
    prompt: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    api_config = APIConfig()
    model_config = ModelConfig()
    _model, _tokenizer = load_finetuned_model(model_config.model_id, api_config.model_path)
    yield
    _model, _tokenizer = None, None


app = FastAPI(title="Fine-Tuned LLM API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": _model is not None}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    response = generate_response(
        _model, _tokenizer, request.prompt,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    return ChatResponse(response=response, prompt=request.prompt)

def start_server():

    """Run the API server"""

    config = APIConfig()
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    start_server()