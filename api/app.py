from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import ModelService

app = FastAPI()
model_service = ModelService()


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 512


class GenerateResponse(BaseModel):
    text: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    try:
        generated_text = model_service.generate(
            request.prompt, max_length=request.max_length
        )
        return GenerateResponse(text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
