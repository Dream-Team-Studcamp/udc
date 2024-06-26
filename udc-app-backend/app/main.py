from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from keywords import get_keywords
from udcs import get_udcs

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=[
        "Access-Control-Allow-Headers",
        "Content-Type",
        "Authorization",
        "Access-Control-Allow-Origin",
        "Set-Cookie",
        "Allow-Origin-With-Credentials",
        "Access-Control-Allow-Credentials",
    ],
)


class KeywordsRequest(BaseModel):
    text: str
    n_keywords: int


@app.post('/keywords')
async def process_keywords(req: KeywordsRequest):
    text = req.text
    n_keywords = req.n_keywords
    if text.strip() == '' or n_keywords == 0:
        keywords = []
    else:
        keywords = get_keywords(text, n_keywords)
    return {"keywords": keywords}


class UDCSRequest(BaseModel):
    text: str


@app.post('/udcs')
async def process_udcs(req: UDCSRequest):
    text = req.text
    udcs = [] if text.strip() == '' else get_udcs(text)
    return {"udcs": udcs}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
