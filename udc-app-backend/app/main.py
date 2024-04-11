from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from keywords import get_keywords

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KeywordsRequest(BaseModel):
    text: str
    n_keywords: int


@app.post('/keywords')
async def process_keywords(req: KeywordsRequest):
    abstract = req.text
    n_keywords = req.n_keywords
    keywords = get_keywords(abstract, n_keywords)
    return {"keywords": keywords}


class UDCSRequest(BaseModel):
    text: str


@app.post('/udcs')
async def process_udcs(req: UDCSRequest):
    text = req.text
    udcs = [("621.317", "https://teacode.com/online/udc/62/621.317.html"),
            ("621.317", "https://teacode.com/online/udc/62/621.313.html")]
    return {"udcs": udcs}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
