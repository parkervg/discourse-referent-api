from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from fastapi.responses import JSONResponse
import json

from dataset import load_tacl_corpus, get_masked_refs
from helpers import load_model, load_best_state
from get_prediction_json import get_prediction_json
from utils import get_example_script as _get_example_script

"""
TODO:
    - Issue with Glove, "token not in vocab"
        - Getting weird tokens, seems like a mistake on the dataset.py script
    - Model isn't deterministic on evaluation: probably from creating the entity embedding
        - Figure out how to load model in deterministic mode
    - Have slide wheel to determine context given by random generated script
    
docker build . -t referent-api
docker run -dp 8000:8000 referent-api
"""
class ModelRequest(BaseModel):
    text: str

class ScriptRequest(BaseModel):
    script_type: str

templates = Jinja2Templates(directory="template")
app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")

origins = [
    "http://localhost",
    "http://localhost:8000" "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_model(model_path: str, corpus):
    base_model = load_model(corpus=corpus, device="cpu", model_load_dir=model_path)
    return load_best_state(model_path, base_model)


TACL_DIR = "../data/taclData"
REF_MODEL_LOAD_DIR = "../resources/ref_model"
COREF_MODEL_LOAD_DIR = "../resources/coref_model"
masked_refs = get_masked_refs(TACL_DIR)
corpus = load_tacl_corpus(TACL_DIR, masked_refs, device="cpu")
ref_model = _load_model(REF_MODEL_LOAD_DIR, corpus)
coref_model = _load_model(COREF_MODEL_LOAD_DIR, corpus)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # return FileResponse('template/index.html')
    return templates.TemplateResponse("index.html", {"request": request, "id": "hello"})


@app.post("/get_json_prediction/")
async def get_json_prediction(request: ModelRequest):
    text = request.text
    print(f"Received request: {text[:10]}")
    prediction_json = get_prediction_json(
        ref_model, coref_model, request.text, corpus.tok2id, corpus.id2tok
    )
    return JSONResponse(content=prediction_json)


@app.post("/get_example_script/")
async def get_example_script(request: ScriptRequest):
    """
    Given a script type, returns a random example of a script with the masked referent left blank.
    """
    script_type = request.script_type
    return JSONResponse({"text": _get_example_script(corpus=corpus, script_type=script_type)})

"""
curl -X 'POST' \
  'http://127.0.0.1:8000/get_json_prediction/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "hey everyone, I love to plant"
}'
"""
