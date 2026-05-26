from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from app.retriever import retrieve_documents

app = FastAPI()

# HARUS STRING LANGSUNG
templates = Jinja2Templates("app/templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "results": []
        }
    )


@app.post("/", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...)
):

    results = retrieve_documents(query)

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "results": results,
            "query": query
        }
    )