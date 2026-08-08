from __future__ import annotations

import json
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import storage
from .agent import run_agent
from .batch import MAX_CONCURRENCY, summarize_files
from .mcp_service import discover_tools
from .models import BatchRequest, ChatRequest, McpServerInput, NewSession
from .workspace import list_files, read_file, search_code


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.initialize()
    yield


app = FastAPI(title="SyncSpace API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/sessions")
def create_session(payload: NewSession) -> dict:
    return storage.create_session(payload.title)


@app.get("/sessions")
def sessions() -> list[dict]:
    return storage.list_sessions()


@app.get("/sessions/{session_id}/messages")
def messages(session_id: str) -> list[dict]:
    return storage.get_messages(session_id)


@app.get("/workspace/files")
def workspace_files(path: str) -> list[str]:
    try:
        return list_files(path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspace/search")
def workspace_search(path: str, query: str) -> list[dict]:
    try:
        return search_code(path, query)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/workspace/read")
def workspace_read(path: str, file: str, start_line: int = 1, end_line: int = 250) -> dict:
    try:
        return read_file(path, file, start_line, end_line)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/chat")
async def chat(payload: ChatRequest):
    title = " ".join(payload.message.split())[:60] or "New conversation"
    session = storage.create_session(title) if not payload.session_id else {"id": payload.session_id}
    history = storage.get_messages(session["id"])
    storage.add_message(session["id"], "user", payload.message)

    async def events():
        final_answer = ""
        try:
            async for event in run_agent(payload.provider, payload.workspace_path, history, payload.message, storage.list_mcp_servers()):
                if event["type"] == "done":
                    final_answer = event["content"]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        if final_answer:
            storage.add_message(session["id"], "assistant", final_answer)
        yield f"data: {json.dumps({'type': 'session', 'session_id': session['id']})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/batch/summarize")
async def batch_summarize(payload: BatchRequest) -> dict:
    """Summarise several files in one batched round of concurrent model calls.

    `total_seconds` is wall-clock for the whole batch while each result carries
    its own duration, so the saving over a sequential run is visible in the
    response itself.
    """
    started = perf_counter()
    results = await summarize_files(payload.provider, payload.workspace_path, payload.paths)
    return {
        "total_seconds": round(perf_counter() - started, 2),
        "sequential_seconds": round(sum(item["seconds"] for item in results), 2),
        "concurrency": MAX_CONCURRENCY,
        "results": results,
    }


@app.get("/mcp/servers")
def mcp_servers() -> list[dict]:
    return storage.list_mcp_servers()


@app.post("/mcp/servers")
def save_mcp_server(payload: McpServerInput) -> dict:
    return storage.save_mcp_server(payload.name, payload.command, payload.args)


@app.post("/mcp/discover")
async def mcp_discover(payload: McpServerInput) -> dict:
    try:
        return {"tools": await discover_tools(payload.command, payload.args)}
    except Exception as exc:
        raise HTTPException(400, f"Could not connect to MCP server: {exc}") from exc
