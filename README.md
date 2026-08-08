# SyncSpace

SyncSpace is a small AI workspace companion built for the Workshop 2 chatbot
exercise. It lets a user chat about a local source tree, keeps conversation
history in SQLite, and can use OpenAI-compatible tool calling to search and
read files safely.

## Stack

- **Backend:** FastAPI, SQLite, OpenAI Python SDK
- **UI:** Streamlit
- **Providers:** Azure OpenAI, OpenAI, and custom OpenAI-compatible endpoints
- **Google Gemini:** direct Generative Language API endpoint with function calling
- **Optional integrations:** stdio MCP servers (for example GitHub, Notion, or
  Jira MCP servers)

## Run

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API and UI in separate terminals:

```bash
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

Open the Streamlit URL, select a workspace, add an API key in the sidebar, and
start chatting. API keys are held only in the Streamlit process for the active
browser session; SyncSpace does not write them to SQLite.

To prefill the OpenAI-compatible provider, copy `.env.example` to `.env` and
set `AZURE_OPENAI_API_KEY`, `OPENAI_BASE_URL`, and optionally
`SYNCSPACE_DEFAULT_MODEL`. `.env` is ignored by Git.

## Batch summarize

The **Batch summarize files** panel takes up to 20 files from the workspace and
sends one summarization prompt per file. The calls are issued concurrently
(`asyncio.gather` behind a semaphore of 4) rather than one after another, so a
batch costs roughly the slowest call instead of the sum of all of them. The
response reports both wall-clock and total model time so the difference is
visible, and a file that cannot be read is reported on its own row without
failing the rest of the batch.

## MCP

The **MCP Servers** expander accepts a local stdio command plus arguments. The
server process inherits your environment, so provider tokens should be supplied
through environment variables rather than saved in SyncSpace. Install the
server command first (for example a GitHub, Notion, or Jira MCP server), then
use **Discover tools** to verify the connection.

The current MVP discovers MCP tools and persists non-secret server metadata.
Local code navigation is always available without MCP.
