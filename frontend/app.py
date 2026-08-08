from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv


API_URL = os.getenv("SYNCSPACE_API_URL", "http://localhost:8000")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DEFAULT_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("SYNCSPACE_DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_GOOGLE_BASE_URL = os.getenv("GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title="SyncSpace", page_icon="🧭", layout="wide")
st.title("🧭 SyncSpace")
st.caption("Your AI Workspace Companion")


def api(method: str, path: str, timeout: float = 30, **kwargs):
    response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Workspace")
    workspace = st.text_input("Local folder", value=st.session_state.get("workspace", os.getcwd()))
    st.session_state.workspace = workspace

    st.header("AI provider")
    provider_options = ["Azure OpenAI", "OpenAI", "Google Gemini", "Other (OpenAI-compatible)"]
    default_index = 3 if DEFAULT_BASE_URL else 0
    kind_label = st.selectbox("Provider", provider_options, index=default_index)
    kind = {"Azure OpenAI": "azure", "OpenAI": "openai", "Google Gemini": "google", "Other (OpenAI-compatible)": "compatible"}[kind_label]
    default_key = DEFAULT_GOOGLE_API_KEY if kind == "google" else DEFAULT_API_KEY
    api_key = st.text_input("API key", value=default_key, type="password", help="Loaded from .env when available; never written to SQLite.")
    if kind == "azure":
        base_url = st.text_input("Azure endpoint", placeholder="https://your-resource.openai.azure.com")
        model = st.text_input("Deployment name")
        api_version = st.text_input("API version", value="2024-10-21")
    elif kind == "google":
        base_url = st.text_input("Google API endpoint", value=DEFAULT_GOOGLE_BASE_URL)
        model = st.text_input("Gemini model", value=DEFAULT_GOOGLE_MODEL)
        api_version = None
    elif kind == "compatible":
        base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, placeholder="https://api.example.com/v1")
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_version = None
    else:
        base_url = None
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_version = None

    st.divider()
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = None
        st.rerun()

    with st.expander("MCP Servers"):
        st.caption("Secrets belong in the server environment, never in this form.")
        mcp_name = st.text_input("Name", key="mcp_name")
        mcp_command = st.text_input("Command", key="mcp_command", placeholder="npx")
        mcp_args = st.text_input("Arguments (space separated)", key="mcp_args", placeholder="-y @modelcontextprotocol/server-github")
        if st.button("Save & discover tools"):
            try:
                payload = {"name": mcp_name, "command": mcp_command, "args": mcp_args.split()}
                api("POST", "/mcp/servers", json=payload)
                result = api("POST", "/mcp/discover", json=payload)
                st.success(f"Connected: {len(result['tools'])} tool(s)")
                st.json(result["tools"])
            except Exception as exc:
                st.error(str(exc))


provider = {"kind": kind, "api_key": api_key, "model": model, "base_url": base_url, "api_version": api_version}
provider_ready = bool(api_key and model) and not (kind in {"azure", "compatible"} and not base_url)


with st.expander("📦 Batch summarize files", expanded=bool(st.session_state.get("batch"))):
    st.caption("Every selected file is summarized in its own model call; the calls run concurrently.")
    try:
        available_files = api("GET", "/workspace/files", params={"path": workspace})
    except Exception as exc:
        available_files = []
        st.warning(f"Could not list the workspace: {exc}")
    chosen = st.multiselect("Files", available_files, max_selections=20, key="batch_paths")
    if st.button("Summarize selected", disabled=not chosen):
        if not provider_ready:
            st.error("Please complete the provider configuration first.")
        else:
            with st.spinner(f"Summarizing {len(chosen)} file(s)…"):
                try:
                    st.session_state.batch = api(
                        "POST", "/batch/summarize",
                        json={"workspace_path": workspace, "paths": chosen, "provider": provider},
                        timeout=300,
                    )
                except Exception as exc:
                    st.session_state.batch = None
                    st.error(str(exc))

    batch = st.session_state.get("batch")
    if batch:
        st.caption(
            f"{len(batch['results'])} file(s) in **{batch['total_seconds']}s** wall clock, "
            f"{batch['sequential_seconds']}s of model time at concurrency {batch['concurrency']}."
        )
        for item in batch["results"]:
            if "error" in item:
                st.error(f"`{item['path']}` — {item['error']}")
            else:
                st.markdown(f"**`{item['path']}`** · {item['seconds']}s")
                st.text(item["summary"])


try:
    existing_messages = api("GET", f"/sessions/{st.session_state.session_id}/messages") if st.session_state.get("session_id") else []
except Exception:
    existing_messages = []

for message in existing_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the selected source code…")
if prompt:
    if not provider_ready:
        st.error("Please complete the provider configuration first.")
        st.stop()
    with st.chat_message("user"):
        st.markdown(prompt)
    payload = {"session_id": st.session_state.get("session_id"), "workspace_path": workspace, "message": prompt, "provider": provider}
    with st.chat_message("assistant"):
        output = st.empty()
        status = st.status("SyncSpace is inspecting your workspace…", expanded=True)
        answer = ""
        try:
            with httpx.stream("POST", f"{API_URL}/chat", json=payload, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        answer += event["content"]
                        output.markdown(answer + "▌")
                    elif event["type"] == "tool":
                        status.write(f"Using `{event['name']}`")
                    elif event["type"] == "session":
                        st.session_state.session_id = event["session_id"]
                    elif event["type"] == "error":
                        raise RuntimeError(event["content"])
            output.markdown(answer)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Request failed", state="error")
            st.error(str(exc))
