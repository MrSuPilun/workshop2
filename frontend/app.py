from __future__ import annotations

import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = os.getenv("SYNCSPACE_API_URL", "http://localhost:8000")
DEFAULT_AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", DEFAULT_AZURE_API_KEY)
DEFAULT_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
DEFAULT_MODEL = os.getenv("SYNCSPACE_DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_GOOGLE_BASE_URL = os.getenv(
    "GOOGLE_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
DEFAULT_GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
DEFAULT_AZURE_BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT", "")

PROVIDER_OPTIONS = [
    "Azure OpenAI",
    "OpenAI",
    "Google Gemini",
    "Other (OpenAI-compatible)",
]
PROVIDER_KINDS = {
    "Azure OpenAI": "azure",
    "OpenAI": "openai",
    "Google Gemini": "google",
    "Other (OpenAI-compatible)": "compatible",
}
DEFAULT_PROVIDER_LABEL = (
    "Other (OpenAI-compatible)" if DEFAULT_BASE_URL else "Azure OpenAI"
)
VALID_PAGES = {"chat", "workspace", "history", "settings", "profile"}

# Streamlit drops a widget's session_state entry on any run where that widget is
# not rendered. Each of these is built on a single page but has to outlive
# navigation, so they are re-asserted as plain state before any widget is built.
PERSISTED_WIDGET_KEYS = (
    "workspace",
    "provider_kind",
    "api_key_azure",
    "api_key_openai",
    "api_key_google",
    "api_key_compatible",
    "azure_base_url",
    "azure_model",
    "azure_api_version",
    "google_base_url",
    "google_model",
    "compatible_base_url",
    "compatible_model",
    "openai_model",
    "batch_paths",
    "history_search",
)

st.set_page_config(
    page_title="SyncSpace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    f"<style>{(Path(__file__).with_name('styles.css')).read_text()}</style>",
    unsafe_allow_html=True,
)


def api(method: str, path: str, **kwargs):
    timeout = kwargs.pop("timeout", 30)
    response = httpx.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10, show_spinner=False)
def backend_is_ready(api_url: str) -> bool:
    try:
        response = httpx.get(f"{api_url}/health", timeout=2)
        response.raise_for_status()
        return response.json().get("status") == "ok"
    except (httpx.HTTPError, ValueError):
        return False


def initialise_state() -> None:
    defaults = {
        "page": "chat",
        "session_id": None,
        "workspace": os.getcwd(),
        "provider_kind": DEFAULT_PROVIDER_LABEL,
        "api_key_azure": DEFAULT_AZURE_API_KEY,
        "api_key_openai": DEFAULT_OPENAI_API_KEY,
        "api_key_google": DEFAULT_GOOGLE_API_KEY,
        "api_key_compatible": DEFAULT_AZURE_API_KEY,
        "azure_base_url": DEFAULT_AZURE_BASE_URL,
        "azure_model": "",
        "azure_api_version": "2024-10-21",
        "google_base_url": DEFAULT_GOOGLE_BASE_URL,
        "google_model": DEFAULT_GOOGLE_MODEL,
        "compatible_base_url": DEFAULT_BASE_URL,
        "compatible_model": DEFAULT_MODEL,
        "openai_model": DEFAULT_MODEL,
        "indexed_workspace": None,
        "indexed_files": [],
        "history_search": "",
        "batch": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    legacy_provider = st.session_state.get("provider_kind")
    if legacy_provider == "Khác (tương thích OpenAI)":
        st.session_state.provider_kind = "Other (OpenAI-compatible)"
    elif legacy_provider not in PROVIDER_OPTIONS:
        st.session_state.provider_kind = DEFAULT_PROVIDER_LABEL


def keep_widget_state() -> None:
    """Re-assert widget values as plain state so navigation cannot discard them.

    Writing a key back to itself moves it out of Streamlit's widget bookkeeping
    and into ordinary session state, which is never cleaned up for being absent
    from a run. Must run before any widget is instantiated.
    """
    for key in PERSISTED_WIDGET_KEYS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def apply_query_navigation() -> None:
    query_page = st.query_params.get("page")
    query_session = st.query_params.get("session")
    signature = (query_page, query_session)
    if signature == st.session_state.get("_query_signature"):
        return

    if query_page in VALID_PAGES:
        st.session_state.page = query_page
    if query_session:
        st.session_state.session_id = query_session
        st.session_state.page = "chat"
    st.session_state._query_signature = signature


def navigate(page: str, session_id: str | None = None) -> None:
    st.query_params.clear()
    st.session_state._query_signature = (None, None)
    st.session_state.page = page
    if page == "chat":
        st.session_state.session_id = session_id
    st.rerun()


def current_provider_config() -> dict:
    kind_label = st.session_state.get("provider_kind", DEFAULT_PROVIDER_LABEL)
    kind = PROVIDER_KINDS[kind_label]
    api_key = st.session_state.get(f"api_key_{kind}", "")

    if kind == "azure":
        return {
            "kind": kind,
            "api_key": api_key,
            "model": st.session_state.get("azure_model", ""),
            "base_url": st.session_state.get("azure_base_url", ""),
            "api_version": st.session_state.get("azure_api_version", "2024-10-21"),
        }
    if kind == "google":
        return {
            "kind": kind,
            "api_key": api_key,
            "model": st.session_state.get("google_model", ""),
            "base_url": st.session_state.get("google_base_url", ""),
            "api_version": None,
        }
    if kind == "compatible":
        return {
            "kind": kind,
            "api_key": api_key,
            "model": st.session_state.get("compatible_model", ""),
            "base_url": st.session_state.get("compatible_base_url", ""),
            "api_version": None,
        }
    return {
        "kind": kind,
        "api_key": api_key,
        "model": st.session_state.get("openai_model", ""),
        "base_url": None,
        "api_version": None,
    }


def provider_is_ready(provider: dict) -> bool:
    return bool(
        provider["api_key"]
        and provider["model"]
        and (
            provider["kind"] not in {"azure", "compatible"}
            or provider["base_url"]
        )
    )


def short_title(value: str, limit: int = 42) -> str:
    cleaned = " ".join(value.split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"


def render_topbar(context: str) -> None:
    st.markdown(
        f"""
        <header class="sync-topbar">
            <div class="sync-topbar-context">
                <span class="sync-terminal">&gt;_</span>
                <span>{html.escape(context)}</span>
            </div>
            <div class="sync-topbar-actions">
                <span title="Notifications">◌</span>
                <span class="sync-help" title="Help">?</span>
                <a class="sync-index-link" href="?page=workspace">Index Source</a>
            </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_page_heading(title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="sync-page-heading">
            <h1 class="sync-page-title">{html.escape(title)}</h1>
            <p class="sync-page-caption">{html.escape(caption)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(backend_ready: bool) -> None:
    page = st.session_state.page
    with st.sidebar:
        st.markdown(
            """
            <div class="sync-brand">
                <div class="sync-logo">S</div>
                <div>
                    <div class="sync-brand-name">SyncSpace</div>
                    <div class="sync-brand-caption">AI Companion</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "New Chat",
            icon=":material/add:",
            key="new_chat",
            type="primary",
            use_container_width=True,
        ):
            navigate("chat", None)

        nav_items = [
            ("workspace", "Workspace", ":material/folder_open:"),
            ("history", "History", ":material/history:"),
            ("settings", "Settings", ":material/settings:"),
            ("profile", "Profile", ":material/person:"),
        ]
        for target, label, icon in nav_items:
            is_active = page == target or (target == "workspace" and page == "chat")
            if st.button(
                label,
                icon=icon,
                key=f"nav_{target}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                navigate(target)

        dot_class = "" if backend_ready else " offline"
        status_text = "Sync Status: Active" if backend_ready else "Sync Status: Offline"
        st.markdown(
            f"""
            <div class="sync-sidebar-footer">
                <div class="status">
                    <span class="status-dot{dot_class}"></span>
                    <span>{status_text}</span>
                </div>
                <div class="sync-user-row">
                    <span class="sync-avatar">HG</span>
                    <span>User profile</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def model_settings_form() -> None:
    provider_column, key_column = st.columns([0.34, 0.66])
    with provider_column:
        kind_label = st.selectbox("Provider", PROVIDER_OPTIONS, key="provider_kind")
    kind = PROVIDER_KINDS[kind_label]
    with key_column:
        st.text_input(
            "API Key",
            type="password",
            key=f"api_key_{kind}",
            placeholder="sk-...",
            help="Kept only in this browser session and never written to SQLite.",
        )

    if kind == "azure":
        endpoint_column, model_column = st.columns([0.66, 0.34])
        with endpoint_column:
            st.text_input(
                "Azure endpoint",
                placeholder="https://your-resource.openai.azure.com",
                key="azure_base_url",
            )
        with model_column:
            st.text_input("Deployment name", key="azure_model")
        with st.expander("Advanced options"):
            st.text_input("API version", key="azure_api_version")
    elif kind == "google":
        endpoint_column, model_column = st.columns([0.66, 0.34])
        with endpoint_column:
            st.text_input("Google API endpoint", key="google_base_url")
        with model_column:
            st.text_input("Gemini model", key="google_model")
    elif kind == "compatible":
        endpoint_column, model_column = st.columns([0.66, 0.34])
        with endpoint_column:
            st.text_input(
                "Base URL",
                placeholder="https://api.example.com/v1",
                key="compatible_base_url",
            )
        with model_column:
            st.text_input("Model", key="compatible_model")
    else:
        st.text_input("Model", key="openai_model")

    _, action_column = st.columns([0.72, 0.28])
    with action_column:
        verify = st.button(
            "Verify Configuration",
            icon=":material/check_circle:",
            key="verify_provider",
            type="primary",
            use_container_width=True,
        )
    if verify:
        provider = current_provider_config()
        if not provider_is_ready(provider):
            st.warning("Add the API key, model, and endpoint required by this provider.")
        else:
            with st.spinner("Sending a test request to the provider…"):
                try:
                    result = api("POST", "/provider/verify", json=provider, timeout=60)
                except Exception as exc:
                    result = {"ok": False, "detail": f"Could not reach the SyncSpace API: {exc}"}
            if result["ok"]:
                st.success(result["detail"])
            else:
                st.error(result["detail"])


def render_workspace_page(backend_ready: bool) -> None:
    workspace = st.session_state.workspace
    render_topbar("Workspace source")
    render_page_heading(
        "Source Indexing",
        "Connect a local repository so SyncSpace can provide contextual AI assistance, code navigation, and intelligent search.",
    )

    with st.container(border=True):
        source_column, safe_column = st.columns([0.68, 0.32])
        with source_column:
            workspace = st.text_input(
                "Source directory",
                key="workspace",
                placeholder="/path/to/project",
                help="SyncSpace only reads files inside this directory.",
            )
            st.markdown(
                f'<div class="sync-source-path">~ / {html.escape(workspace)}</div>',
                unsafe_allow_html=True,
            )
        with safe_column:
            st.markdown(
                """
                <div class="sync-safe-mode">
                    <span class="sync-safe-icon">◇</span>
                    <div>
                        <strong>Safe Mode</strong>
                        <span>Read-only tool calling</span>
                    </div>
                    <span class="sync-toggle-on"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        workspace_ready = Path(workspace).expanduser().is_dir()
        action_column, status_column = st.columns([0.36, 0.64])
        with action_column:
            index_clicked = st.button(
                "Index Source",
                icon=":material/play_arrow:",
                key="index_workspace",
                type="primary",
                disabled=not (backend_ready and workspace_ready),
                use_container_width=True,
            )
        with status_column:
            if not backend_ready:
                st.error("SyncSpace API is offline.")
            elif not workspace_ready:
                st.error("Select an existing directory to continue.")
            else:
                st.caption("Safe mode is active. Source files remain read-only.")

    if index_clicked:
        try:
            files = api("GET", "/workspace/files", params={"path": workspace})
            st.session_state.indexed_workspace = str(Path(workspace).expanduser().resolve())
            st.session_state.indexed_files = files
            # Selections and summaries belong to the previous index; drop them
            # before the multiselect below is built from the new file list.
            st.session_state.pop("batch_paths", None)
            st.session_state.batch = None
        except Exception as exc:
            st.error(f"Could not inspect the workspace: {exc}")

    indexed_path = st.session_state.get("indexed_workspace")
    indexed_files = st.session_state.get("indexed_files", [])
    current_path = str(Path(workspace).expanduser().resolve()) if workspace_ready else None
    is_indexed = bool(current_path and indexed_path == current_path)

    with st.container(border=True):
        state_title = "Source Ready" if is_indexed else "Ready to Index"
        state_icon = "↻" if is_indexed else "⌁"
        st.markdown(
            f'<div class="sync-panel-title"><span class="violet">{state_icon}</span>{state_title}</div>',
            unsafe_allow_html=True,
        )
        if is_indexed:
            st.progress(100, text=f"{len(indexed_files)} files available for contextual search")
            log_lines = "".join(
                f'<div><span class="accent">[READY]</span> {html.escape(path)}</div>'
                for path in indexed_files[:6]
            )
            st.markdown(
                f'<div class="sync-terminal-log">{log_lines}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="sync-section-copy">Choose a valid source directory and run indexing to preview the files available to the assistant.</p>',
                unsafe_allow_html=True,
            )

    provider = current_provider_config()
    provider_ready = provider_is_ready(provider)
    with st.container(border=True):
        st.markdown(
            '<div class="sync-panel-title"><span class="violet">≡</span>Batch Summaries</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sync-section-copy">Every selected file gets its own summarization call. '
            'The calls are issued concurrently, so a batch costs about as much as its slowest '
            'file rather than the sum of them all.</p>',
            unsafe_allow_html=True,
        )
        chosen = st.multiselect(
            "Files to summarize",
            indexed_files,
            max_selections=20,
            key="batch_paths",
            disabled=not is_indexed,
            placeholder="Choose up to 20 files" if is_indexed else "Index the source first",
        )
        if st.button(
            "Summarize selected",
            icon=":material/bolt:",
            key="run_batch",
            type="primary",
            disabled=not (chosen and backend_ready and provider_ready),
            use_container_width=True,
        ):
            with st.spinner(f"Summarizing {len(chosen)} file(s)…"):
                try:
                    st.session_state.batch = api(
                        "POST",
                        "/batch/summarize",
                        timeout=300,
                        json={"workspace_path": workspace, "paths": chosen, "provider": provider},
                    )
                except Exception as exc:
                    st.session_state.batch = None
                    st.error(f"Batch failed: {exc}")
        if is_indexed and not provider_ready:
            st.caption("Add a model and an API key on the Settings page to enable batch summaries.")

        batch = st.session_state.get("batch")
        if batch:
            saved = batch["sequential_seconds"] - batch["total_seconds"]
            wall, model_time, gain = st.columns(3)
            wall.metric("Wall clock", f"{batch['total_seconds']}s")
            model_time.metric("Model time", f"{batch['sequential_seconds']}s")
            gain.metric("Saved", f"{saved:.2f}s", f"concurrency {batch['concurrency']}", delta_color="off")
            for item in batch["results"]:
                if "error" in item:
                    st.error(f"{item['path']} — {item['error']}")
                else:
                    with st.expander(f"{item['path']} · {item['seconds']}s"):
                        st.text(item["summary"])

    st.markdown(
        '<div class="sync-panel-title" style="border:0;margin-top:24px;margin-bottom:12px;">↻ Recent Workspace</div>',
        unsafe_allow_html=True,
    )
    groups = Counter(path.split("/", 1)[0] for path in indexed_files)
    if groups:
        columns = st.columns(min(3, len(groups)))
        for column, (name, count) in zip(columns, groups.most_common(3)):
            with column:
                st.markdown(
                    f"""
                    <div class="sync-workspace-card">
                        <h4>▣ {html.escape(name)}</h4>
                        <div class="sync-card-path">{html.escape(workspace)}</div>
                        <div class="sync-card-meta">{count} indexed files</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            f"""
            <div class="sync-workspace-card">
                <h4>▣ {html.escape(Path(workspace).expanduser().name or 'No source selected')}</h4>
                <div class="sync-card-path">{html.escape(workspace)}</div>
                <div class="sync-card-meta">Waiting for source indexing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history_page(sessions: list[dict]) -> None:
    render_topbar("SyncSpace")
    render_page_heading("Conversation History", "Manage and resume past sessions.")
    search = st.text_input(
        "Search conversations",
        key="history_search",
        placeholder="Search conversations...",
        label_visibility="collapsed",
    ).strip().lower()
    filtered_sessions = [
        session for session in sessions if not search or search in session["title"].lower()
    ]

    if not filtered_sessions:
        st.info("No conversations match this search yet.")
        return

    now = datetime.now().astimezone()
    groups: dict[str, list[tuple[dict, datetime]]] = defaultdict(list)
    for session in filtered_sessions:
        try:
            created = datetime.fromisoformat(session["created_at"]).astimezone()
        except (TypeError, ValueError):
            created = now
        if created.date() == now.date():
            label = "Today"
        elif created.date() == (now - timedelta(days=1)).date():
            label = "Yesterday"
        else:
            label = "Earlier"
        groups[label].append((session, created))

    for label in ("Today", "Yesterday", "Earlier"):
        items = groups.get(label, [])
        if not items:
            continue
        st.markdown(f'<div class="sync-history-group">{label}</div>', unsafe_allow_html=True)
        for start in range(0, len(items), 2):
            columns = st.columns(2)
            for column, (session, created) in zip(columns, items[start : start + 2]):
                with column:
                    title = short_title(session["title"])
                    href = f"?page=chat&session={quote(session['id'])}"
                    st.markdown(
                        f"""
                        <a class="sync-history-card" href="{href}">
                            <h4>{html.escape(title)}</h4>
                            <div class="sync-card-path">⌘ {html.escape(Path(st.session_state.workspace).name)}</div>
                            <div class="sync-card-meta">{created.strftime('%b %d, %Y · %H:%M')}</div>
                        </a>
                        """,
                        unsafe_allow_html=True,
                    )


def render_settings_page(backend_ready: bool, mcp_servers: list[dict]) -> None:
    render_topbar("Preferences")
    render_page_heading(
        "Settings", "Configure your workspace environment and model integrations."
    )

    with st.container(border=True):
        st.markdown(
            '<div class="sync-panel-title"><span class="violet">▣</span>Model Provider</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sync-note">API credentials are kept only in this browser session. SyncSpace never writes them to SQLite.</div>',
            unsafe_allow_html=True,
        )
        model_settings_form()

    tools_column, info_column = st.columns([0.52, 0.48])
    with tools_column:
        with st.container(border=True):
            st.markdown(
                '<div class="sync-panel-title"><span class="orange">⌘</span>Tool Configuration</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="sync-section-copy">Read-only file tools available to the AI agent.</p>',
                unsafe_allow_html=True,
            )
            tools = [
                ("⌕", "File Search", "tool: search_code"),
                ("▤", "Read File Content", "tool: read_file"),
                ("☷", "List Directory", "tool: list_files"),
            ]
            for icon, name, tool_name in tools:
                st.markdown(
                    f"""
                    <div class="sync-tool-row">
                        <span class="sync-tool-icon">{icon}</span>
                        <div class="sync-tool-copy">
                            <strong>{name}</strong><span>{tool_name}</span>
                        </div>
                        <span class="sync-tool-badge">ENABLED</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.expander(f"MCP Servers · {len(mcp_servers)} configured"):
                if mcp_servers:
                    for server in mcp_servers:
                        st.markdown(
                            f"""
                            <div class="sync-tool-row">
                                <span class="sync-tool-icon">◇</span>
                                <div class="sync-tool-copy">
                                    <strong>{html.escape(server['name'])}</strong>
                                    <span>{html.escape(server['command'])}</span>
                                </div>
                                <span class="sync-tool-badge">SAVED</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No MCP servers configured.")

                with st.form("mcp_server_form", clear_on_submit=False):
                    mcp_name = st.text_input("Server name", key="mcp_name")
                    mcp_command = st.text_input(
                        "Command", key="mcp_command", placeholder="npx"
                    )
                    mcp_args = st.text_input(
                        "Arguments",
                        key="mcp_args",
                        placeholder="-y @modelcontextprotocol/server-github",
                    )
                    save_and_discover = st.form_submit_button(
                        "Save and discover tools",
                        icon=":material/extension:",
                        disabled=not (
                            backend_ready and mcp_name.strip() and mcp_command.strip()
                        ),
                        use_container_width=True,
                    )
                if save_and_discover:
                    try:
                        payload = {
                            "name": mcp_name.strip(),
                            "command": mcp_command.strip(),
                            "args": mcp_args.split(),
                        }
                        api("POST", "/mcp/servers", json=payload)
                        result = api("POST", "/mcp/discover", json=payload)
                        st.success(f"Connected {len(result['tools'])} tools.")
                        st.cache_data.clear()
                    except Exception as exc:
                        st.error(f"Could not connect to the MCP server: {exc}")

    with info_column:
        with st.container(border=True):
            st.markdown(
                '<div class="sync-panel-title"><span class="orange">◉</span>Appearance</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="sync-label">Theme</div>
                <div class="sync-tool-row" style="margin-top:8px;justify-content:center;">
                    <span class="sync-tool-badge">DARK</span>
                    <span class="sync-card-meta" style="margin:0;">Developer-centric · high contrast</span>
                </div>
                <div class="sync-label" style="margin-top:18px;">Editor Font Size</div>
                <div class="sync-source-path">14px · JetBrains Mono</div>
                """,
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            st.markdown(
                '<div class="sync-panel-title"><span>ⓘ</span>About</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="sync-about">
                    <strong>SyncSpace Core v0.1.0</strong>
                    Built for the Workshop 2 chatbot exercise. A context-aware AI companion for focused, read-only codebase exploration.
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_profile_page(
    backend_ready: bool, workspace_ready: bool, provider_ready: bool
) -> None:
    render_topbar("Profile")
    render_page_heading("User Profile", "Review the status of your local SyncSpace session.")
    with st.container(border=True):
        st.markdown(
            """
            <div class="sync-panel-title"><span class="violet">HG</span>Local Workspace User</div>
            <p class="sync-section-copy">SyncSpace runs locally and does not require a hosted user account.</p>
            """,
            unsafe_allow_html=True,
        )
        statuses = [
            ("API", backend_ready),
            ("Workspace", workspace_ready),
            ("Model", provider_ready),
        ]
        columns = st.columns(3)
        for column, (label, ready) in zip(columns, statuses):
            with column:
                state = "READY" if ready else "ACTION REQUIRED"
                st.markdown(
                    f"""
                    <div class="sync-workspace-card">
                        <h4>{html.escape(label)}</h4>
                        <div class="sync-card-meta">{state}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_chat_page(
    backend_ready: bool,
    workspace_ready: bool,
    configuration_ready: bool,
    existing_messages: list[dict],
) -> None:
    workspace = st.session_state.workspace
    workspace_name = Path(workspace).expanduser().name or workspace
    render_topbar(workspace)

    indexed_files = st.session_state.get("indexed_files", [])
    context_label = f"Context loaded: {workspace_name}"
    if indexed_files:
        context_label += f" ({len(indexed_files)} files)"
    st.markdown(
        f'<div class="sync-chat-intro"><span class="sync-chip">{html.escape(context_label)}</span></div>',
        unsafe_allow_html=True,
    )

    suggested_prompt = None
    if not existing_messages:
        st.markdown(
            f"""
            <div class="sync-empty-state">
                <div class="sync-empty-icon">AI</div>
                <h2>Ready for {html.escape(workspace_name)}</h2>
                <p>Ask about architecture, trace a bug, or explore this codebase. SyncSpace will inspect only the files needed to help.</p>
            </div>
            <div class="sync-suggestion-label">Start with a focused prompt</div>
            """,
            unsafe_allow_html=True,
        )
        suggestions = [
            ("Summarize project structure", "Summarize the structure and main components of this project."),
            ("Find potential bugs", "Review the codebase and identify potential bugs or risks."),
            ("Explain the main flow", "Explain the application's primary flow from end to end."),
            ("Suggest missing tests", "Analyze the project and suggest important tests that are missing."),
        ]
        for row_start in range(0, len(suggestions), 2):
            columns = st.columns(2)
            for column, (label, value) in zip(
                columns, suggestions[row_start : row_start + 2]
            ):
                with column:
                    if st.button(
                        label,
                        key=f"suggestion_{row_start}_{label}",
                        use_container_width=True,
                    ):
                        suggested_prompt = value

    for message in existing_messages:
        avatar = ":material/person:" if message["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask SyncSpace or type '/' for commands...",
        disabled=not (backend_ready and workspace_ready),
    )
    prompt = suggested_prompt or prompt

    if not backend_ready:
        st.error("SyncSpace API is offline. Start the backend to begin chatting.")
    elif not workspace_ready:
        st.error("Select a valid source directory in Workspace before chatting.")

    if not prompt:
        return
    if not configuration_ready:
        st.error("Open Settings and complete the Model Provider configuration first.")
        return

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    payload = {
        "session_id": st.session_state.get("session_id"),
        "workspace_path": workspace,
        "message": prompt,
        "provider": current_provider_config(),
    }

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        output = st.empty()
        status = st.status("Inspecting workspace...", expanded=True)
        answer = ""
        try:
            with httpx.stream(
                "POST", f"{API_URL}/chat", json=payload, timeout=120
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        answer += event["content"]
                        output.markdown(answer + "▌")
                    elif event["type"] == "tool":
                        status.write(f"Running `{event['name']}`")
                    elif event["type"] == "session":
                        st.session_state.session_id = event["session_id"]
                    elif event["type"] == "error":
                        raise RuntimeError(event["content"])
            output.markdown(answer)
            status.update(label="Complete", state="complete", expanded=False)
            st.cache_data.clear()
        except Exception as exc:
            status.update(label="Request failed", state="error")
            st.error(str(exc))


initialise_state()
keep_widget_state()
apply_query_navigation()

backend_ready = backend_is_ready(API_URL)
try:
    sessions = api("GET", "/sessions", timeout=3) if backend_ready else []
except Exception:
    sessions = []
try:
    mcp_servers = api("GET", "/mcp/servers", timeout=3) if backend_ready else []
except Exception:
    mcp_servers = []

render_sidebar(backend_ready)

workspace = st.session_state.workspace
workspace_ready = Path(workspace).expanduser().is_dir()
configuration_ready = provider_is_ready(current_provider_config())
page = st.session_state.page

if page == "workspace":
    render_workspace_page(backend_ready)
elif page == "history":
    render_history_page(sessions)
elif page == "settings":
    render_settings_page(backend_ready, mcp_servers)
elif page == "profile":
    render_profile_page(backend_ready, workspace_ready, configuration_ready)
else:
    try:
        existing_messages = (
            api("GET", f"/sessions/{st.session_state.session_id}/messages")
            if st.session_state.get("session_id") and backend_ready
            else []
        )
    except Exception:
        existing_messages = []
    render_chat_page(
        backend_ready,
        workspace_ready,
        configuration_ready,
        existing_messages,
    )
