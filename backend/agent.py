from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .models import ProviderConfig
from .mcp_service import call_tool as call_mcp_tool, discover_tools
from .prompts import SYSTEM_PROMPT, build_google_contents, build_openai_messages
from .workspace import list_files, read_file, search_code


TOOLS = [
    {"type": "function", "function": {"name": "list_files", "description": "List files in the selected workspace.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search source files for text. Use this to locate definitions and usages.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a range of lines from a relative source-file path.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
]


def normalize_azure_endpoint(base_url: str) -> str:
    """Reduce a pasted Azure URL to the resource root the SDK expects.

    The SDK appends `/openai/deployments/<deployment>/chat/completions` itself,
    so an endpoint that already carries `/openai` or the `/openai/v1` surface
    produces a doubled path and a 404 from Azure. Both forms are easy to copy
    out of the portal, so they are trimmed here rather than rejected.
    """
    endpoint = base_url.strip().rstrip("/")
    for suffix in ("/openai/v1", "/openai"):
        if endpoint.lower().endswith(suffix):
            return endpoint[: -len(suffix)].rstrip("/")
    return endpoint


def explain_provider_error(config: ProviderConfig, exc: Exception) -> str:
    """Turn a bare provider status code into something the user can act on."""
    message = str(exc)
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status == 404 and config.kind == "azure":
        return (
            f"Azure returned 404 for deployment '{config.model}'. Check that the deployment name matches the "
            f"one in Azure AI Foundry (it is the deployment name, not the model name), that the endpoint is "
            f"the resource root such as https://<resource>.openai.azure.com, and that API version "
            f"'{config.api_version or '2024-10-21'}' is offered by that deployment. Original error: {message}"
        )
    if status == 404:
        return f"The provider has no model named '{config.model}' at this endpoint. Original error: {message}"
    if status == 401:
        return f"The provider rejected this API key. Original error: {message}"
    return message


def create_client(config: ProviderConfig):
    """Build an async OpenAI-family client. Not used by the Gemini path, which is REST."""
    if config.kind == "azure":
        if not config.base_url:
            raise ValueError("Azure OpenAI needs an endpoint")
        return AsyncAzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=normalize_azure_endpoint(config.base_url),
            api_version=config.api_version or "2024-10-21",
        )
    return AsyncOpenAI(api_key=config.api_key, base_url=config.base_url if config.kind == "compatible" else None)


async def verify_provider(config: ProviderConfig) -> dict:
    """Send the smallest real request so misconfiguration surfaces during setup."""
    try:
        if config.kind == "google":
            endpoint = (config.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            async with httpx.AsyncClient(timeout=30) as http_client:
                response = await http_client.post(
                    f"{endpoint}/models/{config.model}:generateContent",
                    headers={"x-goog-api-key": config.api_key},
                    json={"contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
                )
                response.raise_for_status()
        else:
            client = create_client(config)
            await client.chat.completions.create(
                model=config.model, messages=[{"role": "user", "content": "ping"}]
            )
        return {"ok": True, "detail": f"{config.model} answered a test request."}
    except Exception as exc:
        return {"ok": False, "detail": explain_provider_error(config, exc)}


async def _mcp_tools(servers: list[dict]) -> tuple[list[dict], dict[str, tuple[dict, str]]]:
    tools: list[dict] = []
    lookup: dict[str, tuple[dict, str]] = {}
    for server in servers:
        try:
            discovered = await discover_tools(server["command"], server["args"])
        except Exception:
            # A broken optional integration should not prevent local code chat.
            continue
        prefix = re.sub(r"[^a-zA-Z0-9_-]", "_", server["name"])
        for item in discovered:
            exposed_name = f"mcp_{prefix}_{item['name']}"
            lookup[exposed_name] = (server, item["name"])
            tools.append({"type": "function", "function": {
                "name": exposed_name,
                "description": f"MCP server '{server['name']}': {item['description']}",
                "parameters": item["input_schema"],
            }})
    return tools, lookup


async def _run_tool(name: str, args: dict[str, Any], workspace_path: str, mcp_lookup: dict[str, tuple[dict, str]]) -> Any:
    if name == "list_files":
        return list_files(workspace_path)
    if name == "search_code":
        return search_code(workspace_path, args["query"])
    if name == "read_file":
        return read_file(workspace_path, args["path"], args.get("start_line", 1), args.get("end_line", 250))
    if name in mcp_lookup:
        server, original_name = mcp_lookup[name]
        return await call_mcp_tool(server["command"], server["args"], original_name, args)
    raise ValueError(f"Tool is not allowed: {name}")


def _google_tools(openai_tools: list[dict]) -> list[dict]:
    return [{"functionDeclarations": [
        {
            "name": item["function"]["name"],
            "description": item["function"].get("description", ""),
            "parameters": item["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for item in openai_tools
    ]}]


async def _run_google_agent(
    config: ProviderConfig,
    workspace_path: str,
    history: list[dict],
    user_message: str,
    all_tools: list[dict],
    mcp_lookup: dict[str, tuple[dict, str]],
) -> AsyncIterator[dict]:
    """Gemini REST tool-call loop using the Google Generative Language API."""
    endpoint = (config.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    url = f"{endpoint}/models/{config.model}:generateContent"
    contents = build_google_contents(history, user_message)
    payload_base = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "tools": _google_tools(all_tools),
    }
    async with httpx.AsyncClient(timeout=90) as http_client:
        for _ in range(6):
            response = await http_client.post(url, headers={"x-goog-api-key": config.api_key}, json={**payload_base, "contents": contents})
            response.raise_for_status()
            data = response.json()
            candidate = (data.get("candidates") or [{}])[0]
            content = candidate.get("content", {})
            function_calls = [part["functionCall"] for part in content.get("parts", []) if "functionCall" in part]
            if not function_calls:
                answer = "".join(part.get("text", "") for part in content.get("parts", [])) or "I could not produce a response."
                for index in range(0, len(answer), 40):
                    yield {"type": "token", "content": answer[index:index + 40]}
                    await asyncio.sleep(0)
                yield {"type": "done", "content": answer}
                return

            contents.append(content)
            responses = []
            for call in function_calls:
                name, arguments = call["name"], call.get("args", {})
                try:
                    yield {"type": "tool", "name": name, "arguments": arguments}
                    result = await _run_tool(name, arguments, workspace_path, mcp_lookup)
                    result_payload = {"result": result}
                except Exception as exc:
                    result_payload = {"error": str(exc)}
                responses.append({"functionResponse": {"name": name, "response": result_payload}})
            contents.append({"role": "user", "parts": responses})
    yield {"type": "done", "content": "I stopped after too many tool calls. Please narrow the request."}


async def run_agent(config: ProviderConfig, workspace_path: str, history: list[dict], user_message: str, mcp_servers: list[dict] | None = None) -> AsyncIterator[dict]:
    """Run an OpenAI tool-call loop and emit SSE-ready events."""
    mcp_tools, mcp_lookup = await _mcp_tools(mcp_servers or [])
    all_tools = TOOLS + mcp_tools
    if config.kind == "google":
        async for event in _run_google_agent(config, workspace_path, history, user_message, all_tools, mcp_lookup):
            yield event
        return

    client = create_client(config)
    messages: list[dict[str, Any]] = build_openai_messages(history, user_message)

    for _ in range(6):
        completion = await client.chat.completions.create(model=config.model, messages=messages, tools=all_tools, tool_choice="auto")
        message = completion.choices[0].message
        if not message.tool_calls:
            answer = message.content or "I could not produce a response."
            for index in range(0, len(answer), 40):
                yield {"type": "token", "content": answer[index:index + 40]}
                await asyncio.sleep(0)
            yield {"type": "done", "content": answer}
            return

        messages.append(message.model_dump(exclude_none=True))
        for call in message.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
                yield {"type": "tool", "name": call.function.name, "arguments": arguments}
                result = await _run_tool(call.function.name, arguments, workspace_path, mcp_lookup)
                tool_result = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                tool_result = json.dumps({"error": str(exc)})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})
    yield {"type": "done", "content": "I stopped after too many tool calls. Please narrow the request."}
