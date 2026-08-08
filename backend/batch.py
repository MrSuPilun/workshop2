"""Batched multi-file summarisation.

The chat agent handles one question at a time, where each model call depends on
the previous one. Summarising a set of files is the opposite shape: many small,
independent prompts that share nothing. Run sequentially they cost the sum of
their round trips; run through `asyncio.gather` they cost roughly the slowest
one. A semaphore keeps the fan-out from tripping provider rate limits, and each
file's failure is captured per item so one unreadable file cannot sink the run.
"""
from __future__ import annotations

import asyncio
from time import perf_counter
from typing import Any

import httpx

from .agent import create_client, explain_provider_error
from .models import ProviderConfig
from .prompts import build_batch_summary_messages
from .workspace import read_file


MAX_CONCURRENCY = 4
SUMMARY_MAX_LINES = 400
DEFAULT_GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


async def _summarize_openai(client: Any, model: str, path: str, content: str) -> str:
    completion = await client.chat.completions.create(
        model=model,
        messages=build_batch_summary_messages(path, content),
    )
    return (completion.choices[0].message.content or "").strip()


async def _summarize_google(http_client: httpx.AsyncClient, config: ProviderConfig, path: str, content: str) -> str:
    endpoint = (config.base_url or DEFAULT_GOOGLE_ENDPOINT).rstrip("/")
    system, *turns = build_batch_summary_messages(path, content)
    payload = {
        "systemInstruction": {"parts": [{"text": system["content"]}]},
        "contents": [
            {"role": "model" if turn["role"] == "assistant" else "user", "parts": [{"text": turn["content"]}]}
            for turn in turns
        ],
    }
    response = await http_client.post(
        f"{endpoint}/models/{config.model}:generateContent",
        headers={"x-goog-api-key": config.api_key},
        json=payload,
    )
    response.raise_for_status()
    candidate = (response.json().get("candidates") or [{}])[0]
    parts = candidate.get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts).strip()


async def summarize_files(config: ProviderConfig, workspace_path: str, paths: list[str]) -> list[dict]:
    """Summarise every path concurrently and return one result per path, in order."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    is_google = config.kind == "google"
    client = None if is_google else create_client(config)
    http_client = httpx.AsyncClient(timeout=90) if is_google else None

    async def summarize_one(path: str) -> dict:
        async with semaphore:
            started = perf_counter()
            try:
                content = read_file(workspace_path, path, 1, SUMMARY_MAX_LINES)["content"]
                if is_google:
                    summary = await _summarize_google(http_client, config, path, content)
                else:
                    summary = await _summarize_openai(client, config.model, path, content)
                return {"path": path, "summary": summary, "seconds": round(perf_counter() - started, 2)}
            except Exception as exc:
                # Keep the batch alive; the caller reports which files failed.
                detail = explain_provider_error(config, exc)
                return {"path": path, "error": detail, "seconds": round(perf_counter() - started, 2)}

    try:
        return list(await asyncio.gather(*(summarize_one(path) for path in paths)))
    finally:
        if http_client is not None:
            await http_client.aclose()
