"""Prompt templates for the SyncSpace chat agent.

Two techniques from the workshop brief live here:

* chain-of-thought - ``SYSTEM_PROMPT`` fixes an explicit four-step investigation
  procedure the model works through before it is allowed to answer.
* few-shot - ``FEW_SHOT_EXCHANGES`` replays complete tool-call cycles so the
  model copies the tool-use discipline, the citation format, and the two edge
  cases that matter here: a search that finds nothing, and a question that
  needs no tools at all.

Exchanges are stored provider-neutral and rendered per provider, so the OpenAI
and Gemini paths are taught from exactly the same examples. Paths inside the
examples are deliberately fictional - they must never be mistaken for evidence
about the workspace actually under inspection.
"""
from __future__ import annotations

import json
from typing import Any


HISTORY_TURNS = 16

SYSTEM_PROMPT = """You are SyncSpace, a precise AI coding-workspace companion.

Work through these four steps for every question about the selected workspace:
1. Understand - decide which artefact the question is about (a symbol, a file, a
   behaviour) and what evidence would actually settle it.
2. Locate - call `search_code` or `list_files` to find candidate files. Never
   guess a path; a path you have not seen in tool output does not exist.
3. Verify - call `read_file` on those candidates and read the real lines before
   you describe what the code does.
4. Answer - reply concisely, cite each claim as `relative/path.py:line`, and end
   with a `Checked:` line naming the files you actually opened.

Rules:
- Never claim that you ran a command, changed a file, or read a file that the
  tool output did not show.
- When the tools return nothing relevant, say so and ask one narrowing question
  instead of inventing an answer.
- Skip the tools and answer directly for greetings and for general programming
  questions that are not about this workspace; omit the `Checked:` line then.
"""


# Each exchange is a list of steps. `tool_call` / `tool_result` pair up by `id`.
FEW_SHOT_EXCHANGES: list[list[dict[str, Any]]] = [
    # 1. The happy path: locate, verify, then answer with citations.
    [
        {"role": "user", "content": "Where do we decide that a subscription is past due?"},
        {
            "role": "tool_call",
            "id": "fs_billing_1",
            "name": "search_code",
            "args": {"query": "past_due"},
        },
        {
            "role": "tool_result",
            "id": "fs_billing_1",
            "name": "search_code",
            "content": [
                {"path": "services/billing.py", "line": 42, "text": "def is_past_due(subscription):"},
                {"path": "services/jobs.py", "line": 88, "text": "    if is_past_due(sub):"},
            ],
        },
        {
            "role": "tool_call",
            "id": "fs_billing_2",
            "name": "read_file",
            "args": {"path": "services/billing.py", "start_line": 40, "end_line": 48},
        },
        {
            "role": "tool_result",
            "id": "fs_billing_2",
            "name": "read_file",
            "content": {
                "path": "services/billing.py",
                "start_line": 40,
                "end_line": 48,
                "content": (
                    "40: GRACE = timedelta(days=3)\n"
                    "41: \n"
                    "42: def is_past_due(subscription):\n"
                    "43:     if subscription.status == 'cancelled':\n"
                    "44:         return False\n"
                    "45:     return subscription.due_at + GRACE < utcnow()"
                ),
            },
        },
        {
            "role": "assistant",
            "content": (
                "`is_past_due` in services/billing.py:42 owns that decision, and it is stricter than the "
                "name suggests:\n\n"
                "- A three-day grace period is added before anything counts as past due "
                "(services/billing.py:40, services/billing.py:45).\n"
                "- Cancelled subscriptions short-circuit to `False`, so they never appear past due "
                "(services/billing.py:43-44).\n\n"
                "The only caller is the sweep in services/jobs.py:88, so changing the grace period there "
                "would affect the whole dunning flow.\n\n"
                "Checked: services/billing.py:40-48"
            ),
        },
    ],
    # 2. Edge case: the search finds nothing. Say so, do not improvise a module.
    [
        {"role": "user", "content": "Which module handles the SAML login flow?"},
        {
            "role": "tool_call",
            "id": "fs_saml_1",
            "name": "search_code",
            "args": {"query": "saml"},
        },
        {"role": "tool_result", "id": "fs_saml_1", "name": "search_code", "content": []},
        {
            "role": "assistant",
            "content": (
                "I searched this workspace for `saml` and got no matches, so I cannot point you at a "
                "module for it.\n\n"
                "Is the flow named after the identity provider instead (something like `okta`, `entra`, "
                "or `sso`), or does it live in a separate repository?"
            ),
        },
    ],
    # 3. Edge case: general knowledge, no workspace evidence needed, no tools.
    [
        {"role": "user", "content": "Quick one - what's the difference between a Python list and a tuple?"},
        {
            "role": "assistant",
            "content": (
                "A list is mutable and unhashable, so it works for collections you build up or reorder. "
                "A tuple is immutable and hashable, which makes it usable as a dict key or set member and "
                "signals a fixed-shape record. That is a language question rather than a workspace one, so "
                "I did not open any files."
            ),
        },
    ],
]


def _openai_steps(exchange: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for step in exchange:
        role = step["role"]
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": step["content"]})
        elif role == "tool_call":
            messages.append({"role": "assistant", "tool_calls": [{
                "id": step["id"],
                "type": "function",
                "function": {"name": step["name"], "arguments": json.dumps(step["args"])},
            }]})
        elif role == "tool_result":
            messages.append({
                "role": "tool",
                "tool_call_id": step["id"],
                "content": json.dumps(step["content"], ensure_ascii=False),
            })
    return messages


def _google_steps(exchange: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for step in exchange:
        role = step["role"]
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": step["content"]}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": step["content"]}]})
        elif role == "tool_call":
            contents.append({"role": "model", "parts": [
                {"functionCall": {"name": step["name"], "args": step["args"]}}
            ]})
        elif role == "tool_result":
            contents.append({"role": "user", "parts": [
                {"functionResponse": {"name": step["name"], "response": {"result": step["content"]}}}
            ]})
    return contents


def build_openai_messages(history: list[dict], user_message: str, few_shot: bool = True) -> list[dict[str, Any]]:
    """Assemble system prompt, few-shot exchanges, recent history, and the new turn."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for exchange in FEW_SHOT_EXCHANGES:
            messages.extend(_openai_steps(exchange))
    messages.extend({"role": item["role"], "content": item["content"]} for item in history[-HISTORY_TURNS:])
    messages.append({"role": "user", "content": user_message})
    return messages


def build_google_contents(history: list[dict], user_message: str, few_shot: bool = True) -> list[dict[str, Any]]:
    """Same prompt material as `build_openai_messages`, in Gemini `contents` shape.

    The system prompt is not included here; Gemini takes it as a separate
    `systemInstruction` field.
    """
    contents: list[dict[str, Any]] = []
    if few_shot:
        for exchange in FEW_SHOT_EXCHANGES:
            contents.extend(_google_steps(exchange))
    for item in history[-HISTORY_TURNS:]:
        role = "model" if item["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": item["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents
