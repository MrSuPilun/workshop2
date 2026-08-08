from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ProviderKind = Literal["azure", "openai", "compatible", "google"]


class ProviderConfig(BaseModel):
    """Connection configuration submitted per request.

    The API key is deliberately not persisted by this application.
    """

    kind: ProviderKind
    api_key: str = Field(min_length=1)
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_version: str | None = None


class ChatRequest(BaseModel):
    session_id: str | None = None
    workspace_path: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=20_000)
    provider: ProviderConfig


class NewSession(BaseModel):
    title: str = "New conversation"


class McpServerInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)


class BatchRequest(BaseModel):
    workspace_path: str
    paths: list[str] = Field(min_length=1, max_length=20)
    provider: ProviderConfig
