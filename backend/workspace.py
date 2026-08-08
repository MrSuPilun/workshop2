from __future__ import annotations

from pathlib import Path


IGNORED_PARTS = {".git", ".venv", "node_modules", "__pycache__", ".idea", ".next", "dist", "build"}
MAX_FILE_BYTES = 350_000


def root_path(workspace_path: str) -> Path:
    root = Path(workspace_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Workspace path must be an existing directory")
    return root


def _safe_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path must stay inside the selected workspace")
    if not candidate.is_file():
        raise ValueError("File does not exist")
    return candidate


def list_files(workspace_path: str, limit: int = 300) -> list[str]:
    root = root_path(workspace_path)
    files: list[str] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            files.append(str(path.relative_to(root)))
        if len(files) >= limit:
            break
    return files


def search_code(workspace_path: str, query: str, limit: int = 30) -> list[dict]:
    root = root_path(workspace_path)
    query_lower = query.lower()
    findings: list[dict] = []
    for relative in list_files(workspace_path, limit=1000):
        try:
            path = _safe_path(root, relative)
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if query_lower in line.lower():
                    findings.append({"path": relative, "line": number, "text": line[:500]})
                    if len(findings) >= limit:
                        return findings
        except (OSError, UnicodeError):
            continue
    return findings


def read_file(workspace_path: str, relative_path: str, start_line: int = 1, end_line: int = 250) -> dict:
    root = root_path(workspace_path)
    path = _safe_path(root, relative_path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("File is too large to include in context")
    start_line = max(1, start_line)
    end_line = min(max(start_line, end_line), start_line + 500)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    snippet = "\n".join(f"{index}: {line}" for index, line in enumerate(lines[start_line - 1:end_line], start=start_line))
    return {"path": relative_path, "start_line": start_line, "end_line": min(end_line, len(lines)), "content": snippet}
