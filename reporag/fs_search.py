"""Exact filesystem search — the regex/glob complement to semantic ``search_code``.

RepoRAG's hybrid index is great at "find this by *meaning*", but an agent still needs
the other half of the job: "find every literal ``raise TimeoutError``", "list the files
matching ``*_test.py``". That is exactly what coding agents otherwise shell out to
``grep``/``rg``/``find`` for. This module gives them an in-process, ignore-aware exact
search instead — modelled on the Hermes agent's ``search_files`` tool (ripgrep-backed,
``target`` content/files, ``output_mode`` content/files_only/count, context lines,
pagination, secret redaction).

Design: candidate files are always enumerated in Python, honouring RepoRAG's own
``ignore_globs`` via :mod:`reporag._ignore` (so the search sees exactly the same
workspace the indexer does). When ripgrep is on PATH it scans that explicit file list
for the content case — a genuine speed-up with *no* divergence in which files are
searched, since rg is handed the paths directly. Without ripgrep, a pure-Python scan
produces identical results; that fallback is what the test-suite exercises so CI never
depends on rg being installed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from reporag._ignore import walk_files
from reporag._lines import split_lines
from reporag.config import DEFAULT_IGNORE_GLOBS

DEFAULT_LIMIT = 50
_RG_BATCH = 400  # files per ripgrep invocation, to stay under arg-length limits
_MAX_FILE_BYTES = 1_000_000  # skip files larger than this when scanning content

# Conservative secret redaction. Two flavours: "keyed" patterns mask only the value that
# follows a credential-ish key, so searching for the word "token" still shows the line;
# "standalone" patterns mask a whole well-known credential shape.
_KEYED_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key)\b"
    r"(\s*[:=]\s*['\"]?)([^\s'\"]{6,})"
)
_STANDALONE_SECRETS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"-----BEGIN[A-Z ]*PRIVATE KEY-----"),
)


def redact_secrets(text: str) -> str:
    """Mask obvious credential values in a line, conservatively."""
    out = _KEYED_SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
    for pat in _STANDALONE_SECRETS:
        out = pat.sub("***", out)
    return out


@dataclass(slots=True)
class _ContentMatch:
    path: str  # POSIX path relative to root
    line_number: int  # 1-based
    line: str
    before: List[str] = field(default_factory=list)
    after: List[str] = field(default_factory=list)


def _rg_available() -> bool:
    """Whether ripgrep is on PATH. Indirected so tests can force the Python path."""
    return shutil.which("rg") is not None


def _glob_matches(rel: str, glob: str) -> bool:
    """Match a glob against the full relative path or just the basename (``*.py``)."""
    return fnmatch(rel, glob) or fnmatch(rel.rsplit("/", 1)[-1], glob)


def _read_text(abs_path: Path, max_file_bytes: int) -> Optional[str]:
    """Read a file as text, skipping ones that are too large or binary (NUL sniff)."""
    try:
        data = abs_path.read_bytes()
    except OSError:
        return None
    if len(data) > max_file_bytes or b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def _match_python(
    files: Sequence[Tuple[Path, str]],
    regex: "re.Pattern[str]",
    context: int,
    max_file_bytes: int,
) -> List[_ContentMatch]:
    matches: List[_ContentMatch] = []
    for abs_path, rel in files:
        text = _read_text(abs_path, max_file_bytes)
        if text is None:
            continue
        lines = split_lines(text)
        for i, line in enumerate(lines):
            if regex.search(line):
                before = lines[max(0, i - context) : i] if context else []
                after = lines[i + 1 : i + 1 + context] if context else []
                matches.append(_ContentMatch(rel, i + 1, line, before, after))
    return matches


def _match_ripgrep(
    files: Sequence[Tuple[Path, str]],
    pattern: str,
    ignore_case: bool,
) -> List[_ContentMatch]:
    """Scan an explicit file list with ripgrep (context-free fast path).

    Files are passed by path, so ripgrep's own ignore rules never apply — the set of
    searched files is exactly what :func:`_iter_files` produced. Raises on any failure
    so the caller can fall back to the Python scan.
    """
    rel_by_abs = {str(abs_path): rel for abs_path, rel in files}
    matches: List[_ContentMatch] = []
    paths = list(rel_by_abs.keys())
    for start in range(0, len(paths), _RG_BATCH):
        batch = paths[start : start + _RG_BATCH]
        cmd = ["rg", "--json", "-n", "--no-config"]
        if ignore_case:
            cmd.append("-i")
        cmd += ["-e", pattern, "--", *batch]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        # rg exits 1 when there are simply no matches; 2+ is a real error.
        if proc.returncode >= 2:
            raise RuntimeError(proc.stderr.strip() or "ripgrep failed")
        for raw in proc.stdout.splitlines():
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("type") != "match":
                continue
            data = event["data"]
            abs_text = data["path"]["text"]
            rel = rel_by_abs.get(abs_text, abs_text)
            line = data["lines"]["text"].rstrip("\n")
            matches.append(_ContentMatch(rel, data["line_number"], line))
    matches.sort(key=lambda m: (m.path, m.line_number))
    return matches


def _paginate(items: List, offset: int, limit: int) -> Tuple[List, bool, Optional[int]]:
    total = len(items)
    page = items[offset : offset + limit] if limit > 0 else items[offset:]
    truncated = limit > 0 and offset + limit < total
    next_offset = offset + limit if truncated else None
    return page, truncated, next_offset


def _shape_content(
    matches: List[_ContentMatch],
    *,
    output_mode: str,
    offset: int,
    limit: int,
    context: int,
    redact: bool,
) -> Tuple[List[Dict], int]:
    """Project raw content matches into the requested output_mode + page."""
    if output_mode == "files_only":
        seen: List[str] = []
        for m in matches:
            if m.path not in seen:
                seen.append(m.path)
        page, _, _ = _paginate(seen, offset, limit)
        return [{"path": p} for p in page], len(seen)

    if output_mode == "count":
        counts: Dict[str, int] = {}
        for m in matches:
            counts[m.path] = counts.get(m.path, 0) + 1
        rows = [{"path": p, "count": counts[p]} for p in sorted(counts)]
        page, _, _ = _paginate(rows, offset, limit)
        return page, len(rows)

    # default: "content"
    page, _, _ = _paginate(matches, offset, limit)
    rows = []
    for m in page:
        row: Dict = {
            "location": f"{m.path}:{m.line_number}",
            "path": m.path,
            "line_number": m.line_number,
            "line": redact_secrets(m.line) if redact else m.line,
        }
        if context:
            row["before"] = [redact_secrets(x) if redact else x for x in m.before]
            row["after"] = [redact_secrets(x) if redact else x for x in m.after]
        rows.append(row)
    return rows, len(matches)


def search_files(
    root: os.PathLike,
    pattern: str,
    *,
    target: str = "content",
    file_glob: Optional[str] = None,
    output_mode: str = "content",
    context: int = 0,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    ignore_globs: Sequence[str] = DEFAULT_IGNORE_GLOBS,
    use_gitignore: bool = True,
    ignore_case: bool = False,
    max_file_bytes: int = _MAX_FILE_BYTES,
    redact: bool = True,
    use_ripgrep: bool = True,
) -> Dict:
    """Exact regex/glob search over the workspace, honouring RepoRAG's ignore rules.

    Args:
        root: Workspace root to search under.
        pattern: A regex (``target="content"``) or a filename glob (``target="files"``).
        target: ``"content"`` (regex inside files) or ``"files"`` (find by name).
        file_glob: For content search, restrict to files matching this glob (e.g. ``*.py``).
        output_mode: ``"content"`` | ``"files_only"`` | ``"count"`` (content target only).
        context: Lines of context around each match (content + Python path only).
        limit: Page size (``<= 0`` means no limit).
        offset: Page offset, for paginating large result sets.
        ignore_globs: Ignore patterns; defaults to RepoRAG's standard set.
        ignore_case: Case-insensitive matching.
        max_file_bytes: Skip files larger than this when scanning content.
        redact: Mask obvious credential values in returned lines.
        use_ripgrep: Use ripgrep for the content fast path when available.

    Returns a JSON-able dict with ``results`` plus pagination metadata.
    """
    root_path = Path(root).resolve()
    if target not in ("content", "files"):
        return {"error": f"unknown target {target!r} (use 'content' or 'files')"}
    if output_mode not in ("content", "files_only", "count"):
        return {"error": f"unknown output_mode {output_mode!r}"}
    if offset < 0:
        offset = 0

    if target == "files":
        rels = sorted(
            rel
            for _, rel in walk_files(
                root_path, ignore_globs, use_gitignore=use_gitignore
            )
            if _glob_matches(rel, pattern)
        )
        page, truncated, next_offset = _paginate(rels, offset, limit)
        return _envelope(
            pattern,
            target,
            "files",
            [{"path": p} for p in page],
            len(rels),
            offset,
            next_offset,
            truncated,
            ripgrep=False,
        )

    # target == "content"
    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        return {"error": f"invalid regex: {exc}", "pattern": pattern}

    files = [
        (abs_path, rel)
        for abs_path, rel in walk_files(
            root_path, ignore_globs, use_gitignore=use_gitignore
        )
        if file_glob is None or _glob_matches(rel, file_glob)
    ]

    used_rg = False
    matches: Optional[List[_ContentMatch]] = None
    if use_ripgrep and context == 0 and files and _rg_available():
        try:
            matches = _match_ripgrep(files, pattern, ignore_case)
            used_rg = True
        except Exception:  # fall back to the always-correct Python scan
            matches = None
    if matches is None:
        matches = _match_python(files, regex, context, max_file_bytes)

    results, total = _shape_content(
        matches,
        output_mode=output_mode,
        offset=offset,
        limit=limit,
        context=context,
        redact=redact,
    )
    _, truncated, next_offset = _paginate(list(range(total)), offset, limit)
    return _envelope(
        pattern,
        target,
        output_mode,
        results,
        total,
        offset,
        next_offset,
        truncated,
        ripgrep=used_rg,
    )


def _envelope(
    pattern: str,
    target: str,
    output_mode: str,
    results: List[Dict],
    total: int,
    offset: int,
    next_offset: Optional[int],
    truncated: bool,
    *,
    ripgrep: bool,
) -> Dict:
    env: Dict = {
        "pattern": pattern,
        "target": target,
        "output_mode": output_mode,
        "count": len(results),
        "total": total,
        "offset": offset,
        "truncated": truncated,
        "ripgrep": ripgrep,
        "results": results,
    }
    if truncated:
        env["next_offset"] = next_offset
        env["hint"] = f"Results truncated. Use offset={next_offset} to see more."
    return env
