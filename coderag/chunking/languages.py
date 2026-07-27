"""File-extension -> language mapping and the set of languages with symbol parsers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

# Languages for which we extract symbol-aware spans (function/class/method).
# Python uses the stdlib ``ast``; the rest use tree-sitter.
PYTHON = "python"
TREE_SITTER_LANGUAGES = {"javascript", "typescript", "tsx", "go", "rust", "java"}
SYMBOL_LANGUAGES = {PYTHON} | TREE_SITTER_LANGUAGES

# Everything indexable. Languages not in SYMBOL_LANGUAGES are still indexed via the
# line-window fallback (so docs/config/other code remain searchable).
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    # Indexed with the fallback chunker:
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".cfg": "ini",
    ".ini": "ini",
    # Common markup/web/config text — searchable in most repos, line-window chunked.
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".properties": "properties",
    ".gradle": "gradle",
}

# Well-known text files that have no (or an unconventional) extension. Matched on the
# lowercased file *name* when the extension lookup misses.
FILENAME_TO_LANGUAGE = {
    "dockerfile": "dockerfile",
    "makefile": "make",
    "license": "text",
    "notice": "text",
    "readme": "text",
    "codeowners": "text",
    ".env": "text",
    ".gitignore": "text",
    ".dockerignore": "text",
}


def detect_language(path: str | Path, *, all_text: bool = False) -> Optional[str]:
    """Return the language for ``path``, or ``None`` if it should not be indexed.

    With ``all_text=True`` any unrecognized file is treated as plain ``"text"`` so a whole
    directory (docs, notes, config) becomes searchable, not just code. Binary files are
    still rejected later by the indexer's NUL-byte sniff, so this stays safe.
    """
    p = Path(path)
    lang = EXTENSION_TO_LANGUAGE.get(p.suffix.lower())
    if lang:
        return lang
    lang = FILENAME_TO_LANGUAGE.get(p.name.lower())
    if lang:
        return lang
    return "text" if all_text else None


def extensions_for(languages: Iterable[str]) -> List[str]:
    """File extensions that map to any of ``languages`` (the canonical reverse lookup)."""
    wanted = set(languages)
    return sorted(ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang in wanted)
