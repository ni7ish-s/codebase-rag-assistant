"""CodeRAG: a standalone, local-first semantic code-search engine.

Public API::

    from coderag import CodeRAG, Config

    cr = CodeRAG(Config.from_env(watched_dir="/path/to/repo"))
    cr.index()
    for hit in cr.search("where is retry/backoff handled?"):
        print(hit.path, hit.start_line, hit.score)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reporag.config import Config

if TYPE_CHECKING:
    # Re-exported lazily at runtime via __getattr__ below (keeps ``import coderag``
    # light — no chromadb/fastembed pulled in at import). Declared here only so type
    # checkers and static analysis see ``CodeRAG`` as a defined export of __all__.
    from reporag.api import CodeRAG

__version__ = "1.0.0"

__all__ = ["CodeRAG", "Config", "__version__"]


def __getattr__(name: str) -> object:
    # Lazy re-export so ``import coderag`` stays light (no chromadb/fastembed at import).
    if name == "CodeRAG":
        from reporag.api import CodeRAG

        return CodeRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
