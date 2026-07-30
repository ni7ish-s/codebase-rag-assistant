"""RepoRAG: a standalone, local-first semantic code-search engine.

Public API::

    from reporag import RepoRAG, Config

    cr = RepoRAG(Config.from_env(watched_dir="/path/to/repo"))
    cr.index()
    for hit in cr.search("where is retry/backoff handled?"):
        print(hit.path, hit.start_line, hit.score)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reporag.config import Config

if TYPE_CHECKING:
    # Re-exported lazily at runtime via __getattr__ below (keeps ``import reporag``
    # light — no chromadb/fastembed pulled in at import). Declared here only so type
    # checkers and static analysis see ``RepoRAG`` as a defined export of __all__.
    from reporag.api import RepoRAG

__version__ = "1.0.0"

__all__ = ["RepoRAG", "Config", "__version__"]


def __getattr__(name: str) -> object:
    # Lazy re-export so ``import reporag`` stays light (no chromadb/fastembed at import).
    if name == "RepoRAG":
        from reporag.api import RepoRAG

        return RepoRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
