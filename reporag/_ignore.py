"""Shared file-walking + ignore matching for indexing and exact filesystem search.

Both the :class:`~coderag.indexer.Indexer` and the exact filesystem search
(:mod:`coderag.fs_search`) must enumerate the *same* set of paths — skipping vendored
deps, VCS directories, build output, and (optionally) anything matched by ``.gitignore`` —
or the two would disagree about what "the workspace" is. The single :func:`walk_files`
generator below is the one place that decision is made, so both callers stay in lock-step.
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

GITIGNORE_FILE = ".gitignore"


def ignore_dir_names(ignore_globs: Iterable[str]) -> Set[str]:
    """Top-level directory names that can be pruned wholesale during a walk.

    Derived from ``"<name>/*"`` globs (e.g. ``"node_modules/*"`` -> ``"node_modules"``)
    so ``os.walk`` can drop the whole subtree without visiting every entry, and so a
    *nested* ``node_modules`` is ignored too (matched by path component, not just prefix).
    """
    return {g[:-2] for g in ignore_globs if g.endswith("/*") and "/" not in g[:-2]}


def is_ignored(rel: str, ignore_globs: Iterable[str], ignore_dirs: Set[str]) -> bool:
    """True if the POSIX relative path ``rel`` should be skipped.

    A path is ignored if any of its components is an ignored directory name, or if the
    whole relative path matches one of ``ignore_globs``.
    """
    parts = rel.split("/")
    if ignore_dirs.intersection(parts):
        return True
    return any(fnmatch.fnmatch(rel, g) for g in ignore_globs)


def _is_ancestor(base: str, dir_rel: str) -> bool:
    """Whether a ``.gitignore`` at ``base`` still applies at ``dir_rel`` (``""`` = root)."""
    if base == "":
        return True
    return dir_rel == base or dir_rel.startswith(base + "/")


class _GitignoreMatcher:
    """Honor nested ``.gitignore`` files during a top-down walk (nearest rule wins).

    A ``.gitignore`` at directory ``B`` scopes its patterns to paths under ``B``; the
    closest file's rules take precedence and may re-include via ``!``. We keep a stack of
    ``(base_rel, spec)`` ordered root→leaf, trimmed to the current directory's ancestors as
    the (DFS pre-order) walk moves, and test a path nearest-first using pathspec's
    tri-state ``check_file`` (ignore / negated-include / no-match). A no-op if pathspec is
    somehow unavailable, so indexing never hard-fails on a missing optional dependency.
    """

    def __init__(self) -> None:
        try:
            from pathspec import GitIgnoreSpec
        except ImportError:  # pragma: no cover - pathspec is a declared dependency
            logger.warning(
                "pathspec not installed; .gitignore files will not be honored."
            )
            self._spec_cls = None
        else:
            self._spec_cls = GitIgnoreSpec
        self._stack: List[Tuple[str, object]] = []

    @property
    def enabled(self) -> bool:
        return self._spec_cls is not None

    def enter(self, dir_rel: str, dir_abs: Path) -> None:
        """Refresh the active-rule stack for ``dir_rel`` and load its ``.gitignore``."""
        if self._spec_cls is None:
            return
        # Drop rules from sibling subtrees we've left; keep only ancestors of dir_rel.
        self._stack = [
            (base, spec) for base, spec in self._stack if _is_ancestor(base, dir_rel)
        ]
        try:
            text = (dir_abs / GITIGNORE_FILE).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return  # no .gitignore here (or unreadable)
        self._stack.append((dir_rel, self._spec_cls.from_lines(text.splitlines())))

    def match(self, rel: str, *, is_dir: bool) -> bool:
        """True if ``rel`` (root-relative POSIX) is ignored by the active rules."""
        if not self._stack:
            return False
        suffix = "/" if is_dir else ""
        for base, spec in reversed(self._stack):
            sub = rel if base == "" else rel[len(base) + 1 :]
            result = spec.check_file(sub + suffix)  # type: ignore[attr-defined]
            if result.include is not None:
                return bool(result.include)
        return False


def walk_files(
    start: Path,
    ignore_globs: Iterable[str],
    *,
    root: Optional[Path] = None,
    use_gitignore: bool = True,
) -> Iterator[Tuple[Path, str]]:
    """Yield ``(absolute_path, posix_rel)`` for every non-ignored file under ``start``.

    ``rel`` is relative to ``root`` (defaults to ``start``) so every caller shares one
    notion of the workspace. Ignored directories are pruned *before descending* (the big
    win at ``/home`` scale), honoring ``ignore_globs`` (dir-name prune + path globs) and,
    when ``use_gitignore``, nested ``.gitignore`` files.
    """
    start = Path(start)
    root = Path(root) if root is not None else start
    globs = tuple(ignore_globs)
    ignore_dirs = ignore_dir_names(globs)
    matcher = _GitignoreMatcher() if use_gitignore else None
    active = matcher if (matcher is not None and matcher.enabled) else None

    for dirpath, dirnames, filenames in os.walk(start):
        d_abs = Path(dirpath)
        try:
            d_rel = "" if d_abs == root else d_abs.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - start outside root
            continue
        if active is not None:
            active.enter(d_rel, d_abs)

        kept: List[str] = []
        for name in dirnames:
            if name in ignore_dirs:
                continue
            rel = name if d_rel == "" else f"{d_rel}/{name}"
            if is_ignored(rel, globs, ignore_dirs):
                continue
            if active is not None and active.match(rel, is_dir=True):
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in filenames:
            rel = name if d_rel == "" else f"{d_rel}/{name}"
            if is_ignored(rel, globs, ignore_dirs):
                continue
            if active is not None and active.match(rel, is_dir=False):
                continue
            yield d_abs / name, rel
