"""The canonical line-splitting convention, shared by the chunker and file readers.

Chunk ``start_line``/``end_line`` are 1-based indices into ``split_lines(text)``. Any
consumer that maps those numbers back to text (e.g. :meth:`coderag.api.CodeRAG.get_file`)
MUST split the same way, so this lives in one place to prevent the two from drifting.
"""

from __future__ import annotations

from typing import List


def split_lines(text: str) -> List[str]:
    r"""Split ``text`` into lines on ``\n`` only.

    Deliberately not :meth:`str.splitlines`, which additionally breaks on ``\r``,
    ``\f``, ``\v``, U+2028, U+2029, …; that would desync line numbers for files with
    CRLF endings or stray control characters.
    """
    return text.split("\n")
