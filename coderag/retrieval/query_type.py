"""Query-type detection for adaptive fusion weighting.

Symbol-level evaluation showed a fixed 1:1 dense/BM25 fusion is a compromise: on pure
natural-language "where is X handled" queries the dense retriever is much stronger and
equal-weight RRF drags it down with weak BM25. But the first cut keyed on query *shape*
(short + code-looking) and mis-classified the common real case — a prose query that
*mentions* a specific symbol, e.g. a commit message *"Fix tuple order in
`AliasGenerator.generate_aliases()`"*. External validation (pydantic) showed leaning dense
on those hurts, because the discriminating signal is the exact identifier BM25 matches.

So the routing question is simply: **does the query reference a specific code identifier?**
If yes, keep fusion neutral (BM25 carries real signal); if it's pure prose, lean dense. This
makes adaptive fusion fall back to plain hybrid whenever an identifier is named — it can only
help pure-NL queries, never repeat the regression.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from coderag.config import Config

# Query-wide signals, both linear (disjoint character classes — no catastrophic backtracking
# on the user-supplied query string): a `backtick`-quoted span, or a call ``word(``.
_BACKTICK = re.compile(r"`[^`]+`")
_CALL = re.compile(r"\w\(")
# Punctuation stripped from a token before classifying it (query prose, not code).
_STRIP = "`'\".,:;!?()[]{}<>"


def _is_snake_case(token: str) -> bool:
    """An underscore flanked by alphanumerics, e.g. ``foo_bar`` (linear scan)."""
    return any(
        token[i] == "_" and token[i - 1].isalnum() and token[i + 1].isalnum()
        for i in range(1, len(token) - 1)
    )


def _is_dotted_path(token: str) -> bool:
    """Two adjacent identifier parts joined by a dot, e.g. ``Foo.bar`` — excludes ``e.g``/``3.11``."""
    parts = token.split(".")
    return any(
        len(a) >= 2 and len(b) >= 2 and a.isidentifier() and b.isidentifier()
        for a, b in zip(parts, parts[1:], strict=False)
    )


def _is_camel_case(token: str) -> bool:
    """A lower→upper boundary, e.g. ``fooBar`` / ``AliasGenerator`` (linear scan)."""
    return any(
        a.islower() and b.isupper() for a, b in zip(token, token[1:], strict=False)
    )


def references_identifier(query: str) -> bool:
    """True if ``query`` names a specific code identifier (even inside a prose sentence).

    Uses linear token scanning rather than one backtracking regex, so it can't be turned into
    a ReDoS by a crafted query (the query flows in from the HTTP API).
    """
    if not query:
        return False
    if _BACKTICK.search(query) or _CALL.search(query):
        return True
    for raw in query.split():
        token = raw.strip(_STRIP)
        if len(token) >= 2 and (
            _is_snake_case(token) or _is_dotted_path(token) or _is_camel_case(token)
        ):
            return True
    return False


# Back-compat alias: the original name meant "is an identifier lookup"; the detection is now
# "references an identifier", which subsumes the old behavior and also catches embedded ones.
looks_like_identifier = references_identifier


def fusion_weights(query: str, config: "Config") -> Tuple[float, float]:
    """Return ``(dense_weight, lexical_weight)`` for ``query``.

    Without adaptive fusion this is the configured static pair. With it on: a query that
    references a specific identifier uses the (neutral by default) code weights — so BM25's
    exact-match signal is kept; a pure natural-language query uses the dense-leaning weights.
    """
    if not config.adaptive_fusion:
        return config.dense_weight, config.lexical_weight
    if references_identifier(query):
        return config.code_dense_weight, config.code_lexical_weight
    return config.nl_dense_weight, config.nl_lexical_weight
