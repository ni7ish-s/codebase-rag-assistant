"""Structure-aware 1-hop neighbor expansion (callee direction, opt-in).

After hybrid fusion ranks a candidate pool, the top *seed* hits are expanded with their
1-hop neighbors in the call graph: the **definitions of the functions/methods a seed
calls** (its callees). The edges need no schema change — the called names are parsed from
the seed's own text, then resolved to the chunks that define them via the store's symbol
index. Neighbors come back as a ranked id-list to be fused in with a small weight, so they
add recall (the helper a relevant function delegates to is often relevant too) without
overpowering the direct dense + lexical signal.

This is the direction that pays off. On a 3-repo symbol-level eval (flask / requests /
click) callee expansion is a small, consistent lift; the opposite edge — a symbol's
*callers*, found by FTS — measurably *hurt* (callers of a relevant symbol are rarely
relevant themselves, so they only add noise), which is why it isn't used. See
docs/research/code-retrieval-strategy.md (§4) and docs/eval.md. Off by default.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Mapping, Sequence

if TYPE_CHECKING:
    from reporag.store.chroma_store import ChromaStore

# A bare identifier used as a call target, e.g. ``do_thing(`` (≥3 chars). The store's symbol
# index only holds names the repo defines, so language builtins (len, str, …) never resolve.
_CALL = re.compile(r"([A-Za-z_]\w{2,})\s*\(")


def called_names(text: str) -> List[str]:
    """Identifiers invoked as calls in ``text``, in source order, deduped.

    These are the candidate callees to resolve to their definitions. Order is preserved
    (and the result deduped) so neighbor ranking is deterministic.
    """
    out: List[str] = []
    seen: set[str] = set()
    for name in _CALL.findall(text or ""):
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def neighbor_ids(
    store: "ChromaStore",
    seed_ids: Sequence[int],
    seed_texts: Mapping[int, str],
    *,
    per_seed: int,
    max_seeds: int,
) -> List[int]:
    """Return 1-hop callee neighbors of the seeds, ranked best-first, deduped.

    For each of the first ``max_seeds`` seeds we resolve up to ``per_seed`` *new* callee
    definitions (the chunks defining what the seed calls) via the store's symbol index.
    Seeds and already-emitted ids are excluded, so the result is purely new candidate
    signal to fuse alongside dense and lexical.
    """
    if per_seed <= 0 or max_seeds <= 0:
        return []
    index = store.symbol_index()
    if not index:
        return []
    seen = set(seed_ids)
    out: List[int] = []
    for sid in list(seed_ids)[:max_seeds]:
        added = 0
        for name in called_names(seed_texts.get(sid, "")):
            for cid in index.get(name, ()):  # chunks defining this called name
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(cid)
                added += 1
                if added >= per_seed:
                    break
            if added >= per_seed:
                break
    return out
