"""Self-hostable HTTP/REST API over a CodeRAG instance (optional ``[server]`` extra).

Lets custom apps, remote frontends, or a shared team deployment query a big codebase over
the network. Endpoints: ``GET /search``, ``POST /index``, ``GET /status``, ``GET /file``.

Security: the API can read indexed source and file contents, so it is **opt-in
authenticated**. Set ``CODERAG_API_KEY`` (``config.api_key``) and every request must then
present ``Authorization: Bearer <key>`` or ``X-API-Key: <key>`` — except ``GET /status``,
the health-probe target, which stays unauthenticated so a key-protected server can still
pass its own liveness/readiness checks (it exposes only coarse index stats). With no key
configured there is no auth — only safe for trusted, loopback-only use. CORS is disabled
unless ``CODERAG_CORS_ORIGINS`` lists explicit origins (never ``*``).

Note: this module intentionally does NOT use ``from __future__ import annotations`` — FastAPI
must see the real Pydantic model classes (not stringized annotations) to bind request bodies.
"""

import hmac
import ipaddress
import logging
import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from fastapi import FastAPI

    from coderag.api import CodeRAG

logger = logging.getLogger(__name__)


def create_app(cr: "CodeRAG") -> "FastAPI":
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    api_key = cr.config.api_key

    def require_auth(
        request: Request,
        authorization: Optional[str] = Header(default=None),
        x_api_key: Optional[str] = Header(default=None),
    ) -> None:
        """Enforce the API key when one is configured; otherwise a no-op.

        ``GET /status`` is intentionally exempt: it is the liveness/health probe
        target (container probes, the Docker HEALTHCHECK, and readiness gates all
        hit it), so requiring auth here would make a server with CODERAG_API_KEY
        set never pass its own health checks. It exposes only coarse index stats,
        never source or file contents.
        """
        if request.url.path == "/status":
            return
        if not api_key:
            return
        presented = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            presented = authorization[len("bearer ") :].strip()
        # Constant-time compare to avoid leaking the key via timing.
        if not presented or not hmac.compare_digest(presented, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    app = FastAPI(
        title="CodeRAG",
        version="1.0.0",
        description="Semantic code-search engine.",
        dependencies=[Depends(require_auth)],
    )

    # CORS is off by default. Only add the middleware when explicit origins are
    # configured — never a wildcard, since the API exposes source and file reads.
    if cr.config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cr.config.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    # Serialize indexing so a flood of POST /index can't pile up concurrent walks
    # (resource exhaustion) or race the single-writer store.
    index_lock = threading.Lock()

    class IndexRequest(BaseModel):
        path: Optional[str] = None
        full: bool = False

    @app.get("/status")
    def status() -> dict:
        return cr.status()

    @app.get("/search")
    def search(
        q: str = Query(..., description="Search query"),
        k: int = Query(8, ge=1, le=100),
    ) -> dict:
        hits = cr.search(q, top_k=k)
        return {"query": q, "count": len(hits), "results": [h.as_dict() for h in hits]}

    @app.post("/index")
    def index(req: IndexRequest) -> dict:
        if not index_lock.acquire(blocking=False):
            raise HTTPException(
                status_code=409, detail="An index operation is already in progress"
            )
        try:
            stats = cr.index(req.path, full=req.full)
        finally:
            index_lock.release()
        return stats.as_dict()

    @app.get("/file")
    def get_file(
        path: str = Query(...),
        start_line: Optional[int] = Query(None, ge=1),
        end_line: Optional[int] = Query(None, ge=1),
    ) -> dict:
        try:
            content = cr.get_file(path, start_line, end_line)
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"path": path, "content": content}

    return app


def _is_public_host(host: str) -> bool:
    """True if ``host`` is reachable beyond loopback (so auth really matters)."""
    if host in ("127.0.0.1", "localhost", "::1"):
        return False
    if host in ("0.0.0.0", "::"):  # nosec B104 — classifies a host as public; does not bind a socket
        return True
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True  # a hostname — assume it's externally reachable


def run_server(cr: "CodeRAG", host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    if _is_public_host(host) and not cr.config.api_key:
        logger.warning(
            "CodeRAG HTTP API is binding to %s with NO API key set. It is "
            "UNAUTHENTICATED and exposes indexed source and file reads to anyone who "
            "can reach this port. Set CODERAG_API_KEY to require authentication.",
            host,
        )
    # Warm the index/provider AND the embedding model (loads the model + JITs the query
    # path) so the first request isn't slow — matches the UI. status() alone builds the
    # store/provider but never embeds, leaving the first query to pay the cold model load.
    try:
        cr.warm()
    except Exception:  # pragma: no cover - warm-up is best-effort
        logger.exception("HTTP API warm-up failed (continuing).")
    uvicorn.run(create_app(cr), host=host, port=port)
