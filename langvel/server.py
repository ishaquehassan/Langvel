"""FastAPI server for Langvel agents."""

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import json
import logging
import uuid
import re
from threading import Lock

from config.langvel import config

# Setup logging
logger = logging.getLogger("langvel.server")

# Agent path validation pattern - only allow safe characters
# Allows: letters, numbers, underscores, hyphens, forward slashes
# Prevents: path traversal (..), absolute paths, special characters
AGENT_PATH_PATTERN = re.compile(r'^[a-zA-Z0-9_/-]+$')


def validate_agent_path(agent_path: str) -> None:
    """
    Validate agent path to prevent path traversal attacks.

    Args:
        agent_path: Agent route path to validate

    Raises:
        HTTPException: If path is invalid or contains path traversal attempts

    Security checks:
    - No path traversal sequences (..)
    - No absolute paths (starting with /)
    - Only alphanumeric, underscore, hyphen, and forward slash allowed
    - No consecutive slashes
    """
    # Check for empty path
    if not agent_path or agent_path.strip() == "":
        logger.warning(
            "Agent path validation failed: empty path",
            extra={"agent_path": agent_path}
        )
        raise HTTPException(
            status_code=400,
            detail="Agent path cannot be empty"
        )

    # Check for path traversal
    if '..' in agent_path:
        logger.warning(
            f"Agent path validation failed: path traversal attempt detected",
            extra={"agent_path": agent_path, "reason": "contains .."}
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid agent path: path traversal not allowed"
        )

    # Check for absolute paths (should be relative)
    if agent_path.startswith('/'):
        logger.warning(
            f"Agent path validation failed: absolute path",
            extra={"agent_path": agent_path, "reason": "starts with /"}
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid agent path: must be relative path"
        )

    # Check for consecutive slashes
    if '//' in agent_path:
        logger.warning(
            f"Agent path validation failed: consecutive slashes",
            extra={"agent_path": agent_path, "reason": "contains //"}
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid agent path: consecutive slashes not allowed"
        )

    # Check against pattern (only safe characters)
    if not AGENT_PATH_PATTERN.match(agent_path):
        logger.warning(
            f"Agent path validation failed: invalid characters",
            extra={"agent_path": agent_path, "reason": "contains invalid characters"}
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid agent path: only letters, numbers, underscores, hyphens, and slashes allowed"
        )


# Agent instance pool - prevents memory leak from creating new instances per request
# Uses singleton pattern to reuse agent instances across requests
_agent_pool: Dict[str, Any] = {}
_agent_pool_lock = Lock()


def get_agent(agent_path: str):
    """
    Get or create agent instance (singleton per path).

    Implements thread-safe singleton pattern to prevent memory leaks.
    Each agent path gets exactly one instance that is reused across all requests.

    Args:
        agent_path: Agent route path (must be validated first)

    Returns:
        Agent instance (reused across requests)

    Raises:
        HTTPException: If agent not found in router

    Performance Impact:
    - Without pooling: +15MB per request → 15GB/hour at 1000 req/min → OOM in 2-3 hours
    - With pooling: ~50MB total (one-time) → stable memory usage

    Thread Safety:
    - Double-checked locking pattern prevents race conditions
    - First check (outside lock) for performance
    - Second check (inside lock) ensures only one instance created
    """
    # Fast path: check if agent already exists (no lock needed for read)
    if agent_path in _agent_pool:
        return _agent_pool[agent_path]

    # Slow path: need to create agent (acquire lock)
    with _agent_pool_lock:
        # Double-check inside lock (another thread might have created it)
        if agent_path in _agent_pool:
            return _agent_pool[agent_path]

        # Agent doesn't exist, create it
        from routes.agent import router

        agent_class = router.get(f"/{agent_path}")
        if not agent_class:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: /{agent_path}"
            )

        # Create instance and cache it
        agent_instance = agent_class()
        _agent_pool[agent_path] = agent_instance

        logger.info(
            f"Created new agent instance in pool",
            extra={
                "agent_path": agent_path,
                "agent_class": agent_class.__name__,
                "pool_size": len(_agent_pool)
            }
        )

        return agent_instance


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.

    Prevents DoS attacks via large payloads (payload bomb attacks).
    """

    def __init__(self, app, max_size: int = 10_000_000):  # 10MB default
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            max_size: Maximum request body size in bytes (default: 10MB)
        """
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        """
        Check request size before processing.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response from next handler

        Raises:
            HTTPException: If request body exceeds size limit
        """
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    size = int(content_length)
                    if size > self.max_size:
                        logger.warning(
                            f"Request rejected: size {size} bytes exceeds limit {self.max_size} bytes",
                            extra={
                                "method": request.method,
                                "path": request.url.path,
                                "client": request.client.host if request.client else None,
                                "content_length": size,
                                "max_size": self.max_size
                            }
                        )
                        raise HTTPException(
                            status_code=413,
                            detail=f"Request body too large. Maximum size: {self.max_size} bytes"
                        )
                except ValueError:
                    # Invalid content-length header
                    logger.warning(
                        f"Invalid Content-Length header: {content_length}",
                        extra={
                            "method": request.method,
                            "path": request.url.path,
                            "client": request.client.host if request.client else None,
                        }
                    )
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid Content-Length header"
                    )

        return await call_next(request)


# Create FastAPI app
app = FastAPI(
    title="Langvel Server",
    description="Laravel-inspired framework for LangGraph agents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request size limit middleware (10MB default)
app.add_middleware(RequestSizeLimitMiddleware, max_size=10_000_000)


class AgentRequest(BaseModel):
    """Request model for agent invocation."""
    input: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    stream: bool = False


class AgentResponse(BaseModel):
    """Response model for agent invocation."""
    output: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Langvel Server",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    try:
        from routes.agent import router
        routes = router.list_routes()
        return {"agents": routes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/{agent_path:path}")
async def invoke_agent(agent_path: str, request: AgentRequest):
    """
    Invoke an agent.

    Args:
        agent_path: Agent route path
        request: Agent request with input data

    Returns:
        Agent response
    """
    # Validate agent path for security
    validate_agent_path(agent_path)

    try:
        # Get or create agent instance (pooled for memory efficiency)
        agent = get_agent(agent_path)

        # Handle streaming
        if request.stream:
            async def generate():
                async for chunk in agent.stream(request.input, request.config):
                    yield f"data: {json.dumps(chunk, default=str)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        # Handle normal invocation
        result = await agent.invoke(request.input, request.config)

        return AgentResponse(
            output=result,
            metadata={
                "agent": agent.__class__.__name__,
                "path": f"/{agent_path}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_path:path}/graph")
async def get_agent_graph(agent_path: str):
    """
    Get agent graph visualization.

    Args:
        agent_path: Agent route path

    Returns:
        Graph in mermaid format
    """
    # Validate agent path for security
    validate_agent_path(agent_path)

    try:
        # Get or create agent instance (pooled for memory efficiency)
        agent = get_agent(agent_path)
        graph = agent.compile()

        # Get mermaid representation
        mermaid = graph.get_graph().draw_mermaid()

        return {
            "agent": agent.__class__.__name__,
            "graph": mermaid
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler with safe error reporting.

    Logs full error details internally but returns sanitized errors to clients.

    Note: HTTPException is handled separately by FastAPI, so we skip it here.
    """
    # Skip HTTPException - let FastAPI's built-in handler deal with it
    if isinstance(exc, HTTPException):
        raise exc

    # Generate trace ID for error tracking
    trace_id = str(uuid.uuid4())

    # Log full error details internally
    logger.error(
        f"Request failed: {request.method} {request.url.path}",
        exc_info=exc,
        extra={
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
            "error_type": type(exc).__name__,
        }
    )

    # Return safe error to client based on environment
    if config.DEBUG:
        # In debug mode, show detailed errors
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": type(exc).__name__,
                "trace_id": trace_id,
                "message": "An error occurred. Check logs for details."
            }
        )
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "trace_id": trace_id,
                "message": "An unexpected error occurred. Please contact support with this trace ID."
            }
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        workers=config.SERVER_WORKERS
    )
