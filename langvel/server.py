"""FastAPI server for Langvel agents."""

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json

from config.langvel import config

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
    try:
        from routes.agent import router

        # Get agent class
        agent_class = router.get(f"/{agent_path}")
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent not found: /{agent_path}")

        # Create agent instance
        agent = agent_class()

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
                "agent": agent_class.__name__,
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
    try:
        from routes.agent import router

        agent_class = router.get(f"/{agent_path}")
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent not found: /{agent_path}")

        agent = agent_class()
        graph = agent.compile()

        # Get mermaid representation
        mermaid = graph.get_graph().draw_mermaid()

        return {
            "agent": agent_class.__name__,
            "graph": mermaid
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__
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
