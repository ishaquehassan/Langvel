"""Tool decorators - Define tools for agents with elegant syntax."""

from typing import Any, Callable, Dict, List, Optional
from functools import wraps
from langchain_core.tools import tool as langchain_tool


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    return_direct: bool = False,
    retry: int = 3,
    timeout: Optional[float] = None,
    fallback: Optional[Callable] = None
) -> Callable:
    """
    Decorator to mark a method as a custom tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description
        return_direct: Whether to return the tool output directly
        retry: Number of retry attempts on failure (default: 3)
        timeout: Timeout in seconds for tool execution
        fallback: Fallback function if tool fails

    Example:
        @tool(description="Analyzes sentiment of text", retry=5, timeout=30)
        async def analyze_sentiment(self, text: str) -> float:
            # Your logic here
            return sentiment_score
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Mark this as a tool
        wrapper._is_tool = True
        wrapper._tool_type = "custom"
        wrapper._tool_name = name or func.__name__
        wrapper._tool_description = description or func.__doc__ or ""
        wrapper._return_direct = return_direct
        wrapper._tool_retry = retry
        wrapper._tool_timeout = timeout
        wrapper._tool_fallback = fallback

        return wrapper

    return decorator


def mcp_tool(
    server: str,
    tool_name: Optional[str] = None,
    **mcp_kwargs
) -> Callable:
    """
    Decorator to integrate MCP (Model Context Protocol) server tools.

    Args:
        server: MCP server name (configured in config)
        tool_name: Specific tool name from the server
        **mcp_kwargs: Additional MCP configuration

    Example:
        @mcp_tool(server='slack', tool_name='send_message')
        async def send_slack(self, message: str, channel: str):
            pass  # Implementation handled by MCP
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # MCP implementation will be injected
            return await func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_type = "mcp"
        wrapper._mcp_server = server
        wrapper._mcp_tool_name = tool_name or func.__name__
        wrapper._mcp_kwargs = mcp_kwargs

        return wrapper

    return decorator


def rag_tool(
    collection: str,
    k: int = 5,
    similarity_threshold: Optional[float] = None,
    **rag_kwargs
) -> Callable:
    """
    Decorator for RAG (Retrieval Augmented Generation) operations.

    Args:
        collection: Vector store collection name
        k: Number of documents to retrieve
        similarity_threshold: Minimum similarity score
        **rag_kwargs: Additional RAG configuration

    Example:
        @rag_tool(collection='knowledge_base', k=5)
        async def search_knowledge(self, query: str):
            pass  # Returns retrieved documents automatically
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_type = "rag"
        wrapper._rag_collection = collection
        wrapper._rag_k = k
        wrapper._rag_threshold = similarity_threshold
        wrapper._rag_kwargs = rag_kwargs

        return wrapper

    return decorator


def http_tool(
    method: str = "GET",
    url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    **http_kwargs
) -> Callable:
    """
    Decorator for HTTP API calls.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Base URL (can be overridden at runtime)
        headers: Default headers
        **http_kwargs: Additional HTTP configuration

    Example:
        @http_tool(method='POST', url='https://api.example.com/data')
        async def fetch_data(self, params: dict):
            pass  # HTTP call handled automatically
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_type = "http"
        wrapper._http_method = method
        wrapper._http_url = url
        wrapper._http_headers = headers or {}
        wrapper._http_kwargs = http_kwargs

        return wrapper

    return decorator


def llm_tool(
    model: Optional[str] = None,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **llm_kwargs
) -> Callable:
    """
    Decorator for LLM-based tools.

    Args:
        model: LLM model to use
        temperature: Sampling temperature
        system_prompt: System prompt for the LLM
        **llm_kwargs: Additional LLM configuration

    Example:
        @llm_tool(system_prompt="You are a code reviewer")
        async def review_code(self, code: str) -> str:
            pass  # LLM call handled automatically
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._is_tool = True
        wrapper._tool_type = "llm"
        wrapper._llm_model = model
        wrapper._llm_temperature = temperature
        wrapper._llm_system_prompt = system_prompt
        wrapper._llm_kwargs = llm_kwargs

        return wrapper

    return decorator


class ToolDecorators:
    """Namespace for all tool decorators."""

    custom = tool
    mcp = mcp_tool
    rag = rag_tool
    http = http_tool
    llm = llm_tool
