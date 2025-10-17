"""Tool registry - Manages all tools available to agents."""

from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import BaseTool, StructuredTool


class ToolRegistry:
    """
    Registry for managing agent tools.

    Handles registration and retrieval of tools with various types
    (custom, MCP, RAG, HTTP, LLM).
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            func: Tool function with metadata from decorators
        """
        self._tools[name] = func

        # Extract metadata from function attributes
        metadata = {
            'type': getattr(func, '_tool_type', 'custom'),
            'name': getattr(func, '_tool_name', name),
            'description': getattr(func, '_tool_description', ''),
            'return_direct': getattr(func, '_return_direct', False),
        }

        # Add type-specific metadata
        if metadata['type'] == 'mcp':
            metadata['mcp_server'] = getattr(func, '_mcp_server', None)
            metadata['mcp_tool_name'] = getattr(func, '_mcp_tool_name', None)
            metadata['mcp_kwargs'] = getattr(func, '_mcp_kwargs', {})

        elif metadata['type'] == 'rag':
            metadata['rag_collection'] = getattr(func, '_rag_collection', None)
            metadata['rag_k'] = getattr(func, '_rag_k', 5)
            metadata['rag_threshold'] = getattr(func, '_rag_threshold', None)
            metadata['rag_kwargs'] = getattr(func, '_rag_kwargs', {})

        elif metadata['type'] == 'http':
            metadata['http_method'] = getattr(func, '_http_method', 'GET')
            metadata['http_url'] = getattr(func, '_http_url', None)
            metadata['http_headers'] = getattr(func, '_http_headers', {})
            metadata['http_kwargs'] = getattr(func, '_http_kwargs', {})

        elif metadata['type'] == 'llm':
            metadata['llm_model'] = getattr(func, '_llm_model', None)
            metadata['llm_temperature'] = getattr(func, '_llm_temperature', 0.7)
            metadata['llm_system_prompt'] = getattr(func, '_llm_system_prompt', None)
            metadata['llm_kwargs'] = getattr(func, '_llm_kwargs', {})

        self._tool_metadata[name] = metadata

    def get(self, name: str) -> Optional[Callable]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get tool metadata."""
        return self._tool_metadata.get(name, {})

    def get_all(self) -> Dict[str, Callable]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_langchain_tools(self) -> List[BaseTool]:
        """
        Convert registered tools to LangChain tools.

        Returns:
            List of LangChain BaseTool instances
        """
        langchain_tools = []

        for name, func in self._tools.items():
            metadata = self._tool_metadata[name]

            # Create structured tool
            tool = StructuredTool.from_function(
                func=func,
                name=metadata['name'],
                description=metadata['description'],
                return_direct=metadata['return_direct']
            )

            langchain_tools.append(tool)

        return langchain_tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all tools with their metadata.

        Returns:
            List of tool information
        """
        return [
            {
                'name': name,
                **self._tool_metadata[name]
            }
            for name in self._tools.keys()
        ]
