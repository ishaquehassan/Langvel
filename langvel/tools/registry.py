"""Tool registry - Manages all tools available to agents."""

from typing import Any, Callable, Dict, List, Optional, Union
from langchain_core.tools import BaseTool, StructuredTool
import asyncio
import inspect
import time
from functools import wraps


class ToolExecutionError(Exception):
    """Error raised when tool execution fails."""
    pass


class ToolRegistry:
    """
    Registry for managing agent tools.

    Handles registration and retrieval of tools with various types
    (custom, MCP, RAG, HTTP, LLM).
    """

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}

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

    async def execute_tool(
        self,
        name: str,
        agent: Any,
        *args,
        retry: int = 3,
        timeout: Optional[float] = None,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """
        Execute a tool with error handling, retries, and timeout.

        Args:
            name: Tool name
            agent: Agent instance (for accessing managers)
            *args: Positional arguments for the tool
            retry: Number of retry attempts
            timeout: Timeout in seconds
            fallback: Fallback function if tool fails
            **kwargs: Keyword arguments for the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool execution fails after retries
        """
        if name not in self._tools:
            raise ToolExecutionError(f"Tool '{name}' not found")

        tool_func = self._tools[name]
        metadata = self._tool_metadata[name]
        tool_type = metadata['type']

        # Initialize stats
        if name not in self._execution_stats:
            self._execution_stats[name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0
            }

        start_time = time.time()
        last_error = None

        for attempt in range(retry):
            try:
                # Execute based on tool type
                if tool_type == 'custom':
                    result = await self._execute_custom_tool(tool_func, agent, *args, **kwargs)
                elif tool_type == 'rag':
                    result = await self._execute_rag_tool(tool_func, agent, metadata, *args, **kwargs)
                elif tool_type == 'mcp':
                    result = await self._execute_mcp_tool(tool_func, agent, metadata, *args, **kwargs)
                elif tool_type == 'http':
                    result = await self._execute_http_tool(tool_func, agent, metadata, *args, **kwargs)
                elif tool_type == 'llm':
                    result = await self._execute_llm_tool(tool_func, agent, metadata, *args, **kwargs)
                else:
                    raise ToolExecutionError(f"Unknown tool type: {tool_type}")

                # Apply timeout if specified
                if timeout:
                    result = await asyncio.wait_for(
                        asyncio.create_task(asyncio.coroutine(lambda: result)()),
                        timeout=timeout
                    )

                # Update stats on success
                duration = time.time() - start_time
                self._update_stats(name, duration, success=True)

                return result

            except asyncio.TimeoutError:
                last_error = ToolExecutionError(f"Tool '{name}' timed out after {timeout}s")
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

            except Exception as e:
                last_error = ToolExecutionError(f"Tool '{name}' failed: {str(e)}")
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

        # All retries failed
        duration = time.time() - start_time
        self._update_stats(name, duration, success=False)

        # Try fallback if provided
        if fallback:
            try:
                return await self._execute_fallback(fallback, agent, last_error, *args, **kwargs)
            except Exception as fb_error:
                raise ToolExecutionError(
                    f"Tool '{name}' and fallback both failed. "
                    f"Original: {last_error}, Fallback: {fb_error}"
                )

        raise last_error

    async def _execute_custom_tool(
        self,
        tool_func: Callable,
        agent: Any,
        *args,
        **kwargs
    ) -> Any:
        """Execute a custom tool."""
        # Check if it's a coroutine
        if inspect.iscoroutinefunction(tool_func):
            return await tool_func(agent, *args, **kwargs)
        else:
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, tool_func, agent, *args, **kwargs)

    async def _execute_rag_tool(
        self,
        tool_func: Callable,
        agent: Any,
        metadata: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute a RAG tool with automatic document retrieval."""
        # Extract state from args (first arg should be state)
        state = args[0] if args else kwargs.get('state')

        if not state:
            raise ToolExecutionError("RAG tool requires state as first argument")

        # Get query from state
        query = getattr(state, 'query', None) or kwargs.get('query', '')

        if not query:
            raise ToolExecutionError("RAG tool requires 'query' field in state or kwargs")

        # Retrieve documents using RAG manager
        try:
            collection = metadata['rag_collection']
            k = metadata['rag_k']
            threshold = metadata['rag_threshold']

            retrieved_docs = await agent.rag_manager.retrieve(
                collection=collection,
                query=query,
                k=k,
                similarity_threshold=threshold,
                **metadata['rag_kwargs']
            )

            # Add retrieved docs to state
            if hasattr(state, 'retrieved_docs'):
                state.retrieved_docs = retrieved_docs
            if hasattr(state, 'format_context'):
                state.format_context()

        except Exception as e:
            # Don't fail completely if RAG fails, just log
            if hasattr(state, 'add_message'):
                state.add_message("system", f"RAG retrieval failed: {str(e)}")

        # Execute the original function
        return await self._execute_custom_tool(tool_func, agent, *args, **kwargs)

    async def _execute_mcp_tool(
        self,
        tool_func: Callable,
        agent: Any,
        metadata: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute an MCP tool."""
        server = metadata['mcp_server']
        tool_name = metadata['mcp_tool_name']

        # Build arguments for MCP call
        mcp_args = kwargs.copy()
        mcp_args.update(metadata['mcp_kwargs'])

        # Call MCP server
        try:
            result = await agent.mcp_manager.call_tool(
                server=server,
                tool_name=tool_name,
                arguments=mcp_args
            )
            return result

        except Exception as e:
            raise ToolExecutionError(f"MCP tool call failed: {str(e)}")

    async def _execute_http_tool(
        self,
        tool_func: Callable,
        agent: Any,
        metadata: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute an HTTP tool."""
        import aiohttp

        method = metadata['http_method']
        url = metadata['http_url'] or kwargs.pop('url', None)
        headers = metadata['http_headers'].copy()
        headers.update(kwargs.pop('headers', {}))

        if not url:
            raise ToolExecutionError("HTTP tool requires 'url' parameter")

        # Prepare request kwargs
        request_kwargs = {
            'headers': headers,
            **metadata['http_kwargs'],
            **kwargs
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, **request_kwargs) as response:
                    response.raise_for_status()

                    # Try to parse JSON, fallback to text
                    try:
                        return await response.json()
                    except:
                        return await response.text()

        except aiohttp.ClientError as e:
            raise ToolExecutionError(f"HTTP request failed: {str(e)}")

    async def _execute_llm_tool(
        self,
        tool_func: Callable,
        agent: Any,
        metadata: Dict[str, Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute an LLM tool."""
        # Extract prompt from args/kwargs
        prompt = args[0] if args else kwargs.get('prompt', '')
        system_prompt = metadata['llm_system_prompt'] or kwargs.get('system_prompt')

        if not prompt:
            raise ToolExecutionError("LLM tool requires 'prompt' parameter")

        # Get LLM configuration
        llm_kwargs = {
            'model': metadata['llm_model'],
            'temperature': metadata['llm_temperature'],
            **metadata['llm_kwargs'],
            **kwargs
        }

        try:
            # Use agent's LLM manager
            response = await agent.llm.invoke(
                prompt=prompt,
                system_prompt=system_prompt,
                **llm_kwargs
            )
            return response

        except Exception as e:
            raise ToolExecutionError(f"LLM call failed: {str(e)}")

    async def _execute_fallback(
        self,
        fallback_func: Callable,
        agent: Any,
        error: Exception,
        *args,
        **kwargs
    ) -> Any:
        """Execute fallback function."""
        # Add error to kwargs for fallback to handle
        kwargs['error'] = error

        if inspect.iscoroutinefunction(fallback_func):
            return await fallback_func(agent, *args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, fallback_func, agent, *args, **kwargs)

    def _update_stats(self, name: str, duration: float, success: bool) -> None:
        """Update execution statistics for a tool."""
        stats = self._execution_stats[name]
        stats['total_calls'] += 1

        if success:
            stats['successful_calls'] += 1
        else:
            stats['failed_calls'] += 1

        stats['total_duration'] += duration
        stats['avg_duration'] = stats['total_duration'] / stats['total_calls']

    def get_stats(self, name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get execution statistics for tools.

        Args:
            name: Optional tool name. If None, returns stats for all tools.

        Returns:
            Statistics dictionary
        """
        if name:
            return self._execution_stats.get(name, {})
        return self._execution_stats.copy()

    def reset_stats(self, name: Optional[str] = None) -> None:
        """
        Reset execution statistics.

        Args:
            name: Optional tool name. If None, resets all stats.
        """
        if name:
            if name in self._execution_stats:
                self._execution_stats[name] = {
                    'total_calls': 0,
                    'successful_calls': 0,
                    'failed_calls': 0,
                    'total_duration': 0.0,
                    'avg_duration': 0.0
                }
        else:
            self._execution_stats.clear()
