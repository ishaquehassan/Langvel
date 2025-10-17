"""MCP (Model Context Protocol) manager."""

from typing import Any, Dict, List, Optional
import asyncio
import json
import logging

# Setup logging
logger = logging.getLogger("langvel.mcp")


class MCPManager:
    """
    Manages MCP (Model Context Protocol) server connections and tools.

    Handles server lifecycle, tool discovery, and execution.
    """

    def __init__(self):
        self._servers: Dict[str, "MCPServer"] = {}
        self._tools: Dict[str, Dict[str, Any]] = {}

    async def register_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Register an MCP server.

        Args:
            name: Server identifier
            command: Command to start the server
            args: Command arguments
            env: Environment variables
        """
        server = MCPServer(name, command, args or [], env or {})
        await server.start()
        self._servers[name] = server

        # Discover tools
        tools = await server.list_tools()
        for tool in tools:
            tool_key = f"{name}:{tool['name']}"
            self._tools[tool_key] = {
                'server': name,
                'name': tool['name'],
                'description': tool.get('description', ''),
                'parameters': tool.get('parameters', {})
            }

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on an MCP server.

        Args:
            server: Server name
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If server or tool not found
        """
        if server not in self._servers:
            raise ValueError(f"MCP server '{server}' not found")

        mcp_server = self._servers[server]
        return await mcp_server.call_tool(tool_name, arguments)

    def get_tool(self, server: str, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata."""
        tool_key = f"{server}:{tool_name}"
        return self._tools.get(tool_key)

    def list_tools(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available tools.

        Args:
            server: Optional server name to filter by

        Returns:
            List of tool metadata
        """
        if server:
            return [
                tool for key, tool in self._tools.items()
                if tool['server'] == server
            ]
        return list(self._tools.values())

    async def shutdown(self) -> None:
        """Shutdown all MCP servers."""
        for server in self._servers.values():
            await server.stop()


class MCPServer:
    """
    Represents an MCP server connection.

    Manages the server process and communication.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: List[str],
        env: Dict[str, str]
    ):
        """
        Initialize MCP server.

        Args:
            name: Server identifier
            command: Command to start the server
            args: Command arguments
            env: Environment variables
        """
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._tools: List[Dict[str, Any]] = []
        self._stderr_task: Optional[asyncio.Task] = None
        self._request_id_counter: int = 0

    async def start(self) -> None:
        """
        Start the MCP server process.

        Fixes TODO-008: Prevents stderr deadlock by reading stderr in background.
        Implements proper readiness checks instead of fixed sleep.
        """
        import os

        # Merge environment variables
        full_env = os.environ.copy()
        full_env.update(self.env)

        # Start the process
        self.process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env
        )

        # Start background task to read stderr (prevents deadlock)
        # Without this, if server writes >64KB to stderr, process will hang
        self._stderr_task = asyncio.create_task(self._read_stderr())

        # Wait for server to be ready (with timeout)
        try:
            await self._wait_for_ready(timeout=10)
            logger.info(f"MCP server '{self.name}' started successfully")
        except Exception as e:
            logger.error(f"MCP server '{self.name}' failed to start: {e}")
            # Clean up on failure
            await self.stop()
            raise RuntimeError(f"Failed to start MCP server '{self.name}': {e}")

    async def stop(self) -> None:
        """Stop the MCP server process."""
        # Cancel stderr reading task
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        # Terminate process
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning(f"MCP server '{self.name}' did not terminate gracefully, killing")
                self.process.kill()
                await self.process.wait()

        logger.info(f"MCP server '{self.name}' stopped")

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List tools available on this server.

        Returns:
            List of tool metadata
        """
        # Send list_tools request
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/list"
        }

        response = await self._send_request(request)
        self._tools = response.get('result', {}).get('tools', [])
        return self._tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on this server.

        Args:
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        response = await self._send_request(request)
        return response.get('result')

    async def _send_request(
        self,
        request: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Send a JSON-RPC request to the server.

        Fixes TODO-026: Adds timeout, response validation, and error handling.

        Args:
            request: JSON-RPC request
            timeout: Request timeout in seconds (default: 30)

        Returns:
            JSON-RPC response

        Raises:
            RuntimeError: If server not running, timeout, or invalid response
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError(f"MCP server '{self.name}' not running")

        request_id = request.get('id')

        # Send request
        request_str = json.dumps(request) + "\n"
        try:
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
        except Exception as e:
            raise RuntimeError(
                f"Failed to send request to MCP server '{self.name}': {e}"
            )

        # Read response with timeout
        try:
            response_str = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP server '{self.name}' timed out after {timeout}s for request: "
                f"{request.get('method', 'unknown')}"
            )

        # Parse JSON response
        try:
            response = json.loads(response_str.decode())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON-RPC response from MCP server '{self.name}': {e}\n"
                f"Response: {response_str.decode()[:200]}"
            )

        # Validate JSON-RPC error response
        if 'error' in response:
            error = response['error']
            error_message = error.get('message', 'Unknown error')
            error_code = error.get('code', 'N/A')
            error_data = error.get('data', '')

            raise RuntimeError(
                f"MCP server '{self.name}' returned error: {error_message} "
                f"(code: {error_code})"
                f"{f', data: {error_data}' if error_data else ''}"
            )

        # Verify response ID matches request ID
        response_id = response.get('id')
        if response_id != request_id:
            logger.warning(
                f"Response ID mismatch for MCP server '{self.name}': "
                f"expected {request_id}, got {response_id}"
            )

        # Validate response has result
        if 'result' not in response:
            raise RuntimeError(
                f"MCP server '{self.name}' response missing 'result' field"
            )

        return response

    def _next_request_id(self) -> int:
        """
        Generate next request ID.

        Returns:
            Incremented request ID
        """
        self._request_id_counter += 1
        return self._request_id_counter

    async def _read_stderr(self) -> None:
        """
        Continuously read stderr to prevent deadlock.

        Critical fix for TODO-008: Without this, if the MCP server writes more
        than ~64KB to stderr, the process will block waiting for the buffer to
        be read, causing a deadlock.

        This task runs in the background for the lifetime of the process.
        """
        if not self.process or not self.process.stderr:
            return

        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    # EOF reached, process has terminated
                    break

                # Log stderr output for debugging
                stderr_line = line.decode().strip()
                if stderr_line:
                    logger.warning(
                        f"MCP server '{self.name}' stderr: {stderr_line}"
                    )
        except asyncio.CancelledError:
            # Task cancelled during shutdown, this is expected
            pass
        except Exception as e:
            logger.error(
                f"Error reading stderr from MCP server '{self.name}': {e}"
            )

    async def _wait_for_ready(self, timeout: int = 10) -> None:
        """
        Wait for server to be ready by checking for successful response.

        Replaces fixed sleep with actual readiness check.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If server doesn't become ready within timeout
        """
        start_time = asyncio.get_event_loop().time()
        last_error = None

        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Try to list tools as a health check
                await asyncio.wait_for(self.list_tools(), timeout=2)
                # If we get here, server is ready
                return
            except Exception as e:
                last_error = e
                # Wait a bit before retrying
                await asyncio.sleep(0.5)

        # Timeout reached
        error_msg = f"MCP server '{self.name}' failed to become ready within {timeout}s"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise RuntimeError(error_msg)


class MCPConfig:
    """Configuration for MCP servers."""

    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize MCP configuration.

        Args:
            servers: Server configurations
        """
        self.servers = servers or {}

    async def setup(self) -> MCPManager:
        """
        Set up MCP manager based on configuration.

        Returns:
            Configured MCPManager instance
        """
        manager = MCPManager()

        # Register all servers
        for name, config in self.servers.items():
            await manager.register_server(
                name=name,
                command=config.get('command', ''),
                args=config.get('args', []),
                env=config.get('env', {})
            )

        return manager
