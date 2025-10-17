"""MCP (Model Context Protocol) manager."""

from typing import Any, Dict, List, Optional
import asyncio
import json


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

    async def start(self) -> None:
        """Start the MCP server process."""
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

        # Wait a bit for server to start
        await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List tools available on this server.

        Returns:
            List of tool metadata
        """
        # Send list_tools request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
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
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }

        response = await self._send_request(request)
        return response.get('result')

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request to the server.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("MCP server not running")

        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()

        # Read response
        response_str = await self.process.stdout.readline()
        response = json.loads(response_str.decode())

        return response


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
