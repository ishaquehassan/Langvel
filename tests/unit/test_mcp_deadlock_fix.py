"""Test MCP stderr deadlock fix (TODO-008) and JSON-RPC validation (TODO-026)."""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langvel.mcp.manager import MCPServer, MCPManager


class TestMCPStderrDeadlock:
    """Test MCP stderr deadlock prevention (TODO-008)."""

    @pytest.mark.asyncio
    async def test_stderr_task_created_on_start(self):
        """Test that stderr reading task is created when server starts."""
        server = MCPServer("test", "echo", ["hello"], {})

        # Mock the process
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Mock readline to return tool list response
        mock_process.stdout.readline = AsyncMock(return_value=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        }).encode() + b"\n")

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await server.start()

            # Verify stderr task was created
            assert server._stderr_task is not None
            assert not server._stderr_task.done()

            # Cleanup
            await server.stop()

    @pytest.mark.asyncio
    async def test_stderr_prevents_deadlock(self):
        """Test that stderr reading prevents process deadlock."""
        server = MCPServer("test", "echo", ["hello"], {})

        # Mock process with large stderr output
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Simulate large stderr output (>64KB that would cause deadlock)
        large_stderr_lines = [
            b"Error line " + str(i).encode() + b"\n"
            for i in range(10000)
        ]

        stderr_iter = iter(large_stderr_lines + [b""])  # EOF
        mock_process.stderr.readline = AsyncMock(side_effect=lambda: next(stderr_iter))

        # Mock stdout for readiness check
        mock_process.stdout.readline = AsyncMock(return_value=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        }).encode() + b"\n")

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await server.start()

            # Give stderr task time to consume lines
            await asyncio.sleep(0.1)

            # Process should not be deadlocked
            assert server._stderr_task is not None

            # Cleanup
            await server.stop()

    @pytest.mark.asyncio
    async def test_stderr_logs_warning_messages(self):
        """Test that stderr output is logged as warnings."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Simulate stderr with warning messages
        stderr_lines = [
            b"Warning: something happened\n",
            b"Error: another issue\n",
            b""  # EOF
        ]
        stderr_iter = iter(stderr_lines)
        mock_process.stderr.readline = AsyncMock(side_effect=lambda: next(stderr_iter))

        mock_process.stdout.readline = AsyncMock(return_value=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        }).encode() + b"\n")

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch('langvel.mcp.manager.logger') as mock_logger:
                await server.start()

                # Give stderr task time to process
                await asyncio.sleep(0.1)

                # Verify warnings were logged
                assert mock_logger.warning.call_count >= 2

                # Cleanup
                await server.stop()

    @pytest.mark.asyncio
    async def test_stderr_task_cancelled_on_stop(self):
        """Test that stderr task is properly cancelled on server stop."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.terminate = Mock()
        mock_process.kill = Mock()
        mock_process.wait = AsyncMock()

        # Stderr keeps returning data (infinite loop simulation)
        mock_process.stderr.readline = AsyncMock(return_value=b"stderr line\n")

        mock_process.stdout.readline = AsyncMock(return_value=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        }).encode() + b"\n")

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await server.start()

            stderr_task = server._stderr_task
            assert not stderr_task.done()

            # Stop server
            await server.stop()

            # Verify task was cancelled
            assert stderr_task.cancelled() or stderr_task.done()


class TestMCPJSONRPCValidation:
    """Test MCP JSON-RPC response validation (TODO-026)."""

    @pytest.mark.asyncio
    async def test_request_timeout_handling(self):
        """Test that requests timeout properly."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Setup server as if started
        server.process = mock_process
        server._stderr_task = AsyncMock()

        # Simulate timeout - readline never returns
        async def never_returns():
            await asyncio.sleep(100)
            return b""

        mock_process.stdout.readline = never_returns

        # Request should timeout
        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await server._send_request(request, timeout=1)

        assert "timed out" in str(exc_info.value).lower()
        assert server.name in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_json_error_response_handling(self):
        """Test that JSON-RPC error responses are properly handled."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        server.process = mock_process
        server._stderr_task = AsyncMock()

        # Return error response
        error_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32601,
                "message": "Method not found",
                "data": "The method 'test' does not exist"
            }
        }

        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(error_response).encode() + b"\n"
        )

        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await server._send_request(request, timeout=5)

        assert "Method not found" in str(exc_info.value)
        assert "-32601" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self):
        """Test that invalid JSON responses are caught."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        server.process = mock_process
        server._stderr_task = AsyncMock()

        # Return invalid JSON
        mock_process.stdout.readline = AsyncMock(
            return_value=b"This is not valid JSON\n"
        )

        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await server._send_request(request, timeout=5)

        assert "Invalid JSON-RPC response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_response_id_verification(self):
        """Test that response ID is verified against request ID."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        server.process = mock_process
        server._stderr_task = AsyncMock()

        # Return response with mismatched ID
        response = {
            "jsonrpc": "2.0",
            "id": 999,  # Different from request ID
            "result": {"status": "ok"}
        }

        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )

        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        # Should log warning but not fail
        with patch('langvel.mcp.manager.logger') as mock_logger:
            result = await server._send_request(request, timeout=5)

            # Warning should be logged
            assert mock_logger.warning.called
            warning_message = str(mock_logger.warning.call_args)
            assert "mismatch" in warning_message.lower()

    @pytest.mark.asyncio
    async def test_missing_result_field(self):
        """Test that responses without 'result' field are caught."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        server.process = mock_process
        server._stderr_task = AsyncMock()

        # Return response without result field
        response = {
            "jsonrpc": "2.0",
            "id": 1
            # Missing 'result' field
        }

        mock_process.stdout.readline = AsyncMock(
            return_value=json.dumps(response).encode() + b"\n"
        )

        request = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        with pytest.raises(RuntimeError) as exc_info:
            await server._send_request(request, timeout=5)

        assert "missing 'result' field" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_request_id_counter_increments(self):
        """Test that request IDs increment properly."""
        server = MCPServer("test", "echo", ["hello"], {})

        # Get multiple IDs
        id1 = server._next_request_id()
        id2 = server._next_request_id()
        id3 = server._next_request_id()

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3


class TestMCPReadinessCheck:
    """Test MCP server readiness checking."""

    @pytest.mark.asyncio
    async def test_readiness_check_replaces_fixed_sleep(self):
        """Test that readiness check is performed instead of fixed sleep."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Server becomes ready immediately
        mock_process.stdout.readline = AsyncMock(return_value=json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "test_tool"}]}
        }).encode() + b"\n")

        mock_process.stderr.readline = AsyncMock(return_value=b"")

        import time

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            start = time.time()
            await server.start()
            elapsed = time.time() - start

            # Should complete quickly (< 1 second) instead of waiting fixed 1 second
            assert elapsed < 1.0

            # Cleanup
            await server.stop()

    @pytest.mark.asyncio
    async def test_readiness_check_timeout(self):
        """Test that readiness check times out if server never responds."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Server never becomes ready
        async def never_ready():
            await asyncio.sleep(100)
            return b""

        mock_process.stdout.readline = never_ready
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with pytest.raises(RuntimeError) as exc_info:
                await server.start()

            assert "failed to become ready" in str(exc_info.value).lower()
            assert "timeout" in str(exc_info.value).lower() or "10s" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_readiness_check_retries_on_failure(self):
        """Test that readiness check retries on transient failures."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()

        # Fail first 2 times, then succeed
        call_count = 0
        async def readline_with_retries():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Transient error")
            return json.dumps({
                "jsonrpc": "2.0",
                "id": call_count,
                "result": {"tools": []}
            }).encode() + b"\n"

        mock_process.stdout.readline = readline_with_retries
        mock_process.stderr.readline = AsyncMock(return_value=b"")

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            # Should eventually succeed after retries
            await server.start()

            # Verify it retried (call_count > 1)
            assert call_count > 2

            # Cleanup
            await server.stop()


class TestMCPGracefulShutdown:
    """Test MCP server graceful shutdown."""

    @pytest.mark.asyncio
    async def test_graceful_termination(self):
        """Test that server terminates gracefully."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock()

        server.process = mock_process

        # Create a real asyncio task that can be cancelled
        async def dummy_stderr_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                pass

        server._stderr_task = asyncio.create_task(dummy_stderr_task())

        await server.stop()

        # Verify terminate was called
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()

    @pytest.mark.asyncio
    async def test_force_kill_on_timeout(self):
        """Test that server is killed if termination times out."""
        server = MCPServer("test", "echo", ["hello"], {})

        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.terminate = Mock()
        mock_process.kill = Mock()

        # wait() times out first time, succeeds second time
        wait_call_count = 0
        async def wait_with_timeout():
            nonlocal wait_call_count
            wait_call_count += 1
            if wait_call_count == 1:
                await asyncio.sleep(10)  # Simulate timeout
            return None

        mock_process.wait = wait_with_timeout

        server.process = mock_process

        # Create a real asyncio task
        async def dummy_stderr_task():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                pass

        server._stderr_task = asyncio.create_task(dummy_stderr_task())

        with patch('asyncio.wait_for') as mock_wait_for:
            # First call (wait after terminate) raises timeout
            # Second call (wait after kill) succeeds
            mock_wait_for.side_effect = [asyncio.TimeoutError(), None]

            await server.stop()

            # Verify kill was called after terminate timeout
            mock_process.kill.assert_called_once()


if __name__ == "__main__":
    print("Testing MCP Stderr Deadlock Fix (TODO-008) and JSON-RPC Validation (TODO-026)")
    print("=" * 80)

    # Run tests
    pytest.main([__file__, "-v", "-s"])
