"""Observability tracer - LangSmith and Langfuse integration."""

from typing import Any, Callable, Dict, Optional
from functools import wraps
import os
import uuid
from datetime import datetime
import asyncio


class ObservabilityManager:
    """
    Manages observability integrations (LangSmith, Langfuse).

    Provides automatic tracing, logging, and metrics collection for agents.
    """

    def __init__(self):
        self.langsmith_enabled = False
        self.langfuse_enabled = False
        self._langsmith_client = None
        self._langfuse_client = None
        self._current_trace = None

        # Initialize integrations
        self._init_langsmith()
        self._init_langfuse()

    def _init_langsmith(self):
        """Initialize LangSmith integration."""
        api_key = os.getenv('LANGSMITH_API_KEY') or os.getenv('LANGCHAIN_API_KEY')

        if api_key:
            try:
                from langsmith import Client

                self._langsmith_client = Client(
                    api_key=api_key,
                    api_url=os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
                )
                self.langsmith_enabled = True
                print("[Langvel] LangSmith tracing enabled")

            except ImportError:
                print("[Langvel] Warning: langsmith not installed. Run: pip install langsmith")
            except Exception as e:
                print(f"[Langvel] Warning: Could not initialize LangSmith: {e}")

    def _init_langfuse(self):
        """Initialize Langfuse integration."""
        public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
        secret_key = os.getenv('LANGFUSE_SECRET_KEY')

        if public_key and secret_key:
            try:
                from langfuse import Langfuse

                self._langfuse_client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
                )
                self.langfuse_enabled = True
                print("[Langvel] Langfuse tracing enabled")

            except ImportError:
                print("[Langvel] Warning: langfuse not installed. Run: pip install langfuse")
            except Exception as e:
                print(f"[Langvel] Warning: Could not initialize Langfuse: {e}")

    def start_trace(
        self,
        name: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new trace.

        Args:
            name: Trace name (typically agent name)
            input_data: Input data for the trace
            metadata: Optional metadata

        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())

        self._current_trace = {
            'id': trace_id,
            'name': name,
            'start_time': datetime.utcnow(),
            'input': input_data,
            'metadata': metadata or {},
            'spans': []
        }

        # Start LangSmith trace
        if self.langsmith_enabled and self._langsmith_client:
            try:
                self._langsmith_client.create_run(
                    name=name,
                    run_type="chain",
                    inputs=input_data,
                    run_id=trace_id,
                    extra=metadata or {}
                )
            except Exception as e:
                print(f"[Langvel] LangSmith trace start failed: {e}")

        # Start Langfuse trace
        if self.langfuse_enabled and self._langfuse_client:
            try:
                self._langfuse_client.trace(
                    id=trace_id,
                    name=name,
                    input=input_data,
                    metadata=metadata or {}
                )
            except Exception as e:
                print(f"[Langvel] Langfuse trace start failed: {e}")

        return trace_id

    def end_trace(
        self,
        trace_id: str,
        output_data: Dict[str, Any],
        error: Optional[Exception] = None
    ):
        """
        End a trace.

        Args:
            trace_id: Trace ID
            output_data: Output data from the trace
            error: Optional error if trace failed
        """
        if not self._current_trace or self._current_trace['id'] != trace_id:
            return

        self._current_trace['end_time'] = datetime.utcnow()
        self._current_trace['output'] = output_data
        self._current_trace['error'] = str(error) if error else None

        duration = (
            self._current_trace['end_time'] - self._current_trace['start_time']
        ).total_seconds()

        # End LangSmith trace
        if self.langsmith_enabled and self._langsmith_client:
            try:
                self._langsmith_client.update_run(
                    run_id=trace_id,
                    outputs=output_data,
                    error=str(error) if error else None,
                    end_time=datetime.utcnow()
                )
            except Exception as e:
                print(f"[Langvel] LangSmith trace end failed: {e}")

        # End Langfuse trace
        if self.langfuse_enabled and self._langfuse_client:
            try:
                self._langfuse_client.trace(
                    id=trace_id,
                    output=output_data,
                    metadata={
                        'duration_seconds': duration,
                        'error': str(error) if error else None
                    }
                )
            except Exception as e:
                print(f"[Langvel] Langfuse trace end failed: {e}")

        self._current_trace = None

    def add_span(
        self,
        name: str,
        span_type: str,
        input_data: Any,
        output_data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        duration: Optional[float] = None
    ):
        """
        Add a span to the current trace.

        Args:
            name: Span name
            span_type: Type of span (tool, llm, retrieval, etc.)
            input_data: Input data for the span
            output_data: Output data from the span
            metadata: Optional metadata
            error: Optional error
            duration: Duration in seconds
        """
        if not self._current_trace:
            return

        span_id = str(uuid.uuid4())
        span = {
            'id': span_id,
            'name': name,
            'type': span_type,
            'input': input_data,
            'output': output_data,
            'metadata': metadata or {},
            'error': str(error) if error else None,
            'duration': duration
        }

        self._current_trace['spans'].append(span)

        # Add to LangSmith
        if self.langsmith_enabled and self._langsmith_client:
            try:
                self._langsmith_client.create_run(
                    name=name,
                    run_type=span_type,
                    inputs={'input': input_data},
                    outputs={'output': output_data} if output_data else None,
                    error=str(error) if error else None,
                    parent_run_id=self._current_trace['id'],
                    run_id=span_id,
                    extra=metadata or {}
                )
            except Exception as e:
                print(f"[Langvel] LangSmith span failed: {e}")

        # Add to Langfuse
        if self.langfuse_enabled and self._langfuse_client:
            try:
                self._langfuse_client.span(
                    id=span_id,
                    trace_id=self._current_trace['id'],
                    name=name,
                    input=input_data,
                    output=output_data,
                    metadata={
                        'type': span_type,
                        'duration': duration,
                        **(metadata or {})
                    }
                )
            except Exception as e:
                print(f"[Langvel] Langfuse span failed: {e}")

    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        duration: float,
        token_usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an LLM call.

        Args:
            model: Model name
            prompt: Prompt sent to LLM
            response: Response from LLM
            duration: Duration in seconds
            token_usage: Token usage stats
            metadata: Optional metadata
        """
        self.add_span(
            name=f"llm_{model}",
            span_type="llm",
            input_data={'prompt': prompt},
            output_data={'response': response},
            metadata={
                'model': model,
                'token_usage': token_usage or {},
                **(metadata or {})
            },
            duration=duration
        )

    def log_tool_call(
        self,
        tool_name: str,
        tool_type: str,
        input_data: Any,
        output_data: Any,
        duration: float,
        success: bool,
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a tool call.

        Args:
            tool_name: Tool name
            tool_type: Tool type (rag, mcp, http, etc.)
            input_data: Input data
            output_data: Output data
            duration: Duration in seconds
            success: Whether tool succeeded
            error: Optional error
            metadata: Optional metadata
        """
        self.add_span(
            name=tool_name,
            span_type="tool",
            input_data=input_data,
            output_data=output_data if success else None,
            metadata={
                'tool_type': tool_type,
                'success': success,
                **(metadata or {})
            },
            error=error,
            duration=duration
        )

    def flush(self):
        """Flush all pending traces to backends."""
        if self.langfuse_enabled and self._langfuse_client:
            try:
                self._langfuse_client.flush()
            except Exception as e:
                print(f"[Langvel] Langfuse flush failed: {e}")


# Global observability manager instance
_observability_manager = None


def get_observability_manager() -> ObservabilityManager:
    """Get or create global observability manager."""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager()
    return _observability_manager


def trace_agent(func: Callable) -> Callable:
    """
    Decorator to trace agent execution.

    Example:
        @trace_agent
        async def invoke(self, input_data):
            ...
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        manager = get_observability_manager()

        # Extract input data
        input_data = args[0] if args else kwargs.get('input_data', {})

        # Start trace
        trace_id = manager.start_trace(
            name=self.__class__.__name__,
            input_data=input_data,
            metadata={'agent_class': self.__class__.__name__}
        )

        try:
            # Execute function
            result = await func(self, *args, **kwargs)

            # End trace with success
            manager.end_trace(trace_id, result)

            return result

        except Exception as e:
            # End trace with error
            manager.end_trace(trace_id, {}, error=e)
            raise

    return wrapper


def trace_tool(func: Callable) -> Callable:
    """
    Decorator to trace tool execution.

    Example:
        @trace_tool
        async def my_tool(self, state):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        manager = get_observability_manager()

        import time
        start_time = time.time()

        try:
            # Execute function
            result = await func(*args, **kwargs)
            duration = time.time() - start_time

            # Log tool call
            manager.log_tool_call(
                tool_name=func.__name__,
                tool_type=getattr(func, '_tool_type', 'custom'),
                input_data={'args': args, 'kwargs': kwargs},
                output_data=result,
                duration=duration,
                success=True
            )

            return result

        except Exception as e:
            duration = time.time() - start_time

            # Log failed tool call
            manager.log_tool_call(
                tool_name=func.__name__,
                tool_type=getattr(func, '_tool_type', 'custom'),
                input_data={'args': args, 'kwargs': kwargs},
                output_data=None,
                duration=duration,
                success=False,
                error=e
            )
            raise

    return wrapper
