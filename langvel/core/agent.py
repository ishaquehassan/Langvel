"""Base Agent class - The heart of Langvel framework."""

from typing import Any, Callable, Dict, List, Optional, Type
from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from langvel.state.base import StateModel
from langvel.tools.registry import ToolRegistry
from langvel.middleware.manager import MiddlewareManager
from langvel.rag.manager import RAGManager
from langvel.mcp.manager import MCPManager
from langvel.memory.manager import MemoryManager


class Agent(ABC):
    """
    Base Agent class inspired by Laravel's Controller pattern.

    All agents should inherit from this class and implement the build_graph method.

    Example:
        class CustomerSupportAgent(Agent):
            def build_graph(self):
                return (
                    self.start()
                    .then(self.classify_request)
                    .branch({
                        'technical': self.handle_technical,
                        'billing': self.handle_billing,
                    })
                    .end()
                )
    """

    state_model: Type[StateModel] = None
    middleware: List[str] = []
    checkpointer: Optional[str] = "memory"  # memory, postgres, redis
    enable_memory: bool = False  # Enable memory systems

    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.middleware_manager = MiddlewareManager()
        self.rag_manager = RAGManager()
        self.mcp_manager = MCPManager()
        self.memory_manager = None  # Lazy initialization
        self._graph = None
        self._compiled_graph = None

        # Initialize LLM
        self._init_llm()

        # Initialize memory if enabled
        if self.enable_memory:
            self._init_memory()

        # Register tools and middleware
        self._register_tools()
        self._register_middleware()

    def _init_llm(self):
        """Initialize LLM client."""
        from langvel.llm.manager import LLMManager
        from config.langvel import config

        self.llm = LLMManager(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE
        )

    def _init_memory(self):
        """Initialize memory systems."""
        from langvel.memory.manager import MemoryManager
        from langvel.memory.semantic import SemanticMemory
        from langvel.memory.episodic import EpisodicMemory
        from langvel.memory.working import WorkingMemory

        # Create memory instances with configured backends
        semantic = SemanticMemory(backend='postgres')
        episodic = EpisodicMemory(backend='redis')
        working = WorkingMemory(max_tokens=4000)

        # Create unified manager
        self.memory_manager = MemoryManager(
            semantic=semantic,
            episodic=episodic,
            working=working
        )

    def _register_tools(self):
        """Register all tools defined in this agent."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_tool'):
                self.tool_registry.register(attr_name, attr)

    def _register_middleware(self):
        """Register middleware for this agent."""
        for middleware_name in self.middleware:
            self.middleware_manager.register(middleware_name)

    @abstractmethod
    def build_graph(self) -> "GraphBuilder":
        """
        Define the agent's workflow graph.

        This method should return a GraphBuilder instance that defines
        the flow of your agent.

        Returns:
            GraphBuilder: The graph builder instance
        """
        pass

    def compile(self) -> StateGraph:
        """
        Compile the agent into a LangGraph StateGraph with interrupt support.

        Returns:
            StateGraph: The compiled graph ready for execution
        """
        if self._compiled_graph is not None:
            return self._compiled_graph

        # Build the graph using the builder pattern
        builder = self.build_graph()
        graph = builder._compile(self)

        # Set up checkpointer
        checkpointer = self._get_checkpointer()

        # Compile with checkpointer and interrupt support
        compile_kwargs = {'checkpointer': checkpointer}

        # Add interrupt configuration if defined
        if hasattr(graph, '_interrupt_before') and graph._interrupt_before:
            compile_kwargs['interrupt_before'] = graph._interrupt_before

        self._compiled_graph = graph.compile(**compile_kwargs)
        return self._compiled_graph

    def _get_checkpointer(self):
        """Get the appropriate checkpointer based on configuration."""
        if self.checkpointer == "memory":
            return MemorySaver()
        elif self.checkpointer == "postgres":
            from langvel.state.checkpointers import PostgresCheckpointer
            return PostgresCheckpointer()
        elif self.checkpointer == "redis":
            from langvel.state.checkpointers import RedisCheckpointer
            return RedisCheckpointer()
        return None

    async def invoke(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Invoke the agent with input data.

        Args:
            input_data: Input state data
            config: Optional configuration for the run

        Returns:
            The final state after execution
        """
        from langvel.observability.tracer import get_observability_manager

        observability = get_observability_manager()

        # Start trace
        trace_id = observability.start_trace(
            name=self.__class__.__name__,
            input_data=input_data,
            metadata={'agent_class': self.__class__.__name__}
        )

        try:
            # Apply before middleware
            input_data = await self.middleware_manager.run_before(input_data)

            # Compile and run the graph
            graph = self.compile()
            result = await graph.ainvoke(input_data, config)

            # Apply after middleware
            result = await self.middleware_manager.run_after(result)

            # End trace with success
            observability.end_trace(trace_id, result)

            return result

        except Exception as e:
            # End trace with error
            observability.end_trace(trace_id, {}, error=e)
            raise

    async def stream(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None):
        """
        Stream the agent execution.

        Args:
            input_data: Input state data
            config: Optional configuration for the run

        Yields:
            State updates as they occur
        """
        input_data = await self.middleware_manager.run_before(input_data)

        graph = self.compile()
        async for chunk in graph.astream(input_data, config):
            yield chunk

    async def resume(
        self,
        config: RunnableConfig,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Resume an interrupted agent execution.

        Args:
            config: Configuration with thread_id of the interrupted run
            input_data: Optional updated input data to merge with checkpoint

        Returns:
            The final state after resuming execution

        Example:
            # Agent execution interrupted at node
            config = {"configurable": {"thread_id": "thread-123"}}
            result = await agent.resume(config)
        """
        graph = self.compile()

        # If input_data provided, merge with checkpoint
        if input_data:
            result = await graph.ainvoke(input_data, config)
        else:
            # Resume from checkpoint with None to continue
            result = await graph.ainvoke(None, config)

        return result

    def get_state(self, config: RunnableConfig) -> Optional[Dict[str, Any]]:
        """
        Get the current state of an agent execution.

        Args:
            config: Configuration with thread_id

        Returns:
            Current state dict or None if not found

        Example:
            config = {"configurable": {"thread_id": "thread-123"}}
            state = agent.get_state(config)
        """
        graph = self.compile()
        try:
            snapshot = graph.get_state(config)
            return snapshot.values if snapshot else None
        except Exception:
            return None

    def update_state(
        self,
        config: RunnableConfig,
        values: Dict[str, Any],
        as_node: Optional[str] = None
    ) -> None:
        """
        Update the state of an interrupted execution.

        Args:
            config: Configuration with thread_id
            values: State values to update
            as_node: Optional node name to update as

        Example:
            # Update interrupted state
            config = {"configurable": {"thread_id": "thread-123"}}
            agent.update_state(config, {"approved": True})
            result = await agent.resume(config)
        """
        graph = self.compile()
        graph.update_state(config, values, as_node=as_node)

    async def execute_tool(
        self,
        tool_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a registered tool with automatic retry, timeout, and fallback.

        This method integrates with the ToolRegistry to execute tools decorated with
        @tool, @rag_tool, @mcp_tool, @http_tool, or @llm_tool decorators.

        Args:
            tool_name: Name of the tool to execute
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool execution fails

        Example:
            # In your agent node
            async def process(self, state):
                # Tool will automatically retry 3 times with exponential backoff
                result = await self.execute_tool('analyze_sentiment', state)
                return state
        """
        tool = self.tool_registry.get(tool_name)
        if not tool:
            from langvel.tools.registry import ToolExecutionError
            raise ToolExecutionError(f"Tool '{tool_name}' not found")

        # Get tool metadata for retry/timeout/fallback
        metadata = self.tool_registry.get_metadata(tool_name)

        # Extract decorator settings
        retry = getattr(tool, '_tool_retry', 3)
        timeout = getattr(tool, '_tool_timeout', None)
        fallback_func = getattr(tool, '_tool_fallback', None)

        # Execute using ToolRegistry
        return await self.tool_registry.execute_tool(
            name=tool_name,
            agent=self,
            *args,
            retry=retry,
            timeout=timeout,
            fallback=fallback_func,
            **kwargs
        )

    def start(self) -> "GraphBuilder":
        """Start building the graph."""
        from langvel.routing.builder import GraphBuilder
        return GraphBuilder(self.state_model)


class GraphBuilder:
    """
    Fluent interface for building LangGraph workflows.

    Provides a Laravel-like chainable API for defining agent flows.
    """

    def __init__(self, state_model: Type[StateModel]):
        self.state_model = state_model
        self.nodes: List[tuple] = []
        self.edges: List[tuple] = []
        self.conditional_edges: List[tuple] = []
        self.current_node: Optional[str] = None

    def then(self, func: Callable, name: Optional[str] = None) -> "GraphBuilder":
        """Add a sequential node."""
        node_name = name or func.__name__
        self.nodes.append((node_name, func))

        if self.current_node:
            self.edges.append((self.current_node, node_name))
        else:
            self.edges.append((START, node_name))

        self.current_node = node_name
        return self

    def branch(self, conditions: Dict[str, Callable], condition_func: Optional[Callable] = None) -> "GraphBuilder":
        """Add a conditional branch."""
        if condition_func is None:
            # Auto-generate condition function based on state
            def auto_condition(state):
                return state.get('next_step', 'default')
            condition_func = auto_condition

        # Add all branch nodes
        for branch_name, branch_func in conditions.items():
            self.nodes.append((branch_name, branch_func))

        # Add conditional edge
        self.conditional_edges.append((
            self.current_node,
            condition_func,
            conditions
        ))

        # Don't set current_node after branch - requires explicit merge
        self.current_node = None
        return self

    def merge(self, func: Callable, name: Optional[str] = None) -> "GraphBuilder":
        """Merge multiple branches back together."""
        node_name = name or func.__name__
        self.nodes.append((node_name, func))

        # All branches should lead to this merge point
        # This is simplified - in practice, you'd track branch nodes
        self.current_node = node_name
        return self

    def end(self) -> "GraphBuilder":
        """Mark the end of the graph."""
        if self.current_node:
            self.edges.append((self.current_node, END))
        return self

    def _compile(self, agent: Agent) -> StateGraph:
        """Compile the builder into a LangGraph StateGraph."""
        # Create the graph with the state model
        graph = StateGraph(self.state_model)

        # Add all nodes
        for node_name, node_func in self.nodes:
            graph.add_node(node_name, node_func)

        # Add all edges
        for from_node, to_node in self.edges:
            graph.add_edge(from_node, to_node)

        # Add conditional edges
        for from_node, condition_func, branches in self.conditional_edges:
            graph.add_conditional_edges(
                from_node,
                condition_func,
                branches
            )

        return graph
