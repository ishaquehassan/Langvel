"""Graph builder - Fluent interface for building LangGraph workflows."""

from typing import Any, Callable, Dict, List, Optional, Type, Union
from langgraph.graph import StateGraph, START, END

from langvel.state.base import StateModel


class SubGraph:
    """Represents a reusable subgraph that can be nested in other graphs."""

    def __init__(self, name: str, builder: "GraphBuilder"):
        """
        Initialize a subgraph.

        Args:
            name: Subgraph name
            builder: GraphBuilder instance defining the subgraph
        """
        self.name = name
        self.builder = builder
        self.entry_node: Optional[str] = None
        self.exit_node: Optional[str] = None


class GraphBuilder:
    """
    Fluent interface for building LangGraph workflows.

    Provides a Laravel-like chainable API for defining agent flows with advanced features:
    - Sequential flows with .then()
    - Conditional branching with .branch()
    - Parallel execution with .parallel()
    - Loop patterns with .loop(), .until(), .while_()
    - Subgraph composition with .subgraph()
    - Dynamic graph modification
    - Human-in-the-loop interrupts
    """

    def __init__(self, state_model: Type[StateModel], parent_builder: Optional["GraphBuilder"] = None):
        """
        Initialize graph builder.

        Args:
            state_model: Pydantic state model class
            parent_builder: Parent builder for nested subgraphs
        """
        self.state_model = state_model
        self.nodes: List[tuple] = []
        self.edges: List[tuple] = []
        self.conditional_edges: List[tuple] = []
        self.current_node: Optional[str] = None
        self.parallel_nodes: List[str] = []  # Track parallel execution nodes
        self.subgraphs: Dict[str, SubGraph] = {}
        self.interrupt_nodes: List[str] = []  # Nodes that require human approval
        self.parent_builder = parent_builder
        self._is_dynamic = False  # Allow runtime modifications

    def then(self, func: Callable, name: Optional[str] = None) -> "GraphBuilder":
        """
        Add a sequential node.

        Args:
            func: Node function
            name: Optional node name (defaults to function name)

        Returns:
            Self for chaining
        """
        node_name = name or func.__name__
        self.nodes.append((node_name, func))

        if self.current_node:
            self.edges.append((self.current_node, node_name))
        else:
            self.edges.append((START, node_name))

        self.current_node = node_name
        return self

    def branch(
        self,
        conditions: Dict[str, Callable],
        condition_func: Optional[Callable] = None
    ) -> "GraphBuilder":
        """
        Add a conditional branch.

        Args:
            conditions: Dictionary mapping condition names to node functions
            condition_func: Function to determine which branch to take

        Returns:
            Self for chaining
        """
        if condition_func is None:
            # Auto-generate condition function based on state
            def auto_condition(state):
                return state.get('next_step', list(conditions.keys())[0])
            condition_func = auto_condition

        # Add all branch nodes
        for branch_name, branch_func in conditions.items():
            node_name = branch_func.__name__ if hasattr(branch_func, '__name__') else branch_name
            self.nodes.append((node_name, branch_func))

        # Add conditional edge
        self.conditional_edges.append((
            self.current_node,
            condition_func,
            {k: (v.__name__ if hasattr(v, '__name__') else k) for k, v in conditions.items()}
        ))

        # Don't set current_node after branch - requires explicit merge
        self.current_node = None
        return self

    def merge(self, func: Callable, name: Optional[str] = None) -> "GraphBuilder":
        """
        Merge multiple branches back together.

        Args:
            func: Merge function
            name: Optional node name

        Returns:
            Self for chaining
        """
        node_name = name or func.__name__
        self.nodes.append((node_name, func))

        # All branches should lead to this merge point
        self.current_node = node_name
        return self

    def parallel(self, *funcs: Callable, auto_merge: bool = True) -> "GraphBuilder":
        """
        Execute multiple nodes in parallel.

        Args:
            *funcs: Functions to execute in parallel
            auto_merge: If True, automatically merge to END (default: True)

        Returns:
            Self for chaining

        Example:
            .parallel(self.fetch_data, self.validate_user, auto_merge=True)
        """
        # Add all parallel nodes
        parallel_node_names = []
        for func in funcs:
            node_name = func.__name__
            self.nodes.append((node_name, func))
            parallel_node_names.append(node_name)

            # Connect from current node to each parallel node
            if self.current_node:
                self.edges.append((self.current_node, node_name))
            else:
                self.edges.append((START, node_name))

        # Store parallel nodes for potential merging
        self.parallel_nodes = parallel_node_names

        # Auto-merge to END if requested
        if auto_merge:
            for node_name in parallel_node_names:
                self.edges.append((node_name, END))
            self.current_node = None
        else:
            # Require explicit merge
            self.current_node = None

        return self

    def loop(
        self,
        func: Callable,
        condition: Callable,
        max_iterations: Optional[int] = None,
        name: Optional[str] = None
    ) -> "GraphBuilder":
        """
        Create a loop that repeats while condition is True.

        Args:
            func: Function to execute in loop
            condition: Function that returns True to continue, False to exit
            max_iterations: Optional maximum iterations (safety limit)
            name: Optional node name

        Returns:
            Self for chaining

        Example:
            .loop(
                self.process_batch,
                lambda state: len(state.remaining_items) > 0,
                max_iterations=100
            )
        """
        node_name = name or func.__name__

        # Add the loop node
        self.nodes.append((node_name, func))

        # Connect from current node or START
        if self.current_node:
            self.edges.append((self.current_node, node_name))
        else:
            self.edges.append((START, node_name))

        # Create loop condition wrapper that tracks iterations
        def loop_condition_wrapper(state):
            # Check max iterations if specified
            if max_iterations:
                iteration_count = getattr(state, f'_{node_name}_iterations', 0)
                if iteration_count >= max_iterations:
                    return 'exit'
                # Increment counter
                setattr(state, f'_{node_name}_iterations', iteration_count + 1)

            # Check user condition
            if condition(state):
                return 'continue'
            return 'exit'

        # Add conditional edge for loop
        self.conditional_edges.append((
            node_name,
            loop_condition_wrapper,
            {'continue': node_name, 'exit': END}  # Loop back or exit
        ))

        # After a loop, current_node is None (requires explicit next step)
        self.current_node = None
        return self

    def until(
        self,
        func: Callable,
        condition: Callable,
        name: Optional[str] = None
    ) -> "GraphBuilder":
        """
        Execute function repeatedly until condition becomes True (do-while pattern).

        Args:
            func: Function to execute
            condition: Exit condition (opposite of loop - True = exit)
            name: Optional node name

        Returns:
            Self for chaining

        Example:
            .until(self.retry_api_call, lambda state: state.success == True)
        """
        # until is just loop with inverted condition
        def inverted_condition(state):
            return not condition(state)

        return self.loop(func, inverted_condition, name=name)

    def while_(
        self,
        condition: Callable,
        func: Callable,
        name: Optional[str] = None
    ) -> "GraphBuilder":
        """
        Execute function while condition is True (while pattern).
        Alias for loop() with clearer while semantics.

        Args:
            condition: Continue condition
            func: Function to execute
            name: Optional node name

        Returns:
            Self for chaining

        Example:
            .while_(lambda state: state.retry_count < 3, self.attempt_connection)
        """
        return self.loop(func, condition, name=name)

    def subgraph(
        self,
        builder_or_subgraph: Union["GraphBuilder", SubGraph],
        name: Optional[str] = None
    ) -> "GraphBuilder":
        """
        Embed a subgraph within the current graph.

        Args:
            builder_or_subgraph: GraphBuilder or SubGraph instance to embed
            name: Optional name for the subgraph

        Returns:
            Self for chaining

        Example:
            # Define reusable subgraph
            auth_flow = GraphBuilder(AuthState)
            auth_flow.start().then(verify_token).then(load_user).end()

            # Use in main graph
            self.start().subgraph(auth_flow, 'authentication').then(process).end()
        """
        if isinstance(builder_or_subgraph, SubGraph):
            subgraph = builder_or_subgraph
        else:
            # Create subgraph from builder
            subgraph_name = name or f"subgraph_{len(self.subgraphs)}"
            subgraph = SubGraph(subgraph_name, builder_or_subgraph)

        # Store subgraph
        self.subgraphs[subgraph.name] = subgraph

        # Import all nodes from subgraph
        for node_name, node_func in subgraph.builder.nodes:
            prefixed_name = f"{subgraph.name}_{node_name}"
            self.nodes.append((prefixed_name, node_func))

            # Track entry and exit
            if subgraph.entry_node is None:
                subgraph.entry_node = prefixed_name
            subgraph.exit_node = prefixed_name

        # Import edges
        for from_node, to_node in subgraph.builder.edges:
            from_prefixed = f"{subgraph.name}_{from_node}" if from_node != START and from_node != END else from_node
            to_prefixed = f"{subgraph.name}_{to_node}" if to_node != START and to_node != END else to_node

            if from_node == START:
                from_prefixed = self.current_node or START
            if to_node == END:
                to_prefixed = None  # Will connect to next node

            if to_prefixed:
                self.edges.append((from_prefixed, to_prefixed))

        # Import conditional edges
        for from_node, condition_func, branches in subgraph.builder.conditional_edges:
            from_prefixed = f"{subgraph.name}_{from_node}"
            branches_prefixed = {
                k: f"{subgraph.name}_{v}" if v != END else v
                for k, v in branches.items()
            }
            self.conditional_edges.append((from_prefixed, condition_func, branches_prefixed))

        # Set current node to subgraph exit
        self.current_node = subgraph.exit_node
        return self

    def interrupt(self, node: Optional[Callable] = None) -> "GraphBuilder":
        """
        Mark the current or specified node as requiring human approval.

        Args:
            node: Optional node to mark for interrupt. If None, marks current node.

        Returns:
            Self for chaining

        Example:
            .then(self.classify)
            .interrupt()  # Pause here for human review
            .then(self.execute)
        """
        if node:
            node_name = node.__name__
        else:
            node_name = self.current_node

        if node_name and node_name not in self.interrupt_nodes:
            self.interrupt_nodes.append(node_name)

        return self

    def dynamic(self, enabled: bool = True) -> "GraphBuilder":
        """
        Enable or disable dynamic graph modification at runtime.

        Args:
            enabled: Whether to allow runtime modifications

        Returns:
            Self for chaining

        Example:
            builder.dynamic(True)  # Allow runtime modifications
        """
        self._is_dynamic = enabled
        return self

    def add_node_dynamic(
        self,
        func: Callable,
        connect_from: Optional[str] = None,
        connect_to: Optional[str] = None,
        name: Optional[str] = None
    ) -> str:
        """
        Add a node to the graph at runtime (requires dynamic mode).

        Args:
            func: Node function
            connect_from: Node to connect from (None = START)
            connect_to: Node to connect to (None = END)
            name: Optional node name

        Returns:
            Name of the added node

        Raises:
            RuntimeError: If dynamic mode is not enabled
        """
        if not self._is_dynamic:
            raise RuntimeError("Dynamic node addition requires .dynamic(True) to be called first")

        node_name = name or func.__name__
        self.nodes.append((node_name, func))

        # Add edges
        from_node = connect_from or self.current_node or START
        to_node = connect_to or END

        self.edges.append((from_node, node_name))
        if to_node:
            self.edges.append((node_name, to_node))

        self.current_node = node_name
        return node_name

    def remove_node_dynamic(self, node_name: str) -> "GraphBuilder":
        """
        Remove a node from the graph at runtime (requires dynamic mode).

        Args:
            node_name: Name of node to remove

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If dynamic mode is not enabled
        """
        if not self._is_dynamic:
            raise RuntimeError("Dynamic node removal requires .dynamic(True) to be called first")

        # Remove node
        self.nodes = [n for n in self.nodes if n[0] != node_name]

        # Remove associated edges
        self.edges = [
            (f, t) for f, t in self.edges
            if f != node_name and t != node_name
        ]

        # Remove from conditional edges
        self.conditional_edges = [
            (f, c, b) for f, c, b in self.conditional_edges
            if f != node_name and node_name not in b.values()
        ]

        return self

    def validate(self) -> List[str]:
        """
        Validate the graph structure and return any warnings or errors.

        Returns:
            List of validation messages (empty if valid)

        Example:
            warnings = builder.validate()
            if warnings:
                print("Graph issues:", warnings)
        """
        warnings = []

        # Check for unreachable nodes
        reachable = set()
        to_visit = [START]

        # Build adjacency list
        adjacency = {}
        for from_node, to_node in self.edges:
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].append(to_node)

        for from_node, _, branches in self.conditional_edges:
            if from_node not in adjacency:
                adjacency[from_node] = []
            adjacency[from_node].extend(branches.values())

        # BFS to find reachable nodes
        while to_visit:
            current = to_visit.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            if current in adjacency:
                to_visit.extend(adjacency[current])

        # Check if all nodes are reachable
        all_node_names = {name for name, _ in self.nodes}
        unreachable = all_node_names - reachable

        if unreachable:
            warnings.append(f"Unreachable nodes detected: {unreachable}")

        # Check for nodes without edges to END
        has_path_to_end = set()
        reverse_adjacency = {}

        for from_node, to_node in self.edges:
            if to_node not in reverse_adjacency:
                reverse_adjacency[to_node] = []
            reverse_adjacency[to_node].append(from_node)

        # BFS backward from END
        to_visit = [END]
        while to_visit:
            current = to_visit.pop(0)
            if current in has_path_to_end:
                continue
            has_path_to_end.add(current)

            if current in reverse_adjacency:
                to_visit.extend(reverse_adjacency[current])

        no_end_path = all_node_names - has_path_to_end
        if no_end_path:
            warnings.append(f"Nodes with no path to END: {no_end_path}")

        # Check for missing merge after parallel
        if self.parallel_nodes and self.current_node is None:
            warnings.append("Parallel execution detected without explicit merge or end()")

        return warnings

    def end(self) -> "GraphBuilder":
        """
        Mark the end of the graph.

        Returns:
            Self for chaining
        """
        if self.current_node:
            self.edges.append((self.current_node, END))
        elif self.parallel_nodes:
            # Auto-connect parallel nodes if not already connected
            for node in self.parallel_nodes:
                if not any(e[0] == node and e[1] == END for e in self.edges):
                    self.edges.append((node, END))
        return self

    def _compile(self, agent: Any) -> StateGraph:
        """
        Compile the builder into a LangGraph StateGraph with interrupt support.

        Args:
            agent: Agent instance (for binding methods)

        Returns:
            Compiled StateGraph with interrupts configured
        """
        # Create the graph with the state model
        graph = StateGraph(self.state_model)

        # Add all nodes
        for node_name, node_func in self.nodes:
            # Bind method to agent instance if needed
            if hasattr(node_func, '__self__'):
                # Already bound
                graph.add_node(node_name, node_func)
            else:
                # Bind to agent
                bound_func = node_func.__get__(agent, type(agent))
                graph.add_node(node_name, bound_func)

        # Add all edges
        for from_node, to_node in self.edges:
            graph.add_edge(from_node, to_node)

        # Add conditional edges
        for from_node, condition_func, branches in self.conditional_edges:
            # Bind condition function if needed
            if not hasattr(condition_func, '__self__'):
                condition_func = condition_func.__get__(agent, type(agent))

            graph.add_conditional_edges(
                from_node,
                condition_func,
                branches
            )

        # Store interrupt nodes on the graph for compilation
        if self.interrupt_nodes:
            graph._interrupt_before = self.interrupt_nodes

        return graph
