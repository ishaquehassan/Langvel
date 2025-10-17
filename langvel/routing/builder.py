"""Graph builder - Fluent interface for building LangGraph workflows."""

from typing import Any, Callable, Dict, List, Optional, Type
from langgraph.graph import StateGraph, START, END

from langvel.state.base import StateModel


class GraphBuilder:
    """
    Fluent interface for building LangGraph workflows.

    Provides a Laravel-like chainable API for defining agent flows.
    """

    def __init__(self, state_model: Type[StateModel]):
        """
        Initialize graph builder.

        Args:
            state_model: Pydantic state model class
        """
        self.state_model = state_model
        self.nodes: List[tuple] = []
        self.edges: List[tuple] = []
        self.conditional_edges: List[tuple] = []
        self.current_node: Optional[str] = None

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

    def parallel(self, *funcs: Callable) -> "GraphBuilder":
        """
        Execute multiple nodes in parallel.

        Args:
            *funcs: Functions to execute in parallel

        Returns:
            Self for chaining
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

        # Don't set current_node after parallel - requires explicit merge
        self.current_node = None
        return self

    def end(self) -> "GraphBuilder":
        """
        Mark the end of the graph.

        Returns:
            Self for chaining
        """
        if self.current_node:
            self.edges.append((self.current_node, END))
        return self

    def _compile(self, agent: Any) -> StateGraph:
        """
        Compile the builder into a LangGraph StateGraph.

        Args:
            agent: Agent instance (for binding methods)

        Returns:
            Compiled StateGraph
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

        return graph
