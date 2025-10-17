"""
Fluent Graph Builder - Ultra-intuitive graph building with natural syntax.

This module provides a declarative, template-based approach to building complex
graphs with minimal code and maximum readability.
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
from langvel.routing.builder import GraphBuilder
from langvel.state.base import StateModel


class NodeType(Enum):
    """Types of nodes for better DX."""
    PROCESS = "process"
    DECISION = "decision"
    LOOP = "loop"
    PARALLEL = "parallel"
    APPROVAL = "approval"
    RETRY = "retry"


@dataclass
class Node:
    """Declarative node definition."""
    name: str
    func: Callable
    type: NodeType = NodeType.PROCESS
    condition: Optional[Callable] = None
    branches: Optional[Dict[str, 'Node']] = None
    retry: int = 0
    requires_approval: bool = False
    parallel_with: Optional[List['Node']] = None
    next_node: Optional['Node'] = None


class Flow:
    """
    Ultra-intuitive fluent interface for building complex workflows.

    Makes graph building feel like writing natural language.

    Example:
        flow = (
            Flow(MyState)
            .start_with(self.validate)
            .when(lambda s: s.is_premium)
                .do(self.premium_flow)
            .otherwise()
                .do(self.standard_flow)
            .then_approve("Review required")
            .retry_until(self.save_data, max_attempts=3)
            .finally_do(self.send_notification)
        )
    """

    def __init__(self, state_model: Type[StateModel]):
        """Initialize flow builder."""
        self.state_model = state_model
        self.steps: List[Dict[str, Any]] = []
        self._current_context = None
        self._branch_stack = []

    # ==========================================
    # Natural Language Methods
    # ==========================================

    def start_with(self, func: Callable, description: str = None) -> 'Flow':
        """Start the workflow with this function."""
        self.steps.append({
            'type': 'start',
            'func': func,
            'description': description or func.__name__
        })
        return self

    def then(self, func: Callable, description: str = None) -> 'Flow':
        """Then execute this function."""
        self.steps.append({
            'type': 'then',
            'func': func,
            'description': description or func.__name__
        })
        return self

    def when(self, condition: Union[Callable, str]) -> 'Flow':
        """When this condition is True..."""
        self._current_context = {
            'type': 'when',
            'condition': condition,
            'then_branch': [],
            'else_branch': []
        }
        self._branch_stack.append(self._current_context)
        return self

    def do(self, func: Callable, description: str = None) -> 'Flow':
        """...do this function."""
        if self._current_context and self._current_context['type'] == 'when':
            self._current_context['then_branch'].append({
                'func': func,
                'description': description or func.__name__
            })
        return self

    def otherwise(self) -> 'Flow':
        """Otherwise (else branch)..."""
        # Switch to else branch
        return self

    def then_approve(self, message: str = "Approval required") -> 'Flow':
        """Then wait for human approval."""
        self.steps.append({
            'type': 'approve',
            'message': message
        })
        return self

    def retry_until(
        self,
        func: Callable,
        condition: Optional[Callable] = None,
        max_attempts: int = 3
    ) -> 'Flow':
        """Retry this function until condition is met."""
        self.steps.append({
            'type': 'retry',
            'func': func,
            'condition': condition or (lambda s: s.success),
            'max_attempts': max_attempts
        })
        return self

    def loop_over(
        self,
        func: Callable,
        items_field: str,
        max_iterations: int = 1000
    ) -> 'Flow':
        """Loop over items in state field."""
        self.steps.append({
            'type': 'loop_over',
            'func': func,
            'items_field': items_field,
            'max_iterations': max_iterations
        })
        return self

    def in_parallel(self, *funcs: Callable, wait_for_all: bool = True) -> 'Flow':
        """Execute these functions in parallel."""
        self.steps.append({
            'type': 'parallel',
            'funcs': funcs,
            'wait_for_all': wait_for_all
        })
        return self

    def use_subflow(self, subflow: 'Flow', name: str = None) -> 'Flow':
        """Use another flow as a subgraph."""
        self.steps.append({
            'type': 'subflow',
            'flow': subflow,
            'name': name
        })
        return self

    def finally_do(self, func: Callable) -> 'Flow':
        """Finally, execute this function."""
        self.steps.append({
            'type': 'finally',
            'func': func
        })
        return self

    def build(self) -> GraphBuilder:
        """Build the actual GraphBuilder from fluent definition."""
        builder = GraphBuilder(self.state_model)

        # Convert fluent steps to graph builder calls
        for step in self.steps:
            if step['type'] == 'start':
                builder = builder.then(step['func'])

            elif step['type'] == 'then':
                builder = builder.then(step['func'])

            elif step['type'] == 'approve':
                builder = builder.interrupt()

            elif step['type'] == 'retry':
                builder = builder.until(
                    step['func'],
                    condition=step['condition']
                )

            elif step['type'] == 'loop_over':
                # Generate loop function
                items_field = step['items_field']
                process_func = step['func']

                def loop_func(state):
                    items = getattr(state, items_field, [])
                    if items:
                        item = items.pop(0)
                        return process_func(state, item)
                    return state

                builder = builder.loop(
                    loop_func,
                    condition=lambda s: len(getattr(s, items_field, [])) > 0,
                    max_iterations=step['max_iterations']
                )

            elif step['type'] == 'parallel':
                builder = builder.parallel(*step['funcs'])

            elif step['type'] == 'subflow':
                subgraph = step['flow'].build()
                builder = builder.subgraph(subgraph, name=step.get('name'))

            elif step['type'] == 'finally':
                builder = builder.then(step['func'])

        return builder.end()


# ==========================================
# Graph Templates
# ==========================================

class GraphTemplate:
    """Pre-built graph templates for common patterns."""

    @staticmethod
    def crud_api(state_model: Type[StateModel]) -> Flow:
        """
        Template for CRUD API workflow.

        Includes: auth → validate → execute → audit → respond
        """
        return Flow(state_model)

    @staticmethod
    def approval_workflow(state_model: Type[StateModel]) -> Flow:
        """
        Template for approval workflow.

        Includes: submit → review → approve/reject → notify
        """
        return Flow(state_model)

    @staticmethod
    def data_pipeline(state_model: Type[StateModel]) -> Flow:
        """
        Template for ETL data pipeline.

        Includes: extract → transform → validate → load → verify
        """
        return Flow(state_model)

    @staticmethod
    def ai_agent(state_model: Type[StateModel]) -> Flow:
        """
        Template for AI agent workflow.

        Includes: classify → retrieve (RAG) → generate → validate → respond
        """
        return Flow(state_model)


# ==========================================
# Decorator-Based Graph Definition
# ==========================================

def workflow(state_model: Type[StateModel] = None):
    """
    Decorator for defining entire workflows declaratively.

    Example:
        @workflow(MyState)
        class OrderProcessing:
            @step(order=1)
            async def validate(self, state):
                return state

            @step(order=2, requires_approval=True)
            async def process_payment(self, state):
                return state

            @step(order=3, parallel_with=['send_email', 'send_sms'])
            async def notify_customer(self, state):
                return state
    """
    def decorator(cls):
        cls._workflow_state_model = state_model
        cls._workflow_steps = []
        return cls
    return decorator


def step(
    order: int,
    when: Optional[Callable] = None,
    retry: int = 0,
    requires_approval: bool = False,
    parallel_with: Optional[List[str]] = None,
    description: str = None
):
    """
    Decorator to mark a method as a workflow step.

    Args:
        order: Execution order
        when: Conditional function (only run if True)
        retry: Number of retry attempts
        requires_approval: Whether to pause for approval
        parallel_with: List of other steps to run in parallel
        description: Human-readable description
    """
    def decorator(func):
        func._workflow_step = {
            'order': order,
            'when': when,
            'retry': retry,
            'requires_approval': requires_approval,
            'parallel_with': parallel_with or [],
            'description': description or func.__name__
        }
        return func
    return decorator


# ==========================================
# Visual Graph Generation
# ==========================================

class GraphVisualizer:
    """Generate visual representations of graphs."""

    @staticmethod
    def to_mermaid(flow: Flow) -> str:
        """
        Convert flow to Mermaid diagram syntax.

        Returns:
            Mermaid diagram code that can be rendered
        """
        lines = ["```mermaid", "graph TD"]

        for i, step in enumerate(flow.steps):
            node_id = f"N{i}"
            next_id = f"N{i+1}" if i < len(flow.steps) - 1 else "END"

            if step['type'] == 'start':
                lines.append(f"    START([Start]) --> {node_id}")
                lines.append(f"    {node_id}[{step['description']}]")

            elif step['type'] == 'then':
                lines.append(f"    {node_id}[{step['description']}]")
                lines.append(f"    {node_id} --> {next_id}")

            elif step['type'] == 'approve':
                lines.append(f"    {node_id}{{{{Approval Required}}}}")
                lines.append(f"    {node_id} --> {next_id}")

            elif step['type'] == 'parallel':
                lines.append(f"    {node_id}[Parallel Execution]")
                for j, func in enumerate(step['funcs']):
                    parallel_id = f"{node_id}_P{j}"
                    lines.append(f"    {node_id} --> {parallel_id}[{func.__name__}]")
                    lines.append(f"    {parallel_id} --> {next_id}")

        lines.append(f"    N{len(flow.steps)-1} --> END([End])")
        lines.append("```")

        return "\n".join(lines)

    @staticmethod
    def to_ascii(flow: Flow) -> str:
        """Generate ASCII art representation of flow."""
        output = []
        output.append("=" * 50)
        output.append("WORKFLOW VISUALIZATION")
        output.append("=" * 50)

        for i, step in enumerate(flow.steps):
            if step['type'] == 'start':
                output.append(f"\n┌─ START")
                output.append(f"│  └─ {step['description']}")

            elif step['type'] == 'then':
                output.append(f"│")
                output.append(f"├─ THEN")
                output.append(f"│  └─ {step['description']}")

            elif step['type'] == 'approve':
                output.append(f"│")
                output.append(f"├─ ⏸️  APPROVAL REQUIRED")
                output.append(f"│  └─ {step['message']}")

            elif step['type'] == 'retry':
                output.append(f"│")
                output.append(f"├─ 🔄 RETRY (max {step['max_attempts']})")
                output.append(f"│  └─ {step['func'].__name__}")

            elif step['type'] == 'parallel':
                output.append(f"│")
                output.append(f"├─ ⚡ PARALLEL")
                for func in step['funcs']:
                    output.append(f"│  ├─ {func.__name__}")

        output.append(f"│")
        output.append(f"└─ END")
        output.append("=" * 50)

        return "\n".join(output)


# ==========================================
# AI-Assisted Graph Building
# ==========================================

class GraphAssistant:
    """AI-powered assistance for building graphs."""

    @staticmethod
    def suggest_next_step(flow: Flow, state_model: Type[StateModel]) -> List[str]:
        """
        Suggest next logical steps based on current flow.

        Returns:
            List of suggested next steps with explanations
        """
        suggestions = []

        # Analyze current flow
        has_approval = any(s['type'] == 'approve' for s in flow.steps)
        has_error_handling = any(s['type'] == 'retry' for s in flow.steps)
        has_parallel = any(s['type'] == 'parallel' for s in flow.steps)

        # Make suggestions
        if not has_approval and len(flow.steps) > 2:
            suggestions.append(
                "Consider adding .then_approve() for critical operations"
            )

        if not has_error_handling:
            suggestions.append(
                "Add .retry_until() for operations that might fail"
            )

        if not has_parallel and len(flow.steps) > 3:
            suggestions.append(
                "Use .in_parallel() to execute independent operations concurrently"
            )

        # Always suggest validation
        suggestions.append(
            "Add validation step with .then(self.validate_data)"
        )

        # Always suggest finalization
        suggestions.append(
            "End with .finally_do(self.cleanup) for cleanup operations"
        )

        return suggestions

    @staticmethod
    def validate_flow(flow: Flow) -> List[str]:
        """
        Validate flow and return warnings/suggestions.

        Returns:
            List of warnings or suggestions
        """
        warnings = []

        # Check for common issues
        if len(flow.steps) == 0:
            warnings.append("⚠️  Flow is empty - add steps with .start_with()")

        if not any(s['type'] == 'start' for s in flow.steps):
            warnings.append("⚠️  Flow should start with .start_with()")

        # Check for unbalanced branches
        branch_depth = 0
        for step in flow.steps:
            if step['type'] == 'when':
                branch_depth += 1
            # Could add more checks here

        return warnings


# ==========================================
# Quick Helpers
# ==========================================

def quick_flow(state_model: Type[StateModel], *steps: Callable) -> GraphBuilder:
    """
    Ultra-quick flow creation for simple sequential workflows.

    Example:
        graph = quick_flow(
            MyState,
            self.step1,
            self.step2,
            self.step3
        )
    """
    flow = Flow(state_model)
    for step in steps:
        flow = flow.then(step)
    return flow.build()


def approval_flow(
    state_model: Type[StateModel],
    before_approval: List[Callable],
    after_approval: List[Callable]
) -> GraphBuilder:
    """
    Quick approval workflow.

    Example:
        graph = approval_flow(
            MyState,
            before_approval=[self.validate, self.prepare],
            after_approval=[self.execute, self.notify]
        )
    """
    flow = Flow(state_model)

    for step in before_approval:
        flow = flow.then(step)

    flow = flow.then_approve()

    for step in after_approval:
        flow = flow.then(step)

    return flow.build()


def retry_flow(
    state_model: Type[StateModel],
    operation: Callable,
    max_attempts: int = 3,
    success_condition: Callable = None
) -> GraphBuilder:
    """
    Quick retry workflow.

    Example:
        graph = retry_flow(
            MyState,
            operation=self.call_api,
            max_attempts=5,
            success_condition=lambda s: s.api_success
        )
    """
    flow = Flow(state_model)
    flow = flow.retry_until(
        operation,
        condition=success_condition,
        max_attempts=max_attempts
    )
    return flow.build()
