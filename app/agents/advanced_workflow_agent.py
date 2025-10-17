"""
Advanced Workflow Agent - Demonstrates all new LangGraph features in Langvel.

This example showcases:
1. Loop patterns (.loop(), .until(), .while_())
2. Subgraph composition
3. Human-in-the-loop interrupts
4. Dynamic graph modification
5. Tool execution with retry/fallback
6. Graph validation
7. Parallel execution with auto-merge
"""

from typing import Optional
from pydantic import Field
from langvel.core.agent import Agent
from langvel.state.base import StateModel
from langvel.tools.decorators import tool
from langvel.routing.builder import GraphBuilder


# ========================================
# State Models
# ========================================

class AdvancedWorkflowState(StateModel):
    """State for advanced workflow demonstration."""

    query: str
    iteration_count: int = 0
    max_retries: int = 3
    success: bool = False
    remaining_tasks: list = Field(default_factory=list)
    processed_tasks: list = Field(default_factory=list)
    approved: bool = False
    results: dict = Field(default_factory=dict)


class AuthenticationState(StateModel):
    """State for authentication subgraph."""

    token: Optional[str] = None
    user_id: Optional[str] = None
    authenticated: bool = False


# ========================================
# Reusable Subgraph - Authentication Flow
# ========================================

class AuthenticationFlow:
    """Reusable authentication subgraph."""

    @staticmethod
    def build() -> GraphBuilder:
        """Build the authentication subgraph."""
        auth_graph = GraphBuilder(AuthenticationState)

        return (
            auth_graph
            .then(AuthenticationFlow.verify_token, name='verify_token')
            .then(AuthenticationFlow.load_user, name='load_user')
            .end()
        )

    @staticmethod
    async def verify_token(state: AuthenticationState) -> AuthenticationState:
        """Verify authentication token."""
        # Simulate token verification
        if state.token and len(state.token) > 10:
            state.authenticated = True
        return state

    @staticmethod
    async def load_user(state: AuthenticationState) -> AuthenticationState:
        """Load user data after authentication."""
        if state.authenticated:
            state.user_id = "user_123"
        return state


# ========================================
# Main Agent - Advanced Workflow Patterns
# ========================================

class AdvancedWorkflowAgent(Agent):
    """
    Demonstrates advanced LangGraph patterns in Langvel.

    Features showcased:
    - Loop patterns for retry logic
    - Subgraph composition for reusable flows
    - Human-in-the-loop for approval workflows
    - Dynamic graph modification
    - Parallel execution with auto-merge
    - Tool retry/fallback execution
    - Graph validation
    """

    state_model = AdvancedWorkflowState
    middleware = ['logging']
    checkpointer = 'memory'  # Use 'postgres' or 'redis' for production

    def build_graph(self):
        """
        Build advanced workflow with all new features.
        """
        # Example 1: Basic flow with validation
        builder = (
            self.start()
            .then(self.initialize_tasks)

            # Example 2: Subgraph composition
            # .subgraph(AuthenticationFlow.build(), name='auth')

            # Example 3: Loop pattern - process tasks until done
            .loop(
                self.process_next_task,
                condition=lambda state: len(state.remaining_tasks) > 0,
                max_iterations=10  # Safety limit
            )

            # Example 4: Parallel execution with auto-merge
            .then(self.prepare_results)
            .parallel(
                self.generate_summary,
                self.calculate_metrics,
                self.log_completion,
                auto_merge=True  # Automatically merge to END
            )
        )

        # Example 5: Graph validation
        validation_warnings = builder.validate()
        if validation_warnings:
            print(f"⚠️  Graph validation warnings: {validation_warnings}")

        return builder

    def build_graph_with_interrupt(self):
        """
        Alternative graph with human-in-the-loop approval.
        """
        return (
            self.start()
            .then(self.initialize_tasks)
            .then(self.classify_request)

            # Mark this point for human approval
            .interrupt()

            # Only continues after human approves via update_state()
            .then(self.execute_approved_action)
            .end()
        )

    def build_graph_with_retry_loop(self):
        """
        Graph using until() pattern for retry logic.
        """
        return (
            self.start()
            .until(
                self.attempt_api_call,
                condition=lambda state: state.success or state.iteration_count >= state.max_retries
            )
            .then(self.handle_result)
            .end()
        )

    def build_graph_with_dynamic_nodes(self):
        """
        Graph demonstrating dynamic modification.
        """
        builder = self.start().dynamic(True)  # Enable dynamic mode

        # Build initial graph
        builder.then(self.analyze_request)

        # Dynamically add nodes based on runtime conditions
        # (This would typically be done in a node, shown here for example)
        # builder.add_node_dynamic(self.custom_handler, connect_from='analyze_request')

        return builder.end()

    # ========================================
    # Node Implementations
    # ========================================

    async def initialize_tasks(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Initialize task list for processing."""
        state.remaining_tasks = ['task1', 'task2', 'task3', 'task4']
        state.processed_tasks = []
        return state

    async def process_next_task(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """
        Process one task from the queue.
        This demonstrates loop pattern usage.
        """
        if state.remaining_tasks:
            task = state.remaining_tasks.pop(0)

            # Use tool execution with automatic retry
            try:
                result = await self.execute_tool('process_task_tool', task)
                state.processed_tasks.append(task)
                state.results[task] = result
            except Exception as e:
                print(f"Task {task} failed: {e}")

        state.iteration_count += 1
        return state

    async def classify_request(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Classify the request (before interrupt)."""
        # Analyze query and set classification
        state.results['classification'] = 'high_risk' if 'delete' in state.query.lower() else 'normal'
        return state

    async def execute_approved_action(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Execute action after human approval."""
        if not state.approved:
            state.results['error'] = 'Not approved'
            return state

        # Execute the approved action
        state.results['action_executed'] = True
        state.success = True
        return state

    async def attempt_api_call(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """
        Attempt an API call with retry logic.
        Demonstrates until() pattern.
        """
        state.iteration_count += 1

        # Simulate API call with potential failure
        import random
        if random.random() > 0.3:  # 70% success rate
            state.success = True
            state.results['api_response'] = {'status': 'ok', 'data': 'sample'}

        return state

    async def handle_result(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Handle the result after retries."""
        if not state.success:
            state.results['error'] = f'Failed after {state.iteration_count} attempts'
        return state

    async def analyze_request(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Analyze incoming request."""
        state.results['analyzed'] = True
        return state

    async def prepare_results(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Prepare results before parallel processing."""
        state.results['prepared_at'] = 'timestamp'
        return state

    async def generate_summary(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Generate summary (runs in parallel)."""
        state.results['summary'] = f"Processed {len(state.processed_tasks)} tasks"
        return state

    async def calculate_metrics(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Calculate metrics (runs in parallel)."""
        state.results['metrics'] = {
            'total_tasks': len(state.processed_tasks),
            'iterations': state.iteration_count
        }
        return state

    async def log_completion(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
        """Log completion (runs in parallel)."""
        print(f"✅ Workflow completed: {state.results}")
        return state

    # ========================================
    # Tools with Retry/Fallback
    # ========================================

    @tool(
        description="Process a task with automatic retry",
        retry=3,
        timeout=5.0,
        fallback=lambda self, *args, error=None, **kwargs: "fallback_result"
    )
    async def process_task_tool(self, task: str) -> str:
        """
        Process a task with retry and fallback.

        This tool demonstrates:
        - Automatic retry (3 attempts)
        - Timeout (5 seconds)
        - Fallback function if all retries fail
        """
        # Simulate processing
        import asyncio
        await asyncio.sleep(0.1)

        # Simulate occasional failures for retry demonstration
        import random
        if random.random() > 0.7:
            raise Exception("Simulated failure for retry")

        return f"Processed: {task}"


# ========================================
# Usage Examples
# ========================================

async def example_basic_workflow():
    """Example 1: Basic workflow with loops and parallel execution."""
    agent = AdvancedWorkflowAgent()

    result = await agent.invoke({
        'query': 'Process my batch tasks',
        'remaining_tasks': []
    })

    print("Basic workflow result:", result)


async def example_human_in_loop():
    """Example 2: Human-in-the-loop approval workflow."""
    agent = AdvancedWorkflowAgent()

    # Override build_graph for this example
    agent.build_graph = agent.build_graph_with_interrupt

    # Initial invocation - will pause at interrupt
    config = {"configurable": {"thread_id": "workflow-123"}}

    try:
        result = await agent.invoke(
            {'query': 'DELETE user data'},
            config=config
        )
    except Exception as e:
        # Workflow interrupted - waiting for approval
        print(f"⏸️  Workflow paused for approval: {e}")

    # Check state at interrupt point
    current_state = agent.get_state(config)
    print(f"State at interrupt: {current_state}")

    # Human reviews and approves
    agent.update_state(config, {'approved': True})

    # Resume execution
    result = await agent.resume(config)
    print("Approved workflow result:", result)


async def example_retry_pattern():
    """Example 3: Retry pattern with until() loop."""
    agent = AdvancedWorkflowAgent()

    # Override build_graph
    agent.build_graph = agent.build_graph_with_retry_loop

    result = await agent.invoke({
        'query': 'Call external API',
        'max_retries': 5
    })

    print("Retry pattern result:", result)


async def example_dynamic_graph():
    """Example 4: Dynamic graph modification."""
    agent = AdvancedWorkflowAgent()

    # Override build_graph
    agent.build_graph = agent.build_graph_with_dynamic_nodes

    result = await agent.invoke({'query': 'Dynamic workflow'})

    print("Dynamic graph result:", result)


async def example_subgraph_composition():
    """Example 5: Using subgraphs for reusable components."""

    class MainAgent(Agent):
        state_model = AdvancedWorkflowState

        def build_graph(self):
            # Embed authentication subgraph
            return (
                self.start()
                .subgraph(AuthenticationFlow.build(), name='auth')
                .then(self.process_authenticated_request)
                .end()
            )

        async def process_authenticated_request(self, state):
            state.results['authenticated_request'] = True
            return state

    agent = MainAgent()
    result = await agent.invoke({
        'query': 'Secure request',
        'token': 'valid_token_123'
    })

    print("Subgraph composition result:", result)


# ========================================
# Tool Execution Examples
# ========================================

async def example_tool_with_retry():
    """Example 6: Tool execution with automatic retry."""
    agent = AdvancedWorkflowAgent()

    # Tools decorated with @tool(retry=3) automatically retry on failure
    try:
        result = await agent.execute_tool('process_task_tool', 'important_task')
        print(f"Tool result: {result}")
    except Exception as e:
        print(f"Tool failed after retries: {e}")


# Run examples
if __name__ == "__main__":
    import asyncio

    print("=" * 60)
    print("🚀 Langvel Advanced Workflow Examples")
    print("=" * 60)

    # Run all examples
    asyncio.run(example_basic_workflow())
    print("\n" + "=" * 60 + "\n")

    asyncio.run(example_retry_pattern())
    print("\n" + "=" * 60 + "\n")

    # Uncomment to run other examples
    # asyncio.run(example_human_in_loop())
    # asyncio.run(example_dynamic_graph())
    # asyncio.run(example_subgraph_composition())
    # asyncio.run(example_tool_with_retry())
