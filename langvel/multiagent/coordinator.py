"""Multi-agent coordinator - Supervisor and orchestration patterns."""

from typing import Any, Dict, List, Optional, Type
from langvel.core.agent import Agent
from langvel.multiagent.communication import MessageBus, AgentMessage, get_message_bus
import asyncio


class AgentCoordinator:
    """
    Coordinates multiple agents working together.

    Provides patterns for:
    - Sequential workflows
    - Parallel execution
    - Conditional routing between agents
    - Shared state management
    """

    def __init__(self, message_bus: Optional[MessageBus] = None):
        """
        Initialize coordinator.

        Args:
            message_bus: Optional message bus (creates one if not provided)
        """
        self.message_bus = message_bus or get_message_bus()
        self._agents: Dict[str, Agent] = {}
        self._agent_results: Dict[str, Any] = {}

    async def register_agent(
        self,
        agent_id: str,
        agent: Agent
    ):
        """
        Register an agent with the coordinator.

        Args:
            agent_id: Unique agent identifier
            agent: Agent instance
        """
        self._agents[agent_id] = agent

        # Register message handler
        async def handle_message(message: AgentMessage):
            # Execute agent with message content as input
            result = await agent.invoke(message.content)
            self._agent_results[agent_id] = result

            # Send result back if reply_to is specified
            if message.reply_to:
                await self.message_bus.send(
                    sender_id=agent_id,
                    recipient_id=message.reply_to,
                    content=result,
                    message_type="result",
                    correlation_id=message.correlation_id
                )

        self.message_bus.register_agent(agent_id, handle_message)

    async def execute_sequential(
        self,
        agent_ids: List[str],
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agents sequentially, passing output to next agent.

        Args:
            agent_ids: List of agent IDs in execution order
            initial_input: Initial input for first agent

        Returns:
            Final agent output
        """
        current_input = initial_input

        for agent_id in agent_ids:
            if agent_id not in self._agents:
                raise ValueError(f"Agent '{agent_id}' not registered")

            agent = self._agents[agent_id]
            result = await agent.invoke(current_input)
            current_input = result

        return current_input

    async def execute_parallel(
        self,
        agent_ids: List[str],
        inputs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute agents in parallel.

        Args:
            agent_ids: List of agent IDs to execute
            inputs: Dict mapping agent_id to input data

        Returns:
            Dict mapping agent_id to output data
        """
        tasks = []

        for agent_id in agent_ids:
            if agent_id not in self._agents:
                raise ValueError(f"Agent '{agent_id}' not registered")

            agent = self._agents[agent_id]
            input_data = inputs.get(agent_id, {})
            tasks.append(agent.invoke(input_data))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            agent_id: result
            for agent_id, result in zip(agent_ids, results)
        }

    async def execute_conditional(
        self,
        routing_func: callable,
        initial_input: Dict[str, Any],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute agents based on conditional routing.

        Args:
            routing_func: Function that returns next agent_id based on current state
            initial_input: Initial input
            max_iterations: Maximum number of agent transitions

        Returns:
            Final state
        """
        current_state = initial_input
        iterations = 0

        while iterations < max_iterations:
            next_agent_id = routing_func(current_state)

            if next_agent_id is None or next_agent_id == "END":
                break

            if next_agent_id not in self._agents:
                raise ValueError(f"Agent '{next_agent_id}' not registered")

            agent = self._agents[next_agent_id]
            current_state = await agent.invoke(current_state)

            iterations += 1

        return current_state

    def get_agent_results(self) -> Dict[str, Any]:
        """Get results from all agents."""
        return self._agent_results.copy()

    def clear_results(self):
        """Clear agent results."""
        self._agent_results.clear()


class SupervisorAgent(Agent):
    """
    Supervisor agent that coordinates worker agents.

    The supervisor decides which worker should handle each task and
    aggregates results.
    """

    def __init__(self, workers: Optional[List[Type[Agent]]] = None):
        """
        Initialize supervisor agent.

        Args:
            workers: List of worker agent classes
        """
        super().__init__()
        self.coordinator = AgentCoordinator()
        self._worker_instances: Dict[str, Agent] = {}

        if workers:
            self.register_workers(workers)

    def register_workers(self, workers: List[Type[Agent]]):
        """
        Register worker agents.

        Args:
            workers: List of worker agent classes
        """
        for worker_class in workers:
            worker_id = worker_class.__name__
            worker = worker_class()
            self._worker_instances[worker_id] = worker
            asyncio.create_task(
                self.coordinator.register_agent(worker_id, worker)
            )

    def build_graph(self):
        """Build supervisor graph."""
        return (
            self.start()
            .then(self.route_to_workers)
            .then(self.aggregate_results)
            .end()
        )

    async def route_to_workers(self, state):
        """
        Route task to appropriate workers.

        Override this method to implement custom routing logic.
        """
        # Default: send to all workers in parallel
        worker_ids = list(self._worker_instances.keys())

        if not worker_ids:
            return state

        # Prepare inputs for each worker
        inputs = {worker_id: state for worker_id in worker_ids}

        # Execute in parallel
        results = await self.coordinator.execute_parallel(worker_ids, inputs)

        # Store results in state
        state['worker_results'] = results

        return state

    async def aggregate_results(self, state):
        """
        Aggregate results from workers.

        Override this method to implement custom aggregation logic.
        """
        # Default: combine all worker outputs
        worker_results = state.get('worker_results', {})

        aggregated = {
            'supervisor': self.__class__.__name__,
            'workers': list(worker_results.keys()),
            'results': worker_results
        }

        state['aggregated_results'] = aggregated

        return state

    async def invoke_with_routing(
        self,
        input_data: Dict[str, Any],
        routing_func: callable
    ) -> Dict[str, Any]:
        """
        Invoke with custom routing logic.

        Args:
            input_data: Input data
            routing_func: Function to determine next worker

        Returns:
            Final result
        """
        return await self.coordinator.execute_conditional(
            routing_func,
            input_data
        )


# Example usage pattern:
"""
# Define worker agents
class ResearchAgent(Agent):
    def build_graph(self):
        return self.start().then(self.research).end()

    async def research(self, state):
        state['research_data'] = 'Research results...'
        return state


class AnalysisAgent(Agent):
    def build_graph(self):
        return self.start().then(self.analyze).end()

    async def analyze(self, state):
        state['analysis'] = 'Analysis results...'
        return state


# Create supervisor
supervisor = SupervisorAgent(workers=[ResearchAgent, AnalysisAgent])

# Execute
result = await supervisor.invoke({'task': 'Complex research task'})
"""
