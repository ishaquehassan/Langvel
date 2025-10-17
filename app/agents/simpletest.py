"""
SimpleTest Agent - Demonstrates Langvel CLI and structured logging
"""

from langvel.core.agent import Agent
from langvel.state.base import StateModel
from langvel.tools.decorators import tool


class SimpleTest(Agent):
    """
    Simple test agent for demonstrating Langvel features.

    No external dependencies required - perfect for testing!
    """

    state_model = StateModel
    middleware = ['logging']  # Use structured logging middleware

    def build_graph(self):
        """Define the agent workflow."""
        return (
            self.start()
            .then(self.greet)
            .then(self.process)
            .then(self.farewell)
            .end()
        )

    async def greet(self, state: StateModel) -> StateModel:
        """
        Greet the user.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        query = state.messages[-1] if state.messages else "Hello"
        state.add_message("system", f"Processing query: {query}")
        return state

    async def process(self, state: StateModel) -> StateModel:
        """
        Main processing logic.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        # Simple processing - no LLM required
        query = str(state.messages[-2] if len(state.messages) > 1 else "")

        response = f"✅ SimpleTest Agent Response:\n" \
                   f"   • Query received: {query}\n" \
                   f"   • Status: Processed successfully\n" \
                   f"   • Structured logging: ACTIVE\n" \
                   f"   • Middleware: {', '.join(self.middleware)}\n" \
                   f"   • Framework: Langvel v0.2.0"

        state.add_message("assistant", response)
        return state

    async def farewell(self, state: StateModel) -> StateModel:
        """
        Add farewell message.

        Args:
            state: Current state

        Returns:
            Updated state
        """
        state.add_message("system", "Processing complete!")
        return state

    @tool(description="Example tool that uppercases text")
    async def uppercase_tool(self, input_text: str) -> str:
        """
        Example tool - converts text to uppercase.

        Args:
            input_text: Input text to process

        Returns:
            Uppercased text
        """
        return f"PROCESSED: {input_text.upper()}"
