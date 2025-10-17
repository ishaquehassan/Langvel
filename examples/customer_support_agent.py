"""
Example: Customer Support Agent

This example demonstrates the full power of Langvel:
- State management
- RAG integration
- MCP tools
- Custom tools
- Middleware
- Conditional routing
"""

from typing import Optional
from pydantic import Field
from langvel.core.agent import Agent
from langvel.state.base import RAGState, AuthenticatedState
from langvel.tools.decorators import tool, mcp_tool, rag_tool, llm_tool
from langvel.auth.decorators import requires_auth, rate_limit
from langchain_anthropic import ChatAnthropic


class CustomerSupportState(RAGState, AuthenticatedState):
    """State model for customer support agent."""

    ticket_id: Optional[str] = None
    category: Optional[str] = None  # 'technical', 'billing', 'general'
    sentiment: Optional[float] = None
    resolution: Optional[str] = None

    class Config:
        checkpointer = "memory"
        interrupts = ['before_response']  # Allow human review before sending


class CustomerSupportAgent(Agent):
    """
    Customer support agent with full Langvel features.

    Features demonstrated:
    - Automatic classification of requests
    - RAG-powered knowledge base search
    - Sentiment analysis
    - MCP integration for Slack notifications
    - Conditional routing based on category
    - Human-in-the-loop for sensitive issues
    """

    state_model = CustomerSupportState
    middleware = ['logging', 'auth', 'rate_limit']

    def __init__(self):
        super().__init__()
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)

    def build_graph(self):
        """Define the customer support workflow."""
        return (
            self.start()
            .then(self.classify_request)
            .then(self.analyze_sentiment)
            .then(self.search_knowledge)
            .branch(
                {
                    'technical': self.handle_technical,
                    'billing': self.handle_billing,
                    'general': self.handle_general
                },
                condition_func=lambda state: state.get('category', 'general')
            )
            .then(self.generate_response)
            .then(self.notify_slack)
            .end()
        )

    async def classify_request(self, state: CustomerSupportState) -> CustomerSupportState:
        """
        Classify the customer request into categories.

        Args:
            state: Current state with user query

        Returns:
            State with category set
        """
        query = state.query.lower()

        # Simple classification logic (could use LLM)
        if any(word in query for word in ['bug', 'error', 'not working', 'technical']):
            state.category = 'technical'
        elif any(word in query for word in ['bill', 'charge', 'payment', 'refund']):
            state.category = 'billing'
        else:
            state.category = 'general'

        state.add_message("system", f"Classified as: {state.category}")
        return state

    @tool(description="Analyze sentiment of customer message")
    async def analyze_sentiment(self, state: CustomerSupportState) -> CustomerSupportState:
        """
        Analyze sentiment of the customer's message.

        Args:
            state: Current state

        Returns:
            State with sentiment score
        """
        # Simple sentiment analysis (in production, use proper sentiment model)
        query = state.query.lower()
        negative_words = ['angry', 'frustrated', 'terrible', 'awful', 'hate']
        positive_words = ['thanks', 'great', 'love', 'appreciate', 'excellent']

        score = 0.5  # neutral
        for word in negative_words:
            if word in query:
                score -= 0.1
        for word in positive_words:
            if word in query:
                score += 0.1

        state.sentiment = max(0.0, min(1.0, score))
        state.add_message("system", f"Sentiment: {state.sentiment:.2f}")
        return state

    @rag_tool(collection='knowledge_base', k=5)
    async def search_knowledge(self, state: CustomerSupportState) -> CustomerSupportState:
        """
        Search knowledge base for relevant information.

        Args:
            state: Current state

        Returns:
            State with retrieved documents
        """
        # RAG retrieval happens automatically via decorator
        # In production, this would query your vector store
        try:
            results = await self.rag_manager.retrieve(
                collection='knowledge_base',
                query=state.query,
                k=5
            )
            state.retrieved_docs = results
            state.format_context()
        except Exception as e:
            state.add_message("system", f"Knowledge search failed: {str(e)}")
            state.rag_context = ""

        return state

    @rate_limit(max_requests=5, window=60)
    async def handle_technical(self, state: CustomerSupportState) -> CustomerSupportState:
        """Handle technical support requests."""
        state.add_message("system", "Routing to technical support...")

        # Technical-specific logic
        state.resolution = "technical"
        return state

    @requires_auth
    async def handle_billing(self, state: CustomerSupportState) -> CustomerSupportState:
        """Handle billing requests (requires authentication)."""
        state.add_message("system", "Routing to billing department...")

        # Billing-specific logic
        state.resolution = "billing"
        return state

    async def handle_general(self, state: CustomerSupportState) -> CustomerSupportState:
        """Handle general inquiries."""
        state.add_message("system", "Handling general inquiry...")

        state.resolution = "general"
        return state

    @llm_tool(system_prompt="You are a helpful customer support assistant.")
    async def generate_response(self, state: CustomerSupportState) -> CustomerSupportState:
        """
        Generate final response using LLM.

        Args:
            state: Current state with context

        Returns:
            State with response
        """
        # Build prompt with context
        prompt = f"""
        Customer Query: {state.query}
        Category: {state.category}
        Sentiment: {state.sentiment}

        Knowledge Base Context:
        {state.rag_context}

        Generate a helpful, empathetic response to the customer.
        """

        try:
            response = await self.llm.ainvoke(prompt)
            state.response = response.content
            state.add_message("assistant", response.content)
        except Exception as e:
            state.response = "I apologize, but I'm having trouble generating a response. Please try again."
            state.add_message("system", f"Error: {str(e)}")

        return state

    @mcp_tool(server='slack', tool_name='send_message')
    async def notify_slack(self, state: CustomerSupportState) -> CustomerSupportState:
        """
        Send notification to Slack (if configured).

        Args:
            state: Current state

        Returns:
            State unchanged
        """
        # If MCP server is configured, this would send to Slack
        # For now, just log
        state.add_message(
            "system",
            f"Notification sent: Ticket {state.ticket_id} - {state.category}"
        )
        return state


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create agent
        agent = CustomerSupportAgent()

        # Test input
        test_input = {
            "query": "I'm having trouble with my payment, it keeps getting declined",
            "user_id": "user_123",
            "auth_state": "authenticated",
            "permissions": ["customer"]
        }

        # Run agent
        result = await agent.invoke(test_input)

        print("\n=== Result ===")
        print(f"Category: {result.get('category')}")
        print(f"Sentiment: {result.get('sentiment')}")
        print(f"Response: {result.get('response')}")

    asyncio.run(main())
