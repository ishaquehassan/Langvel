"""
Example: Code Review Agent

This example demonstrates LLM integration in Langvel:
- Direct LLM invocation using self.llm
- @llm_tool decorator for LLM-based tools
- Structured output with Pydantic
- Streaming responses
- Multi-turn conversations
"""

from typing import List, Optional
from pydantic import Field, BaseModel
from langvel.core.agent import Agent
from langvel.state.base import StateModel
from langvel.tools.decorators import tool, llm_tool
from langvel.llm.manager import ask_llm


class CodeReviewResult(BaseModel):
    """Structured code review output."""
    issues: List[str]
    suggestions: List[str]
    rating: int = Field(ge=1, le=10, description="Code quality rating")
    summary: str


class CodeReviewState(StateModel):
    """State model for code review agent."""

    code: str
    language: str = "python"
    review: Optional[str] = None
    structured_review: Optional[CodeReviewResult] = None
    severity: Optional[str] = None  # 'low', 'medium', 'high'

    class Config:
        checkpointer = "memory"


class CodeReviewAgent(Agent):
    """
    Code review agent demonstrating LLM usage.

    Shows multiple ways to use LLMs in Langvel:
    1. Direct self.llm usage
    2. @llm_tool decorator
    3. Structured output
    4. Streaming
    """

    state_model = CodeReviewState
    middleware = ['logging']

    def build_graph(self):
        """Define the code review workflow."""
        return (
            self.start()
            .then(self.analyze_code)
            .then(self.generate_review)
            .then(self.assess_severity)
            .end()
        )

    async def analyze_code(self, state: CodeReviewState) -> CodeReviewState:
        """
        Analyze code using direct LLM invocation.

        This shows the most basic LLM usage.
        """
        prompt = f"""
        Analyze this {state.language} code for potential issues:

        ```{state.language}
        {state.code}
        ```

        Provide a brief analysis.
        """

        # Method 1: Direct invocation
        analysis = await self.llm.invoke(
            prompt=prompt,
            system_prompt="You are an expert code reviewer."
        )

        state.add_message("system", f"Analysis: {analysis}")
        return state

    @llm_tool(system_prompt="You are a meticulous code reviewer focused on best practices.")
    async def generate_review(self, state: CodeReviewState) -> CodeReviewState:
        """
        Generate detailed review using @llm_tool decorator.

        The decorator handles the LLM integration automatically.
        """
        prompt = f"""
        Review this {state.language} code in detail:

        ```{state.language}
        {state.code}
        ```

        Consider:
        - Code quality
        - Best practices
        - Potential bugs
        - Performance issues
        - Security concerns

        Provide a comprehensive review.
        """

        review = await self.llm.invoke(prompt)
        state.review = review
        state.add_message("assistant", review)

        return state

    async def assess_severity(self, state: CodeReviewState) -> CodeReviewState:
        """
        Assess severity using structured output.

        This demonstrates LLM with structured Pydantic output.
        """
        # Get structured output
        llm_structured = self.llm.with_structured_output(CodeReviewResult)

        prompt = f"""
        Provide a structured code review for:

        ```{state.language}
        {state.code}
        ```

        Include:
        - List of issues found
        - List of improvement suggestions
        - Rating from 1-10
        - Brief summary
        """

        result = await llm_structured.ainvoke(prompt)
        state.structured_review = result

        # Determine severity based on rating
        if result.rating >= 7:
            state.severity = "low"
        elif result.rating >= 4:
            state.severity = "medium"
        else:
            state.severity = "high"

        return state

    @tool(description="Stream code explanation")
    async def explain_code_streaming(self, code: str, language: str = "python") -> str:
        """
        Example of streaming LLM response.

        Use this when you want to stream the response in real-time.
        """
        prompt = f"Explain this {language} code line by line:\n\n{code}"

        result = ""
        async for chunk in self.llm.stream(prompt):
            result += chunk
            # In real usage, you might yield chunks or send to client

        return result

    @tool(description="Multi-turn code discussion")
    async def discuss_code(
        self,
        code: str,
        conversation_history: List[dict]
    ) -> str:
        """
        Example of multi-turn conversation.

        Useful for back-and-forth discussions about code.
        """
        # Build messages
        messages = conversation_history + [
            {"role": "user", "content": f"Let's discuss this code:\n\n{code}"}
        ]

        # Use chat method for multi-turn
        response = await self.llm.chat(messages)
        return response


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        # Create agent
        agent = CodeReviewAgent()

        # Example code to review
        code = '''
def calculate_total(items):
    total = 0
    for item in items:
        total = total + item['price']
    return total
        '''

        # Test input
        test_input = {
            "code": code,
            "language": "python"
        }

        # Run agent
        print("🔍 Reviewing code...\n")
        result = await agent.invoke(test_input)

        print("\n=== Results ===")
        print(f"Review: {result.get('review', 'N/A')}")
        print(f"Severity: {result.get('severity', 'N/A')}")

        if result.get('structured_review'):
            review = result['structured_review']
            print(f"\nRating: {review.rating}/10")
            print(f"Issues: {len(review.issues)}")
            print(f"Summary: {review.summary}")

        # Example: Direct LLM usage (convenience function)
        print("\n=== Quick LLM Query ===")
        quick_result = await ask_llm(
            "What is the time complexity of bubble sort?",
            system_prompt="You are a computer science expert."
        )
        print(quick_result)

    asyncio.run(main())
