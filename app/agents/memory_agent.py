"""
Memory-Enabled Agent - Demonstrates memory systems in Langvel.

This example shows how to build an agent that:
1. Remembers user information across sessions (semantic memory)
2. Maintains conversation history (episodic memory)
3. Uses working memory for current context
4. Automatically extracts and stores facts
5. Recalls relevant information when needed
"""

from typing import Optional
from pydantic import Field
from langvel.core.agent import Agent
from langvel.state.base import StateModel


class MemoryAgentState(StateModel):
    """State model for memory-enabled agent."""

    query: str = Field(description="User query")
    user_id: str = Field(description="User identifier")
    session_id: str = Field(description="Session identifier")
    response: Optional[str] = Field(default=None, description="Agent response")
    context: Optional[str] = Field(default=None, description="Memory context used")

    class Config:
        checkpointer = "memory"


class MemoryAgent(Agent):
    """
    Memory-enabled agent that remembers information across conversations.

    Features:
    - Semantic memory: Long-term facts about users
    - Episodic memory: Conversation history
    - Working memory: Current context
    - Automatic fact extraction
    - Context-aware responses
    """

    state_model = MemoryAgentState
    middleware = ['logging']
    enable_memory = True  # Enable memory systems

    def build_graph(self):
        """Build agent workflow with memory integration."""
        return (
            self.start()
            .then(self.recall_memories)
            .then(self.update_working_memory)
            .then(self.generate_response)
            .then(self.store_interaction)
            .end()
        )

    async def recall_memories(self, state: MemoryAgentState) -> MemoryAgentState:
        """
        Recall relevant memories from all memory systems.

        This node retrieves:
        - Semantic memory: Facts about the user
        - Episodic memory: Recent conversation
        - Working memory: Current context
        """
        if not self.memory_manager:
            state.context = "No memory system available"
            return state

        # Build context from all memory types
        context = await self.memory_manager.build_context(
            user_id=state.user_id,
            session_id=state.session_id,
            query=state.query,
            max_tokens=2000,
            format_style='detailed'
        )

        state.context = context
        state.add_message("system", f"Retrieved memory context ({len(context)} chars)")

        return state

    async def update_working_memory(self, state: MemoryAgentState) -> MemoryAgentState:
        """
        Update working memory with current query context.

        Working memory is used for the current conversation turn.
        """
        if not self.memory_manager:
            return state

        # Add current query to working memory
        self.memory_manager.working.add(
            key=f'query_{state.session_id}',
            value=state.query,
            priority=10  # High priority
        )

        # Add any extracted intent
        if 'help' in state.query.lower():
            self.memory_manager.working.add('intent', 'help_request', priority=5)
        elif any(word in state.query.lower() for word in ['remember', 'recall', 'what did']):
            self.memory_manager.working.add('intent', 'memory_query', priority=5)

        return state

    async def generate_response(self, state: MemoryAgentState) -> MemoryAgentState:
        """
        Generate response using LLM with memory context.

        The response is informed by:
        - What we know about the user (semantic memory)
        - Recent conversation (episodic memory)
        - Current context (working memory)
        """
        # Build prompt with memory context
        system_prompt = """You are a helpful AI assistant with memory.
You remember facts about users and past conversations.
Use the context provided to give personalized, relevant responses."""

        prompt_parts = []

        if state.context:
            prompt_parts.append("Context from memory:")
            prompt_parts.append(state.context)
            prompt_parts.append("")

        prompt_parts.append(f"User: {state.query}")
        prompt_parts.append("Assistant:")

        prompt = "\n".join(prompt_parts)

        # Generate response
        try:
            response = await self.llm.invoke(
                prompt,
                system_prompt=system_prompt
            )
            state.response = response
            state.add_message("assistant", response)

        except Exception as e:
            state.response = f"I apologize, I encountered an error: {str(e)}"
            state.add_message("system", f"Error generating response: {e}")

        return state

    async def store_interaction(self, state: MemoryAgentState) -> MemoryAgentState:
        """
        Store the interaction in memory.

        This node:
        1. Stores the conversation turn in episodic memory
        2. Extracts and stores facts in semantic memory (if detected)
        3. Updates working memory
        """
        if not self.memory_manager:
            return state

        try:
            # Store user message
            await self.memory_manager.remember(
                user_id=state.user_id,
                session_id=state.session_id,
                content=state.query,
                role='user',
                memory_type='auto'  # Automatically route to appropriate memory
            )

            # Store assistant response
            if state.response:
                await self.memory_manager.remember(
                    user_id=state.user_id,
                    session_id=state.session_id,
                    content=state.response,
                    role='assistant',
                    memory_type='episodic'  # Only conversation history
                )

            state.add_message("system", "Interaction stored in memory")

        except Exception as e:
            state.add_message("system", f"Error storing memory: {e}")

        return state


# ========================================
# Usage Examples
# ========================================

async def example_basic_memory():
    """Example 1: Basic memory usage."""
    agent = MemoryAgent()

    print("=" * 60)
    print("Example 1: Basic Memory Usage")
    print("=" * 60)

    # First conversation
    print("\n📝 Session 1: User introduces themselves")
    result1 = await agent.invoke({
        'query': 'Hi! My name is Alice and I work at TechCorp as a software engineer.',
        'user_id': 'alice123',
        'session_id': 'session1'
    })
    print(f"Response: {result1.get('response')}\n")

    # Second conversation in same session
    print("📝 Session 1: User asks about their job")
    result2 = await agent.invoke({
        'query': 'What did I tell you about my job?',
        'user_id': 'alice123',
        'session_id': 'session1'
    })
    print(f"Response: {result2.get('response')}\n")

    # Third conversation in new session (should still remember)
    print("📝 Session 2: User asks in a new session")
    result3 = await agent.invoke({
        'query': 'Do you remember where I work?',
        'user_id': 'alice123',
        'session_id': 'session2'
    })
    print(f"Response: {result3.get('response')}\n")


async def example_memory_context():
    """Example 2: Memory context in responses."""
    agent = MemoryAgent()

    print("=" * 60)
    print("Example 2: Memory Context in Responses")
    print("=" * 60)

    user_id = 'bob456'
    session_id = 'session3'

    # Store some background information
    print("\n📝 Building user profile")
    await agent.invoke({
        'query': 'I am Bob, I live in San Francisco and I love Python programming.',
        'user_id': user_id,
        'session_id': session_id
    })

    await agent.invoke({
        'query': 'I have 5 years of experience in machine learning.',
        'user_id': user_id,
        'session_id': session_id
    })

    # Ask a question that requires memory
    print("\n📝 Asking question that needs memory")
    result = await agent.invoke({
        'query': 'Can you recommend a Python ML library for my skill level?',
        'user_id': user_id,
        'session_id': session_id
    })

    print(f"\n🧠 Context used:\n{result.get('context')}")
    print(f"\n💬 Response: {result.get('response')}\n")


async def example_entity_memory():
    """Example 3: Explicit entity storage."""
    agent = MemoryAgent()

    print("=" * 60)
    print("Example 3: Entity Memory")
    print("=" * 60)

    user_id = 'charlie789'

    # Store entities explicitly
    if agent.memory_manager:
        print("\n📝 Storing entities")

        await agent.memory_manager.store_entity(
            user_id=user_id,
            entity_type='person',
            entity_name='Charlie',
            properties={
                'occupation': 'Data Scientist',
                'company': 'AI Labs',
                'interests': ['NLP', 'Computer Vision']
            }
        )

        await agent.memory_manager.store_entity(
            user_id=user_id,
            entity_type='project',
            entity_name='ChatBot Project',
            properties={
                'status': 'in_progress',
                'technology': 'Langvel',
                'deadline': '2025-12-01'
            }
        )

        # Retrieve entity
        entity = await agent.memory_manager.get_entity(
            user_id=user_id,
            entity_name='Charlie'
        )

        print(f"✅ Stored entity: {entity}")

        # Use memory in conversation
        result = await agent.invoke({
            'query': 'Tell me about my current projects',
            'user_id': user_id,
            'session_id': 'session4'
        })

        print(f"\n💬 Response: {result.get('response')}\n")


async def example_memory_stats():
    """Example 4: Memory statistics."""
    agent = MemoryAgent()

    print("=" * 60)
    print("Example 4: Memory Statistics")
    print("=" * 60)

    user_id = 'dave101'
    session_id = 'session5'

    # Have some conversations
    print("\n📝 Having conversations")
    conversations = [
        "My name is Dave",
        "I work at DataCorp",
        "I'm learning Langvel",
        "It's a great framework!",
        "I'm building a chatbot"
    ]

    for query in conversations:
        await agent.invoke({
            'query': query,
            'user_id': user_id,
            'session_id': session_id
        })

    # Get stats
    if agent.memory_manager:
        stats = await agent.memory_manager.get_stats(user_id, session_id)
        print(f"\n📊 Memory Statistics:")
        print(f"  - Working Memory: {stats.get('working', {})}")
        print(f"  - Semantic Memory: {stats.get('semantic', {})}")
        print(f"  - Episodic Memory: {stats.get('episodic', {})}")


async def example_memory_clear():
    """Example 5: Clearing memory."""
    agent = MemoryAgent()

    print("=" * 60)
    print("Example 5: Clearing Memory")
    print("=" * 60)

    user_id = 'eve202'
    session_id = 'session6'

    # Store some data
    await agent.invoke({
        'query': 'Remember that I like pizza',
        'user_id': user_id,
        'session_id': session_id
    })

    # Clear specific memory types
    if agent.memory_manager:
        print("\n🗑️  Clearing episodic memory only")
        await agent.memory_manager.clear(
            session_id=session_id,
            clear_episodic=True
        )

        print("🗑️  Clearing all memory for user")
        await agent.memory_manager.clear(
            user_id=user_id,
            session_id=session_id,
            clear_semantic=True,
            clear_episodic=True,
            clear_working=True
        )

        print("✅ Memory cleared\n")


# ========================================
# Run Examples
# ========================================

if __name__ == "__main__":
    import asyncio

    print("\n" + "=" * 60)
    print("🧠 Langvel Memory System Examples")
    print("=" * 60 + "\n")

    # Note: These examples require PostgreSQL and Redis to be running
    # For testing without databases, set backend='memory' in Agent._init_memory()

    try:
        # Run basic example
        asyncio.run(example_basic_memory())

        # Uncomment to run other examples
        # asyncio.run(example_memory_context())
        # asyncio.run(example_entity_memory())
        # asyncio.run(example_memory_stats())
        # asyncio.run(example_memory_clear())

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: Memory system requires PostgreSQL and Redis.")
        print("Update backend='memory' in Agent._init_memory() for testing without databases.")
