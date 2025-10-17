"""Memory manager - Unified interface for all memory types."""

from typing import Any, Dict, List, Optional
import re
from langvel.memory.semantic import SemanticMemory
from langvel.memory.episodic import EpisodicMemory
from langvel.memory.working import WorkingMemory
from langvel.logging import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """
    Unified memory manager combining all memory types.

    Provides simple interface for agents to store/recall information
    across semantic (long-term), episodic (conversation), and working (current) memory.

    Example:
        memory = MemoryManager()

        # Initialize memory systems
        await memory.initialize()

        # Store information (automatically routes to appropriate memory)
        await memory.remember(
            user_id='user123',
            session_id='session456',
            content='I work at Acme Corp as a data scientist'
        )

        # Recall information
        memories = await memory.recall(
            user_id='user123',
            session_id='session456',
            query='Where does the user work?'
        )

        # Build context for LLM
        context = memory.build_context(
            user_id='user123',
            session_id='session456',
            query='Tell me about my job'
        )
    """

    def __init__(
        self,
        semantic: Optional[SemanticMemory] = None,
        episodic: Optional[EpisodicMemory] = None,
        working: Optional[WorkingMemory] = None,
        auto_detect_facts: bool = True
    ):
        """
        Initialize memory manager.

        Args:
            semantic: Semantic memory instance (created if not provided)
            episodic: Episodic memory instance (created if not provided)
            working: Working memory instance (created if not provided)
            auto_detect_facts: Automatically detect and store facts
        """
        self.semantic = semantic or SemanticMemory()
        self.episodic = episodic or EpisodicMemory()
        self.working = working or WorkingMemory()
        self.auto_detect_facts = auto_detect_facts
        self._initialized = False

        logger.info(
            "Memory manager created",
            extra={'auto_detect_facts': auto_detect_facts}
        )

    async def initialize(self):
        """Initialize all memory systems."""
        if self._initialized:
            return

        await self.semantic.initialize()
        await self.episodic.initialize()

        self._initialized = True

        logger.info("Memory manager initialized")

    async def remember(
        self,
        user_id: str,
        session_id: str,
        content: str,
        role: str = 'user',
        memory_type: str = 'auto',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store information in appropriate memory.

        Args:
            user_id: User identifier
            session_id: Session identifier
            content: Content to remember
            role: Message role ('user', 'assistant', 'system')
            memory_type: Memory type ('auto', 'semantic', 'episodic', 'working')
            metadata: Optional metadata
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Always store in episodic (conversation history)
            if memory_type in ('auto', 'episodic'):
                await self.episodic.add_turn(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata
                )

            # Store in semantic if it's a fact
            if memory_type == 'semantic' or (memory_type == 'auto' and self._is_fact(content)):
                # Extract entities and facts
                if self.auto_detect_facts:
                    await self._extract_and_store_facts(user_id, content, metadata)
                else:
                    await self.semantic.store_fact(user_id, content, metadata)

            # Store in working memory for current context
            if memory_type == 'working':
                key = f"{session_id}:{role}"
                self.working.add(key, content)

            logger.debug(
                "Information stored",
                extra={
                    'user_id': user_id,
                    'session_id': session_id,
                    'memory_type': memory_type,
                    'content_length': len(content)
                }
            )

        except Exception as e:
            logger.error(
                "Failed to store information",
                extra={'user_id': user_id, 'session_id': session_id, 'error': str(e)},
                exc_info=True
            )

    async def recall(
        self,
        user_id: str,
        session_id: str,
        query: Optional[str] = None,
        include_semantic: bool = True,
        include_episodic: bool = True,
        include_working: bool = True
    ) -> Dict[str, Any]:
        """
        Recall information from all memory types.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Optional search query
            include_semantic: Include semantic memory
            include_episodic: Include episodic memory
            include_working: Include working memory

        Returns:
            Dictionary with memories from all types
        """
        if not self._initialized:
            await self.initialize()

        memories = {}

        try:
            if include_semantic:
                memories['semantic'] = await self.semantic.recall_facts(
                    user_id, query, limit=5
                )

            if include_episodic:
                memories['episodic'] = await self.episodic.get_recent(
                    session_id, limit=10
                )

            if include_working:
                memories['working'] = self.working.to_dict()

            logger.debug(
                "Memories recalled",
                extra={
                    'user_id': user_id,
                    'session_id': session_id,
                    'semantic_count': len(memories.get('semantic', [])),
                    'episodic_count': len(memories.get('episodic', [])),
                    'working_count': len(memories.get('working', {}))
                }
            )

            return memories

        except Exception as e:
            logger.error(
                "Failed to recall memories",
                extra={'user_id': user_id, 'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            return {'semantic': [], 'episodic': [], 'working': {}}

    async def build_context(
        self,
        user_id: str,
        session_id: str,
        query: Optional[str] = None,
        max_tokens: int = 2000,
        format_style: str = 'detailed'
    ) -> str:
        """
        Build context string for LLM from all memory types.

        Args:
            user_id: User identifier
            session_id: Session identifier
            query: Optional query for semantic search
            max_tokens: Maximum tokens for context
            format_style: Format style ('simple', 'detailed')

        Returns:
            Formatted context string
        """
        if not self._initialized:
            await self.initialize()

        try:
            memories = await self.recall(user_id, session_id, query)

            context_parts = []

            # Semantic memory (facts about user)
            if memories.get('semantic'):
                if format_style == 'detailed':
                    context_parts.append("## What I Know About You ##")
                else:
                    context_parts.append("Facts:")

                for fact in memories['semantic'][:5]:  # Limit to 5 facts
                    content = fact.get('content') or fact.get('fact', '')
                    context_parts.append(f"- {content}")

            # Episodic memory (recent conversation)
            if memories.get('episodic'):
                if format_style == 'detailed':
                    context_parts.append("\n## Recent Conversation ##")
                else:
                    context_parts.append("\nRecent:")

                # Get conversation context window
                conversation_turns = await self.episodic.get_context_window(
                    session_id,
                    max_tokens=max_tokens // 2  # Allocate half tokens to conversation
                )

                for turn in conversation_turns[-10:]:  # Last 10 turns
                    role = turn['role'].capitalize()
                    content = turn['content']
                    if len(content) > 200:
                        content = content[:200] + "..."
                    context_parts.append(f"{role}: {content}")

            # Working memory (current context)
            if memories.get('working') and format_style == 'detailed':
                context_parts.append("\n## Current Context ##")
                working_context = self.working.to_context_string(format_style='simple')
                if working_context:
                    context_parts.append(working_context)

            context = "\n".join(context_parts)

            logger.debug(
                "Context built",
                extra={
                    'user_id': user_id,
                    'session_id': session_id,
                    'context_length': len(context)
                }
            )

            return context

        except Exception as e:
            logger.error(
                "Failed to build context",
                extra={'user_id': user_id, 'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            return ""

    async def store_entity(
        self,
        user_id: str,
        entity_type: str,
        entity_name: str,
        properties: Optional[Dict[str, Any]] = None
    ):
        """
        Store entity in semantic memory.

        Args:
            user_id: User identifier
            entity_type: Entity type
            entity_name: Entity name
            properties: Entity properties
        """
        if not self._initialized:
            await self.initialize()

        await self.semantic.store_entity(user_id, entity_type, entity_name, properties)

    async def get_entity(
        self,
        user_id: str,
        entity_name: str,
        entity_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get entity from semantic memory.

        Args:
            user_id: User identifier
            entity_name: Entity name
            entity_type: Optional entity type

        Returns:
            Entity data or None
        """
        if not self._initialized:
            await self.initialize()

        return await self.semantic.get_entity(user_id, entity_name, entity_type)

    async def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        clear_semantic: bool = False,
        clear_episodic: bool = False,
        clear_working: bool = False
    ):
        """
        Clear memory.

        Args:
            user_id: User identifier (required for semantic)
            session_id: Session identifier (required for episodic)
            clear_semantic: Clear semantic memory
            clear_episodic: Clear episodic memory
            clear_working: Clear working memory
        """
        if not self._initialized:
            await self.initialize()

        try:
            if clear_semantic and user_id:
                await self.semantic.clear_user_memory(user_id)
                logger.info("Semantic memory cleared", extra={'user_id': user_id})

            if clear_episodic and session_id:
                await self.episodic.clear_session(session_id)
                logger.info("Episodic memory cleared", extra={'session_id': session_id})

            if clear_working:
                self.working.clear()
                logger.info("Working memory cleared")

        except Exception as e:
            logger.error(
                "Failed to clear memory",
                extra={'error': str(e)},
                exc_info=True
            )

    def _is_fact(self, content: str) -> bool:
        """
        Heuristic to determine if content is a fact.

        Args:
            content: Content to check

        Returns:
            True if content appears to be a fact
        """
        content_lower = content.lower()

        # Fact indicators
        fact_patterns = [
            r'\b(my name is|i am|i\'m)\b',
            r'\b(i work at|i work for|i\'m employed by)\b',
            r'\b(i live in|i\'m from|i reside in)\b',
            r'\b(my (job|role|position|title) is)\b',
            r'\b(i have|i own|i possess)\b',
            r'\b(i like|i love|i enjoy|i prefer)\b',
            r'\b(my (email|phone|address) is)\b',
        ]

        for pattern in fact_patterns:
            if re.search(pattern, content_lower):
                return True

        return False

    async def _extract_and_store_facts(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Extract entities and facts from content and store in semantic memory.

        Args:
            user_id: User identifier
            content: Content to extract from
            metadata: Optional metadata
        """
        # Simple extraction patterns (could be enhanced with NER/LLM)
        content_lower = content.lower()

        # Extract name
        name_match = re.search(r'my name is ([a-z\s]+)', content_lower)
        if name_match:
            name = name_match.group(1).strip().title()
            await self.semantic.store_entity(
                user_id,
                entity_type='person',
                entity_name=name,
                properties={'type': 'self'}
            )

        # Extract company
        company_match = re.search(r'i work (?:at|for) ([a-z\s&.]+?)(?:\s+as|\.|$)', content_lower)
        if company_match:
            company = company_match.group(1).strip().title()
            await self.semantic.store_entity(
                user_id,
                entity_type='company',
                entity_name=company,
                properties={'relationship': 'employer'}
            )

            # Extract role if present
            role_match = re.search(r'as (?:a |an )?([a-z\s]+)', content_lower)
            if role_match:
                role = role_match.group(1).strip()
                await self.semantic.store_relationship(
                    user_id,
                    subject=user_id,
                    relationship='works_as',
                    object_entity=role,
                    properties={'company': company}
                )

        # Extract location
        location_match = re.search(r'i live in ([a-z\s,]+)', content_lower)
        if location_match:
            location = location_match.group(1).strip().title()
            await self.semantic.store_entity(
                user_id,
                entity_type='location',
                entity_name=location,
                properties={'type': 'residence'}
            )

        # Store full content as fact
        await self.semantic.store_fact(user_id, content, metadata)

    async def get_stats(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get memory statistics.

        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            Statistics dictionary
        """
        stats = {
            'working': self.working.get_stats()
        }

        if user_id and self._initialized:
            # Count semantic memory items
            facts = await self.semantic.recall_facts(user_id, limit=1000)
            stats['semantic'] = {
                'fact_count': len(facts)
            }

        if session_id and self._initialized:
            # Count episodic memory items
            turns = await self.episodic.get_all(session_id)
            stats['episodic'] = {
                'turn_count': len(turns)
            }

        return stats

    async def close(self):
        """Close all memory connections."""
        await self.semantic.close()
        await self.episodic.close()
        logger.info("Memory manager closed")
