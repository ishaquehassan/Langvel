"""Episodic memory - Short-term conversation history storage."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
from langvel.logging import get_logger

logger = get_logger(__name__)


class EpisodicMemory:
    """
    Short-term episodic memory for conversation history.

    Stores recent interactions with automatic pruning/summarization.
    Uses Redis for fast access with TTL support.

    Example:
        memory = EpisodicMemory(backend='redis', ttl=86400)

        # Add conversation turns
        await memory.add_turn(
            session_id='session123',
            role='user',
            content='Hello, how are you?'
        )

        await memory.add_turn(
            session_id='session123',
            role='assistant',
            content='I am doing well, thank you!'
        )

        # Get recent conversation
        recent = await memory.get_recent('session123', limit=10)

        # Generate summary
        summary = await memory.get_summary('session123', llm)
    """

    def __init__(
        self,
        backend: str = 'redis',
        ttl: int = 86400,  # 24 hours
        max_turns: int = 100,
        connection_string: Optional[str] = None
    ):
        """
        Initialize episodic memory.

        Args:
            backend: Storage backend ('redis', 'memory')
            ttl: Time-to-live in seconds for conversations
            max_turns: Maximum conversation turns to store
            connection_string: Optional Redis connection string
        """
        self.backend = backend
        self.ttl = ttl
        self.max_turns = max_turns
        self.connection_string = connection_string
        self._store = None
        self._initialized = False

        logger.info(
            "Episodic memory initialized",
            extra={'backend': backend, 'ttl': ttl, 'max_turns': max_turns}
        )

    async def initialize(self):
        """Initialize storage backend."""
        if self._initialized:
            return

        if self.backend == 'redis':
            await self._init_redis()
        elif self.backend == 'memory':
            await self._init_memory()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._initialized = True
        logger.info("Episodic memory storage initialized", extra={'backend': self.backend})

    async def _init_redis(self):
        """Initialize Redis backend."""
        try:
            import redis.asyncio as redis
            from config.langvel import config

            redis_url = self.connection_string or config.REDIS_URL

            self._store = await redis.from_url(
                redis_url,
                encoding='utf-8',
                decode_responses=True
            )

            # Test connection
            await self._store.ping()

            logger.info("Redis episodic memory connected", extra={'url': redis_url.split('@')[-1]})

        except Exception as e:
            logger.error(
                "Failed to initialize Redis episodic memory",
                extra={'error': str(e)},
                exc_info=True
            )
            raise

    async def _init_memory(self):
        """Initialize in-memory backend (for testing)."""
        self._store = {}
        logger.info("In-memory episodic storage initialized")

    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a conversation turn.

        Args:
            session_id: Session identifier
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional metadata

        Returns:
            Turn index (position in conversation)
        """
        if not self._initialized:
            await self.initialize()

        turn = {
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'timestamp': datetime.utcnow().isoformat()
        }

        try:
            if self.backend == 'redis':
                key = f"episodes:{session_id}"

                # Add turn to list
                turn_json = json.dumps(turn)
                index = await self._store.rpush(key, turn_json)

                # Set TTL
                await self._store.expire(key, self.ttl)

                # Trim to max_turns
                length = await self._store.llen(key)
                if length > self.max_turns:
                    await self._store.ltrim(key, -self.max_turns, -1)
                    index = self.max_turns

            elif self.backend == 'memory':
                if session_id not in self._store:
                    self._store[session_id] = {
                        'turns': [],
                        'created_at': datetime.utcnow()
                    }

                self._store[session_id]['turns'].append(turn)

                # Trim to max_turns
                if len(self._store[session_id]['turns']) > self.max_turns:
                    self._store[session_id]['turns'] = self._store[session_id]['turns'][-self.max_turns:]

                index = len(self._store[session_id]['turns'])

                # Clean old sessions (simulate TTL)
                await self._clean_old_sessions()

            logger.debug(
                "Conversation turn added",
                extra={
                    'session_id': session_id,
                    'role': role,
                    'turn_index': index,
                    'content_length': len(content)
                }
            )

            return index

        except Exception as e:
            logger.error(
                "Failed to add conversation turn",
                extra={'session_id': session_id, 'role': role, 'error': str(e)},
                exc_info=True
            )
            raise

    async def get_recent(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation turns.

        Args:
            session_id: Session identifier
            limit: Maximum number of turns to return

        Returns:
            List of conversation turns (most recent first)
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'redis':
                key = f"episodes:{session_id}"

                # Get last N turns
                turns_json = await self._store.lrange(key, -limit, -1)

                if not turns_json:
                    return []

                turns = [json.loads(turn_json) for turn_json in turns_json]

                logger.debug(
                    "Retrieved recent conversation",
                    extra={'session_id': session_id, 'count': len(turns)}
                )

                return turns

            elif self.backend == 'memory':
                if session_id not in self._store:
                    return []

                turns = self._store[session_id]['turns'][-limit:]

                return turns

            return []

        except Exception as e:
            logger.error(
                "Failed to get recent conversation",
                extra={'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            return []

    async def get_all(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all conversation turns for a session.

        Args:
            session_id: Session identifier

        Returns:
            Complete conversation history
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'redis':
                key = f"episodes:{session_id}"
                turns_json = await self._store.lrange(key, 0, -1)

                if not turns_json:
                    return []

                return [json.loads(turn_json) for turn_json in turns_json]

            elif self.backend == 'memory':
                if session_id not in self._store:
                    return []

                return self._store[session_id]['turns']

            return []

        except Exception as e:
            logger.error(
                "Failed to get all conversation turns",
                extra={'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            return []

    async def get_summary(
        self,
        session_id: str,
        llm: Any,
        max_turns: Optional[int] = None
    ) -> str:
        """
        Generate conversation summary using LLM.

        Args:
            session_id: Session identifier
            llm: LLM instance for summarization
            max_turns: Optional limit on turns to summarize

        Returns:
            Conversation summary
        """
        if not self._initialized:
            await self.initialize()

        try:
            turns = await self.get_all(session_id)

            if not turns:
                return "No conversation history."

            if max_turns:
                turns = turns[-max_turns:]

            # Format conversation
            conversation = "\n".join([
                f"{turn['role']}: {turn['content']}"
                for turn in turns
            ])

            # Generate summary
            summary = await llm.invoke(
                f"Summarize this conversation concisely in 2-3 sentences:\n\n{conversation}",
                system_prompt="You are a helpful summarizer. Provide brief, accurate summaries."
            )

            logger.info(
                "Conversation summary generated",
                extra={
                    'session_id': session_id,
                    'turns_count': len(turns),
                    'summary_length': len(summary)
                }
            )

            return summary

        except Exception as e:
            logger.error(
                "Failed to generate conversation summary",
                extra={'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            return "Error generating summary."

    async def get_context_window(
        self,
        session_id: str,
        max_tokens: int = 2000
    ) -> List[Dict[str, Any]]:
        """
        Get conversation turns that fit within a token budget.

        Args:
            session_id: Session identifier
            max_tokens: Maximum tokens (rough approximation: 1 token ~= 4 chars)

        Returns:
            List of turns that fit within budget (most recent first)
        """
        if not self._initialized:
            await self.initialize()

        turns = await self.get_all(session_id)

        if not turns:
            return []

        # Estimate tokens (rough: 1 token ~= 4 characters)
        result = []
        total_chars = 0

        for turn in reversed(turns):
            turn_chars = len(turn['content'])

            if total_chars + turn_chars > max_tokens * 4:
                break

            result.insert(0, turn)
            total_chars += turn_chars

        logger.debug(
            "Context window created",
            extra={
                'session_id': session_id,
                'turns': len(result),
                'estimated_tokens': total_chars // 4
            }
        )

        return result

    async def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'redis':
                key = f"episodes:{session_id}"
                await self._store.delete(key)

            elif self.backend == 'memory':
                if session_id in self._store:
                    del self._store[session_id]

            logger.info("Session cleared", extra={'session_id': session_id})

        except Exception as e:
            logger.error(
                "Failed to clear session",
                extra={'session_id': session_id, 'error': str(e)},
                exc_info=True
            )
            raise

    async def list_sessions(self) -> List[str]:
        """
        List all active sessions.

        Returns:
            List of session IDs
        """
        if not self._initialized:
            await self.initialize()

        try:
            if self.backend == 'redis':
                # Scan for episode keys
                sessions = []
                async for key in self._store.scan_iter(match='episodes:*'):
                    session_id = key.replace('episodes:', '')
                    sessions.append(session_id)

                return sessions

            elif self.backend == 'memory':
                return list(self._store.keys())

            return []

        except Exception as e:
            logger.error(
                "Failed to list sessions",
                extra={'error': str(e)},
                exc_info=True
            )
            return []

    async def _clean_old_sessions(self):
        """Clean old sessions in memory backend (simulate TTL)."""
        if self.backend != 'memory':
            return

        cutoff = datetime.utcnow() - timedelta(seconds=self.ttl)

        sessions_to_remove = [
            session_id
            for session_id, data in self._store.items()
            if data['created_at'] < cutoff
        ]

        for session_id in sessions_to_remove:
            del self._store[session_id]

        if sessions_to_remove:
            logger.debug(
                "Old sessions cleaned",
                extra={'count': len(sessions_to_remove)}
            )

    async def close(self):
        """Close connections."""
        if self.backend == 'redis' and self._store:
            await self._store.close()
            logger.info("Episodic memory connection closed")
