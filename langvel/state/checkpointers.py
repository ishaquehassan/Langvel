"""Custom checkpointers for state persistence."""

from typing import Any, Dict, Optional, Tuple, Sequence
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
import json
import asyncpg
import asyncio
from datetime import datetime


class PostgresCheckpointer(BaseCheckpointSaver):
    """
    PostgreSQL-based checkpointer.

    Stores agent state in PostgreSQL for persistence and recovery.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize PostgreSQL checkpointer.

        Args:
            connection_string: PostgreSQL connection string
        """
        super().__init__()
        self.connection_string = connection_string or self._get_connection_string()
        self._pool = None
        self._setup_complete = False

    def _get_connection_string(self) -> str:
        """Get connection string from config."""
        from config.langvel import config
        return config.DATABASE_URL

    async def _ensure_setup(self):
        """Ensure database and connection pool are set up."""
        if not self._setup_complete:
            await self._setup_database()
            self._setup_complete = True

    async def _get_pool(self):
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=10
            )
        return self._pool

    async def _setup_database(self):
        """Set up database tables."""
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Create checkpoints table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS langvel_checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_data JSONB NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            ''')

            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                ON langvel_checkpoints(thread_id, created_at DESC)
            ''')

            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_checkpoints_parent
                ON langvel_checkpoints(parent_checkpoint_id)
            ''')

    async def aget(
        self,
        config: Dict[str, Any]
    ) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint asynchronously.

        Args:
            config: Configuration containing thread_id and checkpoint_id

        Returns:
            Checkpoint or None
        """
        await self._ensure_setup()
        pool = await self._get_pool()

        thread_id = config.get('configurable', {}).get('thread_id')
        checkpoint_id = config.get('configurable', {}).get('checkpoint_id')

        if not thread_id:
            return None

        async with pool.acquire() as conn:
            if checkpoint_id:
                # Get specific checkpoint
                row = await conn.fetchrow('''
                    SELECT checkpoint_data, metadata, parent_checkpoint_id
                    FROM langvel_checkpoints
                    WHERE thread_id = $1 AND checkpoint_id = $2
                ''', thread_id, checkpoint_id)
            else:
                # Get latest checkpoint for thread
                row = await conn.fetchrow('''
                    SELECT checkpoint_data, metadata, parent_checkpoint_id
                    FROM langvel_checkpoints
                    WHERE thread_id = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', thread_id)

            if row:
                return Checkpoint(
                    v=1,
                    id=checkpoint_id or row['checkpoint_data'].get('id'),
                    ts=row['checkpoint_data'].get('ts'),
                    channel_values=row['checkpoint_data'].get('channel_values', {}),
                    channel_versions=row['checkpoint_data'].get('channel_versions', {}),
                    versions_seen=row['checkpoint_data'].get('versions_seen', {})
                )

            return None

    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint synchronously.

        Args:
            config: Configuration containing thread_id and checkpoint_id

        Returns:
            Checkpoint or None
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aget(config))

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None
    ) -> Dict[str, Any]:
        """
        Store a checkpoint asynchronously.

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data
            metadata: Optional metadata

        Returns:
            Updated config
        """
        await self._ensure_setup()
        pool = await self._get_pool()

        thread_id = config.get('configurable', {}).get('thread_id')
        if not thread_id:
            raise ValueError("thread_id is required in config")

        checkpoint_id = checkpoint.id
        parent_checkpoint_id = config.get('configurable', {}).get('checkpoint_id')

        checkpoint_data = {
            'id': checkpoint.id,
            'ts': checkpoint.ts,
            'channel_values': checkpoint.channel_values,
            'channel_versions': checkpoint.channel_versions,
            'versions_seen': checkpoint.versions_seen
        }

        async with pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO langvel_checkpoints
                (thread_id, checkpoint_id, parent_checkpoint_id, checkpoint_data, metadata)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (thread_id, checkpoint_id)
                DO UPDATE SET
                    checkpoint_data = EXCLUDED.checkpoint_data,
                    metadata = EXCLUDED.metadata
            ''',
                thread_id,
                checkpoint_id,
                parent_checkpoint_id,
                json.dumps(checkpoint_data),
                json.dumps(metadata or {})
            )

        # Return updated config with checkpoint_id
        new_config = config.copy()
        if 'configurable' not in new_config:
            new_config['configurable'] = {}
        new_config['configurable']['checkpoint_id'] = checkpoint_id

        return new_config

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None
    ) -> Dict[str, Any]:
        """
        Store a checkpoint synchronously.

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data
            metadata: Optional metadata

        Returns:
            Updated config
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aput(config, checkpoint, metadata))

    async def alist(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[Dict[str, Any]] = None
    ) -> Sequence[Checkpoint]:
        """
        List checkpoints asynchronously.

        Args:
            config: Configuration containing thread_id
            limit: Maximum number of checkpoints to return
            before: List checkpoints before this config

        Returns:
            List of checkpoints
        """
        await self._ensure_setup()
        pool = await self._get_pool()

        thread_id = config.get('configurable', {}).get('thread_id')
        if not thread_id:
            return []

        query = '''
            SELECT checkpoint_data
            FROM langvel_checkpoints
            WHERE thread_id = $1
        '''
        params = [thread_id]

        if before:
            before_checkpoint_id = before.get('configurable', {}).get('checkpoint_id')
            if before_checkpoint_id:
                query += ' AND created_at < (SELECT created_at FROM langvel_checkpoints WHERE checkpoint_id = $2)'
                params.append(before_checkpoint_id)

        query += ' ORDER BY created_at DESC'

        if limit:
            query += f' LIMIT {limit}'

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [
                Checkpoint(
                    v=1,
                    id=row['checkpoint_data'].get('id'),
                    ts=row['checkpoint_data'].get('ts'),
                    channel_values=row['checkpoint_data'].get('channel_values', {}),
                    channel_versions=row['checkpoint_data'].get('channel_versions', {}),
                    versions_seen=row['checkpoint_data'].get('versions_seen', {})
                )
                for row in rows
            ]

    def list(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[Dict[str, Any]] = None
    ) -> Sequence[Checkpoint]:
        """List checkpoints synchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.alist(config, limit, before))

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


class RedisCheckpointer(BaseCheckpointSaver):
    """
    Redis-based checkpointer.

    Stores agent state in Redis for fast access and persistence.
    """

    def __init__(self, redis_url: Optional[str] = None, ttl: int = 86400):
        """
        Initialize Redis checkpointer.

        Args:
            redis_url: Redis connection URL
            ttl: Time to live for checkpoints in seconds (default: 24 hours)
        """
        super().__init__()
        self.redis_url = redis_url or self._get_redis_url()
        self.ttl = ttl
        self._client = None
        self._setup_complete = False

    def _get_redis_url(self) -> str:
        """Get Redis URL from config."""
        from config.langvel import config
        return config.REDIS_URL

    async def _ensure_setup(self):
        """Ensure Redis connection is set up."""
        if not self._setup_complete:
            await self._setup_redis()
            self._setup_complete = True

    async def _setup_redis(self):
        """Set up Redis connection."""
        import redis.asyncio as redis
        self._client = await redis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    def _get_checkpoint_key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint."""
        return f"langvel:checkpoint:{thread_id}:{checkpoint_id}"

    def _get_thread_key(self, thread_id: str) -> str:
        """Generate Redis key for thread checkpoint list."""
        return f"langvel:thread:{thread_id}:checkpoints"

    async def aget(
        self,
        config: Dict[str, Any]
    ) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint asynchronously.

        Args:
            config: Configuration containing thread_id and checkpoint_id

        Returns:
            Checkpoint or None
        """
        await self._ensure_setup()

        thread_id = config.get('configurable', {}).get('thread_id')
        checkpoint_id = config.get('configurable', {}).get('checkpoint_id')

        if not thread_id:
            return None

        if not checkpoint_id:
            # Get latest checkpoint ID for thread
            checkpoint_id = await self._client.lindex(self._get_thread_key(thread_id), 0)
            if not checkpoint_id:
                return None

        # Get checkpoint data
        key = self._get_checkpoint_key(thread_id, checkpoint_id)
        data = await self._client.get(key)

        if data:
            checkpoint_data = json.loads(data)
            return Checkpoint(
                v=1,
                id=checkpoint_data.get('id'),
                ts=checkpoint_data.get('ts'),
                channel_values=checkpoint_data.get('channel_values', {}),
                channel_versions=checkpoint_data.get('channel_versions', {}),
                versions_seen=checkpoint_data.get('versions_seen', {})
            )

        return None

    def get(self, config: Dict[str, Any]) -> Optional[Checkpoint]:
        """
        Retrieve a checkpoint synchronously.

        Args:
            config: Configuration containing thread_id and checkpoint_id

        Returns:
            Checkpoint or None
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aget(config))

    async def aput(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None
    ) -> Dict[str, Any]:
        """
        Store a checkpoint asynchronously.

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data
            metadata: Optional metadata

        Returns:
            Updated config
        """
        await self._ensure_setup()

        thread_id = config.get('configurable', {}).get('thread_id')
        if not thread_id:
            raise ValueError("thread_id is required in config")

        checkpoint_id = checkpoint.id

        checkpoint_data = {
            'id': checkpoint.id,
            'ts': checkpoint.ts,
            'channel_values': checkpoint.channel_values,
            'channel_versions': checkpoint.channel_versions,
            'versions_seen': checkpoint.versions_seen,
            'metadata': metadata or {}
        }

        # Store checkpoint data
        key = self._get_checkpoint_key(thread_id, checkpoint_id)
        await self._client.set(
            key,
            json.dumps(checkpoint_data, default=str),
            ex=self.ttl
        )

        # Add to thread's checkpoint list (latest first)
        thread_key = self._get_thread_key(thread_id)
        await self._client.lpush(thread_key, checkpoint_id)
        await self._client.expire(thread_key, self.ttl)

        # Trim old checkpoints (keep last 100)
        await self._client.ltrim(thread_key, 0, 99)

        # Return updated config with checkpoint_id
        new_config = config.copy()
        if 'configurable' not in new_config:
            new_config['configurable'] = {}
        new_config['configurable']['checkpoint_id'] = checkpoint_id

        return new_config

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Optional[CheckpointMetadata] = None
    ) -> Dict[str, Any]:
        """
        Store a checkpoint synchronously.

        Args:
            config: Configuration containing thread_id
            checkpoint: Checkpoint data
            metadata: Optional metadata

        Returns:
            Updated config
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aput(config, checkpoint, metadata))

    async def alist(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[Dict[str, Any]] = None
    ) -> Sequence[Checkpoint]:
        """
        List checkpoints asynchronously.

        Args:
            config: Configuration containing thread_id
            limit: Maximum number of checkpoints to return
            before: List checkpoints before this config

        Returns:
            List of checkpoints
        """
        await self._ensure_setup()

        thread_id = config.get('configurable', {}).get('thread_id')
        if not thread_id:
            return []

        # Get checkpoint IDs from thread list
        thread_key = self._get_thread_key(thread_id)
        checkpoint_ids = await self._client.lrange(thread_key, 0, (limit or 100) - 1)

        if before:
            before_checkpoint_id = before.get('configurable', {}).get('checkpoint_id')
            if before_checkpoint_id and before_checkpoint_id in checkpoint_ids:
                # Get IDs after the before checkpoint
                before_index = checkpoint_ids.index(before_checkpoint_id)
                checkpoint_ids = checkpoint_ids[before_index + 1:]

        # Get checkpoint data
        checkpoints = []
        for checkpoint_id in checkpoint_ids[:limit] if limit else checkpoint_ids:
            key = self._get_checkpoint_key(thread_id, checkpoint_id)
            data = await self._client.get(key)

            if data:
                checkpoint_data = json.loads(data)
                checkpoints.append(
                    Checkpoint(
                        v=1,
                        id=checkpoint_data.get('id'),
                        ts=checkpoint_data.get('ts'),
                        channel_values=checkpoint_data.get('channel_values', {}),
                        channel_versions=checkpoint_data.get('channel_versions', {}),
                        versions_seen=checkpoint_data.get('versions_seen', {})
                    )
                )

        return checkpoints

    def list(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[Dict[str, Any]] = None
    ) -> Sequence[Checkpoint]:
        """List checkpoints synchronously."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.alist(config, limit, before))

    async def close(self):
        """Close the Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
