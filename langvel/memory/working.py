"""Working memory - Short-term current context storage."""

from typing import Any, Dict, Optional
from collections import OrderedDict
from langvel.logging import get_logger

logger = get_logger(__name__)


class WorkingMemory:
    """
    Working memory for current conversation context.

    Automatically managed, cleared after task completion.
    Similar to short-term memory or "RAM" for the agent.

    Example:
        memory = WorkingMemory(max_tokens=4000)

        # Add to working memory
        memory.add('user_intent', 'book_flight')
        memory.add('destination', 'New York')
        memory.add('departure_date', '2025-11-01')

        # Get from working memory
        intent = memory.get('user_intent')

        # Convert to context string for LLM
        context = memory.to_context_string()
        # Output: "user_intent: book_flight\\ndestination: New York\\n..."

        # Clear when task is done
        memory.clear()
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        auto_prune: bool = True
    ):
        """
        Initialize working memory.

        Args:
            max_tokens: Maximum tokens to store (rough estimate)
            auto_prune: Automatically prune when exceeding max_tokens
        """
        self.max_tokens = max_tokens
        self.auto_prune = auto_prune
        # Use OrderedDict to maintain insertion order for LRU
        self._context: OrderedDict[str, Any] = OrderedDict()

        logger.debug(
            "Working memory initialized",
            extra={'max_tokens': max_tokens, 'auto_prune': auto_prune}
        )

    def add(self, key: str, value: Any, priority: int = 0):
        """
        Add item to working memory.

        Args:
            key: Context key
            value: Context value
            priority: Priority (higher = more important, kept longer during pruning)
        """
        # Store with metadata
        self._context[key] = {
            'value': value,
            'priority': priority,
            'size': self._estimate_tokens(value)
        }

        # Move to end (most recent)
        self._context.move_to_end(key)

        # Auto-prune if needed
        if self.auto_prune:
            self._prune_if_needed()

        logger.debug(
            "Added to working memory",
            extra={
                'key': key,
                'priority': priority,
                'estimated_tokens': self._context[key]['size']
            }
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from working memory.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Value or default
        """
        if key in self._context:
            # Move to end (mark as recently accessed)
            self._context.move_to_end(key)
            return self._context[key]['value']

        return default

    def update(self, key: str, value: Any):
        """
        Update existing item in working memory.

        Args:
            key: Context key
            value: New value
        """
        if key in self._context:
            priority = self._context[key]['priority']
            self.add(key, value, priority)
        else:
            self.add(key, value)

    def remove(self, key: str):
        """
        Remove item from working memory.

        Args:
            key: Context key
        """
        if key in self._context:
            del self._context[key]
            logger.debug("Removed from working memory", extra={'key': key})

    def has(self, key: str) -> bool:
        """
        Check if key exists in working memory.

        Args:
            key: Context key

        Returns:
            True if key exists
        """
        return key in self._context

    def keys(self):
        """Get all keys in working memory."""
        return self._context.keys()

    def items(self):
        """Get all items as (key, value) pairs."""
        return [(k, v['value']) for k, v in self._context.items()]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert working memory to dictionary.

        Returns:
            Dictionary of key-value pairs
        """
        return {k: v['value'] for k, v in self._context.items()}

    def to_context_string(
        self,
        format_style: str = 'simple',
        max_value_length: int = 200
    ) -> str:
        """
        Convert working memory to context string for LLM.

        Args:
            format_style: Format style ('simple', 'detailed', 'json')
            max_value_length: Maximum length for values (truncate longer values)

        Returns:
            Formatted context string
        """
        if not self._context:
            return ""

        if format_style == 'simple':
            # Simple key: value format
            lines = []
            for key, data in self._context.items():
                value = str(data['value'])
                if len(value) > max_value_length:
                    value = value[:max_value_length] + "..."
                lines.append(f"{key}: {value}")
            return "\n".join(lines)

        elif format_style == 'detailed':
            # Detailed format with metadata
            lines = ["## Working Memory Context ##"]
            for key, data in self._context.items():
                value = str(data['value'])
                if len(value) > max_value_length:
                    value = value[:max_value_length] + "..."
                lines.append(f"- {key} (priority: {data['priority']}): {value}")
            return "\n".join(lines)

        elif format_style == 'json':
            # JSON format
            import json
            return json.dumps(self.to_dict(), indent=2)

        else:
            raise ValueError(f"Unknown format style: {format_style}")

    def clear(self):
        """Clear all working memory."""
        size = len(self._context)
        self._context.clear()

        logger.debug("Working memory cleared", extra={'items_removed': size})

    def get_size(self) -> int:
        """
        Get current size in estimated tokens.

        Returns:
            Estimated token count
        """
        return sum(item['size'] for item in self._context.values())

    def get_item_count(self) -> int:
        """
        Get number of items in working memory.

        Returns:
            Item count
        """
        return len(self._context)

    def _estimate_tokens(self, value: Any) -> int:
        """
        Estimate token count for a value.

        Uses rough approximation: 1 token ~= 4 characters.

        Args:
            value: Value to estimate

        Returns:
            Estimated token count
        """
        text = str(value)
        return len(text) // 4

    def _prune_if_needed(self):
        """
        Prune working memory if exceeding max_tokens.

        Removes least recently used items with lower priority first.
        """
        current_size = self.get_size()

        if current_size <= self.max_tokens:
            return

        # Sort by priority (ascending) and recency (oldest first)
        # Items with lower priority and older access get pruned first
        items_by_priority = sorted(
            self._context.items(),
            key=lambda x: (x[1]['priority'], -list(self._context.keys()).index(x[0]))
        )

        tokens_to_remove = current_size - self.max_tokens
        tokens_removed = 0
        items_removed = []

        for key, data in items_by_priority:
            if tokens_removed >= tokens_to_remove:
                break

            tokens_removed += data['size']
            items_removed.append(key)

        # Remove items
        for key in items_removed:
            del self._context[key]

        if items_removed:
            logger.debug(
                "Working memory pruned",
                extra={
                    'items_removed': len(items_removed),
                    'tokens_removed': tokens_removed,
                    'new_size': self.get_size()
                }
            )

    def set_max_tokens(self, max_tokens: int):
        """
        Update max tokens limit.

        Args:
            max_tokens: New maximum tokens
        """
        self.max_tokens = max_tokens

        if self.auto_prune:
            self._prune_if_needed()

        logger.debug("Max tokens updated", extra={'max_tokens': max_tokens})

    def prioritize(self, key: str, priority: int):
        """
        Update priority for an existing item.

        Args:
            key: Context key
            priority: New priority value
        """
        if key in self._context:
            self._context[key]['priority'] = priority
            logger.debug(
                "Priority updated",
                extra={'key': key, 'priority': priority}
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get working memory statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'item_count': self.get_item_count(),
            'estimated_tokens': self.get_size(),
            'max_tokens': self.max_tokens,
            'usage_percent': (self.get_size() / self.max_tokens * 100) if self.max_tokens > 0 else 0,
            'keys': list(self._context.keys())
        }

    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of current working memory.

        Returns:
            Snapshot dictionary with all data and metadata
        """
        return {
            'timestamp': __import__('datetime').datetime.utcnow().isoformat(),
            'items': {
                k: {
                    'value': v['value'],
                    'priority': v['priority'],
                    'size': v['size']
                }
                for k, v in self._context.items()
            },
            'stats': self.get_stats()
        }

    def restore(self, snapshot: Dict[str, Any]):
        """
        Restore working memory from a snapshot.

        Args:
            snapshot: Previously created snapshot
        """
        self._context.clear()

        if 'items' in snapshot:
            for key, data in snapshot['items'].items():
                self._context[key] = data

        logger.info(
            "Working memory restored",
            extra={'items_count': len(self._context)}
        )

    def merge(self, other: 'WorkingMemory', strategy: str = 'keep_existing'):
        """
        Merge another working memory into this one.

        Args:
            other: Another WorkingMemory instance
            strategy: Merge strategy ('keep_existing', 'overwrite', 'higher_priority')
        """
        for key, data in other._context.items():
            if strategy == 'overwrite':
                self._context[key] = data
            elif strategy == 'keep_existing':
                if key not in self._context:
                    self._context[key] = data
            elif strategy == 'higher_priority':
                if key not in self._context or data['priority'] > self._context[key]['priority']:
                    self._context[key] = data

        if self.auto_prune:
            self._prune_if_needed()

        logger.debug(
            "Working memory merged",
            extra={'strategy': strategy, 'items_count': len(self._context)}
        )
