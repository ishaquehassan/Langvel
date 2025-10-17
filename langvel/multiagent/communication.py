"""Multi-agent communication - Message bus and protocols."""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from collections import defaultdict
from langvel.logging import get_logger


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class AgentMessage:
    """Message passed between agents."""

    sender_id: str
    recipient_id: str
    content: Any
    message_type: str = "data"
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': self.content,
            'message_type': self.message_type,
            'priority': self.priority.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to
        }


class MessageBus:
    """
    Central message bus for agent communication.

    Provides pub/sub messaging, direct messaging, and broadcast capabilities.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._agent_handlers: Dict[str, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._message_history: List[AgentMessage] = []
        self._max_history = 1000
        self.logger = get_logger('langvel.multiagent.bus')

    async def start(self):
        """Start the message bus processing loop."""
        self._running = True
        asyncio.create_task(self._process_messages())

    async def stop(self):
        """Stop the message bus."""
        self._running = False

    async def _process_messages(self):
        """Process messages from the queue."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                await self._deliver_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(
                    "Error processing message",
                    extra={'error': str(e)},
                    exc_info=True
                )

    async def _deliver_message(self, message: AgentMessage):
        """Deliver a message to its recipient."""
        # Add to history
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history.pop(0)

        # Direct message delivery
        if message.recipient_id in self._agent_handlers:
            handler = self._agent_handlers[message.recipient_id]
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(
                    "Error delivering message",
                    extra={
                        'recipient_id': message.recipient_id,
                        'sender_id': message.sender_id,
                        'message_type': message.message_type,
                        'error': str(e)
                    },
                    exc_info=True
                )

        # Topic-based delivery
        topic = f"{message.message_type}:{message.recipient_id}"
        for subscriber in self._subscribers.get(topic, []):
            try:
                await subscriber(message)
            except Exception as e:
                self.logger.error(
                    "Error in subscriber",
                    extra={
                        'topic': topic,
                        'sender_id': message.sender_id,
                        'error': str(e)
                    },
                    exc_info=True
                )

    async def send(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        message_type: str = "data",
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ):
        """
        Send a message to a specific agent.

        Args:
            sender_id: Sender agent ID
            recipient_id: Recipient agent ID
            content: Message content
            message_type: Type of message
            priority: Message priority
            **kwargs: Additional message fields
        """
        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type,
            priority=priority,
            **kwargs
        )

        # Add to queue (priority queue would be better for production)
        await self._message_queue.put(message)

    async def broadcast(
        self,
        sender_id: str,
        content: Any,
        message_type: str = "broadcast",
        **kwargs
    ):
        """
        Broadcast a message to all agents.

        Args:
            sender_id: Sender agent ID
            content: Message content
            message_type: Type of message
            **kwargs: Additional message fields
        """
        for agent_id in self._agent_handlers.keys():
            if agent_id != sender_id:
                await self.send(
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    content=content,
                    message_type=message_type,
                    **kwargs
                )

    async def publish(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        **kwargs
    ):
        """
        Publish a message to a topic.

        Args:
            sender_id: Sender agent ID
            topic: Topic name
            content: Message content
            **kwargs: Additional message fields
        """
        message = AgentMessage(
            sender_id=sender_id,
            recipient_id=topic,
            content=content,
            message_type="topic",
            **kwargs
        )

        # Deliver to all subscribers
        for subscriber in self._subscribers.get(topic, []):
            try:
                await subscriber(message)
            except Exception as e:
                self.logger.error(
                    "Error in topic subscriber",
                    extra={
                        'topic': topic,
                        'sender_id': sender_id,
                        'error': str(e)
                    },
                    exc_info=True
                )

    def subscribe(
        self,
        topic: str,
        handler: Callable[[AgentMessage], Any]
    ):
        """
        Subscribe to a topic.

        Args:
            topic: Topic name
            handler: Async callback function
        """
        self._subscribers[topic].append(handler)

    def unsubscribe(
        self,
        topic: str,
        handler: Callable[[AgentMessage], Any]
    ):
        """
        Unsubscribe from a topic.

        Args:
            topic: Topic name
            handler: Callback function to remove
        """
        if topic in self._subscribers:
            self._subscribers[topic].remove(handler)

    def register_agent(
        self,
        agent_id: str,
        handler: Callable[[AgentMessage], Any]
    ):
        """
        Register an agent's message handler.

        Args:
            agent_id: Agent identifier
            handler: Async callback function
        """
        self._agent_handlers[agent_id] = handler

    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agent_handlers:
            del self._agent_handlers[agent_id]

    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[AgentMessage]:
        """
        Get message history.

        Args:
            agent_id: Filter by agent ID (sender or recipient)
            limit: Maximum number of messages

        Returns:
            List of messages
        """
        history = self._message_history

        if agent_id:
            history = [
                m for m in history
                if m.sender_id == agent_id or m.recipient_id == agent_id
            ]

        if limit:
            history = history[-limit:]

        return history


# Global message bus instance
_message_bus = None


def get_message_bus() -> MessageBus:
    """Get or create global message bus."""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus
