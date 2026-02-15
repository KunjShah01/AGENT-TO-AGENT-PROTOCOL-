"""
Unified Protocol Implementation
Main protocol handler that routes between different protocol adapters
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from .message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from .agent_card import AgentCard, AgentCapability, CapabilityType
from .task import Task, TaskResult


class ProtocolCapability(Enum):
    """Capabilities supported by the unified protocol"""
    TASK_MANAGEMENT = "task_management"
    TOOL_CALLING = "tool_calling"
    RESOURCE_ACCESS = "resource_access"
    STREAMING = "streaming"
    GROUP_CHAT = "group_chat"
    WORKFLOW = "workflow"
    STATE_MANAGEMENT = "state_management"
    COLLABORATION = "collaboration"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    SECURITY = "security"
    RL_INTEGRATION = "rl_integration"


@dataclass
class ProtocolAdapter:
    """
    Protocol adapter configuration
    
    Defines how to adapt between unified protocol and specific protocols
    """
    name: str
    protocol_type: ProtocolType
    version: str = "1.0.0"
    
    # Capabilities supported by this adapter
    capabilities: List[ProtocolCapability] = field(default_factory=list)
    
    # Message converters
    to_unified: Optional[Callable[[Any], UnifiedMessage]] = None
    from_unified: Optional[Callable[[UnifiedMessage], Any]] = None
    
    # Handler functions
    handle_message: Optional[Callable[[UnifiedMessage], Any]] = None
    handle_task: Optional[Callable[[Task], TaskResult]] = None
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def can_handle(self, message: UnifiedMessage) -> bool:
        """Check if this adapter can handle the message"""
        return self.enabled and message.protocol == self.protocol_type
    
    def supports_capability(self, capability: ProtocolCapability) -> bool:
        """Check if adapter supports a capability"""
        return capability in self.capabilities


class UnifiedProtocol:
    """
    Unified Protocol Handler
    
    Routes messages between different protocol adapters and provides
    a common interface for all A2A protocols.
    """
    
    def __init__(self):
        self._adapters: Dict[ProtocolType, ProtocolAdapter] = {}
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._tasks: Dict[str, Task] = {}
        self._logger = logging.getLogger(__name__)
        
        # Event callbacks
        self._event_handlers: List[Callable] = []
        
        # Statistics
        self._stats = {
            "messages_processed": 0,
            "tasks_created": 0,
            "tasks_completed": 0,
            "errors": 0,
        }
    
    def register_adapter(self, adapter: ProtocolAdapter) -> None:
        """
        Register a protocol adapter
        
        Args:
            adapter: Protocol adapter to register
        """
        self._adapters[adapter.protocol_type] = adapter
        self._logger.info(f"Registered protocol adapter: {adapter.name} ({adapter.protocol_type.value})")
    
    def unregister_adapter(self, protocol_type: ProtocolType) -> None:
        """
        Unregister a protocol adapter
        
        Args:
            protocol_type: Protocol type to unregister
        """
        if protocol_type in self._adapters:
            del self._adapters[protocol_type]
            self._logger.info(f"Unregistered protocol adapter: {protocol_type.value}")
    
    def get_adapter(self, protocol_type: ProtocolType) -> Optional[ProtocolAdapter]:
        """
        Get protocol adapter
        
        Args:
            protocol_type: Protocol type
            
        Returns:
            Protocol adapter or None
        """
        return self._adapters.get(protocol_type)
    
    def register_handler(self, message_type: MessageType, handler: Callable) -> None:
        """
        Register a message handler
        
        Args:
            message_type: Type of message to handle
            handler: Handler function
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []
        self._handlers[message_type].append(handler)
    
    async def process_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Process a unified message
        
        Args:
            message: Message to process
            
        Returns:
            Response message or None
        """
        self._stats["messages_processed"] += 1
        
        try:
            # Route to appropriate adapter
            adapter = self._adapters.get(message.protocol)
            
            if adapter and adapter.handle_message:
                # Use adapter's handler
                result = await adapter.handle_message(message)
                
                # Convert result to UnifiedMessage if needed
                if result and not isinstance(result, UnifiedMessage):
                    result = UnifiedMessage(
                        id=str(uuid.uuid4()),
                        protocol=message.protocol,
                        message_type=MessageType.RESPONSE,
                        sender_id=message.receiver_id,
                        receiver_id=message.sender_id,
                        content=result,
                        correlation_id=message.id,
                    )
                
                return result
            
            # Use registered handlers
            handlers = self._handlers.get(message.message_type, [])
            for handler in handlers:
                result = await handler(message)
                if result:
                    return result
            
            return None
            
        except Exception as e:
            self._logger.error(f"Error processing message: {e}", exc_info=True)
            self._stats["errors"] += 1
            
            # Return error message
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=message.protocol,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": str(e), "original_message_id": message.id},
                correlation_id=message.id,
            )
    
    async def create_task(self, task: Task) -> Task:
        """
        Create and execute a task
        
        Args:
            task: Task to create
            
        Returns:
            Created task
        """
        self._tasks[task.id] = task
        self._stats["tasks_created"] += 1
        
        # Route to appropriate adapter
        adapter = self._adapters.get(ProtocolType.A2A)
        
        if adapter and adapter.handle_task:
            result = await adapter.handle_task(task)
            task.result = result
            task.update_status(TaskStatus.COMPLETED if result.is_success() else TaskStatus.FAILED, result)
            self._stats["tasks_completed"] += 1
        else:
            # Default: mark as pending for manual execution
            task.update_status(TaskStatus.PENDING)
        
        return task
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task or None
        """
        return self._tasks.get(task_id)
    
    def get_stats(self) -> Dict[str, int]:
        """Get protocol statistics"""
        return self._stats.copy()


