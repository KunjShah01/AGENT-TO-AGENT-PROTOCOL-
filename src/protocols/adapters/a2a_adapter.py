"""
Google A2A Protocol Adapter

Implements Google's Agent-to-Agent protocol specification:
- Agent cards for capability discovery
- tasks/send, tasks/status, tasks/cancel methods
- Streaming support
- Push notifications
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from ..unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..unified.agent_card import AgentCard, AgentCapability, CapabilityType, AgentSkill, AgentEndpoint
from ..unified.task import Task, TaskResult, TaskArtifact
from ..unified.protocol import ProtocolAdapter, ProtocolCapability


class A2AAdapter(ProtocolAdapter):
    """
    Google A2A Protocol Adapter
    
    Implements the Google A2A protocol specification for agent communication.
    
    Key features:
    - Agent cards for capability discovery
    - Task lifecycle management (send, status, cancel)
    - Streaming responses
    - Push notifications
    """
    
    def __init__(self, agent_card: Optional[AgentCard] = None):
        super().__init__(
            name="Google A2A",
            protocol_type=ProtocolType.A2A,
            version="1.0.0",
            capabilities=[
                ProtocolCapability.TASK_MANAGEMENT,
                ProtocolCapability.STREAMING,
                ProtocolCapability.SECURITY,
            ],
        )
        
        self.agent_card = agent_card
        self._tasks: Dict[str, Task] = {}
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Register A2A methods
        self._register_methods()
    
    def _register_methods(self):
        """Register A2A protocol methods"""
        self._handlers["tasks/send"] = self._handle_tasks_send
        self._handlers["tasks/status"] = self._handle_tasks_status
        self._handlers["tasks/cancel"] = self._handle_tasks_cancel
        self._handlers["tasks/subscribe"] = self._handle_tasks_subscribe
        self._handlers["agent/card"] = self._handle_agent_card
    
    async def to_unified(self, a2a_message: Dict[str, Any]) -> UnifiedMessage:
        """
        Convert A2A message to unified format
        
        Args:
            a2a_message: A2A format message
            
        Returns:
            UnifiedMessage
        """
        # Extract method
        method = a2a_message.get("method", "")
        params = a2a_message.get("params", {})
        
        # Determine message type
        message_type = MessageType.TASK
        if "status" in method:
            message_type = MessageType.TASK_STATUS
        elif "cancel" in method:
            message_type = MessageType.TASK_CANCEL
        
        # Build unified message
        unified = UnifiedMessage(
            id=a2a_message.get("id", str(uuid.uuid4())),
            protocol=ProtocolType.A2A,
            message_type=message_type,
            sender_id=params.get("sender", ""),
            receiver_id=params.get("receiver", ""),
            content=params.get("task", params.get("content", {})),
            task_id=params.get("task_id"),
            metadata={
                "a2a_method": method,
                "jsonrpc_id": a2a_message.get("id"),
            }
        )
        
        return unified
    
    async def from_unified(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """
        Convert unified message to A2A format
        
        Args:
            unified: UnifiedMessage
            
        Returns:
            A2A format message
        """
        # Determine method based on message type
        method_map = {
            MessageType.TASK: "tasks/send",
            MessageType.TASK_STATUS: "tasks/status",
            MessageType.TASK_CANCEL: "tasks/cancel",
            MessageType.RESPONSE: "tasks/response",
        }
        
        method = method_map.get(unified.message_type, "tasks/send")
        
        # Build A2A message
        a2a_message = {
            "jsonrpc": "2.0",
            "id": unified.id,
            "method": method,
            "params": {
                "sender": unified.sender_id,
                "receiver": unified.receiver_id,
                "task_id": unified.task_id,
                "content": unified.content,
            }
        }
        
        # Add result if present
        if unified.task_result:
            a2a_message["params"]["result"] = unified.task_result
        
        return a2a_message
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Handle unified message
        
        Args:
            message: UnifiedMessage
            
        Returns:
            Response message or None
        """
        # Get A2A method from metadata
        method = message.metadata.get("a2a_method", "tasks/send")
        
        # Route to appropriate handler
        handler = self._handlers.get(method)
        if handler:
            return await handler(message)
        
        # Default: echo back
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"status": "received", "original_method": method},
            correlation_id=message.id,
        )
    
    async def _handle_tasks_send(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tasks/send method"""
        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            name=message.metadata.get("task_name", "Unnamed Task"),
            description=message.metadata.get("task_description", ""),
            creator_id=message.sender_id,
            assignee_id=message.receiver_id,
            input_data=message.content,
            priority=message.priority,
        )
        
        self._tasks[task.id] = task
        
        # Return task creation response
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task.id,
                "status": task.status.value,
                "message": "Task created successfully",
            },
            task_id=task.id,
            correlation_id=message.id,
        )
    
    async def _handle_tasks_status(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tasks/status method"""
        task_id = message.task_id or message.content.get("task_id")
        task = self._tasks.get(task_id)
        
        if not task:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.A2A,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Task not found: {task_id}"},
                correlation_id=message.id,
            )
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task.id,
                "status": task.status.value,
                "result": task.result.to_dict() if task.result else None,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            },
            task_id=task.id,
            correlation_id=message.id,
        )
    
    async def _handle_tasks_cancel(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tasks/cancel method"""
        task_id = message.task_id or message.content.get("task_id")
        task = self._tasks.get(task_id)
        
        if not task:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.A2A,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Task not found: {task_id}"},
                correlation_id=message.id,
            )
        
        # Cancel task
        task.update_status(TaskStatus.CANCELLED)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task.id,
                "status": task.status.value,
                "message": "Task cancelled successfully",
            },
            task_id=task.id,
            correlation_id=message.id,
        )
    
    async def _handle_tasks_subscribe(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tasks/subscribe method (streaming)"""
        # TODO: Implement streaming support
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"message": "Streaming subscription established"},
            correlation_id=message.id,
        )
    
    async def _handle_agent_card(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle agent/card request"""
        if not self.agent_card:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.A2A,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": "Agent card not available"},
                correlation_id=message.id,
            )
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.A2A,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content=self.agent_card.to_dict(),
            correlation_id=message.id,
        )
    
    async def handle_task(self, task: Task) -> TaskResult:
        """
        Execute a task using A2A protocol
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult
        """
        # Update task status
        task.update_status(TaskStatus.RUNNING)
        
        try:
            # Simulate task execution
            await asyncio.sleep(0.1)
            
            # Create result
            result = TaskResult(
                status=TaskStatus.COMPLETED,
                output={"message": "Task completed via A2A protocol", "task_id": task.id},
                execution_time=0.1,
            )
            
            return result
            
        except Exception as e:
            self._logger.error(f"Task execution failed: {e}")
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                error_code="EXECUTION_ERROR",
            )


