"""
AutoGen Protocol Adapter

Implements Microsoft's AutoGen multi-agent framework:
- Group chat orchestration
- Code execution
- Message passing between agents
- Event-driven architecture
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from ..unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..unified.agent_card import AgentCard, AgentCapability, CapabilityType
from ..unified.task import Task, TaskResult, TaskArtifact
from ..unified.protocol import ProtocolAdapter as BaseProtocolAdapter, ProtocolCapability


class AutoGenAdapter(BaseProtocolAdapter):
    """
    AutoGen Protocol Adapter
    
    Implements Microsoft's AutoGen multi-agent framework.
    
    Key features:
    - Group chat orchestration
    - Code execution
    - Message passing
    - Event-driven agents
    """
    
    def __init__(self, agent_card: Optional[AgentCard] = None):
        super().__init__(
            name="AutoGen",
            protocol_type=ProtocolType.AUTOGEN,
            version="0.7.5",
            capabilities=[
                ProtocolCapability.GROUP_CHAT,
                ProtocolCapability.CODE_EXECUTION,
                ProtocolCapability.COLLABORATION,
            ],
        )
        
        self.agent_card = agent_card
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Register AutoGen methods
        self._register_methods()
    
    def _register_methods(self):
        """Register AutoGen protocol methods"""
        self._handlers["chat/send"] = self._handle_chat_send
        self._handlers["chat/group"] = self._handle_group_chat
        self._handlers["code/execute"] = self._handle_code_execute
        self._handlers["orchestrate"] = self._handle_orchestrate
    
    async def to_unified(self, autogen_message: Dict[str, Any]) -> UnifiedMessage:
        """
        Convert AutoGen message to unified format
        
        Args:
            autogen_message: AutoGen format message
            
        Returns:
            UnifiedMessage
        """
        content = autogen_message.get("content", "")
        role = autogen_message.get("role", "user")
        name = autogen_message.get("name", "")
        
        # Determine message type
        message_type = MessageType.CHAT
        if autogen_message.get("function_call"):
            message_type = MessageType.TOOL_CALL
        elif autogen_message.get("group_id"):
            message_type = MessageType.GROUP_CHAT
        
        unified = UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.AUTOGEN,
            message_type=message_type,
            sender_id=name,
            receiver_id="",  # Group or broadcast
            content=content,
            tool_calls=[autogen_message["function_call"]] if autogen_message.get("function_call") else [],
            crew_id=autogen_message.get("group_id"),
            metadata={
                "role": role,
                "function_call": autogen_message.get("function_call"),
                "metadata": autogen_message.get("metadata", {}),
            }
        )
        
        return unified
    
    async def from_unified(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """
        Convert unified message to AutoGen format
        
        Args:
            unified: UnifiedMessage
            
        Returns:
            AutoGen format message
        """
        # Determine role
        role_map = {
            MessageType.CHAT: "assistant",
            MessageType.GROUP_CHAT: "assistant",
            MessageType.TOOL_CALL: "function",
            MessageType.TOOL_RESULT: "function",
        }
        
        role = role_map.get(unified.message_type, "assistant")
        
        autogen_message = {
            "content": unified.content,
            "role": role,
            "name": unified.sender_id,
            "metadata": {
                "group_id": unified.crew_id,
                "checkpoint_id": unified.checkpoint_id,
                **unified.metadata
            }
        }
        
        if unified.tool_calls:
            autogen_message["function_call"] = unified.tool_calls[0]
        
        return autogen_message
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Handle unified message
        
        Args:
            message: UnifiedMessage
            
        Returns:
            Response message or None
        """
        method = message.metadata.get("autogen_method", "chat/send")
        handler = self._handlers.get(method)
        
        if handler:
            return await handler(message)
        
        return None
    
    async def _handle_chat_send(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle chat/send"""
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.AUTOGEN,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "message": "Chat message received",
                "original_content": message.content,
            },
            correlation_id=message.id,
        )
    
    async def _handle_group_chat(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle group chat"""
        group_id = message.crew_id or str(uuid.uuid4())
        
        if group_id not in self._groups:
            self._groups[group_id] = {
                "id": group_id,
                "participants": [],
                "messages": [],
                "created_at": datetime.now().isoformat(),
            }
        
        # Add message to group
        self._groups[group_id]["messages"].append({
            "sender": message.sender_id,
            "content": message.content,
            "timestamp": datetime.now().isoformat(),
        })
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.AUTOGEN,
            message_type=MessageType.GROUP_CHAT,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "group_id": group_id,
                "message": "Group chat message processed",
                "participants": self._groups[group_id]["participants"],
            },
            crew_id=group_id,
            correlation_id=message.id,
        )
    
    async def _handle_code_execute(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle code execution"""
        code = message.content.get("code") if isinstance(message.content, dict) else str(message.content)
        
        # Simulate code execution (in production, use proper sandbox)
        try:
            # For safety, just simulate execution
            result = f"Simulated execution of: {code[:50]}..."
            
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.AUTOGEN,
                message_type=MessageType.RESPONSE,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={
                    "output": result,
                    "exit_code": 0,
                },
                correlation_id=message.id,
            )
            
        except Exception as e:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.AUTOGEN,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": str(e), "exit_code": 1},
                correlation_id=message.id,
            )
    
    async def _handle_orchestrate(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle orchestration request"""
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.AUTOGEN,
            message_type=MessageType.ORCHESTRATION,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "message": "Orchestration request processed",
                "agents": message.metadata.get("agents", []),
            },
            correlation_id=message.id,
        )
    
    async def handle_task(self, task: Task) -> TaskResult:
        """
        Execute task using AutoGen
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult
        """
        try:
            # Simulate group chat execution
            if task.crew_id and task.crew_id in self._groups:
                group = self._groups[task.crew_id]
                
                # Add task to group messages
                group["messages"].append({
                    "type": "task",
                    "task_id": task.id,
                    "content": task.input_data,
                    "timestamp": datetime.now().isoformat(),
                })
                
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    output={
                        "message": "Task executed in group chat",
                        "group_id": task.crew_id,
                        "participants": group["participants"],
                    },
                )
            
            # Default execution
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output={"message": "Task executed via AutoGen"},
            )
            
        except Exception as e:
            self._logger.error(f"AutoGen task execution failed: {e}")
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                error_code="AUTOGEN_ERROR",
            )


