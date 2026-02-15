"""
LangGraph Protocol Adapter

Implements LangChain's LangGraph workflow orchestration:
- State management
- Workflow graphs
- Checkpoints
- Conditional edges
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


class LangGraphAdapter(BaseProtocolAdapter):
    """
    LangGraph Protocol Adapter
    
    Implements LangChain's LangGraph workflow orchestration.
    
    Key features:
    - State management
    - Workflow graphs
    - Checkpoints
    - Conditional edges
    """
    
    def __init__(self, agent_card: Optional[AgentCard] = None):
        super().__init__(
            name="LangGraph",
            protocol_type=ProtocolType.LANGGRAPH,
            version="0.2.0",
            capabilities=[
                ProtocolCapability.WORKFLOW,
                ProtocolCapability.STATE_MANAGEMENT,
                ProtocolCapability.COLLABORATION,
            ],
        )
        
        self.agent_card = agent_card
        self._workflows: Dict[str, Dict[str, Any]] = {}
        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Register LangGraph methods
        self._register_methods()
    
    def _register_methods(self):
        """Register LangGraph protocol methods"""
        self._handlers["workflow/start"] = self._handle_workflow_start
        self._handlers["workflow/step"] = self._handle_workflow_step
        self._handlers["workflow/resume"] = self._handle_workflow_resume
        self._handlers["state/update"] = self._handle_state_update
        self._handlers["checkpoint/save"] = self._handle_checkpoint_save
        self._handlers["checkpoint/load"] = self._handle_checkpoint_load
    
    async def to_unified(self, langgraph_message: Dict[str, Any]) -> UnifiedMessage:
        """
        Convert LangGraph message to unified format
        
        Args:
            langgraph_message: LangGraph format message
            
        Returns:
            UnifiedMessage
        """
        messages = langgraph_message.get("messages", [])
        state = langgraph_message.get("state", {})
        checkpoint = langgraph_message.get("checkpoint")
        
        # Extract content from messages
        content = ""
        if messages:
            if isinstance(messages, list):
                content = messages[-1].get("content", "") if messages else ""
            else:
                content = str(messages)
        
        # Determine message type
        message_type = MessageType.WORKFLOW_EVENT
        if checkpoint:
            message_type = MessageType.CHECKPOINT
        elif state:
            message_type = MessageType.STATE_UPDATE
        
        unified = UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=message_type,
            sender_id=state.get("sender", ""),
            receiver_id=state.get("receiver", ""),
            content=content,
            state=state,
            checkpoint_id=checkpoint,
            metadata={
                "workflow_id": langgraph_message.get("workflow_id"),
                "step": langgraph_message.get("step"),
                "metadata": langgraph_message.get("metadata", {}),
            }
        )
        
        return unified
    
    async def from_unified(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """
        Convert unified message to LangGraph format
        
        Args:
            unified: UnifiedMessage
            
        Returns:
            LangGraph format message
        """
        langgraph_message = {
            "messages": [{"role": "assistant", "content": unified.content}],
            "state": unified.state or {},
            "checkpoint": unified.checkpoint_id,
            "metadata": {
                "sender": unified.sender_id,
                "receiver": unified.receiver_id,
                "task_id": unified.task_id,
                **unified.metadata
            }
        }
        
        return langgraph_message
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Handle unified message
        
        Args:
            message: UnifiedMessage
            
        Returns:
            Response message or None
        """
        method = message.metadata.get("langgraph_method", "workflow/start")
        handler = self._handlers.get(method)
        
        if handler:
            return await handler(message)
        
        return None
    
    async def _handle_workflow_start(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle workflow start"""
        workflow_id = str(uuid.uuid4())
        
        self._workflows[workflow_id] = {
            "id": workflow_id,
            "status": "running",
            "current_step": 0,
            "state": message.state or {},
            "created_at": datetime.now().isoformat(),
        }
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.WORKFLOW_EVENT,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "workflow_id": workflow_id,
                "status": "started",
                "message": "Workflow started successfully",
            },
            state=self._workflows[workflow_id]["state"],
            correlation_id=message.id,
        )
    
    async def _handle_workflow_step(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle workflow step"""
        workflow_id = message.metadata.get("workflow_id")
        
        if not workflow_id or workflow_id not in self._workflows:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.LANGGRAPH,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Workflow not found: {workflow_id}"},
                correlation_id=message.id,
            )
        
        workflow = self._workflows[workflow_id]
        workflow["current_step"] += 1
        
        # Update state
        if message.state:
            workflow["state"].update(message.state)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.WORKFLOW_EVENT,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "workflow_id": workflow_id,
                "step": workflow["current_step"],
                "status": "running",
            },
            state=workflow["state"],
            correlation_id=message.id,
        )
    
    async def _handle_workflow_resume(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle workflow resume from checkpoint"""
        workflow_id = message.metadata.get("workflow_id")
        checkpoint_id = message.checkpoint_id
        
        if checkpoint_id and checkpoint_id in self._checkpoints:
            checkpoint = self._checkpoints[checkpoint_id]
            
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.LANGGRAPH,
                message_type=MessageType.WORKFLOW_EVENT,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={
                    "workflow_id": workflow_id,
                    "checkpoint_id": checkpoint_id,
                    "status": "resumed",
                },
                state=checkpoint.get("state", {}),
                correlation_id=message.id,
            )
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.ERROR,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"error": f"Checkpoint not found: {checkpoint_id}"},
            correlation_id=message.id,
        )
    
    async def _handle_state_update(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle state update"""
        workflow_id = message.metadata.get("workflow_id")
        
        if workflow_id and workflow_id in self._workflows:
            if message.state:
                self._workflows[workflow_id]["state"].update(message.state)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.STATE_UPDATE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={"message": "State updated"},
            state=message.state,
            correlation_id=message.id,
        )
    
    async def _handle_checkpoint_save(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle checkpoint save"""
        checkpoint_id = str(uuid.uuid4())
        
        self._checkpoints[checkpoint_id] = {
            "id": checkpoint_id,
            "state": message.state or {},
            "workflow_id": message.metadata.get("workflow_id"),
            "created_at": datetime.now().isoformat(),
        }
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.CHECKPOINT,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "checkpoint_id": checkpoint_id,
                "message": "Checkpoint saved",
            },
            checkpoint_id=checkpoint_id,
            correlation_id=message.id,
        )
    
    async def _handle_checkpoint_load(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle checkpoint load"""
        checkpoint_id = message.checkpoint_id
        
        if checkpoint_id not in self._checkpoints:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.LANGGRAPH,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Checkpoint not found: {checkpoint_id}"},
                correlation_id=message.id,
            )
        
        checkpoint = self._checkpoints[checkpoint_id]
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.LANGGRAPH,
            message_type=MessageType.CHECKPOINT,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "checkpoint_id": checkpoint_id,
                "message": "Checkpoint loaded",
            },
            state=checkpoint.get("state", {}),
            checkpoint_id=checkpoint_id,
            correlation_id=message.id,
        )
    
    async def handle_task(self, task: Task) -> TaskResult:
        """
        Execute task using LangGraph
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult
        """
        try:
            # Create workflow for task
            workflow_id = str(uuid.uuid4())
            
            self._workflows[workflow_id] = {
                "id": workflow_id,
                "task_id": task.id,
                "status": "running",
                "current_step": 0,
                "state": task.state or {},
                "created_at": datetime.now().isoformat(),
            }
            
            # Simulate workflow execution
            await asyncio.sleep(0.1)
            
            # Update workflow
            self._workflows[workflow_id]["status"] = "completed"
            self._workflows[workflow_id]["current_step"] = 1
            
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output={
                    "message": "Task executed via LangGraph workflow",
                    "workflow_id": workflow_id,
                    "steps": 1,
                },
                state_updates=self._workflows[workflow_id]["state"],
                checkpoint_id=workflow_id,
            )
            
        except Exception as e:
            self._logger.error(f"LangGraph task execution failed: {e}")
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                error_code="LANGGRAPH_ERROR",
            )


