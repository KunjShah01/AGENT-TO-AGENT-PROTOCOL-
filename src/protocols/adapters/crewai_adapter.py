"""
CrewAI Protocol Adapter

Implements CrewAI's crew-based collaboration:
- Crew management
- Tool sharing
- Collaborative tasks
- Workflow tracing
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


class CrewAIAdapter(BaseProtocolAdapter):
    """
    CrewAI Protocol Adapter
    
    Implements CrewAI's crew-based collaboration framework.
    
    Key features:
    - Crew management
    - Tool sharing
    - Collaborative tasks
    - Workflow tracing
    """
    
    def __init__(self, agent_card: Optional[AgentCard] = None):
        super().__init__(
            name="CrewAI",
            protocol_type=ProtocolType.CREWAI,
            version="0.100.0",
            capabilities=[
                ProtocolCapability.COLLABORATION,
                ProtocolCapability.TOOL_SHARING,
                ProtocolCapability.WORKFLOW,
            ],
        )
        
        self.agent_card = agent_card
        self._crews: Dict[str, Dict[str, Any]] = {}
        self._shared_tools: Dict[str, Dict[str, Any]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._logger = logging.getLogger(__name__)
        
        # Register CrewAI methods
        self._register_methods()
    
    def _register_methods(self):
        """Register CrewAI protocol methods"""
        self._handlers["crew/create"] = self._handle_crew_create
        self._handlers["crew/join"] = self._handle_crew_join
        self._handlers["crew/leave"] = self._handle_crew_leave
        self._handlers["crew/task"] = self._handle_crew_task
        self._handlers["tool/share"] = self._handle_tool_share
        self._handlers["collaborate"] = self._handle_collaborate
    
    async def to_unified(self, crewai_message: Dict[str, Any]) -> UnifiedMessage:
        """
        Convert CrewAI message to unified format
        
        Args:
            crewai_message: CrewAI format message
            
        Returns:
            UnifiedMessage
        """
        description = crewai_message.get("description", "")
        agent = crewai_message.get("agent", "")
        crew = crewai_message.get("crew")
        context = crewai_message.get("context", {})
        tools = crewai_message.get("tools", [])
        
        # Determine message type
        message_type = MessageType.CREW_TASK
        if crewai_message.get("collaboration"):
            message_type = MessageType.COLLABORATION
        elif tools:
            message_type = MessageType.TOOL_SHARE
        
        unified = UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=message_type,
            sender_id=agent,
            receiver_id="",  # Crew or broadcast
            content=description,
            crew_id=crew,
            collaboration_context=context,
            tool_calls=tools,
            metadata={
                "async_execution": crewai_message.get("async_execution", False),
                "expected_output": crewai_message.get("expected_output"),
                **crewai_message.get("metadata", {}),
            }
        )
        
        return unified
    
    async def from_unified(self, unified: UnifiedMessage) -> Dict[str, Any]:
        """
        Convert unified message to CrewAI format
        
        Args:
            unified: UnifiedMessage
            
        Returns:
            CrewAI format message
        """
        crewai_message = {
            "description": unified.content,
            "agent": unified.sender_id,
            "crew": unified.crew_id,
            "context": unified.collaboration_context or {},
            "tools": unified.tool_calls,
            "async_execution": unified.metadata.get("async_execution", False),
            "metadata": unified.metadata,
        }
        
        return crewai_message
    
    async def handle_message(self, message: UnifiedMessage) -> Optional[UnifiedMessage]:
        """
        Handle unified message
        
        Args:
            message: UnifiedMessage
            
        Returns:
            Response message or None
        """
        method = message.metadata.get("crewai_method", "crew/task")
        handler = self._handlers.get(method)
        
        if handler:
            return await handler(message)
        
        return None
    
    async def _handle_crew_create(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle crew creation"""
        crew_id = str(uuid.uuid4())
        
        self._crews[crew_id] = {
            "id": crew_id,
            "name": message.metadata.get("crew_name", f"Crew-{crew_id[:8]}"),
            "agents": [],
            "tasks": [],
            "shared_tools": [],
            "created_at": datetime.now().isoformat(),
        }
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "crew_id": crew_id,
                "message": "Crew created successfully",
                "name": self._crews[crew_id]["name"],
            },
            crew_id=crew_id,
            correlation_id=message.id,
        )
    
    async def _handle_crew_join(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle agent joining crew"""
        crew_id = message.crew_id
        agent_id = message.sender_id
        
        if not crew_id or crew_id not in self._crews:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.CREWAI,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Crew not found: {crew_id}"},
                correlation_id=message.id,
            )
        
        crew = self._crews[crew_id]
        
        if agent_id not in crew["agents"]:
            crew["agents"].append(agent_id)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "crew_id": crew_id,
                "agent_id": agent_id,
                "message": "Agent joined crew successfully",
                "crew_size": len(crew["agents"]),
            },
            crew_id=crew_id,
            correlation_id=message.id,
        )
    
    async def _handle_crew_leave(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle agent leaving crew"""
        crew_id = message.crew_id
        agent_id = message.sender_id
        
        if not crew_id or crew_id not in self._crews:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.CREWAI,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Crew not found: {crew_id}"},
                correlation_id=message.id,
            )
        
        crew = self._crews[crew_id]
        
        if agent_id in crew["agents"]:
            crew["agents"].remove(agent_id)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.RESPONSE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "crew_id": crew_id,
                "agent_id": agent_id,
                "message": "Agent left crew successfully",
                "crew_size": len(crew["agents"]),
            },
            crew_id=crew_id,
            correlation_id=message.id,
        )
    
    async def _handle_crew_task(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle crew task assignment"""
        crew_id = message.crew_id
        
        if not crew_id or crew_id not in self._crews:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.CREWAI,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Crew not found: {crew_id}"},
                correlation_id=message.id,
            )
        
        crew = self._crews[crew_id]
        task_id = str(uuid.uuid4())
        
        crew["tasks"].append({
            "id": task_id,
            "description": message.content,
            "agent": message.sender_id,
            "context": message.collaboration_context,
            "tools": message.tool_calls,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        })
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.CREW_TASK,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "task_id": task_id,
                "crew_id": crew_id,
                "message": "Task assigned to crew",
                "crew_agents": crew["agents"],
            },
            crew_id=crew_id,
            correlation_id=message.id,
        )
    
    async def _handle_tool_share(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle tool sharing"""
        tool_name = message.metadata.get("tool_name")
        crew_id = message.crew_id
        
        if not tool_name:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.CREWAI,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": "Tool name required"},
                correlation_id=message.id,
            )
        
        # Register shared tool
        self._shared_tools[tool_name] = {
            "name": tool_name,
            "owner": message.sender_id,
            "crew_id": crew_id,
            "description": message.content,
            "shared_at": datetime.now().isoformat(),
        }
        
        # Add to crew if specified
        if crew_id and crew_id in self._crews:
            if tool_name not in self._crews[crew_id]["shared_tools"]:
                self._crews[crew_id]["shared_tools"].append(tool_name)
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.TOOL_SHARE,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "tool_name": tool_name,
                "message": "Tool shared successfully",
                "crew_id": crew_id,
            },
            crew_id=crew_id,
            correlation_id=message.id,
        )
    
    async def _handle_collaborate(self, message: UnifiedMessage) -> UnifiedMessage:
        """Handle collaboration request"""
        crew_id = message.crew_id
        
        if not crew_id or crew_id not in self._crews:
            return UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=ProtocolType.CREWAI,
                message_type=MessageType.ERROR,
                sender_id=message.receiver_id,
                receiver_id=message.sender_id,
                content={"error": f"Crew not found: {crew_id}"},
                correlation_id=message.id,
            )
        
        crew = self._crews[crew_id]
        
        return UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=ProtocolType.CREWAI,
            message_type=MessageType.COLLABORATION,
            sender_id=message.receiver_id,
            receiver_id=message.sender_id,
            content={
                "crew_id": crew_id,
                "message": "Collaboration initiated",
                "crew_agents": crew["agents"],
                "shared_tools": crew["shared_tools"],
                "active_tasks": len(crew["tasks"]),
            },
            crew_id=crew_id,
            collaboration_context=message.collaboration_context,
            correlation_id=message.id,
        )
    
    async def handle_task(self, task: Task) -> TaskResult:
        """
        Execute task using CrewAI
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult
        """
        try:
            crew_id = task.crew_id
            
            if crew_id and crew_id in self._crews:
                crew = self._crews[crew_id]
                
                # Simulate collaborative execution
                await asyncio.sleep(0.1)
                
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    output={
                        "message": "Task executed via CrewAI collaboration",
                        "crew_id": crew_id,
                        "collaborating_agents": crew["agents"],
                        "shared_tools_used": crew["shared_tools"],
                    },
                )
            
            # Default execution
            return TaskResult(
                status=TaskStatus.COMPLETED,
                output={"message": "Task executed via CrewAI"},
            )
            
        except Exception as e:
            self._logger.error(f"CrewAI task execution failed: {e}")
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                error_code="CREWAI_ERROR",
            )


