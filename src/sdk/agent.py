"""
Agent Client

Client for individual agent interactions
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import uuid

from ..protocols.unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..protocols.unified.agent_card import AgentCard, AgentCapability
from ..protocols.unified.task import Task, TaskResult
from .client import UnifiedClient


class AgentClient:
    """
    Agent Client
    
    Client for interacting with a specific agent.
    Provides high-level methods for common operations.
    
    Example:
        >>> agent = AgentClient("https://agent.example.com")
        >>> 
        >>> # Send message
        >>> response = await agent.send("Hello!")
        >>> 
        >>> # Execute task
        >>> result = await agent.execute_task({"prompt": "Process this"})
        >>> 
        >>> # Call tool
        >>> tool_result = await agent.call_tool("calculator", {"x": 5, "y": 3})
    """
    
    def __init__(
        self,
        endpoint: str,
        agent_id: Optional[str] = None,
        agent_card: Optional[AgentCard] = None,
        protocol: ProtocolType = ProtocolType.A2A,
        timeout: float = 30.0,
    ):
        """
        Initialize Agent Client
        
        Args:
            endpoint: Agent endpoint URL
            agent_id: Agent ID (discovered if not provided)
            agent_card: Agent card (discovered if not provided)
            protocol: Protocol to use
            timeout: Request timeout
        """
        self.endpoint = endpoint
        self.agent_id = agent_id
        self.agent_card = agent_card
        self.protocol = protocol
        self.timeout = timeout
        
        # Initialize unified client
        self._client = UnifiedClient(
            agent_id=agent_id or str(uuid.uuid4()),
            default_protocol=protocol,
            timeout=timeout,
        )
        
        self._logger = logging.getLogger(__name__)
        self._initialized = False
    
    async def initialize(self) -> bool:
        """
        Initialize client by discovering agent
        
        Returns:
            True if successful
        """
        if self._initialized:
            return True
        
        try:
            # Discover agent
            if not self.agent_card:
                self.agent_card = await self._client.discover(self.endpoint)
                
                if self.agent_card:
                    self.agent_id = self.agent_card.id
            
            self._initialized = True
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize agent client: {e}")
            return False
    
    async def send(
        self,
        content: Any,
        message_type: MessageType = MessageType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedMessage:
        """
        Send message to agent
        
        Args:
            content: Message content
            message_type: Type of message
            metadata: Additional metadata
            
        Returns:
            Response message
        """
        await self.initialize()
        
        return await self._client.send_message(
            agent_id=self.agent_id,
            content=content,
            message_type=message_type,
            protocol=self.protocol,
            metadata=metadata,
        )
    
    async def execute_task(
        self,
        task_data: Dict[str, Any],
        priority: int = 2,
    ) -> TaskResult:
        """
        Execute task on agent
        
        Args:
            task_data: Task data
            priority: Task priority (1-5)
            
        Returns:
            TaskResult
        """
        await self.initialize()
        
        return await self._client.create_task(
            agent_id=self.agent_id,
            task_data=task_data,
            priority=priority,
            protocol=self.protocol,
        )
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call tool on agent (MCP-style)
        
        Args:
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        await self.initialize()
        
        return await self._client.call_tool(
            agent_id=self.agent_id,
            tool_name=tool_name,
            arguments=arguments,
        )
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """
        Get agent capabilities
        
        Returns:
            List of capabilities
        """
        await self.initialize()
        
        if self.agent_card:
            return self.agent_card.capabilities
        
        return []
    
    async def supports_capability(self, capability_type: CapabilityType) -> bool:
        """
        Check if agent supports a capability
        
        Args:
            capability_type: Capability to check
            
        Returns:
            True if supported
        """
        capabilities = await self.get_capabilities()
        
        for cap in capabilities:
            if cap.type == capability_type and cap.enabled:
                return True
        
        return False


