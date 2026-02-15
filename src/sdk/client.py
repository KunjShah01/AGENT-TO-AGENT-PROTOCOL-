"""
Unified A2A Client

Main client for interacting with the Unified A2A Protocol.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid

from ..protocols.unified.message import UnifiedMessage, MessageType, ProtocolType, TaskStatus
from ..protocols.unified.agent_card import AgentCard, AgentCapability, CapabilityType
from ..protocols.unified.task import Task, TaskResult
from ..protocols.unified.protocol import UnifiedProtocol, ProtocolAdapter
from ..protocols.adapters import A2AAdapter, MCPAdapter, AutoGenAdapter, LangGraphAdapter, CrewAIAdapter


class UnifiedClient:
    """
    Unified A2A Client
    
    Main client for interacting with agents using the Unified A2A Protocol.
    
    Features:
    - Multi-protocol support
    - Agent discovery
    - Task management
    - Async/await support
    - Automatic protocol detection
    
    Example:
        >>> client = UnifiedClient()
        >>> 
        >>> # Discover agent
        >>> agent = await client.discover("https://agent.example.com")
        >>> 
        >>> # Send message
        >>> response = await client.send_message(
        ...     agent_id=agent.id,
        ...     content="Hello!"
        ... )
        >>> 
        >>> # Create task
        >>> task = await client.create_task(
        ...     agent_id=agent.id,
        ...     task_data={"prompt": "Process this data"}
        ... )
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_card: Optional[AgentCard] = None,
        default_protocol: ProtocolType = ProtocolType.A2A,
        timeout: float = 30.0,
    ):
        """
        Initialize Unified A2A Client
        
        Args:
            agent_id: Client agent ID
            agent_card: Client agent card
            default_protocol: Default protocol to use
            timeout: Request timeout in seconds
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_card = agent_card
        self.default_protocol = default_protocol
        self.timeout = timeout
        
        # Initialize protocol
        self._protocol = UnifiedProtocol()
        self._setup_adapters()
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Logger
        self._logger = logging.getLogger(__name__)
        
        # Discovered agents cache
        self._discovered_agents: Dict[str, AgentCard] = {}
    
    def _setup_adapters(self):
        """Setup protocol adapters"""
        # Register all adapters
        self._protocol.register_adapter(A2AAdapter(self.agent_card))
        self._protocol.register_adapter(MCPAdapter(self.agent_card))
        self._protocol.register_adapter(AutoGenAdapter(self.agent_card))
        self._protocol.register_adapter(LangGraphAdapter(self.agent_card))
        self._protocol.register_adapter(CrewAIAdapter(self.agent_card))
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self._session:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def discover(self, url: str) -> Optional[AgentCard]:
        """
        Discover agent at URL
        
        Args:
            url: Agent URL
            
        Returns:
            AgentCard or None
        """
        try:
            session = await self._get_session()
            
            # Try well-known endpoint first (A2A spec)
            well_known_url = f"{url}/.well-known/agent.json"
            
            async with session.get(well_known_url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    agent_card = AgentCard.from_dict(data)
                    self._discovered_agents[agent_card.id] = agent_card
                    return agent_card
            
            # Try root endpoint
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if "agent_card" in data:
                        agent_card = AgentCard.from_dict(data["agent_card"])
                        self._discovered_agents[agent_card.id] = agent_card
                        return agent_card
            
            return None
            
        except Exception as e:
            self._logger.error(f"Agent discovery failed: {e}")
            return None
    
    async def send_message(
        self,
        agent_id: str,
        content: Any,
        message_type: MessageType = MessageType.TEXT,
        protocol: Optional[ProtocolType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedMessage:
        """
        Send message to agent
        
        Args:
            agent_id: Target agent ID
            content: Message content
            message_type: Type of message
            protocol: Protocol to use (auto-detect if None)
            metadata: Additional metadata
            
        Returns:
            Response message
        """
        # Get agent card
        agent_card = self._discovered_agents.get(agent_id)
        
        # Determine protocol
        if not protocol:
            protocol = self._detect_protocol(agent_card)
        
        # Create message
        message = UnifiedMessage(
            id=str(uuid.uuid4()),
            protocol=protocol,
            message_type=message_type,
            sender_id=self.agent_id,
            receiver_id=agent_id,
            content=content,
            metadata=metadata or {},
        )
        
        # Process through protocol
        response = await self._protocol.process_message(message)
        
        if not response:
            # Create error response
            response = UnifiedMessage(
                id=str(uuid.uuid4()),
                protocol=protocol,
                message_type=MessageType.ERROR,
                sender_id=agent_id,
                receiver_id=self.agent_id,
                content={"error": "No response from agent"},
                correlation_id=message.id,
            )
        
        return response
    
    async def create_task(
        self,
        agent_id: str,
        task_data: Dict[str, Any],
        priority: int = 2,
        protocol: Optional[ProtocolType] = None,
    ) -> Task:
        """
        Create and execute task
        
        Args:
            agent_id: Target agent ID
            task_data: Task data
            priority: Task priority (1-5)
            protocol: Protocol to use
            
        Returns:
            Task
        """
        # Create task
        task = Task(
            id=str(uuid.uuid4()),
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            creator_id=self.agent_id,
            assignee_id=agent_id,
            input_data=task_data,
            priority=priority,
        )
        
        # Execute through protocol
        result = await self._protocol.create_task(task)
        
        return result
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """
        Get task status
        
        Args:
            task_id: Task ID
            
        Returns:
            Task or None
        """
        return self._protocol.get_task(task_id)
    
    async def call_tool(
        self,
        agent_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call tool on agent (MCP-style)
        
        Args:
            agent_id: Target agent ID
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        message = await self.send_message(
            agent_id=agent_id,
            content={"tool": tool_name, "arguments": arguments},
            message_type=MessageType.TOOL_CALL,
            protocol=ProtocolType.MCP,
        )
        
        return message.content
    
    async def join_crew(
        self,
        crew_id: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Join a crew (CrewAI-style)
        
        Args:
            crew_id: Crew ID
            agent_id: Agent ID (defaults to self)
            
        Returns:
            Join result
        """
        message = await self.send_message(
            agent_id=crew_id,
            content={"action": "join"},
            message_type=MessageType.COLLABORATION,
            protocol=ProtocolType.CREWAI,
            metadata={"crewai_method": "crew/join"},
        )
        
        return message.content
    
    async def start_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start workflow (LangGraph-style)
        
        Args:
            workflow_id: Workflow ID
            initial_state: Initial workflow state
            
        Returns:
            Workflow result
        """
        message = await self.send_message(
            agent_id=workflow_id,
            content=initial_state,
            message_type=MessageType.WORKFLOW_EVENT,
            protocol=ProtocolType.LANGGRAPH,
            metadata={"langgraph_method": "workflow/start"},
        )
        
        return message.content
    
    def _detect_protocol(self, agent_card: Optional[AgentCard]) -> ProtocolType:
        """
        Detect best protocol for agent
        
        Args:
            agent_card: Agent card
            
        Returns:
            Protocol type
        """
        if not agent_card:
            return self.default_protocol
        
        # Check capabilities
        for capability in agent_card.capabilities:
            if capability.type == CapabilityType.TASK_EXECUTION:
                return ProtocolType.A2A
            elif capability.type == CapabilityType.TOOLS:
                return ProtocolType.MCP
            elif capability.type == CapabilityType.GROUP_CHAT:
                return ProtocolType.AUTOGEN
            elif capability.type == CapabilityType.WORKFLOW:
                return ProtocolType.LANGGRAPH
            elif capability.type == CapabilityType.COLLABORATION:
                return ProtocolType.CREWAI
        
        return self.default_protocol
    
    def get_stats(self) -> Dict[str, int]:
        """Get client statistics"""
        return self._protocol.get_stats()


