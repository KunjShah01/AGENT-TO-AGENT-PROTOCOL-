"""
Agent Discovery Service

Service for discovering and managing agents in the network.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..protocols.unified.agent_card import AgentCard, AgentCapability, CapabilityType


@dataclass
class DiscoveredAgent:
    """Discovered agent information"""
    agent_card: AgentCard
    endpoint: str
    discovered_at: datetime
    last_seen: datetime
    health_status: str = "unknown"
    response_time: Optional[float] = None


class AgentDiscovery:
    """
    Agent Discovery Service
    
    Discovers and manages agents in the network.
    
    Features:
    - Agent discovery via well-known endpoints
    - Health monitoring
    - Capability filtering
    - Caching
    
    Example:
        >>> discovery = AgentDiscovery()
        >>> 
        >>> # Discover agent
        >>> agent = await discovery.discover("https://agent.example.com")
        >>> 
        >>> # Search by capability
        >>> agents = await discovery.find_by_capability(CapabilityType.TOOLS)
        >>> 
        >>> # Health check
        >>> healthy = await discovery.health_check(agent.id)
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        cache_ttl: int = 300,  # 5 minutes
    ):
        """
        Initialize Agent Discovery
        
        Args:
            timeout: Request timeout
            cache_ttl: Cache time-to-live in seconds
        """
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # Discovered agents cache
        self._agents: Dict[str, DiscoveredAgent] = {}
        
        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._logger = logging.getLogger(__name__)
    
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
            url: Agent endpoint URL
            
        Returns:
            AgentCard or None
        """
        try:
            session = await self._get_session()
            
            # Try well-known endpoint (A2A spec)
            well_known_url = f"{url.rstrip('/')}/.well-known/agent.json"
            
            start_time = datetime.now()
            
            async with session.get(well_known_url, timeout=self.timeout) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                if response.status == 200:
                    data = await response.json()
                    agent_card = AgentCard.from_dict(data)
                    
                    # Cache discovered agent
                    self._agents[agent_card.id] = DiscoveredAgent(
                        agent_card=agent_card,
                        endpoint=url,
                        discovered_at=datetime.now(),
                        last_seen=datetime.now(),
                        health_status="healthy",
                        response_time=response_time,
                    )
                    
                    self._logger.info(f"Discovered agent: {agent_card.name} ({agent_card.id})")
                    return agent_card
            
            # Try root endpoint
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    if "agent_card" in data:
                        agent_card = AgentCard.from_dict(data["agent_card"])
                        self._agents[agent_card.id] = DiscoveredAgent(
                            agent_card=agent_card,
                            endpoint=url,
                            discovered_at=datetime.now(),
                            last_seen=datetime.now(),
                            health_status="healthy",
                        )
                        return agent_card
            
            return None
            
        except Exception as e:
            self._logger.error(f"Agent discovery failed for {url}: {e}")
            return None
    
    async def health_check(self, agent_id: str) -> bool:
        """
        Check agent health
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if healthy
        """
        discovered = self._agents.get(agent_id)
        if not discovered:
            return False
        
        try:
            session = await self._get_session()
            
            # Try health endpoint
            health_url = f"{discovered.endpoint}/health"
            
            async with session.get(health_url, timeout=10) as response:
                healthy = response.status == 200
                
                discovered.health_status = "healthy" if healthy else "unhealthy"
                discovered.last_seen = datetime.now()
                
                return healthy
                
        except Exception as e:
            self._logger.error(f"Health check failed for {agent_id}: {e}")
            discovered.health_status = "unreachable"
            return False
    
    async def find_by_capability(
        self,
        capability: CapabilityType,
        healthy_only: bool = True,
    ) -> List[AgentCard]:
        """
        Find agents by capability
        
        Args:
            capability: Capability to search for
            healthy_only: Only return healthy agents
            
        Returns:
            List of agent cards
        """
        results = []
        
        for discovered in self._agents.values():
            # Check health
            if healthy_only and discovered.health_status != "healthy":
                continue
            
            # Check capability
            for cap in discovered.agent_card.capabilities:
                if cap.type == capability and cap.enabled:
                    results.append(discovered.agent_card)
                    break
        
        return results
    
    async def list_agents(
        self,
        healthy_only: bool = False,
    ) -> List[DiscoveredAgent]:
        """
        List all discovered agents
        
        Args:
            healthy_only: Only return healthy agents
            
        Returns:
            List of discovered agents
        """
        if healthy_only:
            return [a for a in self._agents.values() if a.health_status == "healthy"]
        
        return list(self._agents.values())
    
    async def refresh(self, agent_id: str) -> Optional[AgentCard]:
        """
        Refresh agent information
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Updated agent card or None
        """
        discovered = self._agents.get(agent_id)
        if not discovered:
            return None
        
        return await self.discover(discovered.endpoint)
    
    async def clear_cache(self) -> None:
        """Clear discovered agents cache"""
        self._agents.clear()


