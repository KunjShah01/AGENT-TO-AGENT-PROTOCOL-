"""
Agent Marketplace - Community-shared agents
==========================================

A comprehensive marketplace for sharing, discovering, and managing AI agents
within the RL-A2A ecosystem.

Features:
- Agent discovery and search
- Community ratings and reviews
- Version management
- Security validation
- Usage analytics
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AgentMetadata:
    """Agent metadata structure"""
    id: str
    name: str
    description: str
    author: str
    version: str
    category: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    downloads: int = 0
    rating: float = 0.0
    reviews: int = 0
    verified: bool = False
    
class AgentMarketplace:
    """Community agent marketplace"""
    
    def __init__(self, data_dir: str = "marketplace/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.agents_file = self.data_dir / "agents.json"
        self.reviews_file = self.data_dir / "reviews.json"
        self.analytics_file = self.data_dir / "analytics.json"
        
        # Initialize data files
        self._init_data_files()
        
    def _init_data_files(self):
        """Initialize marketplace data files"""
        if not self.agents_file.exists():
            self._save_json(self.agents_file, {})
        if not self.reviews_file.exists():
            self._save_json(self.reviews_file, {})
        if not self.analytics_file.exists():
            self._save_json(self.analytics_file, {
                "total_agents": 0,
                "total_downloads": 0,
                "popular_categories": {},
                "trending_agents": []
            })
    
    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_json(self, file_path: Path, data: Dict):
        """Save JSON data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    async def publish_agent(self, agent_data: Dict) -> str:
        """Publish a new agent to marketplace"""
        agent_id = str(uuid.uuid4())
        
        # Create agent metadata
        metadata = AgentMetadata(
            id=agent_id,
            name=agent_data["name"],
            description=agent_data["description"],
            author=agent_data["author"],
            version=agent_data.get("version", "1.0.0"),
            category=agent_data.get("category", "general"),
            tags=agent_data.get("tags", []),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save agent
        agents = self._load_json(self.agents_file)
        agents[agent_id] = asdict(metadata)
        self._save_json(self.agents_file, agents)
        
        # Update analytics
        await self._update_analytics("agent_published", agent_id)
        
        return agent_id
    
    async def search_agents(self, 
                          query: str = "", 
                          category: str = "", 
                          tags: List[str] = None,
                          sort_by: str = "rating") -> List[Dict]:
        """Search agents in marketplace"""
        agents = self._load_json(self.agents_file)
        results = []
        
        for agent_id, agent_data in agents.items():
            # Apply filters
            if query and query.lower() not in agent_data["name"].lower() and \
               query.lower() not in agent_data["description"].lower():
                continue
                
            if category and agent_data["category"] != category:
                continue
                
            if tags and not any(tag in agent_data["tags"] for tag in tags):
                continue
            
            results.append(agent_data)
        
        # Sort results
        if sort_by == "rating":
            results.sort(key=lambda x: x["rating"], reverse=True)
        elif sort_by == "downloads":
            results.sort(key=lambda x: x["downloads"], reverse=True)
        elif sort_by == "recent":
            results.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return results
    
    async def get_agent(self, agent_id: str) -> Optional[Dict]:
        """Get specific agent details"""
        agents = self._load_json(self.agents_file)
        return agents.get(agent_id)
    
    async def download_agent(self, agent_id: str, user_id: str) -> bool:
        """Download agent and track analytics"""
        agents = self._load_json(self.agents_file)
        
        if agent_id not in agents:
            return False
        
        # Increment download count
        agents[agent_id]["downloads"] += 1
        self._save_json(self.agents_file, agents)
        
        # Track analytics
        await self._update_analytics("agent_downloaded", agent_id, user_id)
        
        return True
    
    async def rate_agent(self, agent_id: str, user_id: str, rating: float, review: str = "") -> bool:
        """Rate and review an agent"""
        if not 1 <= rating <= 5:
            return False
        
        # Load reviews
        reviews = self._load_json(self.reviews_file)
        
        if agent_id not in reviews:
            reviews[agent_id] = []
        
        # Add review
        review_data = {
            "user_id": user_id,
            "rating": rating,
            "review": review,
            "timestamp": datetime.now().isoformat()
        }
        reviews[agent_id].append(review_data)
        self._save_json(self.reviews_file, reviews)
        
        # Update agent rating
        await self._update_agent_rating(agent_id)
        
        return True
    
    async def _update_agent_rating(self, agent_id: str):
        """Update agent's average rating"""
        reviews = self._load_json(self.reviews_file)
        agents = self._load_json(self.agents_file)
        
        if agent_id in reviews and agent_id in agents:
            ratings = [r["rating"] for r in reviews[agent_id]]
            avg_rating = sum(ratings) / len(ratings)
            
            agents[agent_id]["rating"] = round(avg_rating, 2)
            agents[agent_id]["reviews"] = len(ratings)
            self._save_json(self.agents_file, agents)
    
    async def get_trending_agents(self, limit: int = 10) -> List[Dict]:
        """Get trending agents based on recent activity"""
        agents = self._load_json(self.agents_file)
        
        # Calculate trending score (downloads + rating + recency)
        trending = []
        for agent_id, agent_data in agents.items():
            days_old = (datetime.now() - datetime.fromisoformat(agent_data["created_at"])).days
            recency_score = max(0, 30 - days_old) / 30  # Higher for newer agents
            
            trending_score = (
                agent_data["downloads"] * 0.4 +
                agent_data["rating"] * 20 * 0.4 +
                recency_score * 100 * 0.2
            )
            
            trending.append({
                **agent_data,
                "trending_score": trending_score
            })
        
        trending.sort(key=lambda x: x["trending_score"], reverse=True)
        return trending[:limit]
    
    async def get_categories(self) -> Dict[str, int]:
        """Get all categories with agent counts"""
        agents = self._load_json(self.agents_file)
        categories = {}
        
        for agent_data in agents.values():
            category = agent_data["category"]
            categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    async def _update_analytics(self, event: str, agent_id: str, user_id: str = None):
        """Update marketplace analytics"""
        analytics = self._load_json(self.analytics_file)
        
        if event == "agent_published":
            analytics["total_agents"] += 1
        elif event == "agent_downloaded":
            analytics["total_downloads"] += 1
        
        self._save_json(self.analytics_file, analytics)
    
    async def get_analytics(self) -> Dict:
        """Get marketplace analytics"""
        return self._load_json(self.analytics_file)

# Example usage and demo agents
async def setup_demo_marketplace():
    """Setup demo marketplace with sample agents"""
    marketplace = AgentMarketplace()
    
    demo_agents = [
        {
            "name": "ChatBot Pro",
            "description": "Advanced conversational AI agent with memory",
            "author": "AI_Developer",
            "category": "conversational",
            "tags": ["chat", "memory", "nlp"]
        },
        {
            "name": "Data Analyzer",
            "description": "Intelligent data analysis and visualization agent",
            "author": "DataScientist",
            "category": "analytics",
            "tags": ["data", "analysis", "visualization"]
        },
        {
            "name": "Code Assistant",
            "description": "AI-powered coding companion and debugger",
            "author": "CodeMaster",
            "category": "development",
            "tags": ["coding", "debugging", "assistance"]
        }
    ]
    
    for agent_data in demo_agents:
        await marketplace.publish_agent(agent_data)
    
    return marketplace

if __name__ == "__main__":
    # Demo setup
    asyncio.run(setup_demo_marketplace())