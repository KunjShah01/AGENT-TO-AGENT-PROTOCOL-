"""
Live Integration Module - Bringing everything together
====================================================

This module integrates all the new features into the existing RL-A2A system:
- Agent Marketplace
- Advanced Analytics
- Plugin System
- Live Dashboard

Author: KUNJ SHAH
Version: 3.1.0 Live
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import our new modules
from marketplace.agent_marketplace import AgentMarketplace
from analytics.advanced_analytics import AdvancedAnalytics, AnalyticsEvent
from plugins.plugin_system import PluginManager

class LiveIntegration:
    """Main integration class for live features"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        
        # Initialize components
        self.marketplace = AgentMarketplace()
        self.analytics = AdvancedAnalytics()
        self.plugin_manager = PluginManager()
        
        # Live features state
        self.live_features_enabled = True
        self.real_time_updates = True
        
    def _setup_logger(self):
        """Setup logging for live integration"""
        logger = logging.getLogger("RL-A2A.Live")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def initialize(self) -> bool:
        """Initialize all live features"""
        try:
            self.logger.info("Initializing RL-A2A Live Features...")
            
            # Setup demo marketplace
            await self._setup_demo_marketplace()
            
            # Load plugins
            await self.plugin_manager.auto_load_plugins()
            
            # Start analytics background tasks
            await self.analytics.start_background_tasks()
            
            self.logger.info("Live features initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize live features: {e}")
            return False
    
    async def _setup_demo_marketplace(self):
        """Setup demo agents in marketplace"""
        demo_agents = [
            {
                "name": "ChatBot Pro",
                "description": "Advanced conversational AI agent with memory and context awareness",
                "author": "RL-A2A Team",
                "version": "2.1.0",
                "category": "conversational",
                "tags": ["chat", "memory", "nlp", "context"]
            },
            {
                "name": "Data Analyzer",
                "description": "Intelligent data analysis and visualization agent with ML capabilities",
                "author": "DataScience Team",
                "version": "1.5.2",
                "category": "analytics",
                "tags": ["data", "analysis", "visualization", "ml"]
            },
            {
                "name": "Code Assistant",
                "description": "AI-powered coding companion with debugging and optimization features",
                "author": "DevTools Team",
                "version": "3.0.1",
                "category": "development",
                "tags": ["coding", "debugging", "optimization", "ai"]
            },
            {
                "name": "Security Guardian",
                "description": "Advanced security monitoring and threat detection agent",
                "author": "Security Team",
                "version": "1.8.0",
                "category": "security",
                "tags": ["security", "monitoring", "threats", "protection"]
            },
            {
                "name": "Performance Optimizer",
                "description": "System performance monitoring and optimization agent",
                "author": "Performance Team",
                "version": "2.3.1",
                "category": "optimization",
                "tags": ["performance", "optimization", "monitoring", "system"]
            }
        ]
        
        for agent_data in demo_agents:
            await self.marketplace.publish_agent(agent_data)
            
        # Add some demo ratings
        agents = await self.marketplace.search_agents()
        for i, agent in enumerate(agents[:3]):
            await self.marketplace.rate_agent(
                agent["id"], 
                f"demo_user_{i}", 
                4.5 + (i * 0.2), 
                f"Great agent! Very useful for {agent['category']} tasks."
            )
    
    async def track_agent_interaction(self, 
                                    agent_id: str, 
                                    user_id: str, 
                                    session_id: str,
                                    interaction_data: Dict[str, Any]):
        """Track agent interaction for analytics"""
        if not self.live_features_enabled:
            return
        
        event = AnalyticsEvent(
            event_id=f"interaction_{datetime.now().timestamp()}",
            event_type="agent_interaction",
            timestamp=datetime.now(),
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            data=interaction_data
        )
        
        await self.analytics.track_event(event)
    
    async def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get real-time analytics
            dashboard_data = await self.analytics.get_real_time_dashboard()
            
            # Get marketplace data
            trending_agents = await self.marketplace.get_trending_agents(5)
            categories = await self.marketplace.get_categories()
            
            # Get plugin data
            plugin_stats = await self.plugin_manager.get_plugin_stats()
            
            # Get performance insights
            insights = await self.analytics.get_performance_insights()
            
            # Combine all data
            live_data = {
                "timestamp": datetime.now().isoformat(),
                "analytics": dashboard_data,
                "marketplace": {
                    "trending_agents": trending_agents,
                    "categories": categories,
                    "total_agents": len(trending_agents)
                },
                "plugins": plugin_stats,
                "insights": insights,
                "system_status": {
                    "live_features": self.live_features_enabled,
                    "real_time_updates": self.real_time_updates,
                    "components": {
                        "marketplace": True,
                        "analytics": True,
                        "plugins": True
                    }
                }
            }
            
            return live_data
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return {"error": str(e)}
    
    async def search_marketplace(self, 
                               query: str = "", 
                               category: str = "", 
                               tags: List[str] = None) -> List[Dict]:
        """Search agents in marketplace"""
        return await self.marketplace.search_agents(query, category, tags)
    
    async def install_marketplace_agent(self, agent_id: str, user_id: str) -> bool:
        """Install agent from marketplace"""
        try:
            # Download agent
            success = await self.marketplace.download_agent(agent_id, user_id)
            
            if success:
                # Track installation
                await self.track_agent_interaction(
                    agent_id, user_id, f"install_session_{datetime.now().timestamp()}",
                    {"action": "install", "source": "marketplace"}
                )
                
                self.logger.info(f"Agent {agent_id} installed by user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to install agent {agent_id}: {e}")
            return False
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin"""
        return await self.plugin_manager.load_plugin(plugin_name)
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        return await self.plugin_manager.unload_plugin(plugin_name)
    
    async def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        return await self.plugin_manager.execute_plugin(plugin_name, *args, **kwargs)
    
    async def get_usage_analytics(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get usage analytics"""
        return await self.analytics.get_usage_metrics(time_range)
    
    async def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get analytics for specific agent"""
        return await self.analytics.get_agent_analytics(agent_id)
    
    async def export_data(self, data_type: str = "all") -> str:
        """Export system data"""
        if data_type == "analytics":
            return await self.analytics.export_analytics()
        elif data_type == "marketplace":
            # Export marketplace data
            agents = await self.marketplace.search_agents()
            return json.dumps(agents, indent=2, default=str)
        elif data_type == "plugins":
            # Export plugin data
            plugins = self.plugin_manager.list_plugins()
            return json.dumps(plugins, indent=2, default=str)
        else:
            # Export all data
            all_data = {
                "analytics": await self.analytics.export_analytics(),
                "marketplace": await self.marketplace.search_agents(),
                "plugins": self.plugin_manager.list_plugins(),
                "export_timestamp": datetime.now().isoformat()
            }
            return json.dumps(all_data, indent=2, default=str)
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "metrics": {}
        }
        
        try:
            # Check marketplace
            agents = await self.marketplace.search_agents()
            health["components"]["marketplace"] = {
                "status": "healthy",
                "agent_count": len(agents)
            }
        except Exception as e:
            health["components"]["marketplace"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        try:
            # Check analytics
            dashboard = await self.analytics.get_real_time_dashboard()
            health["components"]["analytics"] = {
                "status": "healthy",
                "active_sessions": dashboard.get("active_sessions", 0)
            }
        except Exception as e:
            health["components"]["analytics"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        try:
            # Check plugins
            plugin_stats = await self.plugin_manager.get_plugin_stats()
            health["components"]["plugins"] = {
                "status": "healthy",
                "loaded_plugins": plugin_stats["loaded_plugins"]
            }
        except Exception as e:
            health["components"]["plugins"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"
        
        return health

# FastAPI integration endpoints
def create_live_api_routes(app, live_integration: LiveIntegration):
    """Create API routes for live features"""
    
    @app.get("/api/live/dashboard")
    async def get_dashboard():
        """Get live dashboard data"""
        return await live_integration.get_live_dashboard_data()
    
    @app.get("/api/live/marketplace/search")
    async def search_marketplace(query: str = "", category: str = "", tags: str = ""):
        """Search marketplace agents"""
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
        return await live_integration.search_marketplace(query, category, tag_list)
    
    @app.post("/api/live/marketplace/install/{agent_id}")
    async def install_agent(agent_id: str, user_id: str):
        """Install agent from marketplace"""
        success = await live_integration.install_marketplace_agent(agent_id, user_id)
        return {"success": success, "agent_id": agent_id}
    
    @app.get("/api/live/plugins")
    async def list_plugins():
        """List all plugins"""
        return live_integration.plugin_manager.list_plugins()
    
    @app.post("/api/live/plugins/{plugin_name}/load")
    async def load_plugin(plugin_name: str):
        """Load a plugin"""
        success = await live_integration.load_plugin(plugin_name)
        return {"success": success, "plugin": plugin_name}
    
    @app.post("/api/live/plugins/{plugin_name}/unload")
    async def unload_plugin(plugin_name: str):
        """Unload a plugin"""
        success = await live_integration.unload_plugin(plugin_name)
        return {"success": success, "plugin": plugin_name}
    
    @app.get("/api/live/analytics/{time_range}")
    async def get_analytics(time_range: str = "24h"):
        """Get usage analytics"""
        return await live_integration.get_usage_analytics(time_range)
    
    @app.get("/api/live/health")
    async def health_check():
        """Health check endpoint"""
        return await live_integration.health_check()

# Example usage
async def demo_live_integration():
    """Demo the live integration"""
    live = LiveIntegration()
    
    # Initialize
    await live.initialize()
    
    # Get dashboard data
    dashboard = await live.get_live_dashboard_data()
    print("Dashboard Data:", json.dumps(dashboard, indent=2, default=str))
    
    # Search marketplace
    agents = await live.search_marketplace("chat")
    print(f"Found {len(agents)} chat agents")
    
    # Health check
    health = await live.health_check()
    print("Health Status:", health["status"])

if __name__ == "__main__":
    asyncio.run(demo_live_integration())