"""
Advanced Analytics - Usage insights and performance monitoring
============================================================

Comprehensive analytics system for RL-A2A providing:
- Real-time usage metrics
- Performance monitoring
- Agent behavior analysis
- System health tracking
- Predictive insights
"""

import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None

@dataclass
class AnalyticsEvent:
    """Analytics event structure"""
    event_id: str
    event_type: str
    timestamp: datetime
    agent_id: str
    user_id: str
    session_id: str
    data: Dict[str, Any]

class AdvancedAnalytics:
    """Advanced analytics and monitoring system"""
    
    def __init__(self, data_dir: str = "analytics/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.events_file = self.data_dir / "events.json"
        self.metrics_file = self.data_dir / "metrics.json"
        self.sessions_file = self.data_dir / "sessions.json"
        
        # In-memory caches for real-time data
        self.real_time_metrics = defaultdict(deque)
        self.active_sessions = {}
        self.performance_cache = {}
        
        # Initialize data files
        self._init_data_files()
        
        # Start background tasks
        self.cleanup_task = None
        self.aggregation_task = None
    
    def _init_data_files(self):
        """Initialize analytics data files"""
        for file_path in [self.events_file, self.metrics_file, self.sessions_file]:
            if not file_path.exists():
                self._save_json(file_path, {})
    
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
    
    async def track_event(self, event: AnalyticsEvent):
        """Track analytics event"""
        # Store event
        events = self._load_json(self.events_file)
        event_data = asdict(event)
        events[event.event_id] = event_data
        self._save_json(self.events_file, events)
        
        # Update real-time metrics
        await self._update_real_time_metrics(event)
        
        # Update session data
        await self._update_session_data(event)
    
    async def _update_real_time_metrics(self, event: AnalyticsEvent):
        """Update real-time metrics cache"""
        current_time = datetime.now()
        
        # Agent usage metrics
        self.real_time_metrics[f"agent_{event.agent_id}_usage"].append(
            MetricPoint(current_time, 1)
        )
        
        # Event type metrics
        self.real_time_metrics[f"event_{event.event_type}"].append(
            MetricPoint(current_time, 1)
        )
        
        # Performance metrics
        if "response_time" in event.data:
            self.real_time_metrics["response_times"].append(
                MetricPoint(current_time, event.data["response_time"])
            )
        
        # Keep only last 1000 points for real-time metrics
        for metric_name, points in self.real_time_metrics.items():
            if len(points) > 1000:
                points.popleft()
    
    async def _update_session_data(self, event: AnalyticsEvent):
        """Update session tracking data"""
        sessions = self._load_json(self.sessions_file)
        
        if event.session_id not in sessions:
            sessions[event.session_id] = {
                "session_id": event.session_id,
                "user_id": event.user_id,
                "start_time": event.timestamp.isoformat(),
                "last_activity": event.timestamp.isoformat(),
                "events_count": 0,
                "agents_used": set(),
                "total_response_time": 0.0
            }
        
        session = sessions[event.session_id]
        session["last_activity"] = event.timestamp.isoformat()
        session["events_count"] += 1
        
        if isinstance(session["agents_used"], set):
            session["agents_used"] = list(session["agents_used"])
        session["agents_used"].append(event.agent_id)
        session["agents_used"] = list(set(session["agents_used"]))
        
        if "response_time" in event.data:
            session["total_response_time"] += event.data["response_time"]
        
        self._save_json(self.sessions_file, sessions)
    
    async def get_usage_metrics(self, 
                              time_range: str = "24h",
                              agent_id: str = None) -> Dict[str, Any]:
        """Get usage metrics for specified time range"""
        end_time = datetime.now()
        
        if time_range == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_range == "24h":
            start_time = end_time - timedelta(days=1)
        elif time_range == "7d":
            start_time = end_time - timedelta(days=7)
        elif time_range == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(days=1)
        
        events = self._load_json(self.events_file)
        filtered_events = []
        
        for event_data in events.values():
            event_time = datetime.fromisoformat(event_data["timestamp"])
            if start_time <= event_time <= end_time:
                if agent_id is None or event_data["agent_id"] == agent_id:
                    filtered_events.append(event_data)
        
        # Calculate metrics
        metrics = {
            "total_events": len(filtered_events),
            "unique_users": len(set(e["user_id"] for e in filtered_events)),
            "unique_sessions": len(set(e["session_id"] for e in filtered_events)),
            "event_types": defaultdict(int),
            "agents_usage": defaultdict(int),
            "hourly_distribution": defaultdict(int),
            "response_times": []
        }
        
        for event in filtered_events:
            metrics["event_types"][event["event_type"]] += 1
            metrics["agents_usage"][event["agent_id"]] += 1
            
            # Hourly distribution
            hour = datetime.fromisoformat(event["timestamp"]).hour
            metrics["hourly_distribution"][hour] += 1
            
            # Response times
            if "response_time" in event["data"]:
                metrics["response_times"].append(event["data"]["response_time"])
        
        # Calculate response time statistics
        if metrics["response_times"]:
            metrics["avg_response_time"] = statistics.mean(metrics["response_times"])
            metrics["median_response_time"] = statistics.median(metrics["response_times"])
            metrics["p95_response_time"] = statistics.quantiles(metrics["response_times"], n=20)[18]
        
        return dict(metrics)
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        # Get recent performance data
        recent_metrics = await self.get_usage_metrics("24h")
        
        insights = {
            "performance_score": 0.0,
            "bottlenecks": [],
            "recommendations": [],
            "trends": {},
            "alerts": []
        }
        
        # Calculate performance score
        if recent_metrics["response_times"]:
            avg_response = recent_metrics["avg_response_time"]
            if avg_response < 1.0:
                insights["performance_score"] = 95
            elif avg_response < 2.0:
                insights["performance_score"] = 85
            elif avg_response < 5.0:
                insights["performance_score"] = 70
            else:
                insights["performance_score"] = 50
                insights["alerts"].append("High response times detected")
        
        # Identify bottlenecks
        if recent_metrics["response_times"]:
            p95_time = recent_metrics.get("p95_response_time", 0)
            if p95_time > 5.0:
                insights["bottlenecks"].append({
                    "type": "high_latency",
                    "description": f"95th percentile response time: {p95_time:.2f}s",
                    "severity": "high"
                })
        
        # Generate recommendations
        if recent_metrics["total_events"] > 1000:
            insights["recommendations"].append({
                "type": "scaling",
                "description": "Consider implementing caching for high-traffic agents",
                "priority": "medium"
            })
        
        # Usage trends
        insights["trends"] = {
            "most_used_agent": max(recent_metrics["agents_usage"].items(), 
                                 key=lambda x: x[1], default=("none", 0))[0],
            "peak_hour": max(recent_metrics["hourly_distribution"].items(), 
                           key=lambda x: x[1], default=(0, 0))[0],
            "growth_rate": self._calculate_growth_rate()
        }
        
        return insights
    
    def _calculate_growth_rate(self) -> float:
        """Calculate usage growth rate"""
        try:
            current_week = datetime.now() - timedelta(days=7)
            previous_week = current_week - timedelta(days=7)
            
            events = self._load_json(self.events_file)
            
            current_count = sum(1 for e in events.values() 
                              if datetime.fromisoformat(e["timestamp"]) >= current_week)
            previous_count = sum(1 for e in events.values() 
                               if previous_week <= datetime.fromisoformat(e["timestamp"]) < current_week)
            
            if previous_count == 0:
                return 100.0 if current_count > 0 else 0.0
            
            return ((current_count - previous_count) / previous_count) * 100
        except:
            return 0.0
    
    async def get_agent_analytics(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed analytics for specific agent"""
        agent_metrics = await self.get_usage_metrics("30d", agent_id)
        
        analytics = {
            "agent_id": agent_id,
            "usage_stats": agent_metrics,
            "performance_profile": {},
            "user_engagement": {},
            "optimization_suggestions": []
        }
        
        # Performance profile
        if agent_metrics["response_times"]:
            analytics["performance_profile"] = {
                "avg_response_time": agent_metrics["avg_response_time"],
                "reliability_score": min(100, max(0, 100 - (agent_metrics["avg_response_time"] * 10))),
                "usage_frequency": agent_metrics["total_events"] / 30  # per day
            }
        
        # User engagement
        sessions = self._load_json(self.sessions_file)
        agent_sessions = [s for s in sessions.values() if agent_id in s.get("agents_used", [])]
        
        if agent_sessions:
            avg_session_length = statistics.mean([
                (datetime.fromisoformat(s["last_activity"]) - 
                 datetime.fromisoformat(s["start_time"])).total_seconds()
                for s in agent_sessions
            ])
            
            analytics["user_engagement"] = {
                "avg_session_length": avg_session_length,
                "repeat_users": len(set(s["user_id"] for s in agent_sessions)),
                "engagement_score": min(100, avg_session_length / 60 * 10)  # Score based on minutes
            }
        
        return analytics
    
    async def get_real_time_dashboard(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        current_time = datetime.now()
        last_hour = current_time - timedelta(hours=1)
        
        dashboard = {
            "timestamp": current_time.isoformat(),
            "active_sessions": len(self.active_sessions),
            "events_last_hour": 0,
            "avg_response_time": 0.0,
            "top_agents": [],
            "system_health": "healthy",
            "alerts": []
        }
        
        # Count recent events
        events = self._load_json(self.events_file)
        recent_events = [
            e for e in events.values()
            if datetime.fromisoformat(e["timestamp"]) >= last_hour
        ]
        
        dashboard["events_last_hour"] = len(recent_events)
        
        # Calculate average response time
        response_times = [
            e["data"]["response_time"] for e in recent_events
            if "response_time" in e["data"]
        ]
        
        if response_times:
            dashboard["avg_response_time"] = statistics.mean(response_times)
        
        # Top agents
        agent_counts = defaultdict(int)
        for event in recent_events:
            agent_counts[event["agent_id"]] += 1
        
        dashboard["top_agents"] = sorted(
            agent_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # System health check
        if dashboard["avg_response_time"] > 5.0:
            dashboard["system_health"] = "degraded"
            dashboard["alerts"].append("High response times detected")
        
        if dashboard["events_last_hour"] > 10000:
            dashboard["alerts"].append("High traffic volume")
        
        return dashboard
    
    async def export_analytics(self, format: str = "json") -> str:
        """Export analytics data"""
        if format == "json":
            export_data = {
                "events": self._load_json(self.events_file),
                "metrics": self._load_json(self.metrics_file),
                "sessions": self._load_json(self.sessions_file),
                "export_timestamp": datetime.now().isoformat()
            }
            return json.dumps(export_data, indent=2, default=str)
        
        # Add other formats (CSV, etc.) as needed
        return ""
    
    async def start_background_tasks(self):
        """Start background analytics tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_old_data())
        self.aggregation_task = asyncio.create_task(self._aggregate_metrics())
    
    async def _cleanup_old_data(self):
        """Clean up old analytics data"""
        while True:
            try:
                cutoff_date = datetime.now() - timedelta(days=90)
                
                # Clean events
                events = self._load_json(self.events_file)
                cleaned_events = {
                    k: v for k, v in events.items()
                    if datetime.fromisoformat(v["timestamp"]) > cutoff_date
                }
                self._save_json(self.events_file, cleaned_events)
                
                # Clean sessions
                sessions = self._load_json(self.sessions_file)
                cleaned_sessions = {
                    k: v for k, v in sessions.items()
                    if datetime.fromisoformat(v["start_time"]) > cutoff_date
                }
                self._save_json(self.sessions_file, cleaned_sessions)
                
                await asyncio.sleep(86400)  # Run daily
            except Exception as e:
                print(f"Cleanup error: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _aggregate_metrics(self):
        """Aggregate metrics for faster queries"""
        while True:
            try:
                # Aggregate hourly, daily, weekly metrics
                # Implementation depends on specific requirements
                await asyncio.sleep(3600)  # Run hourly
            except Exception as e:
                print(f"Aggregation error: {e}")
                await asyncio.sleep(1800)  # Retry in 30 minutes

# Example usage
async def demo_analytics():
    """Demo analytics system"""
    analytics = AdvancedAnalytics()
    
    # Simulate some events
    for i in range(10):
        event = AnalyticsEvent(
            event_id=f"event_{i}",
            event_type="agent_interaction",
            timestamp=datetime.now(),
            agent_id=f"agent_{i % 3}",
            user_id=f"user_{i % 5}",
            session_id=f"session_{i % 3}",
            data={"response_time": 1.5 + (i * 0.1)}
        )
        await analytics.track_event(event)
    
    # Get metrics
    metrics = await analytics.get_usage_metrics("24h")
    insights = await analytics.get_performance_insights()
    dashboard = await analytics.get_real_time_dashboard()
    
    print("Usage Metrics:", json.dumps(metrics, indent=2, default=str))
    print("Performance Insights:", json.dumps(insights, indent=2, default=str))
    print("Dashboard:", json.dumps(dashboard, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(demo_analytics())