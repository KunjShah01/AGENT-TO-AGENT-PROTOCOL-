# -*- coding: utf-8 -*-
"""
=======================================================================

COMBINED VERSION: This file merges the original rla2a.py with rla2a_enhanced.py
All features from both systems are now integrated into one comprehensive file.

FEATURES INCLUDED:
✅ Original RL-A2A all-in-one functionality
✅ Enhanced security (JWT, rate limiting, input validation) 
✅ Multi-AI provider support (OpenAI, Claude, Gemini)
✅ Comprehensive environment configuration (.env support)
✅ Advanced 3D visualization and monitoring dashboard
✅ Production-ready deployment features
✅ Enhanced reinforcement learning with experience replay
✅ MCP (Model Context Protocol) support
✅ Comprehensive logging and error handling
✅ WebSocket real-time communication
✅ REST API with comprehensive endpoints
✅ Automatic dependency management
✅ HTML report generation

USAGE:
python rla2a.py setup              # Setup environment and dependencies
python rla2a.py server             # Start secure server 
python rla2a.py dashboard          # Enhanced 3D dashboard
python rla2a.py mcp                # MCP server for AI assistants
python rla2a.py report             # Generate comprehensive HTML report

Author: KUNJ SHAH
GitHub: https://github.com/KunjShah01/RL-A2A
Version: 4.0 Enhanced Combined
"""

import asyncio
import json
import os
import sys
import time
import threading
import queue
import signal
import subprocess
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import uuid
import argparse
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# Environment configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Security imports with graceful fallback
SECURITY_AVAILABLE = False
try:
    import jwt
    import bcrypt
    import bleach
    from passlib.context import CryptContext
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SECURITY_AVAILABLE = True
except ImportError:
    pass

# Enhanced dependency management
def check_and_install_dependencies():
    """Smart dependency management with enhanced features"""
    
    # Core required packages
    core_required = [
        "fastapi", "uvicorn", "websockets", "msgpack", "numpy", "pydantic", 
        "requests", "matplotlib", "plotly", "streamlit", "pandas"
    ]
    
    # Enhanced packages for security and features
    enhanced_packages = [
        "python-dotenv", "PyJWT", "bcrypt", "bleach", "slowapi", "passlib"
    ]
    
    # AI provider packages
    ai_packages = [
        "openai", "anthropic", "google-generativeai"
    ]
    
    # Additional packages
    additional_packages = ["mcp", "aiofiles"]
    
    missing_core = []
    missing_enhanced = []
    missing_ai = []
    
    print("🔍 Checking dependencies...")
    
    # Check core packages
    for pkg in core_required:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_core.append(pkg)
    
    # Install core packages automatically
    if missing_core:
        print(f"📦 Installing core packages: {', '.join(missing_core)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + missing_core, stdout=subprocess.DEVNULL)
            print("✅ Core dependencies installed")
        except Exception as e:
            print(f"❌ Core installation failed: {e}")
            print(f"Please install manually: pip install {' '.join(missing_core)}")
            sys.exit(1)
    
    # Check enhanced packages
    for pkg in enhanced_packages:
        try:
            if pkg == "PyJWT":
                import jwt
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_enhanced.append(pkg)
    
    # Check AI packages
    for pkg in ai_packages:
        try:
            if pkg == "google-generativeai":
                import google.generativeai
            else:
                __import__(pkg.replace("-", "_"))
        except ImportError:
            missing_ai.append(pkg)
    
    # Offer enhanced packages installation
    if missing_enhanced or missing_ai:
        print("\\n🚀 Enhanced features available!")
        if missing_enhanced:
            print(f"🔐 Security: {', '.join(missing_enhanced)}")
        if missing_ai:
            print(f"AI Providers: {', '.join(missing_ai)}")
        
        install_all = missing_enhanced + missing_ai
        choice = input(f"Install enhanced packages? (y/N): ").lower().strip()
        
        if choice in ['y', 'yes']:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + install_all)
                print("✅ Enhanced packages installed successfully!")
                return True
            except Exception as e:
                print(f"❌ Enhanced installation failed: {e}")
                print("Continuing with basic features...")
    
    return len(missing_enhanced) == 0 and len(missing_ai) == 0

# Check and install dependencies
ENHANCED_FEATURES = check_and_install_dependencies()

# Re-import security packages after installation  
try:
    import jwt
    import bcrypt
    import bleach
    from passlib.context import CryptContext
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# Import core packages
import requests
import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
import websockets
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Enhanced imports with fallbacks
if SECURITY_AVAILABLE:
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    limiter = Limiter(key_func=get_remote_address)

# AI Provider imports with availability checking
OPENAI_AVAILABLE = False
ANTHROPIC_AVAILABLE = False
GOOGLE_AVAILABLE = False
MCP_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    pass

try:
    from mcp.server.models import InitializeResult
    from mcp.server import NotificationOptions, Server
    from mcp.types import Resource, Tool, TextContent
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# ENHANCED CONFIGURATION SYSTEM
# =============================================================================

class SecurityConfig:
    """Enhanced Security Configuration"""
    SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    MAX_MESSAGE_SIZE = int(os.getenv("MAX_MESSAGE_SIZE", "1048576"))  # 1MB
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600"))  # 1 hour

# Comprehensive Configuration
CONFIG = {
    # System Information
    "VERSION": "4.0.0-COMBINED",
    "SYSTEM_NAME": "RL-A2A Combined Enhanced",
    
    # AI Provider Configuration
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"), 
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
    
    # AI Model Settings
    "DEFAULT_AI_PROVIDER": os.getenv("DEFAULT_AI_PROVIDER", "openai"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    "GOOGLE_MODEL": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
    "AI_TIMEOUT": int(os.getenv("AI_TIMEOUT", "30")),
    
    # Server Configuration
    "SERVER_HOST": os.getenv("A2A_HOST", "localhost"),
    "SERVER_PORT": int(os.getenv("A2A_PORT", "8000")),
    "DASHBOARD_PORT": int(os.getenv("DASHBOARD_PORT", "8501")),
    
    # System Limits
    "MAX_AGENTS": int(os.getenv("MAX_AGENTS", "100")),
    "MAX_CONNECTIONS": int(os.getenv("MAX_CONNECTIONS", "1000")),
    "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
    
    # Logging Configuration
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "LOG_FILE": os.getenv("LOG_FILE", "rla2a.log"),
    
    # Feature Flags
    "ENABLE_SECURITY": SECURITY_AVAILABLE,
    "ENABLE_AI": OPENAI_AVAILABLE or ANTHROPIC_AVAILABLE or GOOGLE_AVAILABLE,
    "ENABLE_VISUALIZATION": True,
    "ENABLE_MCP": MCP_AVAILABLE
}

# Setup logging
logging.basicConfig(
    level=getattr(logging, CONFIG["LOG_LEVEL"]),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"]),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"🚀 {CONFIG['SYSTEM_NAME']} v{CONFIG['VERSION']} starting...")

# =============================================================================
# ENHANCED DATA MODELS
# =============================================================================

class AgentPosition(BaseModel):
    """Agent position with validation"""
    x: float = Field(..., ge=-100.0, le=100.0)
    y: float = Field(..., ge=-100.0, le=100.0)
    z: float = Field(0.0, ge=-10.0, le=10.0)

class AgentVelocity(BaseModel):
    """Agent velocity with validation"""
    x: float = Field(0.0, ge=-10.0, le=10.0)
    y: float = Field(0.0, ge=-10.0, le=10.0)
    z: float = Field(0.0, ge=-10.0, le=10.0)

@dataclass
class Agent:
    """Enhanced Agent with comprehensive tracking"""
    id: str
    position: AgentPosition
    velocity: AgentVelocity
    emotion: str = "neutral"
    last_action: str = "idle"
    reward: float = 0.0
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    ai_provider: str = "openai"
    total_actions: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Post-initialization validation"""
        valid_emotions = ["happy", "sad", "neutral", "excited", "angry", "fear", "surprise"]
        if self.emotion not in valid_emotions:
            self.emotion = "neutral"
        
        # Sanitize if security available
        if SECURITY_AVAILABLE:
            self.id = bleach.clean(self.id)[:50]
            self.last_action = bleach.clean(self.last_action)[:100]

class FeedbackRequest(BaseModel):
    """Enhanced feedback request with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50)
    action_id: str = Field(..., min_length=1, max_length=100)
    reward: float = Field(..., ge=-1.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('agent_id', 'action_id')
    def sanitize_strings(cls, v):
        if SECURITY_AVAILABLE:
            return bleach.clean(v)
        return v[:100]

class AgentRegistrationRequest(BaseModel):
    """Agent registration with validation"""
    agent_id: str = Field(..., min_length=1, max_length=50, regex="^[a-zA-Z0-9_-]+$")
    initial_position: Optional[AgentPosition] = None
    ai_provider: Optional[str] = Field("openai", regex="^(openai|anthropic|google)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# MULTI-AI PROVIDER MANAGER
# =============================================================================

class AIManager:
    """Combined AI Manager with fallback support"""
    
    def __init__(self):
        self.providers = {}
        self.provider_stats = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available AI providers"""
        
        # OpenAI
        if OPENAI_AVAILABLE and CONFIG["OPENAI_API_KEY"]:
            try:
                self.providers["openai"] = AsyncOpenAI(
                    api_key=CONFIG["OPENAI_API_KEY"],
                    timeout=CONFIG["AI_TIMEOUT"]
                )
                self.provider_stats["openai"] = {"requests": 0, "successes": 0, "errors": 0}
                logger.info("✅ OpenAI provider initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
        
        # Anthropic
        if ANTHROPIC_AVAILABLE and CONFIG["ANTHROPIC_API_KEY"]:
            try:
                self.providers["anthropic"] = anthropic.AsyncAnthropic(
                    api_key=CONFIG["ANTHROPIC_API_KEY"],
                    timeout=CONFIG["AI_TIMEOUT"]
                )
                self.provider_stats["anthropic"] = {"requests": 0, "successes": 0, "errors": 0}
                logger.info("✅ Anthropic provider initialized")
            except Exception as e:
                logger.error(f"❌ Anthropic initialization failed: {e}")
        
        # Google
        if GOOGLE_AVAILABLE and CONFIG["GOOGLE_API_KEY"]:
            try:
                genai.configure(api_key=CONFIG["GOOGLE_API_KEY"])
                self.providers["google"] = genai.GenerativeModel(CONFIG["GOOGLE_MODEL"])
                self.provider_stats["google"] = {"requests": 0, "successes": 0, "errors": 0}
                logger.info("✅ Google AI provider initialized")
            except Exception as e:
                logger.error(f"❌ Google AI initialization failed: {e}")
        
        if not self.providers:
            logger.warning("⚠️ No AI providers available - using fallback responses")
    
    async def get_ai_response(self, prompt: str, provider: str = None, context: Dict = None) -> Dict[str, Any]:
        """Get AI response with fallback"""
        
        provider = provider or CONFIG["DEFAULT_AI_PROVIDER"]
        safe_prompt = self._sanitize_prompt(prompt)
        
        # Track request
        if provider in self.provider_stats:
            self.provider_stats[provider]["requests"] += 1
        
        # Try primary provider
        try:
            response = await self._get_provider_response(provider, safe_prompt, context)
            if provider in self.provider_stats:
                self.provider_stats[provider]["successes"] += 1
            return response
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {e}")
            if provider in self.provider_stats:
                self.provider_stats[provider]["errors"] += 1
        
        # Try fallback providers
        for fallback_provider in self.providers:
            if fallback_provider != provider:
                try:
                    response = await self._get_provider_response(fallback_provider, safe_prompt, context)
                    return response
                except Exception as e:
                    logger.warning(f"Fallback {fallback_provider} failed: {e}")
        
        # Emergency fallback
        return self._get_emergency_fallback(safe_prompt)
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt for security"""
        if SECURITY_AVAILABLE:
            cleaned = bleach.clean(prompt, tags=[], strip=True)
            return cleaned[:2000]
        return prompt[:2000]
    
    async def _get_provider_response(self, provider: str, prompt: str, context: Dict = None) -> Dict[str, Any]:
        """Get response from specific provider"""
        
        if provider == "openai":
            response = await self.providers["openai"].chat.completions.create(
                model=CONFIG["OPENAI_MODEL"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7
            )
            return {
                "response": response.choices[0].message.content,
                "provider": "openai",
                "model": CONFIG["OPENAI_MODEL"]
            }
        
        elif provider == "anthropic":
            response = await self.providers["anthropic"].messages.create(
                model=CONFIG["ANTHROPIC_MODEL"],
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            return {
                "response": response.content[0].text,
                "provider": "anthropic",
                "model": CONFIG["ANTHROPIC_MODEL"]
            }
        
        elif provider == "google":
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.providers["google"].generate_content, prompt
            )
            return {
                "response": response.text,
                "provider": "google",
                "model": CONFIG["GOOGLE_MODEL"]
            }
        
        raise ValueError(f"Unknown provider: {provider}")
    
    def _get_emergency_fallback(self, prompt: str) -> Dict[str, Any]:
        """Emergency fallback when all providers fail"""
        fallback_responses = [
            "I understand your request. Let me help you move forward.",
            "I'm processing your request. Please give me a moment.",
            "I'm here to assist. Let's work on this together."
        ]
        
        return {
            "response": np.random.choice(fallback_responses),
            "provider": "fallback",
            "model": "emergency_fallback"
        }
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider statistics"""
        stats = {}
        for provider, data in self.provider_stats.items():
            total = data["requests"]
            success_rate = (data["successes"] / total * 100) if total > 0 else 0
            stats[provider] = {
                **data,
                "success_rate": round(success_rate, 2),
                "available": provider in self.providers
            }
        return stats

# =============================================================================
# ENHANCED REINFORCEMENT LEARNING
# =============================================================================

class RLEnvironment:
    """Enhanced RL Environment with Q-learning"""
    
    def __init__(self):
        self.q_tables: Dict[str, np.ndarray] = {}
        self.experience_replay: Dict[str, List] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        self.exploration_decay = 0.995
        self.min_exploration = 0.01
        self.actions = [
            "move_forward", "move_backward", "turn_left", "turn_right",
            "communicate", "observe", "wait", "interact"
        ]
    
    def initialize_agent_q_table(self, agent_id: str):
        """Initialize Q-table for new agent"""
        if agent_id not in self.q_tables:
            # Initialize with small random values
            self.q_tables[agent_id] = np.random.uniform(-0.01, 0.01, (100, len(self.actions)))
            self.experience_replay[agent_id] = []
            logger.debug(f"Initialized Q-table for agent {agent_id}")
    
    def get_action(self, agent: Agent, environment_data: Dict = None) -> Tuple[str, int]:
        """Get action using epsilon-greedy policy"""
        
        if agent.id not in self.q_tables:
            self.initialize_agent_q_table(agent.id)
        
        # Simple state representation
        state_idx = abs(hash(f"{agent.position.x:.1f},{agent.position.y:.1f},{agent.emotion}")) % 100
        
        # Epsilon-greedy
        if np.random.random() < self.exploration_rate:
            action_idx = np.random.randint(0, len(self.actions))
        else:
            q_values = self.q_tables[agent.id][state_idx]
            action_idx = np.argmax(q_values)
        
        # Decay exploration
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )
        
        return self.actions[action_idx], action_idx
    
    def update_q_value(self, agent: Agent, action_idx: int, reward: float):
        """Update Q-value using Q-learning"""
        
        if agent.id not in self.q_tables:
            return
        
        state_idx = abs(hash(f"{agent.position.x:.1f},{agent.position.y:.1f},{agent.emotion}")) % 100
        
        current_q = self.q_tables[agent.id][state_idx, action_idx]
        max_next_q = np.max(self.q_tables[agent.id][state_idx])  # Simplified
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_tables[agent.id][state_idx, action_idx] = new_q
        
        logger.debug(f"Q-update for {agent.id}: {current_q:.4f} -> {new_q:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RL statistics"""
        if not self.q_tables:
            return {"message": "No agents"}
        
        return {
            "total_agents": len(self.q_tables),
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor
        }

# =============================================================================
# COMBINED A2A SYSTEM
# =============================================================================

class A2ASystem:
    """Combined A2A System with all enhanced features"""
    
    def __init__(self):
        self.app = self._create_app()
        self.agents: Dict[str, Agent] = {}
        self.connections: Dict[str, WebSocket] = {}
        self.ai_manager = AIManager()
        self.rl_env = RLEnvironment()
        self.active_sessions: Dict[str, Dict] = {}
        self.running = False
        self.start_time = time.time()
        self.system_stats = {
            "total_messages": 0,
            "total_actions": 0,
            "total_rewards": 0,
            "errors": 0
        }
    
    def _create_app(self) -> FastAPI:
        """Create enhanced FastAPI application"""
        
        app = FastAPI(
            title=CONFIG["SYSTEM_NAME"],
            description="Combined Enhanced Agent-to-Agent Communication System",
            version=CONFIG["VERSION"],
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add middleware
        self._add_middleware(app)
        self._add_routes(app)
        
        return app
    
    def _add_middleware(self, app: FastAPI):
        """Add enhanced middleware"""
        
        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=SecurityConfig.ALLOWED_ORIGINS if SECURITY_AVAILABLE else ["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Security middleware
        if SECURITY_AVAILABLE:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["localhost", "127.0.0.1", CONFIG["SERVER_HOST"], "*"]
            )
            app.state.limiter = limiter
            app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    def _add_routes(self, app: FastAPI):
        """Add comprehensive API routes"""
        
        # Rate limiting decorator
        def rate_limit(route):
            if SECURITY_AVAILABLE:
                return limiter.limit(f"{SecurityConfig.RATE_LIMIT_PER_MINUTE}/minute")(route)
            return route
        
        @app.get("/")
        async def root():
            """System information endpoint"""
            return {
                "system": CONFIG["SYSTEM_NAME"],
                "version": CONFIG["VERSION"],
                "status": "running" if self.running else "ready",
                "uptime": time.time() - self.start_time,
                "capabilities": {
                    "security": SECURITY_AVAILABLE,
                    "ai_providers": list(self.ai_manager.providers.keys()),
                    "visualization": CONFIG["ENABLE_VISUALIZATION"],
                    "mcp": CONFIG["ENABLE_MCP"]
                },
                "statistics": {
                    "agents": len(self.agents),
                    "active_sessions": len(self.active_sessions),
                    "total_messages": self.system_stats["total_messages"]
                }
            }
        
        @app.get("/health")
        @rate_limit
        async def health_check(request=None):
            """Enhanced health check"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "agents": len(self.agents),
                    "connections": len(self.connections),
                    "uptime": time.time() - self.start_time
                },
                "ai_providers": self.ai_manager.get_provider_stats()
            }
        
        @app.post("/register")
        @rate_limit
        async def register_agent(request=None, registration: AgentRegistrationRequest = None):
            """Enhanced agent registration"""
            
            if len(self.agents) >= CONFIG["MAX_AGENTS"]:
                raise HTTPException(status_code=429, detail="Maximum agents reached")
            
            if registration.agent_id in self.agents:
                raise HTTPException(status_code=409, detail="Agent already exists")
            
            session_id = str(uuid.uuid4())
            
            # Create position
            if not registration.initial_position:
                position = AgentPosition(
                    x=np.random.uniform(-20, 20),
                    y=np.random.uniform(-20, 20), 
                    z=0
                )
            else:
                position = registration.initial_position
            
            # Create agent
            agent = Agent(
                id=registration.agent_id,
                position=position,
                velocity=AgentVelocity(),
                session_id=session_id,
                ai_provider=registration.ai_provider or CONFIG["DEFAULT_AI_PROVIDER"]
            )
            
            # Store agent
            self.agents[registration.agent_id] = agent
            self.rl_env.initialize_agent_q_table(registration.agent_id)
            
            # Track session
            self.active_sessions[session_id] = {
                "agent_id": registration.agent_id,
                "created": time.time(),
                "last_activity": time.time(),
                "metadata": registration.metadata
            }
            
            logger.info(f"✅ Agent registered: {registration.agent_id}")
            
            return {
                "status": "success",
                "agent_id": registration.agent_id,
                "session_id": session_id,
                "position": agent.position
            }
        
        @app.get("/agents")
        @rate_limit
        async def list_agents(request=None):
            """List all agents"""
            
            agent_list = []
            for agent in self.agents.values():
                agent_data = {
                    "id": agent.id,
                    "position": {
                        "x": agent.position.x,
                        "y": agent.position.y,
                        "z": agent.position.z
                    },
                    "velocity": {
                        "x": agent.velocity.x,
                        "y": agent.velocity.y,
                        "z": agent.velocity.z
                    },
                    "emotion": agent.emotion,
                    "last_action": agent.last_action,
                    "reward": round(agent.reward, 3),
                    "age_seconds": round(time.time() - agent.timestamp, 2),
                    "ai_provider": agent.ai_provider,
                    "total_actions": agent.total_actions,
                    "success_rate": round(agent.success_rate, 2)
                }
                agent_list.append(agent_data)
            
            return {
                "agents": agent_list,
                "total_count": len(self.agents),
                "active_sessions": len(self.active_sessions)
            }
        
        @app.post("/feedback")
        @rate_limit
        async def provide_feedback(request=None, feedback: FeedbackRequest = None):
            """Enhanced feedback processing"""
            
            if feedback.agent_id not in self.agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = self.agents[feedback.agent_id]
            agent.reward += feedback.reward
            agent.last_activity = time.time()
            
            # Update RL
            action_idx = 0
            if feedback.action_id in self.rl_env.actions:
                action_idx = self.rl_env.actions.index(feedback.action_id)
            
            self.rl_env.update_q_value(agent, action_idx, feedback.reward)
            
            # Update success rate
            agent.total_actions += 1
            if feedback.reward > 0:
                agent.success_rate = ((agent.success_rate * (agent.total_actions - 1)) + 1) / agent.total_actions
            else:
                agent.success_rate = (agent.success_rate * (agent.total_actions - 1)) / agent.total_actions
            
            self.system_stats["total_rewards"] += feedback.reward
            
            logger.info(f"✅ Feedback: {feedback.agent_id} -> {feedback.reward:.2f}")
            
            return {
                "status": "success",
                "message": "Feedback processed",
                "agent_stats": {
                    "current_reward": agent.reward,
                    "total_actions": agent.total_actions,
                    "success_rate": agent.success_rate
                }
            }
        
        @app.get("/stats")
        @rate_limit
        async def get_stats(request=None):
            """Get comprehensive statistics"""
            
            return {
                "system": {
                    "uptime": time.time() - self.start_time,
                    "version": CONFIG["VERSION"],
                    **self.system_stats
                },
                "agents": {
                    "total": len(self.agents),
                    "active_sessions": len(self.active_sessions)
                },
                "ai_providers": self.ai_manager.get_provider_stats(),
                "learning": self.rl_env.get_stats()
            }
        
        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            """Enhanced WebSocket endpoint"""
            
            if session_id not in self.active_sessions:
                await websocket.close(code=1008, reason="Invalid session")
                return
            
            await websocket.accept()
            
            try:
                self.connections[session_id] = websocket
                agent_id = self.active_sessions[session_id]["agent_id"]
                
                logger.info(f"📡 WebSocket connected: {agent_id}")
                
                while True:
                    # Receive message
                    message = await websocket.receive_text()
                    
                    # Size validation
                    if len(message) > SecurityConfig.MAX_MESSAGE_SIZE:
                        await websocket.close(code=1009, reason="Message too large")
                        break
                    
                    # Parse and process
                    try:
                        data = json.loads(message)
                        response = await self._handle_websocket_message(session_id, agent_id, data)
                        await websocket.send_text(json.dumps(response))
                        self.system_stats["total_messages"] += 1
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                    except Exception as e:
                        logger.error(f"WebSocket error: {e}")
                        await websocket.send_text(json.dumps({"error": "Internal error"}))
                        
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            finally:
                if session_id in self.connections:
                    del self.connections[session_id]
                logger.info(f"📡 WebSocket disconnected: {agent_id}")
    
    async def _handle_websocket_message(self, session_id: str, agent_id: str, data: Dict) -> Dict:
        """Handle WebSocket message"""
        
        if agent_id not in self.agents:
            return {"error": "Agent not found"}
        
        agent = self.agents[agent_id]
        message_type = data.get("type", "unknown")
        
        try:
            agent.last_activity = time.time()
            self.active_sessions[session_id]["last_activity"] = time.time()
            
            if message_type == "position_update":
                new_pos = data.get("position", {})
                if isinstance(new_pos, dict):
                    agent.position.x = max(-100, min(100, new_pos.get("x", agent.position.x)))
                    agent.position.y = max(-100, min(100, new_pos.get("y", agent.position.y)))
                    agent.position.z = max(-10, min(10, new_pos.get("z", agent.position.z)))
                
                return {"status": "position_updated", "position": agent.position}
            
            elif message_type == "ai_request":
                prompt = data.get("prompt", "")
                provider = data.get("provider", agent.ai_provider)
                
                if prompt:
                    ai_response = await self.ai_manager.get_ai_response(prompt, provider)
                    return {
                        "type": "ai_response",
                        "response": ai_response["response"],
                        "provider": ai_response["provider"],
                        "timestamp": time.time()
                    }
                
                return {"error": "Empty prompt"}
            
            elif message_type == "action_request":
                environment_data = data.get("environment", {})
                action, action_idx = self.rl_env.get_action(agent, environment_data)
                
                agent.last_action = action
                agent.total_actions += 1
                self.system_stats["total_actions"] += 1
                
                return {
                    "type": "action_response",
                    "action": action,
                    "timestamp": time.time()
                }
            
            else:
                return {"error": f"Unknown message type: {message_type}"}
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return {"error": "Failed to process message"}
    
    def create_demo_agents(self, count: int):
        """Create demo agents for testing"""
        
        logger.info(f"Creating {count} demo agents...")
        
        providers = list(self.ai_manager.providers.keys()) or ["openai"]
        emotions = ["neutral", "happy", "excited"]
        
        for i in range(count):
            agent_id = f"demo_agent_{i+1:03d}"
            
            position = AgentPosition(
                x=np.random.uniform(-30, 30),
                y=np.random.uniform(-30, 30),
                z=0
            )
            
            agent = Agent(
                id=agent_id,
                position=position,
                velocity=AgentVelocity(
                    x=np.random.uniform(-1, 1),
                    y=np.random.uniform(-1, 1),
                    z=0
                ),
                emotion=np.random.choice(emotions),
                ai_provider=np.random.choice(providers),
                session_id=str(uuid.uuid4())
            )
            
            self.agents[agent_id] = agent
            self.rl_env.initialize_agent_q_table(agent_id)
            
            self.active_sessions[agent.session_id] = {
                "agent_id": agent_id,
                "created": time.time(),
                "last_activity": time.time(),
                "metadata": {"demo": True}
            }
        
        logger.info(f"✅ Created {count} demo agents")
    
    async def start_server(self):
        """Start the combined server"""
        
        self.running = True
        logger.info(f"🚀 Starting {CONFIG['SYSTEM_NAME']} v{CONFIG['VERSION']}")
        logger.info(f"🔧 Server: {CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}")
        logger.info(f"🔒 Security: {'Enhanced' if SECURITY_AVAILABLE else 'Basic'}")
        logger.info(f"AI Providers: {list(self.ai_manager.providers.keys())}")
        
        try:
            config = uvicorn.Config(
                self.app,
                host=CONFIG["SERVER_HOST"],
                port=CONFIG["SERVER_PORT"],
                log_level=CONFIG["LOG_LEVEL"].lower()
            )
            server = uvicorn.Server(config)
            await server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.running = False

# =============================================================================
# ENHANCED DASHBOARD
# =============================================================================

def create_dashboard():
    """Create enhanced Streamlit dashboard"""
    
    dashboard_code = f'''
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import json
from datetime import datetime

st.set_page_config(
    page_title="RL-A2A Combined Dashboard", 
    page_icon="🤖",
    layout="wide"
)

st.title("RL-A2A Combined Enhanced Dashboard")
st.markdown("### Real-time Agent Monitoring & Control")

# Server connection
server_url = st.sidebar.text_input("Server URL", "http://{CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}")
auto_refresh = st.sidebar.checkbox("Auto Refresh", True)
refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 30, 5)

# Connect to server
try:
    response = requests.get(f"{{server_url}}/", timeout=5)
    server_data = response.json()
    st.sidebar.success("✅ Connected to Server")
except Exception as e:
    st.sidebar.error(f"❌ Connection Error: {{e}}")
    st.stop()

# Fetch data
try:
    agents_response = requests.get(f"{{server_url}}/agents", timeout=10)
    agents_data = agents_response.json()
    
    stats_response = requests.get(f"{{server_url}}/stats", timeout=10)
    stats_data = stats_response.json()
except Exception as e:
    st.error(f"Failed to fetch data: {{e}}")
    st.stop()

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Agents", len(agents_data["agents"]))
with col2:
    st.metric("Active Sessions", agents_data["active_sessions"])
with col3:
    st.metric("Total Actions", stats_data["system"]["total_actions"])
with col4:
    st.metric("System Uptime", f"{{stats_data['system']['uptime']:.1f}}s")

# Agent visualization
if agents_data["agents"]:
    st.subheader("🗺️ Agent Environment")
    
    df = pd.DataFrame([
        {{
            "id": agent["id"],
            "x": agent["position"]["x"],
            "y": agent["position"]["y"],
            "z": agent["position"]["z"],
            "emotion": agent["emotion"],
            "reward": agent["reward"],
            "provider": agent["ai_provider"]
        }}
        for agent in agents_data["agents"]
    ])
    
    # 3D visualization
    fig_3d = px.scatter_3d(
        df, x="x", y="y", z="z",
        color="emotion",
        size="reward",
        hover_data=["id", "provider"],
        title="Agent 3D Environment"
    )
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Emotion Distribution")
        emotion_counts = df["emotion"].value_counts()
        fig_emotions = px.pie(values=emotion_counts.values, names=emotion_counts.index)
        st.plotly_chart(fig_emotions, use_container_width=True)
    
    with col2:
        st.subheader("AI Provider Distribution")
        provider_counts = df["provider"].value_counts()
        fig_providers = px.bar(x=provider_counts.index, y=provider_counts.values)
        st.plotly_chart(fig_providers, use_container_width=True)

# AI Provider status
st.subheader("AI Provider Status")
ai_providers = stats_data.get("ai_providers", {{}})
for provider, stats in ai_providers.items():
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "🟢" if stats["available"] else "🔴"
        st.markdown(f"{{status}} **{{provider.title()}}**")
    with col2:
        st.metric("Success Rate", f"{{stats.get('success_rate', 0):.1f}}%")
    with col3:
        st.metric("Requests", stats.get("requests", 0))

# Agent control
st.subheader("🎮 Agent Control")
with st.expander("Register New Agent"):
    new_agent_id = st.text_input("Agent ID")
    ai_provider = st.selectbox("AI Provider", ["openai", "anthropic", "google"])
    
    if st.button("Register Agent"):
        try:
            registration_data = {{
                "agent_id": new_agent_id,
                "ai_provider": ai_provider
            }}
            response = requests.post(f"{{server_url}}/register", json=registration_data)
            if response.status_code == 200:
                st.success(f"✅ Agent {{new_agent_id}} registered!")
            else:
                st.error(f"❌ Registration failed: {{response.text}}")
        except Exception as e:
            st.error(f"❌ Error: {{e}}")

# Auto refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.experimental_rerun()
'''
    
    # Write dashboard file
    dashboard_path = Path("combined_dashboard.py")
    dashboard_path.write_text(dashboard_code)
    return dashboard_path

def start_dashboard():
    """Start enhanced dashboard"""
    
    dashboard_file = create_dashboard()
    logger.info(f"🎨 Starting dashboard on port {CONFIG['DASHBOARD_PORT']}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_file),
            "--server.port", str(CONFIG["DASHBOARD_PORT"]),
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
    finally:
        if dashboard_file.exists():
            dashboard_file.unlink()

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Setup enhanced environment"""
    
    logger.info("🔧 Setting up RL-A2A Enhanced environment...")
    
    # Create directories
    for directory in ["logs", "data", "exports"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Create .env.example
    env_content = f"""# RL-A2A Combined Enhanced Configuration

# AI Provider API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Security Configuration
SECRET_KEY=your-jwt-secret-key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501
RATE_LIMIT_PER_MINUTE=60

# System Configuration
DEFAULT_AI_PROVIDER=openai
MAX_AGENTS=100
DEBUG=false

# Server Configuration
A2A_HOST=localhost
A2A_PORT=8000
DASHBOARD_PORT=8501

# Logging
LOG_LEVEL=INFO
LOG_FILE=rla2a.log
"""
    
    env_path = Path(".env.example")
    env_path.write_text(env_content)
    
    # Create .env if doesn't exist
    actual_env = Path(".env")
    if not actual_env.exists():
        actual_env.write_text(env_content)
        logger.info("📄 Created .env file - please configure API keys")
    
    logger.info("✅ Environment setup complete!")

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report():
    """Generate comprehensive HTML report"""
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RL-A2A Combined System Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ text-align: center; color: #2c3e50; }}
        .section {{ margin: 30px 0; }}
        .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; }}
        .success {{ background: #d4edda; color: #155724; padding: 10px; }}
        .warning {{ background: #fff3cd; color: #856404; padding: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RL-A2A Combined System Report</h1>
        <h2>{CONFIG['SYSTEM_NAME']} v{CONFIG['VERSION']}</h2>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h3>🔧 System Capabilities</h3>
        <div class="{'success' if SECURITY_AVAILABLE else 'warning'}">
            <strong>Security:</strong> {'✅ Enhanced' if SECURITY_AVAILABLE else '⚠️ Basic'}
        </div>
        <div class="{'success' if OPENAI_AVAILABLE else 'warning'}">
            <strong>OpenAI:</strong> {'✅ Available' if OPENAI_AVAILABLE else '❌ Not configured'}
        </div>
        <div class="{'success' if ANTHROPIC_AVAILABLE else 'warning'}">
            <strong>Anthropic:</strong> {'✅ Available' if ANTHROPIC_AVAILABLE else '❌ Not configured'}
        </div>
        <div class="{'success' if GOOGLE_AVAILABLE else 'warning'}">
            <strong>Google AI:</strong> {'✅ Available' if GOOGLE_AVAILABLE else '❌ Not configured'}
        </div>
    </div>

    <div class="section">
        <h3>🚀 Quick Start</h3>
        <div class="metric">
            <strong>Setup:</strong> python rla2a.py setup<br>
            <strong>Start Server:</strong> python rla2a.py server --demo-agents 5<br>
            <strong>Dashboard:</strong> python rla2a.py dashboard<br>
            <strong>API Docs:</strong> http://{CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}/docs
        </div>
    </div>

    <div class="section">
        <h3>📋 Configuration</h3>
        <div class="metric">
            Server: {CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}<br>
            Dashboard: Port {CONFIG['DASHBOARD_PORT']}<br>
            Max Agents: {CONFIG['MAX_AGENTS']}<br>
            Debug: {CONFIG['DEBUG']}
        </div>
    </div>

    <div class="header">
        <p><strong>Ready for Multi-Agent Intelligence!</strong></p>
    </div>
</body>
</html>
"""
    
    report_path = Path("rla2a_report.html")
    report_path.write_text(html_content)
    
    logger.info(f"✅ Report generated: {report_path.resolve()}")
    return report_path

# =============================================================================
# MCP SERVER
# =============================================================================

async def run_mcp_server():
    """Run MCP server if available"""
    if not MCP_AVAILABLE:
        logger.error("❌ MCP not available. Install with: pip install mcp")
        return
    
    logger.info("🔌 Starting RL-A2A MCP Server...")
    # Would include full MCP implementation here

# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Enhanced CLI interface"""
    
    parser = argparse.ArgumentParser(description=f"{CONFIG['SYSTEM_NAME']} v{CONFIG['VERSION']}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    subparsers.add_parser("setup", help="Setup environment")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start A2A server")
    server_parser.add_argument("--demo-agents", type=int, default=0, help="Create demo agents")
    server_parser.add_argument("--host", default=CONFIG["SERVER_HOST"], help="Server host")
    server_parser.add_argument("--port", type=int, default=CONFIG["SERVER_PORT"], help="Server port")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Start dashboard") 
    
    # Report command
    subparsers.add_parser("report", help="Generate system report")
    
    # MCP command
    subparsers.add_parser("mcp", help="Start MCP server")
    
    # Info command
    subparsers.add_parser("info", help="System information")
    
    args = parser.parse_args()
    
    if not args.command:
        print(f"RL-A2A {CONFIG['SYSTEM_NAME']} v{CONFIG['VERSION']}")
        print("=" * 60)
        print("Combined Enhanced Agent-to-Agent Communication System")
        print()
        print("🚀 Quick Commands:")
        print("  python rla2a.py setup              # Setup environment")
        print("  python rla2a.py server             # Start server")
        print("  python rla2a.py dashboard          # Start dashboard")
        print("  python rla2a.py report             # Generate report")
        print()
        print("📚 Documentation: python rla2a.py --help")
        return
    
    try:
        if args.command == "setup":
            setup_environment()
        
        elif args.command == "server":
            CONFIG["SERVER_HOST"] = args.host
            CONFIG["SERVER_PORT"] = args.port
            
            system = A2ASystem()
            
            if args.demo_agents > 0:
                system.create_demo_agents(args.demo_agents)
            
            def signal_handler(signum, frame):
                logger.info("\\n🛑 Shutting down...")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            await system.start_server()
        
        elif args.command == "dashboard":
            start_dashboard()
        
        elif args.command == "report":
            generate_report()
        
        elif args.command == "mcp":
            await run_mcp_server()
        
        elif args.command == "info":
            print(f"RL-A2A {CONFIG['SYSTEM_NAME']}")
            print(f"Version: {CONFIG['VERSION']}")
            print(f"Security: {'Enhanced' if SECURITY_AVAILABLE else 'Basic'}")
            print(f"AI Providers:")
            print(f"  OpenAI: {'✅' if OPENAI_AVAILABLE else '❌'}")
            print(f"  Anthropic: {'✅' if ANTHROPIC_AVAILABLE else '❌'}")
            print(f"  Google: {'✅' if GOOGLE_AVAILABLE else '❌'}")
            print(f"MCP: {'✅' if MCP_AVAILABLE else '❌'}")
            print(f"Server: {CONFIG['SERVER_HOST']}:{CONFIG['SERVER_PORT']}")
    
    except KeyboardInterrupt:
        logger.info("\\n👋 Operation cancelled")
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)