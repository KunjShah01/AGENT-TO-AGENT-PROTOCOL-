#!/usr/bin/env python3
"""
Ollama Integration for RL-A2A
Adds local model support alongside cloud providers
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
import os

class OllamaProvider:
    """Ollama local model provider"""
    
    def __init__(self, base_url: str = None, model: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self.logger = logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Ollama not available: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            self.logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def generate_response(self, prompt: str, context: str = None) -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if context:
                payload["context"] = context
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": self.model,
                    "context": data.get("context"),
                    "provider": "ollama"
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "provider": "ollama"
                }
                
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "ollama"
            }
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Chat completion using Ollama"""
        try:
            # Convert messages to single prompt for Ollama
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "content": data.get("response", ""),
                    "model": self.model,
                    "provider": "ollama",
                    "usage": {
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": len(data.get("response", "").split()),
                        "total_tokens": len(prompt.split()) + len(data.get("response", "").split())
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama chat error: {response.status_code}",
                    "provider": "ollama"
                }
                
        except Exception as e:
            self.logger.error(f"Ollama chat completion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": "ollama"
            }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to single prompt"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

def setup_ollama_integration():
    """Setup Ollama integration in RL-A2A"""
    print("🦙 Setting up Ollama integration...")
    
    # Check if Ollama is installed
    ollama = OllamaProvider()
    
    if not ollama.is_available():
        print("❌ Ollama not detected. Install from: https://ollama.ai")
        print("💡 After installation, run: ollama pull llama2")
        return False
    
    # List available models
    models = ollama.list_models()
    if not models:
        print("⚠️ No Ollama models found. Pull a model first:")
        print("   ollama pull llama2")
        print("   ollama pull codellama")
        return False
    
    print(f"✅ Ollama detected with {len(models)} models:")
    for model in models[:5]:  # Show first 5 models
        print(f"   - {model}")
    
    # Test generation
    print("🧪 Testing Ollama generation...")
    test_response = ollama.generate_response("Hello, how are you?")
    
    if test_response["success"]:
        print("✅ Ollama integration working!")
        print(f"📝 Test response: {test_response['response'][:100]}...")
        return True
    else:
        print(f"❌ Ollama test failed: {test_response['error']}")
        return False

def add_ollama_to_env():
    """Add Ollama configuration to .env file"""
    env_file = ".env"
    
    ollama_config = """
# Ollama Local Model Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
ENABLE_OLLAMA=true
"""
    
    try:
        with open(env_file, "a") as f:
            f.write(ollama_config)
        print("✅ Ollama configuration added to .env")
    except Exception as e:
        print(f"❌ Failed to update .env: {e}")

if __name__ == "__main__":
    print("🦙 Ollama Integration Setup")
    print("=" * 40)
    
    if setup_ollama_integration():
        add_ollama_to_env()
        print("\n🎉 Ollama integration complete!")
        print("💡 Restart RL-A2A to use local models")
    else:
        print("\n🔧 Setup Ollama first, then run this script again")