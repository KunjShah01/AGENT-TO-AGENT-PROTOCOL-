"""
Plugin System - Extensible architecture for RL-A2A
==================================================

A comprehensive plugin system that allows:
- Dynamic plugin loading and unloading
- Plugin dependency management
- Secure plugin execution
- Plugin marketplace integration
- Hot-swappable functionality
"""

import os
import sys
import json
import importlib
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Callable, Type
from dataclasses import dataclass, asdict
from pathlib import Path
from abc import ABC, abstractmethod
import traceback
from datetime import datetime

@dataclass
class PluginMetadata:
    """Plugin metadata structure"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    entry_point: str
    permissions: List[str]
    category: str
    tags: List[str]
    min_rla2a_version: str = "1.0.0"
    enabled: bool = True

class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    @abstractmethod
    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        pass
    
    @property
    @abstractmethod
    def metadata(self) -> PluginMetadata:
        """Plugin metadata"""
        pass

class PluginContext:
    """Plugin execution context"""
    
    def __init__(self, plugin_manager: 'PluginManager'):
        self.plugin_manager = plugin_manager
        self.shared_data = {}
        self.event_bus = EventBus()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup plugin logger"""
        import logging
        logger = logging.getLogger("RL-A2A.Plugins")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get another plugin instance"""
        return self.plugin_manager.get_plugin(name)
    
    def emit_event(self, event_name: str, data: Any):
        """Emit event to event bus"""
        self.event_bus.emit(event_name, data)
    
    def subscribe_event(self, event_name: str, callback: Callable):
        """Subscribe to event"""
        self.event_bus.subscribe(event_name, callback)

class EventBus:
    """Simple event bus for plugin communication"""
    
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_name: str, callback: Callable):
        """Subscribe to an event"""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)
    
    def emit(self, event_name: str, data: Any):
        """Emit an event"""
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        asyncio.create_task(callback(data))
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Event callback error: {e}")

class PluginSandbox:
    """Secure plugin execution sandbox"""
    
    def __init__(self, allowed_modules: List[str] = None):
        self.allowed_modules = allowed_modules or [
            'json', 'datetime', 'math', 'random', 'string', 'uuid',
            'asyncio', 'typing', 'dataclasses', 'pathlib'
        ]
        self.restricted_functions = [
            'exec', 'eval', 'compile', '__import__', 'open', 'file'
        ]
    
    def is_safe_import(self, module_name: str) -> bool:
        """Check if module import is safe"""
        return module_name in self.allowed_modules
    
    def validate_plugin_code(self, plugin_code: str) -> bool:
        """Validate plugin code for security"""
        # Basic security checks
        for restricted in self.restricted_functions:
            if restricted in plugin_code:
                return False
        
        # Check for dangerous imports
        import ast
        try:
            tree = ast.parse(plugin_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self.is_safe_import(alias.name):
                            return False
                elif isinstance(node, ast.ImportFrom):
                    if not self.is_safe_import(node.module):
                        return False
        except SyntaxError:
            return False
        
        return True

class PluginManager:
    """Main plugin management system"""
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(exist_ok=True)
        
        # Plugin storage
        self.loaded_plugins: Dict[str, PluginInterface] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_modules: Dict[str, Any] = {}
        
        # Plugin system components
        self.context = PluginContext(self)
        self.sandbox = PluginSandbox()
        
        # Plugin registry file
        self.registry_file = self.plugins_dir / "registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load plugin registry"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    for name, metadata_dict in registry_data.items():
                        self.plugin_metadata[name] = PluginMetadata(**metadata_dict)
            except Exception as e:
                print(f"Error loading plugin registry: {e}")
    
    def _save_registry(self):
        """Save plugin registry"""
        registry_data = {
            name: asdict(metadata) 
            for name, metadata in self.plugin_metadata.items()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    async def install_plugin(self, plugin_path: str, validate: bool = True) -> bool:
        """Install a new plugin"""
        try:
            plugin_dir = Path(plugin_path)
            
            # Load plugin manifest
            manifest_file = plugin_dir / "manifest.json"
            if not manifest_file.exists():
                raise ValueError("Plugin manifest not found")
            
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            metadata = PluginMetadata(**manifest_data)
            
            # Validate plugin code if required
            if validate:
                plugin_file = plugin_dir / f"{metadata.entry_point}.py"
                if plugin_file.exists():
                    with open(plugin_file, 'r') as f:
                        plugin_code = f.read()
                    
                    if not self.sandbox.validate_plugin_code(plugin_code):
                        raise ValueError("Plugin failed security validation")
            
            # Copy plugin to plugins directory
            import shutil
            target_dir = self.plugins_dir / metadata.name
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(plugin_dir, target_dir)
            
            # Register plugin
            self.plugin_metadata[metadata.name] = metadata
            self._save_registry()
            
            self.context.logger.info(f"Plugin '{metadata.name}' installed successfully")
            return True
            
        except Exception as e:
            self.context.logger.error(f"Plugin installation failed: {e}")
            return False
    
    async def load_plugin(self, plugin_name: str) -> bool:
        """Load and initialize a plugin"""
        try:
            if plugin_name in self.loaded_plugins:
                return True
            
            if plugin_name not in self.plugin_metadata:
                raise ValueError(f"Plugin '{plugin_name}' not found in registry")
            
            metadata = self.plugin_metadata[plugin_name]
            
            if not metadata.enabled:
                raise ValueError(f"Plugin '{plugin_name}' is disabled")
            
            # Check dependencies
            for dep in metadata.dependencies:
                if dep not in self.loaded_plugins:
                    if not await self.load_plugin(dep):
                        raise ValueError(f"Failed to load dependency: {dep}")
            
            # Load plugin module
            plugin_dir = self.plugins_dir / plugin_name
            plugin_file = plugin_dir / f"{metadata.entry_point}.py"
            
            if not plugin_file.exists():
                raise ValueError(f"Plugin entry point not found: {plugin_file}")
            
            # Dynamic import
            spec = importlib.util.spec_from_file_location(
                f"plugin_{plugin_name}", plugin_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise ValueError("No valid plugin class found")
            
            # Instantiate and initialize plugin
            plugin_instance = plugin_class()
            
            # Initialize plugin with context
            init_success = await plugin_instance.initialize(asdict(self.context))
            
            if not init_success:
                raise ValueError("Plugin initialization failed")
            
            # Store plugin
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_modules[plugin_name] = module
            
            self.context.logger.info(f"Plugin '{plugin_name}' loaded successfully")
            self.context.emit_event("plugin_loaded", {"name": plugin_name})
            
            return True
            
        except Exception as e:
            self.context.logger.error(f"Failed to load plugin '{plugin_name}': {e}")
            traceback.print_exc()
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        try:
            if plugin_name not in self.loaded_plugins:
                return True
            
            plugin = self.loaded_plugins[plugin_name]
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            del self.plugin_modules[plugin_name]
            
            # Remove from sys.modules if present
            module_name = f"plugin_{plugin_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            self.context.logger.info(f"Plugin '{plugin_name}' unloaded successfully")
            self.context.emit_event("plugin_unloaded", {"name": plugin_name})
            
            return True
            
        except Exception as e:
            self.context.logger.error(f"Failed to unload plugin '{plugin_name}': {e}")
            return False
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin"""
        if await self.unload_plugin(plugin_name):
            return await self.load_plugin(plugin_name)
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get loaded plugin instance"""
        return self.loaded_plugins.get(plugin_name)
    
    async def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_name}' not loaded")
        
        return await plugin.execute(*args, **kwargs)
    
    def list_plugins(self, loaded_only: bool = False) -> List[Dict[str, Any]]:
        """List available plugins"""
        plugins = []
        
        for name, metadata in self.plugin_metadata.items():
            plugin_info = {
                "name": name,
                "metadata": asdict(metadata),
                "loaded": name in self.loaded_plugins,
                "status": "loaded" if name in self.loaded_plugins else "available"
            }
            
            if not loaded_only or plugin_info["loaded"]:
                plugins.append(plugin_info)
        
        return plugins
    
    async def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name in self.plugin_metadata:
            self.plugin_metadata[plugin_name].enabled = True
            self._save_registry()
            return True
        return False
    
    async def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name in self.plugin_metadata:
            await self.unload_plugin(plugin_name)
            self.plugin_metadata[plugin_name].enabled = False
            self._save_registry()
            return True
        return False
    
    async def auto_load_plugins(self):
        """Auto-load all enabled plugins"""
        for name, metadata in self.plugin_metadata.items():
            if metadata.enabled:
                await self.load_plugin(name)
    
    async def get_plugin_stats(self) -> Dict[str, Any]:
        """Get plugin system statistics"""
        return {
            "total_plugins": len(self.plugin_metadata),
            "loaded_plugins": len(self.loaded_plugins),
            "enabled_plugins": sum(1 for m in self.plugin_metadata.values() if m.enabled),
            "categories": list(set(m.category for m in self.plugin_metadata.values())),
            "plugin_list": self.list_plugins()
        }

# Example plugin implementation
class ExamplePlugin(PluginInterface):
    """Example plugin implementation"""
    
    def __init__(self):
        self._metadata = PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="RL-A2A Team",
            dependencies=[],
            entry_point="example_plugin",
            permissions=["read"],
            category="utility",
            tags=["example", "demo"]
        )
        self.context = None
    
    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata
    
    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        self.context = context
        print(f"Example plugin initialized")
        return True
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        return {
            "message": "Hello from example plugin!",
            "args": args,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources"""
        print("Example plugin cleaned up")
        return True

# Demo function
async def demo_plugin_system():
    """Demo the plugin system"""
    manager = PluginManager()
    
    # Create example plugin directory
    example_dir = Path("plugins/example_plugin")
    example_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest = {
        "name": "example_plugin",
        "version": "1.0.0",
        "description": "Example plugin for demonstration",
        "author": "RL-A2A Team",
        "dependencies": [],
        "entry_point": "example_plugin",
        "permissions": ["read"],
        "category": "utility",
        "tags": ["example", "demo"]
    }
    
    with open(example_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create plugin code
    plugin_code = '''
from plugins.plugin_system import PluginInterface, PluginMetadata
from datetime import datetime

class ExamplePlugin(PluginInterface):
    def __init__(self):
        self._metadata = PluginMetadata(
            name="example_plugin",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="RL-A2A Team",
            dependencies=[],
            entry_point="example_plugin",
            permissions=["read"],
            category="utility",
            tags=["example", "demo"]
        )
    
    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata
    
    async def initialize(self, context) -> bool:
        print("Example plugin initialized")
        return True
    
    async def execute(self, *args, **kwargs):
        return {
            "message": "Hello from example plugin!",
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self) -> bool:
        print("Example plugin cleaned up")
        return True
'''
    
    with open(example_dir / "example_plugin.py", 'w') as f:
        f.write(plugin_code)
    
    # Install and load plugin
    await manager.install_plugin(str(example_dir))
    await manager.load_plugin("example_plugin")
    
    # Execute plugin
    result = await manager.execute_plugin("example_plugin", "test", value=42)
    print("Plugin result:", result)
    
    # Get stats
    stats = await manager.get_plugin_stats()
    print("Plugin stats:", json.dumps(stats, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(demo_plugin_system())