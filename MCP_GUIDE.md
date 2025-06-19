# MCP (Model Context Protocol) Integration Guide

## 🚀 **Complete MCP Support Added**

RL-A2A now includes comprehensive MCP (Model Context Protocol) support for seamless AI assistant integration.

### ✅ **What's Included**

- **🔧 MCP Server** - Full implementation in `rla2a.py`
- **📋 Configuration** - Ready-to-use `mcp_config.json`
- **🛠️ Tools** - 5 comprehensive tools for agent management
- **📚 Resources** - 3 resources for system data access
- **🔌 Auto-Detection** - Automatic MCP package detection and installation

### 🛠️ **Available MCP Tools**

1. **`create_agent`** - Create new AI agents
   - Parameters: `name`, `role`, `ai_provider`
   - Example: Create a researcher agent with OpenAI

2. **`list_agents`** - List all active agents
   - No parameters required
   - Returns: Agent names, roles, and providers

3. **`send_message`** - Send messages between agents
   - Parameters: `sender_id`, `receiver_id`, `content`
   - Enables inter-agent communication

4. **`get_system_status`** - Get comprehensive system status
   - No parameters required
   - Returns: Version, agent count, AI providers, features

5. **`generate_ai_response`** - Generate AI responses
   - Parameters: `prompt`, `provider` (optional)
   - Supports OpenAI, Anthropic, Google

### 📚 **Available MCP Resources**

1. **`rl-a2a://system/config`** - System configuration
   - JSON format with all system settings

2. **`rl-a2a://agents/list`** - Detailed agents list
   - JSON format with agent details and metadata

3. **`rl-a2a://system/logs`** - Recent system logs
   - Text format with last 50 log entries

### 🚀 **Quick Start**

#### 1. Install MCP Support
```bash
# MCP will be auto-detected and installed
python rla2a.py setup

# Or install manually
pip install mcp
```

#### 2. Start MCP Server
```bash
python rla2a.py mcp
```

#### 3. Configure Your AI Assistant

**For Claude Desktop:**
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rl-a2a-enhanced": {
      "command": "python",
      "args": ["rla2a.py", "mcp"],
      "cwd": "/path/to/your/rl-a2a/directory"
    }
  }
}
```

**For Other MCP Clients:**
Use the provided `mcp_config.json` as a reference.

### 💡 **Usage Examples**

Once connected, you can ask your AI assistant:

```
"Create a researcher agent named Alice using OpenAI"
"List all active agents in the system"
"Get the current system status"
"Generate an AI response about machine learning"
"Show me the system configuration"
```

### 🔧 **Advanced Configuration**

#### Environment Variables
```bash
# Set AI provider API keys
export OPENAI_API_KEY="your_key_here"
export ANTHROPIC_API_KEY="your_key_here"
export GOOGLE_API_KEY="your_key_here"

# Configure server settings
export A2A_HOST="localhost"
export A2A_PORT="8000"
```

#### Custom MCP Configuration
Edit `mcp_config.json` to customize:
- Tool descriptions
- Resource URIs
- Server parameters

### 🐛 **Troubleshooting**

#### MCP Not Available
```bash
# Install MCP package
pip install mcp

# Verify installation
python -c "import mcp; print('MCP installed successfully')"
```

#### Connection Issues
1. Ensure `rla2a.py` is in the correct directory
2. Check that Python can find the script
3. Verify MCP server is running: `python rla2a.py mcp`

#### Tool Errors
- Check system logs: `python rla2a.py info`
- Verify AI provider API keys are set
- Ensure required dependencies are installed

### 📋 **Feature Status**

- ✅ **MCP Server** - Fully implemented
- ✅ **Tools** - 5 comprehensive tools available
- ✅ **Resources** - 3 system resources accessible
- ✅ **Error Handling** - Comprehensive error management
- ✅ **Auto-Detection** - Automatic MCP package detection
- ✅ **Configuration** - Ready-to-use config files

### 🔗 **Integration Examples**

#### Claude Desktop Integration
1. Add RL-A2A to Claude Desktop config
2. Restart Claude Desktop
3. Start using RL-A2A tools in conversations

#### Custom MCP Client
```python
# Example MCP client integration
import asyncio
from mcp.client import Client

async def use_rl_a2a():
    client = Client()
    await client.connect("python", ["rla2a.py", "mcp"])
    
    # List available tools
    tools = await client.list_tools()
    print(f"Available tools: {[t.name for t in tools]}")
    
    # Create an agent
    result = await client.call_tool("create_agent", {
        "name": "TestAgent",
        "role": "researcher"
    })
    print(result)

asyncio.run(use_rl_a2a())
```

---

**🎉 MCP support is now fully integrated into RL-A2A!**

Start the MCP server with `python rla2a.py mcp` and connect your AI assistant for seamless agent management and communication.
      "command": "python",
      "args": ["rla2a.py", "mcp"],
      "env": {
        "A2A_SERVER_URL": "http://localhost:8000"
      }
    },
    "rl-a2a-prod": {
      "command": "python", 
      "args": ["rla2a.py", "mcp"],
      "env": {
        "A2A_SERVER_URL": "https://your-prod-server.com"
      }
    }
  }
}
```

## Next Steps

1. **Extend Tools:** Add more MCP tools for your specific use cases
2. **Custom Resources:** Create additional resources for monitoring
3. **Authentication:** Add security features for production use
4. **Scaling:** Implement load balancing for multiple A2A servers
5. **Integration:** Connect with other MCP servers and tools

## Support

For questions or issues:
- Open an issue on GitHub
- Check the rla2a.py file for diagnostic information
- Review A2A server logs for connection issues

## Contributing

Contributions to improve MCP integration are welcome:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

**Note:** This MCP integration is designed to be a starting point. You can extend it with additional tools, resources, and functionality specific to your agent communication needs.