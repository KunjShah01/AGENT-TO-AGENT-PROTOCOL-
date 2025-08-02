# 🚀 RL-A2A MVP Quick Start Guide

## Overview
RL-A2A is a production-ready multi-agent communication system with reinforcement learning capabilities, supporting multiple AI providers and real-time visualization.

## 🎯 MVP Features
- ✅ Multi-AI Provider Support (OpenAI, Claude, Gemini, Ollama)
- ✅ Real-time Agent Communication
- ✅ 3D Visualization Dashboard
- ✅ Security Features (JWT, Rate Limiting)
- ✅ Docker Deployment
- ✅ MCP Integration

## 🚀 Quick Start (5 Minutes)

### Option 1: Automated Setup
```bash
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A
python setup.py
```

### Option 2: Docker Deployment
```bash
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A
cp .env .env.local  # Edit with your API keys
docker-compose up -d
```

### Option 3: Manual Setup
```bash
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A
pip install -r requirements.txt
python rla2a.py setup
```

## 🔧 Configuration

Edit `.env` file:
```bash
# Required: Add at least one AI provider
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
GOOGLE_API_KEY=your-google-key-here

# Optional: Local Ollama support
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Security (generate secure keys)
SECRET_KEY=your-secure-secret-key
```

## 🎮 Usage

### Start the System
```bash
# Start server with demo agents
python rla2a.py server --demo-agents 3

# Start dashboard (new terminal)
python rla2a.py dashboard
```

### Access Points
- 🌐 **Dashboard**: http://localhost:8501
- 📄 **API Docs**: http://localhost:8000/docs
- 📊 **Health Check**: http://localhost:8000/health

### Basic Commands
```bash
python rla2a.py info          # System information
python rla2a.py report        # Generate HTML report
python rla2a.py mcp           # Start MCP server
python test_rla2a.py          # Run tests
```

## 🐳 Docker Commands

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build -d
```

## 🔍 Troubleshooting

### Common Issues
1. **Import Errors**: Run `python setup.py` to install dependencies
2. **API Key Issues**: Check `.env` file configuration
3. **Port Conflicts**: Change ports in `.env` file
4. **Docker Issues**: Ensure Docker is running

### Health Checks
```bash
# Test basic functionality
python test_rla2a.py

# Check API health
curl http://localhost:8000/health

# View system status
python rla2a.py info
```

## 📈 Next Steps

1. **Add Your API Keys** to `.env`
2. **Customize Agents** in the dashboard
3. **Integrate with Your Apps** using the REST API
4. **Scale with Docker** for production deployment
5. **Contribute** to the project on GitHub

## 🔗 Resources

- [API Documentation](http://localhost:8000/docs)
- [MCP Integration Guide](../MCP_GUIDE.md)
- [Deployment Guide](DEPLOYMENT.md)
- [Security Guide](../SECURITY.md)

## 🆘 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/KunjShah01/RL-A2A/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KunjShah01/RL-A2A/discussions)
- 📧 **Contact**: Create an issue for support

---

**🎉 You're ready to build amazing multi-agent systems!**