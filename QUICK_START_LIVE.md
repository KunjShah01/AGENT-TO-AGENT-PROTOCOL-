# 🚀 RL-A2A Live - Quick Start Guide

Get your RL-A2A system running live in minutes with all the new features!

## ⚡ One-Command Deployment

```bash
# Clone and deploy locally
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A
chmod +x scripts/deploy.sh
./scripts/deploy.sh local
```

## 🌟 What's New in Live Version

### 🏪 **Agent Marketplace**
- **Community-shared agents** with ratings and reviews
- **Search and discovery** by category, tags, and popularity
- **One-click installation** from the marketplace
- **Version management** and dependency tracking

### 📊 **Advanced Analytics**
- **Real-time usage metrics** and performance monitoring
- **Agent behavior analysis** and optimization insights
- **Predictive analytics** for system scaling
- **Comprehensive reporting** and data export

### 🔌 **Plugin System**
- **Hot-swappable plugins** without server restart
- **Secure plugin execution** with sandboxing
- **Plugin marketplace integration**
- **Dependency management** and version control

### 📈 **Live Dashboard**
- **Real-time monitoring** with WebSocket updates
- **Interactive charts** and visualizations
- **System health monitoring** with alerts
- **Mobile-responsive design**

## 🚀 Deployment Options

### 1. Local Development
```bash
# Quick local setup
./scripts/deploy.sh local

# Access your application
# 🌐 Main App: http://localhost:8000
# 📊 Dashboard: http://localhost:8000/dashboard
# 📚 API Docs: http://localhost:8000/docs
```

### 2. Vercel (Recommended for Frontend)
```bash
# Deploy to Vercel
./scripts/deploy.sh vercel

# Or manually:
npm install -g vercel
vercel --prod
```

### 3. Render (Full-Stack)
```bash
# Setup Render deployment
./scripts/deploy.sh render

# Then connect your GitHub repo to Render
# Use the provided render.yaml configuration
```

### 4. Netlify (Static Frontend)
```bash
# Deploy frontend to Netlify
./scripts/deploy.sh netlify
```

### 5. Docker Deployment
```bash
# Build and run with Docker
docker build -t rla2a:latest .
docker run -p 8000:8000 rla2a:latest

# Or use Docker Compose
docker-compose up -d
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with your configuration:

```env
# Server Configuration
ENVIRONMENT=production
PORT=8000
HOST=0.0.0.0

# Security
JWT_SECRET_KEY=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# AI Providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Database
REDIS_URL=redis://localhost:6379

# Features
ENABLE_MARKETPLACE=true
ENABLE_ANALYTICS=true
ENABLE_PLUGINS=true
ENABLE_LIVE_DASHBOARD=true

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
```

## 📱 Using the Live Features

### Agent Marketplace
1. **Browse Agents**: Visit `/dashboard` and click "Show Marketplace"
2. **Search**: Use filters by category, tags, or search terms
3. **Install**: Click "Install" on any agent to add it to your system
4. **Rate & Review**: Provide feedback to help the community

### Analytics Dashboard
1. **Real-time Metrics**: View live usage statistics
2. **Performance Insights**: Monitor response times and system health
3. **Usage Trends**: Analyze agent popularity and user engagement
4. **Export Data**: Download analytics for external analysis

### Plugin Management
1. **View Plugins**: Check loaded and available plugins
2. **Load/Unload**: Manage plugins without server restart
3. **Install New**: Add plugins from the marketplace
4. **Monitor Status**: Track plugin health and performance

## 🔌 API Endpoints

### Core Endpoints
- `GET /` - Main application
- `GET /dashboard` - Live dashboard
- `GET /health` - Health check
- `GET /docs` - API documentation

### Live Features API
- `GET /api/live/dashboard` - Dashboard data
- `GET /api/live/marketplace/search` - Search agents
- `POST /api/live/marketplace/install/{agent_id}` - Install agent
- `GET /api/live/plugins` - List plugins
- `POST /api/live/plugins/{name}/load` - Load plugin
- `GET /api/live/analytics/{time_range}` - Usage analytics

### WebSocket
- `ws://localhost:8000/ws` - Real-time updates

## 🛠 Development

### Running in Development Mode
```bash
# Install dependencies
pip install -r requirements.txt

# Run enhanced server
python enhanced_server.py --reload

# Or run original server
python a2a_server.py
```

### Adding Custom Agents
```python
# Create agent in marketplace/
agent_data = {
    "name": "My Custom Agent",
    "description": "Does amazing things",
    "author": "Your Name",
    "category": "utility",
    "tags": ["custom", "utility"]
}

# Publish to marketplace
await marketplace.publish_agent(agent_data)
```

### Creating Plugins
```python
# Create plugin class
from plugins.plugin_system import PluginInterface, PluginMetadata

class MyPlugin(PluginInterface):
    @property
    def metadata(self):
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
            author="Your Name",
            dependencies=[],
            entry_point="my_plugin",
            permissions=["read"],
            category="utility",
            tags=["custom"]
        )
    
    async def initialize(self, context):
        return True
    
    async def execute(self, *args, **kwargs):
        return {"message": "Hello from my plugin!"}
    
    async def cleanup(self):
        return True
```

## 📊 Monitoring & Analytics

### Real-time Monitoring
- **System Health**: Monitor server status and performance
- **Active Sessions**: Track concurrent users
- **Response Times**: Monitor API performance
- **Error Rates**: Track and alert on failures

### Usage Analytics
- **Agent Popularity**: See which agents are most used
- **User Engagement**: Track session lengths and interactions
- **Performance Trends**: Identify optimization opportunities
- **Growth Metrics**: Monitor system adoption

## 🔒 Security Features

### Built-in Security
- **JWT Authentication** for API access
- **Rate Limiting** to prevent abuse
- **Input Validation** and sanitization
- **Plugin Sandboxing** for safe execution
- **CORS Protection** for web security

### Best Practices
1. **Use HTTPS** in production
2. **Set strong JWT secrets**
3. **Enable rate limiting**
4. **Validate all inputs**
5. **Monitor for suspicious activity**

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Kill process on port 8000
sudo lsof -t -i tcp:8000 | xargs kill -9
```

**Dependencies Missing**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

**Docker Issues**
```bash
# Clean Docker cache
docker system prune -a
docker-compose down && docker-compose up --build
```

**WebSocket Connection Failed**
- Check firewall settings
- Verify WebSocket support in your browser
- Ensure server is running on correct port

### Getting Help
1. **Check Logs**: Look at server console output
2. **Health Check**: Visit `/health` endpoint
3. **API Docs**: Check `/docs` for API reference
4. **GitHub Issues**: Report bugs on GitHub

## 🎯 Next Steps

1. **Customize**: Modify the dashboard and add your branding
2. **Extend**: Create custom agents and plugins
3. **Scale**: Deploy to cloud platforms for production
4. **Monitor**: Set up alerts and monitoring
5. **Contribute**: Share your agents and plugins with the community

## 📚 Additional Resources

- **Full Documentation**: See `/docs` directory
- **API Reference**: Visit `/docs` endpoint
- **Examples**: Check `/examples` directory
- **Community**: Join our Discord/Slack
- **Support**: GitHub Issues and Discussions

---

**🎉 Congratulations! Your RL-A2A system is now live with all advanced features enabled!**

Visit your dashboard at `http://localhost:8000/dashboard` to start exploring the new capabilities.