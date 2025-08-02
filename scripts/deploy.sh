#!/bin/bash

# RL-A2A Production Deployment Script
# ==================================
# Automated deployment script for multiple platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="RL-A2A"
REPO_URL="https://github.com/KunjShah01/RL-A2A"
DOCKER_IMAGE="rla2a:latest"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3 first."
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        log_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file..."
        cat > .env << EOF
# RL-A2A Production Configuration
ENVIRONMENT=production
PORT=8000
HOST=0.0.0.0

# Security
JWT_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)

# Database
REDIS_URL=redis://localhost:6379

# AI Providers (Add your API keys)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Monitoring
ENABLE_ANALYTICS=true
ENABLE_METRICS=true

# Rate Limiting
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600
EOF
        log_warning "Please update the .env file with your API keys and configuration"
    fi
    
    log_success "Environment setup complete"
}

install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    log_success "Dependencies installed"
}

build_docker_image() {
    log_info "Building Docker image..."
    
    docker build -t $DOCKER_IMAGE .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run tests
    python -m pytest tests/ -v --cov=. --cov-report=html
    
    if [ $? -eq 0 ]; then
        log_success "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
}

deploy_local() {
    log_info "Deploying locally..."
    
    # Stop existing containers
    docker-compose down 2>/dev/null || true
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Local deployment successful"
        log_info "Application is running at: http://localhost:8000"
        log_info "Dashboard is available at: http://localhost:8000/dashboard"
    else
        log_error "Health check failed"
        exit 1
    fi
}

deploy_vercel() {
    log_info "Deploying to Vercel..."
    
    # Check if Vercel CLI is installed
    if ! command -v vercel &> /dev/null; then
        log_info "Installing Vercel CLI..."
        npm install -g vercel
    fi
    
    # Deploy to Vercel
    vercel --prod
    
    log_success "Deployed to Vercel"
}

deploy_render() {
    log_info "Deploying to Render..."
    
    # Check if render.yaml exists
    if [ ! -f "deploy/render.yaml" ]; then
        log_error "render.yaml not found. Please create deployment configuration."
        exit 1
    fi
    
    log_info "Please connect your GitHub repository to Render and use the render.yaml configuration"
    log_info "Render deployment URL: https://dashboard.render.com"
}

deploy_netlify() {
    log_info "Deploying to Netlify..."
    
    # Check if Netlify CLI is installed
    if ! command -v netlify &> /dev/null; then
        log_info "Installing Netlify CLI..."
        npm install -g netlify-cli
    fi
    
    # Build static files
    mkdir -p dist
    cp index.html dist/
    cp frontend/dashboard.html dist/dashboard.html
    
    # Deploy to Netlify
    netlify deploy --prod --dir=dist
    
    log_success "Deployed to Netlify"
}

setup_monitoring() {
    log_info "Setting up monitoring..."
    
    # Create monitoring directory
    mkdir -p monitoring
    
    # Create basic monitoring script
    cat > monitoring/health_check.sh << 'EOF'
#!/bin/bash

# Health check script
ENDPOINT="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $ENDPOINT)

if [ $RESPONSE -eq 200 ]; then
    echo "$(date): Service is healthy"
else
    echo "$(date): Service is unhealthy (HTTP $RESPONSE)"
    # Add alerting logic here
fi
EOF

    chmod +x monitoring/health_check.sh
    
    # Create systemd service for monitoring (Linux only)
    if [ -f /etc/systemd/system ]; then
        cat > monitoring/rla2a-monitor.service << EOF
[Unit]
Description=RL-A2A Health Monitor
After=network.target

[Service]
Type=simple
ExecStart=/bin/bash $(pwd)/monitoring/health_check.sh
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
        log_info "Systemd service created at monitoring/rla2a-monitor.service"
        log_info "To enable: sudo cp monitoring/rla2a-monitor.service /etc/systemd/system/ && sudo systemctl enable rla2a-monitor"
    fi
    
    log_success "Monitoring setup complete"
}

show_help() {
    echo "RL-A2A Deployment Script"
    echo "========================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local     Deploy locally using Docker"
    echo "  vercel    Deploy to Vercel"
    echo "  render    Deploy to Render"
    echo "  netlify   Deploy to Netlify (frontend only)"
    echo "  test      Run tests only"
    echo "  build     Build Docker image only"
    echo "  setup     Setup environment only"
    echo "  monitor   Setup monitoring only"
    echo "  all       Full deployment (local)"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local          # Deploy locally"
    echo "  $0 vercel         # Deploy to Vercel"
    echo "  $0 test           # Run tests"
    echo ""
}

# Main execution
main() {
    case "${1:-help}" in
        "local")
            check_dependencies
            setup_environment
            install_dependencies
            build_docker_image
            run_tests
            deploy_local
            setup_monitoring
            ;;
        "vercel")
            check_dependencies
            setup_environment
            install_dependencies
            run_tests
            deploy_vercel
            ;;
        "render")
            check_dependencies
            setup_environment
            deploy_render
            ;;
        "netlify")
            check_dependencies
            deploy_netlify
            ;;
        "test")
            check_dependencies
            install_dependencies
            run_tests
            ;;
        "build")
            check_dependencies
            build_docker_image
            ;;
        "setup")
            setup_environment
            install_dependencies
            ;;
        "monitor")
            setup_monitoring
            ;;
        "all")
            check_dependencies
            setup_environment
            install_dependencies
            build_docker_image
            run_tests
            deploy_local
            setup_monitoring
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"