# 🚀 RL-A2A Production Deployment Guide

## Overview
This guide covers deploying RL-A2A in production environments with security, scalability, and monitoring best practices.

## 🏗️ Architecture Options

### 1. Single Server Deployment
**Best for**: Small teams, prototypes, low traffic
```bash
# Direct deployment
python rla2a.py server --demo-agents 5
```

### 2. Docker Deployment
**Best for**: Consistent environments, easy scaling
```bash
docker-compose up -d
```

### 3. Kubernetes Deployment
**Best for**: High availability, auto-scaling, enterprise

## 🔒 Security Checklist

### Environment Security
- [ ] Use strong `SECRET_KEY` (32+ characters)
- [ ] Set `DEBUG=false` in production
- [ ] Configure `ALLOWED_ORIGINS` properly
- [ ] Use HTTPS in production
- [ ] Rotate API keys regularly

### Network Security
- [ ] Use reverse proxy (nginx/traefik)
- [ ] Enable rate limiting
- [ ] Configure firewall rules
- [ ] Use VPN for internal access

### Data Security
- [ ] Encrypt sensitive data at rest
- [ ] Use secure Redis configuration
- [ ] Regular security audits
- [ ] Monitor for vulnerabilities

## 🐳 Docker Production Setup

### 1. Production Docker Compose
```yaml
version: '3.8'

services:
  rla2a:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=false
      - A2A_HOST=0.0.0.0
    env_file:
      - .env.production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rla2a
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### 2. Nginx Configuration
```nginx
upstream rla2a_backend {
    server rla2a:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    location / {
        proxy_pass http://rla2a_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://rla2a_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## ☸️ Kubernetes Deployment

### 1. Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rla2a
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rla2a
  template:
    metadata:
      labels:
        app: rla2a
    spec:
      containers:
      - name: rla2a
        image: rla2a:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEBUG
          value: "false"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        envFrom:
        - secretRef:
            name: rla2a-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 2. Service & Ingress
```yaml
apiVersion: v1
kind: Service
metadata:
  name: rla2a-service
spec:
  selector:
    app: rla2a
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rla2a-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: rla2a-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rla2a-service
            port:
              number: 80
```

## 📊 Monitoring & Logging

### 1. Health Monitoring
```bash
# Health check endpoint
curl https://your-domain.com/health

# System metrics
python rla2a.py report
```

### 2. Log Management
```bash
# Docker logs
docker-compose logs -f rla2a

# Kubernetes logs
kubectl logs -f deployment/rla2a
```

### 3. Metrics Collection
- Use Prometheus for metrics
- Grafana for visualization
- ELK stack for log analysis

## 🔧 Performance Optimization

### 1. Resource Allocation
```yaml
# Recommended resources
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### 2. Caching Strategy
- Redis for session storage
- CDN for static assets
- API response caching

### 3. Database Optimization
- Connection pooling
- Query optimization
- Regular maintenance

## 🚨 Backup & Recovery

### 1. Data Backup
```bash
# Redis backup
docker exec redis redis-cli BGSAVE

# Application data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ logs/
```

### 2. Disaster Recovery
- Regular automated backups
- Multi-region deployment
- Recovery procedures documentation

## 📈 Scaling Strategies

### 1. Horizontal Scaling
```bash
# Docker Compose
docker-compose up --scale rla2a=3

# Kubernetes
kubectl scale deployment rla2a --replicas=5
```

### 2. Load Balancing
- Use nginx or cloud load balancers
- Session affinity for WebSocket connections
- Health check configuration

## 🔍 Troubleshooting

### Common Issues
1. **High Memory Usage**: Increase limits or optimize agents
2. **Connection Timeouts**: Check network configuration
3. **API Rate Limits**: Implement proper rate limiting
4. **WebSocket Issues**: Verify proxy configuration

### Debug Commands
```bash
# Check system status
python rla2a.py info

# View logs
tail -f logs/rla2a.log

# Test connectivity
curl -f http://localhost:8000/health
```

## 📋 Production Checklist

### Pre-Deployment
- [ ] Security review completed
- [ ] Performance testing done
- [ ] Backup strategy implemented
- [ ] Monitoring configured
- [ ] SSL certificates installed

### Post-Deployment
- [ ] Health checks passing
- [ ] Logs being collected
- [ ] Metrics being monitored
- [ ] Backup verification
- [ ] Security scan completed

## 🆘 Support & Maintenance

### Regular Tasks
- Monitor system health
- Update dependencies
- Rotate secrets
- Review logs
- Performance optimization

### Emergency Procedures
- Incident response plan
- Rollback procedures
- Contact information
- Escalation matrix

---

**🎯 Your RL-A2A system is now production-ready!**