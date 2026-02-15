"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Copy, Check } from "lucide-react"
import { useState } from "react"

function EndpointCard({ 
  method, 
  path, 
  title, 
  description, 
  request, 
  response 
}: { 
  method: "GET" | "POST" | "PUT" | "DELETE"
  path: string
  title: string
  description: string
  request?: string
  response: string
}) {
  const [copied, setCopied] = useState(false)

  const methodColors = {
    GET: "text-green-500 bg-green-500/10",
    POST: "text-blue-500 bg-blue-500/10",
    PUT: "text-yellow-500 bg-yellow-500/10",
    DELETE: "text-red-500 bg-red-500/10",
  }

  return (
    <Card className="bg-card/50 border-border overflow-hidden">
      <CardContent className="p-0">
        {/* Header */}
        <div className="flex items-center gap-3 p-4 border-b border-border bg-card/80">
          <Badge className={methodColors[method]}>
            {method}
          </Badge>
          <code className="text-sm font-mono text-foreground">{path}</code>
        </div>
        
        {/* Description */}
        <div className="p-4 border-b border-border">
          <h3 className="font-semibold text-foreground mb-1">{title}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>

        {/* Request/Response Tabs */}
        <Tabs defaultValue="response" className="p-4">
          <TabsList className="bg-muted">
            {request && (
              <TabsTrigger value="request" className="data-[state=active]:bg-primary/20">
                Request
              </TabsTrigger>
            )}
            <TabsTrigger value="response" className="data-[state=active]:bg-primary/20">
              Response
            </TabsTrigger>
          </TabsList>

          {request && (
            <TabsContent value="request" className="mt-4">
              <div className="relative">
                <pre className="p-3 bg-muted rounded-lg text-sm font-mono text-foreground overflow-x-auto">
                  <code>{request}</code>
                </pre>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(request)
                    setCopied(true)
                    setTimeout(() => setCopied(false), 2000)
                  }}
                  className="absolute top-2 right-2 p-1.5 rounded bg-card hover:bg-card/80 transition-colors"
                >
                  {copied ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4 text-muted-foreground" />
                  )}
                </button>
              </div>
            </TabsContent>
          )}

          <TabsContent value="response" className="mt-4">
            <div className="relative">
              <pre className="p-3 bg-muted rounded-lg text-sm font-mono text-foreground overflow-x-auto">
                <code>{response}</code>
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

export default function RestAPIPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">API Reference</Badge>
        <h1 className="text-4xl font-bold text-foreground">REST API</h1>
        <p className="text-xl text-muted-foreground">
          Complete reference for all REST API endpoints. Use these endpoints for
          agent management, messaging, and system configuration.
        </p>
      </div>

      {/* Agents Endpoints */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Agents</h2>
        <div className="space-y-4">
          <EndpointCard
            method="GET"
            path="/agents"
            title="List All Agents"
            description="Retrieve a list of all registered agents in the system."
            response={`{
  "agents": [
    {
      "id": "agent_abc123",
      "name": "Alice",
      "role": "researcher",
      "ai_provider": "openai",
      "status": "active",
      "created_at": "2024-01-15T10:00:00Z",
      "last_active": "2024-01-15T10:30:00Z"
    }
  ]
}`}
          />

          <EndpointCard
            method="POST"
            path="/agents"
            title="Create Agent"
            description="Create a new agent with specified configuration."
            request={`{
  "name": "NewAgent",
  "role": "analyst",
  "ai_provider": "anthropic"
}`}
            response={`{
  "agent_id": "agent_xyz789",
  "status": "created"
}`}
          />

          <EndpointCard
            method="GET"
            path="/agents/{agent_id}"
            title="Get Agent"
            description="Retrieve detailed information about a specific agent."
            response={`{
  "id": "agent_abc123",
  "name": "Alice",
  "role": "researcher",
  "ai_provider": "openai",
  "capabilities": ["communication", "learning", "reasoning"],
  "performance_metrics": {
    "success_rate": 0.85,
    "response_time": 120.5,
    "learning_rate": 0.02,
    "collaboration_score": 0.78
  },
  "status": "active"
}`}
          />
        </div>
      </section>

      {/* Messages Endpoints */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Messages</h2>
        <div className="space-y-4">
          <EndpointCard
            method="POST"
            path="/messages"
            title="Send Message"
            description="Send a message from one agent to another."
            request={`{
  "sender_id": "agent_001",
  "receiver_id": "agent_002",
  "content": "Task completed successfully",
  "message_type": "text",
  "priority": 1
}`}
            response={`{
  "message_id": "msg_abc123",
  "status": "queued"
}`}
          />
        </div>
      </section>

      {/* System Endpoints */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">System</h2>
        <div className="space-y-4">
          <EndpointCard
            method="GET"
            path="/status"
            title="System Status"
            description="Get current system status and health information."
            response={`{
  "status": "operational",
  "version": "2.0.0",
  "system_name": "RL-A2A Combined Enhanced",
  "agents_count": 5,
  "active_connections": 3,
  "ai_providers": ["openai", "anthropic"],
  "features": {
    "security_enabled": true,
    "ai_enabled": true,
    "visualization_enabled": true,
    "mcp_enabled": false
  },
  "uptime": 3600.5
}`}
          />

          <EndpointCard
            method="GET"
            path="/health"
            title="Health Check"
            description="Simple health check endpoint for monitoring."
            response={`{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}`}
          />
        </div>
      </section>

      {/* Feedback Endpoints */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Reinforcement Learning</h2>
        <div className="space-y-4">
          <EndpointCard
            method="POST"
            path="/feedback"
            title="Submit RL Feedback"
            description="Submit reward feedback for reinforcement learning updates."
            request={`{
  "agent_id": "agent_001",
  "action_id": "action_abc123",
  "reward": 0.85,
  "context": {
    "position": {"x": 10, "y": 20},
    "energy": 80,
    "emotion": "satisfied"
  }
}`}
            response={`{
  "status": "received",
  "cumulative_reward": 12.5
}`}
          />
        </div>
      </section>

      {/* Registration */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Authentication</h2>
        <div className="space-y-4">
          <EndpointCard
            method="POST"
            path="/register"
            title="Register Agent"
            description="Register a new agent session and get authentication token."
            request={`{
  "agent_id": "optional_custom_id"
}`}
            response={`{
  "session_id": "agent_xyz789",
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_at": "2024-01-16T10:00:00Z"
}`}
          />
        </div>
      </section>

      {/* Error Codes */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Error Codes</h2>
        <Card className="bg-card/50 border-border overflow-hidden">
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-card/80">
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-muted-foreground">Code</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">HTTP Status</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">Description</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">VALIDATION_ERROR</td>
                    <td className="py-3 px-4 text-muted-foreground">400</td>
                    <td className="py-3 px-4 text-muted-foreground">Invalid request parameters</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">UNAUTHORIZED</td>
                    <td className="py-3 px-4 text-muted-foreground">401</td>
                    <td className="py-3 px-4 text-muted-foreground">Missing or invalid authentication</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">FORBIDDEN</td>
                    <td className="py-3 px-4 text-muted-foreground">403</td>
                    <td className="py-3 px-4 text-muted-foreground">Insufficient permissions</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">NOT_FOUND</td>
                    <td className="py-3 px-4 text-muted-foreground">404</td>
                    <td className="py-3 px-4 text-muted-foreground">Resource not found</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">RATE_LIMITED</td>
                    <td className="py-3 px-4 text-muted-foreground">429</td>
                    <td className="py-3 px-4 text-muted-foreground">Rate limit exceeded</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">INTERNAL_ERROR</td>
                    <td className="py-3 px-4 text-muted-foreground">500</td>
                    <td className="py-3 px-4 text-muted-foreground">Internal server error</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>
    </motion.div>
  )
}
