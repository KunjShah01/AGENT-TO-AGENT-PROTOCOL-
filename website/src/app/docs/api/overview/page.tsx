"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Code, Globe, Radio, Package, ArrowRight, Webhook } from "lucide-react"
import Link from "next/link"

export default function APIOverviewPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">API Reference</Badge>
        <h1 className="text-4xl font-bold text-foreground">API Overview</h1>
        <p className="text-xl text-muted-foreground">
          RL-A2A provides multiple interfaces for interacting with the platform:
          REST API, WebSocket API, and native SDKs for Python and TypeScript.
        </p>
      </div>

      {/* API Types */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Available Interfaces</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-blue-500/10">
                  <Globe className="w-5 h-5 text-blue-500" />
                </div>
                <h3 className="font-semibold text-foreground">REST API</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                HTTP-based API for agent management, messaging, and system
                configuration. Ideal for integration with existing systems.
              </p>
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="secondary" className="text-xs">HTTP/HTTPS</Badge>
                <Badge variant="secondary" className="text-xs">JSON</Badge>
                <Badge variant="secondary" className="text-xs">OpenAPI</Badge>
              </div>
              <Link href="/docs/api/rest">
                <Button variant="ghost" size="sm" className="text-primary">
                  View REST API <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-green-500/10">
                  <Webhook className="w-5 h-5 text-green-500" />
                </div>
                <h3 className="font-semibold text-foreground">WebSocket API</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Real-time bidirectional communication for agent interactions.
                Perfect for high-frequency message exchange.
              </p>
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="secondary" className="text-xs">WS/WSS</Badge>
                <Badge variant="secondary" className="text-xs">Real-time</Badge>
                <Badge variant="secondary" className="text-xs">Low Latency</Badge>
              </div>
              <Link href="/docs/api/websocket">
                <Button variant="ghost" size="sm" className="text-primary">
                  View WebSocket API <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-purple-500/10">
                  <Package className="w-5 h-5 text-purple-500" />
                </div>
                <h3 className="font-semibold text-foreground">Python SDK</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                Native Python library with full type hints and async support.
                The recommended way to build agents in Python.
              </p>
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="secondary" className="text-xs">Python 3.8+</Badge>
                <Badge variant="secondary" className="text-xs">Async</Badge>
                <Badge variant="secondary" className="text-xs">Type Hints</Badge>
              </div>
              <Link href="/docs/api/python-sdk">
                <Button variant="ghost" size="sm" className="text-primary">
                  View Python SDK <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="p-2 rounded-lg bg-yellow-500/10">
                  <Code className="w-5 h-5 text-yellow-500" />
                </div>
                <h3 className="font-semibold text-foreground">TypeScript SDK</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-4">
                TypeScript/JavaScript SDK for building web-based agents and
                frontend integrations.
              </p>
              <div className="flex items-center gap-2 mb-4">
                <Badge variant="secondary" className="text-xs">TypeScript</Badge>
                <Badge variant="secondary" className="text-xs">Node.js</Badge>
                <Badge variant="secondary" className="text-xs">Browser</Badge>
              </div>
              <Link href="/docs/api/typescript-sdk">
                <Button variant="ghost" size="sm" className="text-primary">
                  View TypeScript SDK <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Base URL */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Base URL</h2>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Development</h3>
                <code className="block p-3 bg-muted rounded-lg text-foreground font-mono">
                  http://localhost:8000
                </code>
              </div>
              <div>
                <h3 className="text-sm font-medium text-muted-foreground mb-2">Production</h3>
                <code className="block p-3 bg-muted rounded-lg text-foreground font-mono">
                  https://api.rl-a2a.io
                </code>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Authentication */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Authentication</h2>
        <p className="text-muted-foreground">
          Most API endpoints require authentication using JWT tokens:
        </p>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Authorization Header</h3>
            <pre className="p-3 bg-muted rounded-lg text-foreground font-mono text-sm overflow-x-auto">
{`Authorization: Bearer <your_jwt_token>`}
            </pre>
          </CardContent>
        </Card>
      </section>

      {/* Response Format */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Response Format</h2>
        <p className="text-muted-foreground">
          All API responses are in JSON format with consistent structure:
        </p>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Success Response</h3>
            <pre className="p-3 bg-muted rounded-lg text-foreground font-mono text-sm overflow-x-auto">
{`{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}`}
            </pre>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <h3 className="text-sm font-medium text-muted-foreground mb-2">Error Response</h3>
            <pre className="p-3 bg-muted rounded-lg text-foreground font-mono text-sm overflow-x-auto">
{`{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": { ... }
  }
}`}
            </pre>
          </CardContent>
        </Card>
      </section>

      {/* Rate Limits */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Rate Limits</h2>
        <Card className="bg-card/50 border-border overflow-hidden">
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-card/80">
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-muted-foreground">Endpoint Type</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">Rate Limit</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">Burst</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">REST API</td>
                    <td className="py-3 px-4 text-muted-foreground">60 requests/minute</td>
                    <td className="py-3 px-4 text-muted-foreground">10 requests</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">WebSocket</td>
                    <td className="py-3 px-4 text-muted-foreground">100 messages/second</td>
                    <td className="py-3 px-4 text-muted-foreground">50 messages</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4">AI Generation</td>
                    <td className="py-3 px-4 text-muted-foreground">30 requests/minute</td>
                    <td className="py-3 px-4 text-muted-foreground">5 requests</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* OpenAPI Spec */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">OpenAPI Specification</h2>
        <p className="text-muted-foreground">
          Access the full OpenAPI specification for API documentation and client generation:
        </p>
        <div className="flex gap-4">
          <Button variant="outline">
            <Globe className="mr-2 w-4 h-4" />
            View Swagger UI
          </Button>
          <Button variant="outline">
            <Code className="mr-2 w-4 h-4" />
            Download OpenAPI Spec
          </Button>
        </div>
      </section>
    </motion.div>
  )
}
