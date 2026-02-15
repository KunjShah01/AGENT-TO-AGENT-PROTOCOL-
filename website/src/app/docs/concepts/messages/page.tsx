"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Copy, Check, MessageSquare, Send, ArrowRight } from "lucide-react"
import { useState } from "react"

function CopyableCode({ code, title }: { code: string; title?: string }) {
  const [copied, setCopied] = useState(false)

  return (
    <div className="relative">
      {title && (
        <div className="px-4 py-2 border-b border-border bg-card/80 rounded-t-lg">
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
      )}
      <pre className={`p-4 bg-card/50 border border-border ${title ? 'rounded-b-lg border-t-0' : 'rounded-lg'} text-sm font-mono text-foreground overflow-x-auto`}>
        <code>{code}</code>
      </pre>
      <button
        onClick={() => {
          navigator.clipboard.writeText(code)
          setCopied(true)
          setTimeout(() => setCopied(false), 2000)
        }}
        className="absolute top-2 right-2 p-2 rounded bg-muted hover:bg-muted/80 transition-colors"
      >
        {copied ? (
          <Check className="w-4 h-4 text-green-500" />
        ) : (
          <Copy className="w-4 h-4 text-muted-foreground" />
        )}
      </button>
    </div>
  )
}

export default function MessagesPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Core Concepts</Badge>
        <h1 className="text-4xl font-bold text-foreground">Messages</h1>
        <p className="text-xl text-muted-foreground">
          Messages are the primary means of communication between agents. They
          carry information, commands, and feedback across the multi-agent system.
        </p>
      </div>

      {/* Message Structure */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Message Structure</h2>
        <p className="text-muted-foreground">
          Every message in RL-A2A follows a standardized structure:
        </p>
        <CopyableCode
          title="Message Schema"
          code={`{
  "id": "msg_abc123",
  "sender_id": "agent_001",
  "receiver_id": "agent_002",
  "content": "Hello, ready to collaborate?",
  "message_type": "text",
  "priority": 1,
  "metadata": {
    "context": "task_assignment",
    "session_id": "session_xyz"
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "encrypted": false,
  "signature": null
}`}
        />
      </section>

      {/* Message Properties */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Message Properties</h2>
        <Card className="bg-card/50 border-border overflow-hidden">
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-card/80">
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-muted-foreground">Property</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">Type</th>
                    <th className="text-left py-3 px-4 text-muted-foreground">Description</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">id</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">Unique message identifier</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">sender_id</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">ID of the sending agent</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">receiver_id</td>
                    <td className="py-3 px-4 text-muted-foreground">str | null</td>
                    <td className="py-3 px-4 text-muted-foreground">ID of receiving agent (null for broadcast)</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">content</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">Message payload/content</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">message_type</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">Type: text, command, observation, feedback</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">priority</td>
                    <td className="py-3 px-4 text-muted-foreground">int</td>
                    <td className="py-3 px-4 text-muted-foreground">Message priority (1-10, higher = more urgent)</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">metadata</td>
                    <td className="py-3 px-4 text-muted-foreground">Dict</td>
                    <td className="py-3 px-4 text-muted-foreground">Additional context and metadata</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Message Types */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Message Types</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <MessageSquare className="w-5 h-5 text-blue-500" />
                <h3 className="font-semibold text-foreground">text</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Standard text communication between agents. Used for general
                information exchange and collaboration.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <ArrowRight className="w-5 h-5 text-purple-500" />
                <h3 className="font-semibold text-foreground">command</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Action directives sent to agents. Contains instructions for
                specific actions to be performed.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Send className="w-5 h-5 text-green-500" />
                <h3 className="font-semibold text-foreground">observation</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                State information sent from agents to the environment or
                other agents. Contains current observations.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Badge className="w-5 h-5 text-yellow-500" />
                <h3 className="font-semibold text-foreground">feedback</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                RL feedback messages containing rewards and performance
                metrics for learning updates.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Sending Messages */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Sending Messages</h2>
        <CopyableCode
          title="Python"
          code={`from rla2a import Message

# Create a message
message = Message(
    id="msg_001",
    sender_id="agent_alice",
    receiver_id="agent_bob",
    content="Task completed successfully",
    message_type="text",
    priority=1
)

# Send via system
await system.send_message(message)

# Or use the REST API
import requests

response = requests.post(
    "http://localhost:8000/messages",
    json={
        "sender_id": "agent_alice",
        "receiver_id": "agent_bob",
        "content": "Ready for next task",
        "message_type": "text",
        "priority": 1
    }
)`}
        />
      </section>

      {/* WebSocket Communication */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">WebSocket Communication</h2>
        <p className="text-muted-foreground">
          For real-time communication, use WebSocket connections:
        </p>
        <CopyableCode
          title="Python"
          code={`import asyncio
import websockets
import json

async def agent_communication():
    # Connect to WebSocket
    ws_url = "ws://localhost:8000/ws/agent_id"
    async with websockets.connect(ws_url) as websocket:
        # Send observation
        observation = {
            "agent_id": "agent_001",
            "position": {"x": 10, "y": 20},
            "energy": 85
        }
        await websocket.send(json.dumps(observation))
        
        # Receive response
        response = await websocket.recv()
        action = json.loads(response)
        print(f"Received action: {action['command']}")`}
        />
      </section>

      {/* Message Flow */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Message Flow</h2>
        <div className="bg-card/50 border border-border rounded-lg p-6">
          <pre className="text-sm text-muted-foreground font-mono overflow-x-auto">
{`┌──────────┐     Message      ┌──────────┐
│  Agent A │ ───────────────► │  Agent B │
└──────────┘                  └──────────┘
     │                             │
     │ 1. Create Message            │ 4. Process Message
     │ 2. Send to Hub               │ 5. Generate Response
     │                              │
     ▼                              ▼
┌──────────────────────────────────────────┐
│           Communication Hub              │
│  ┌─────────────────────────────────────┐ │
│  │  Message Queue                      │ │
│  │  • Priority-based ordering          │ │
│  │  • Delivery guarantees              │ │
│  │  • Retry logic                      │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │  Message Router                     │ │
│  │  • Direct routing                   │ │
│  │  • Broadcast                        │ │
│  │  • Topic-based pub/sub              │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘`}
          </pre>
        </div>
      </section>

      {/* Best Practices */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Best Practices</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Message Size</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Keep messages under 1MB</li>
                <li>• Use compression for large payloads</li>
                <li>• Consider batching for high volume</li>
              </ul>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Priority Usage</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Use priority 1-3 for normal messages</li>
                <li>• Use priority 4-7 for important updates</li>
                <li>• Use priority 8-10 for critical alerts</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>
    </motion.div>
  )
}
