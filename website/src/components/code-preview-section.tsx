"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Copy, Check } from "lucide-react"

const codeSnippets = {
  python: `# Create an A2A System
from rla2a import A2ASystem, Agent

# Initialize the system
system = A2ASystem()

# Create agents with different roles
alice = system.create_agent(
    name="Alice",
    role="researcher",
    ai_provider="openai"
)

bob = system.create_agent(
    name="Bob", 
    role="analyst",
    ai_provider="anthropic"
)

# Start the server
await system.start_server()`,

  typescript: `// Create an A2A Client
import { A2AClient } from 'rl-a2a-sdk';

// Initialize client
const client = new A2AClient({
  serverUrl: 'http://localhost:8000',
  agentId: 'my-agent'
});

// Register agent
await client.register();

// Send observation and get action
const response = await client.sendObservation({
  position: { x: 10, y: 20 },
  energy: 85,
  emotion: 'curious'
});

console.log('Action:', response.command);`,

  api: `# REST API Endpoints

# Create a new agent
POST /agents
{
  "name": "Agent-1",
  "role": "coordinator",
  "ai_provider": "openai"
}

# List all agents
GET /agents

# Send a message
POST /messages
{
  "sender_id": "agent-1",
  "receiver_id": "agent-2",
  "content": "Hello, ready to collaborate?"
}

# Get system status
GET /status`,
}

const tabs = [
  { id: "python", label: "Python" },
  { id: "typescript", label: "TypeScript" },
  { id: "api", label: "REST API" },
]

export function CodePreviewSection() {
  const [activeTab, setActiveTab] = useState<"python" | "typescript" | "api">("python")
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(codeSnippets[activeTab])
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <section className="relative py-24 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 grid-pattern opacity-20" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <Badge variant="outline" className="mb-4">
            Developer Experience
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Simple, Powerful{" "}
            <span className="gradient-text">APIs</span>
          </h2>
          <p className="max-w-2xl mx-auto text-muted-foreground text-lg">
            Get started in minutes with our intuitive SDKs and comprehensive
            documentation.
          </p>
        </motion.div>

        {/* Code Editor */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <Card className="overflow-hidden bg-card/80 backdrop-blur-sm border-border">
            {/* Editor Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-card">
              {/* Tabs */}
              <div className="flex items-center gap-1">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as typeof activeTab)}
                    className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                      activeTab === tab.id
                        ? "bg-primary/20 text-primary"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Copy Button */}
              <button
                onClick={copyToClipboard}
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
              >
                {copied ? (
                  <>
                    <Check className="w-4 h-4 text-green-500" />
                    <span className="text-green-500">Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-4 h-4" />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>

            {/* Code Content */}
            <div className="relative">
              {/* Line Numbers */}
              <div className="absolute left-0 top-0 bottom-0 w-12 bg-card/50 border-r border-border flex flex-col items-end pr-3 py-4 text-muted-foreground/50 text-sm font-mono select-none">
                {codeSnippets[activeTab].split("\n").map((_, i) => (
                  <div key={i} className="leading-6">
                    {i + 1}
                  </div>
                ))}
              </div>

              {/* Code */}
              <div className="pl-14 pr-4 py-4 overflow-x-auto">
                <AnimatePresence mode="wait">
                  <motion.pre
                    key={activeTab}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                    className="text-sm font-mono leading-6 text-foreground"
                  >
                    {codeSnippets[activeTab]}
                  </motion.pre>
                </AnimatePresence>
              </div>
            </div>
          </Card>
        </motion.div>

        {/* Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          {[
            { label: "npm install rl-a2a-sdk", value: "Install" },
            { label: "TypeScript Support", value: "100%" },
            { label: "API Reference", value: "Complete" },
            { label: "Example Projects", value: "10+" },
          ].map((item, index) => (
            <Card
              key={index}
              className="bg-card/50 backdrop-blur-sm border-border p-4 text-center"
            >
              <div className="text-lg font-semibold text-primary mb-1">
                {item.value}
              </div>
              <div className="text-sm text-muted-foreground">{item.label}</div>
            </Card>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
