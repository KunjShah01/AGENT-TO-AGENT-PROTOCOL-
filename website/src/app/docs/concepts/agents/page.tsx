"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Copy, Check, Users, Brain, Zap, Settings } from "lucide-react"
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

export default function AgentsPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Core Concepts</Badge>
        <h1 className="text-4xl font-bold text-foreground">Agents</h1>
        <p className="text-xl text-muted-foreground">
          Agents are the fundamental building blocks of RL-A2A. Each agent is an
          autonomous entity that can perceive, reason, act, and learn from its
          environment.
        </p>
      </div>

      {/* What is an Agent */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">What is an Agent?</h2>
        <p className="text-muted-foreground">
          In RL-A2A, an agent is an intelligent entity with the following characteristics:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Brain className="w-5 h-5 text-purple-500" />
                <h3 className="font-semibold text-foreground">Autonomous</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Agents operate independently, making decisions based on their
                observations and learned policies without constant human oversight.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Zap className="w-5 h-5 text-yellow-500" />
                <h3 className="font-semibold text-foreground">Reactive</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Agents respond to changes in their environment in real-time,
                adapting their behavior based on new information.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Users className="w-5 h-5 text-blue-500" />
                <h3 className="font-semibold text-foreground">Social</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Agents can communicate with other agents, sharing information
                and coordinating actions to achieve common goals.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Settings className="w-5 h-5 text-green-500" />
                <h3 className="font-semibold text-foreground">Adaptive</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Agents learn from experience using reinforcement learning,
                improving their decision-making over time.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Agent Properties */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Agent Properties</h2>
        <p className="text-muted-foreground">
          Each agent has the following core properties:
        </p>
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
                    <td className="py-3 px-4 text-muted-foreground">Unique identifier for the agent</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">name</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">Human-readable name</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">role</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">Agent's role (researcher, analyst, coordinator, etc.)</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">capabilities</td>
                    <td className="py-3 px-4 text-muted-foreground">List[str]</td>
                    <td className="py-3 px-4 text-muted-foreground">List of agent capabilities</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">state</td>
                    <td className="py-3 px-4 text-muted-foreground">Dict</td>
                    <td className="py-3 px-4 text-muted-foreground">Current state of the agent</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">memory</td>
                    <td className="py-3 px-4 text-muted-foreground">List[Dict]</td>
                    <td className="py-3 px-4 text-muted-foreground">Agent's memory/history</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">ai_provider</td>
                    <td className="py-3 px-4 text-muted-foreground">str</td>
                    <td className="py-3 px-4 text-muted-foreground">AI provider (openai, anthropic, google)</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-3 px-4 font-mono text-primary">performance_metrics</td>
                    <td className="py-3 px-4 text-muted-foreground">Dict</td>
                    <td className="py-3 px-4 text-muted-foreground">Performance tracking metrics</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Creating Agents */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Creating Agents</h2>
        <p className="text-muted-foreground">
          Create agents using the A2ASystem:
        </p>
        <CopyableCode
          title="Python"
          code={`from rla2a import A2ASystem

system = A2ASystem()

# Basic agent creation
agent_id = system.create_agent(
    name="Alice",
    role="researcher"
)

# Agent with AI provider
agent_id = system.create_agent(
    name="Bob",
    role="analyst",
    ai_provider="anthropic"
)

# Agent with custom capabilities
agent_id = system.create_agent(
    name="Charlie",
    role="coordinator",
    capabilities=["communication", "task_routing", "monitoring"],
    ai_provider="openai"
)

# Agent with security level
agent_id = system.create_agent(
    name="SecureAgent",
    role="specialist",
    security_level="high"
)`}
        />
      </section>

      {/* Agent Roles */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Agent Roles</h2>
        <p className="text-muted-foreground">
          Roles define the agent's primary function in the system:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            { role: "researcher", desc: "Gathers and analyzes information", color: "text-blue-500" },
            { role: "analyst", desc: "Processes and interprets data", color: "text-purple-500" },
            { role: "coordinator", desc: "Manages task distribution", color: "text-green-500" },
            { role: "specialist", desc: "Domain-specific expertise", color: "text-yellow-500" },
            { role: "monitor", desc: "Observes and reports status", color: "text-red-500" },
            { role: "executor", desc: "Performs actions and tasks", color: "text-cyan-500" },
          ].map((item) => (
            <Card key={item.role} className="bg-card/50 border-border">
              <CardContent className="p-4">
                <h3 className={`font-mono font-semibold ${item.color}`}>{item.role}</h3>
                <p className="text-sm text-muted-foreground mt-1">{item.desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Agent Lifecycle */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Agent Lifecycle</h2>
        <div className="bg-card/50 border border-border rounded-lg p-6">
          <pre className="text-sm text-muted-foreground font-mono overflow-x-auto">
{`┌─────────────┐
│   Created   │  Agent is initialized with properties
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Register   │  Agent registers with the A2A system
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Active    │  Agent is connected and processing
└──────┬──────┘
       │
       ├──────► Observe ──────┐
       │                      │
       ├──────► Decide ───────┤
       │                      │
       ├──────► Act ──────────┤
       │                      │
       └──────► Learn ◄───────┘
              (Loop)
       │
       ▼
┌─────────────┐
│  Inactive   │  Agent disconnected but preserved
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Removed   │  Agent is deleted from the system
└─────────────┘`}
          </pre>
        </div>
      </section>

      {/* Managing Agents */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Managing Agents</h2>
        <CopyableCode
          title="Python"
          code={`# List all agents
agents = system.list_agents()
for agent in agents:
    print(f"{agent.name} ({agent.role}) - {agent.status}")

# Get specific agent
agent = system.get_agent(agent_id)
print(f"Agent: {agent.name}")
print(f"Capabilities: {agent.capabilities}")
print(f"Performance: {agent.performance_metrics}")

# Update agent state
agent.state["energy"] = 85
agent.state["emotion"] = "focused"

# Remove agent
system.remove_agent(agent_id)`}
        />
      </section>
    </motion.div>
  )
}
