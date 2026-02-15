"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowRight, Github, BookOpen, Zap, Network, Brain, Rocket, Code } from "lucide-react"
import Link from "next/link"

export default function DocsIntroduction() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Documentation</Badge>
        <h1 className="text-4xl font-bold text-foreground">
          Welcome to RL-A2A
        </h1>
        <p className="text-xl text-muted-foreground">
          The Reinforcement Learning Agent-to-Agent Communication Platform.
          Build intelligent multi-agent systems with decentralized learning and
          real-time communication.
        </p>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Rocket className="w-5 h-5 text-primary" />
              </div>
              <h3 className="font-semibold text-foreground">Quick Start</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Get up and running in minutes with our step-by-step guide.
            </p>
            <Link href="/docs/quick-start">
              <Button variant="ghost" size="sm" className="text-primary">
                Get Started <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <Code className="w-5 h-5 text-purple-500" />
              </div>
              <h3 className="font-semibold text-foreground">API Reference</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              Complete API documentation for REST, WebSocket, and SDKs.
            </p>
            <Link href="/docs/api/overview">
              <Button variant="ghost" size="sm" className="text-purple-500">
                Explore APIs <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>

        <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <BookOpen className="w-5 h-5 text-green-500" />
              </div>
              <h3 className="font-semibold text-foreground">Guides</h3>
            </div>
            <p className="text-sm text-muted-foreground mb-4">
              In-depth tutorials for building multi-agent systems.
            </p>
            <Link href="/docs/guides/first-agent">
              <Button variant="ghost" size="sm" className="text-green-500">
                View Guides <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>

      {/* What is RL-A2A */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">What is RL-A2A?</h2>
        <div className="prose prose-invert max-w-none">
          <p className="text-muted-foreground">
            RL-A2A is a comprehensive platform for building multi-agent systems
            powered by reinforcement learning. It provides the infrastructure
            for agents to communicate, learn from each other, and collaborate
            to solve complex problems.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Network className="w-5 h-5 text-primary" />
              <h3 className="font-semibold text-foreground">Decentralized Architecture</h3>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              Agents operate independently while maintaining cohesive collaboration.
              No single point of failure, enhanced robustness and scalability.
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Brain className="w-5 h-5 text-purple-500" />
              <h3 className="font-semibold text-foreground">Reinforcement Learning</h3>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              Advanced RL algorithms enable agents to learn optimal strategies
              through interaction and feedback from the environment.
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Zap className="w-5 h-5 text-yellow-500" />
              <h3 className="font-semibold text-foreground">Real-Time Communication</h3>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              WebSocket-based messaging with sub-millisecond latency. Support
              for 10,000+ messages per second.
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <Github className="w-5 h-5 text-green-500" />
              <h3 className="font-semibold text-foreground">Open Source</h3>
            </div>
            <p className="text-sm text-muted-foreground pl-8">
              MIT licensed and fully open source. Contribute, customize, and
              extend to fit your needs.
            </p>
          </div>
        </div>
      </section>

      {/* System Architecture */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">System Architecture</h2>
        <div className="bg-card/50 border border-border rounded-lg p-6">
          <pre className="text-sm text-muted-foreground font-mono overflow-x-auto">
{`┌─────────────────────────────────────────────────────────────┐
│                        RL-A2A Platform                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agent A   │  │   Agent B   │  │   Agent C   │  ...    │
│  │  (OpenAI)   │  │ (Anthropic) │  │  (Google)   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │   Communication Hub   │                      │
│              │    (WebSocket/REST)   │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │   RL Learning Engine  │                      │
│              │  (Reward Processing)  │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘`}
          </pre>
        </div>
      </section>

      {/* Key Features */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[
            { title: "Multi-Agent Orchestration", desc: "Coordinate hundreds of agents with different roles and capabilities" },
            { title: "AI Provider Integration", desc: "Seamlessly switch between OpenAI, Anthropic, Google AI, and Ollama" },
            { title: "Security & Authentication", desc: "JWT-based auth, rate limiting, and encrypted communications" },
            { title: "Performance Monitoring", desc: "Real-time dashboards with agent metrics and system health" },
            { title: "MCP Support", desc: "Model Context Protocol integration for AI assistants" },
            { title: "Cloud Native", desc: "Docker and Kubernetes ready for easy deployment" },
          ].map((feature) => (
            <div
              key={feature.title}
              className="p-4 rounded-lg bg-card/30 border border-border"
            >
              <h3 className="font-medium text-foreground mb-1">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Next Steps */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Next Steps</h2>
        <div className="flex flex-wrap gap-4">
          <Link href="/docs/quick-start">
            <Button className="bg-primary hover:bg-primary/90">
              Quick Start Guide <ArrowRight className="ml-2 w-4 h-4" />
            </Button>
          </Link>
          <Link href="/docs/installation">
            <Button variant="outline">Installation</Button>
          </Link>
          <Link href="https://github.com/KunjShah01/RL-A2A" target="_blank">
            <Button variant="ghost">
              <Github className="mr-2 w-4 h-4" /> View on GitHub
            </Button>
          </Link>
        </div>
      </section>
    </motion.div>
  )
}
