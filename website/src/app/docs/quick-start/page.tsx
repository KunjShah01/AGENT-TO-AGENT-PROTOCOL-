"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Copy, Check, Terminal, Play, Zap } from "lucide-react"
import { useState } from "react"

const codeBlocks = [
  {
    title: "Install RL-A2A",
    language: "bash",
    code: `# Using pip
pip install rl-a2a

# Or using poetry
poetry add rl-a2a`,
  },
  {
    title: "Create Your First Agent",
    language: "python",
    code: `from rla2a import A2ASystem

# Initialize the system
system = A2ASystem()

# Create an agent
agent_id = system.create_agent(
    name="MyFirstAgent",
    role="assistant",
    ai_provider="openai"
)

print(f"Agent created: {agent_id}")`,
  },
  {
    title: "Start the Server",
    language: "python",
    code: `# Start the A2A server
await system.start_server()

# Or use the CLI
# python -m rla2a server --port 8000`,
  },
]

function CodeBlock({ title, language, code }: { title: string; language: string; code: string }) {
  const [copied, setCopied] = useState(false)

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card className="bg-card/50 border-border overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card/80">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-foreground">{title}</span>
        </div>
        <button
          onClick={copyToClipboard}
          className="flex items-center gap-1 px-2 py-1 rounded text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-green-500" />
              <span className="text-green-500">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <CardContent className="p-0">
        <pre className="p-4 text-sm font-mono text-foreground overflow-x-auto">
          <code>{code}</code>
        </pre>
      </CardContent>
    </Card>
  )
}

export default function QuickStartPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Getting Started</Badge>
        <h1 className="text-4xl font-bold text-foreground">Quick Start</h1>
        <p className="text-xl text-muted-foreground">
          Get RL-A2A up and running in under 5 minutes. This guide will walk
          you through the basics of creating and running your first multi-agent system.
        </p>
      </div>

      {/* Prerequisites */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Prerequisites</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-lg bg-card/30 border border-border">
            <h3 className="font-medium text-foreground mb-2">Python 3.8+</h3>
            <p className="text-sm text-muted-foreground">
              RL-A2A requires Python 3.8 or higher
            </p>
          </div>
          <div className="p-4 rounded-lg bg-card/30 border border-border">
            <h3 className="font-medium text-foreground mb-2">API Key (Optional)</h3>
            <p className="text-sm text-muted-foreground">
              OpenAI, Anthropic, or Google AI key for AI features
            </p>
          </div>
          <div className="p-4 rounded-lg bg-card/30 border border-border">
            <h3 className="font-medium text-foreground mb-2">pip or poetry</h3>
            <p className="text-sm text-muted-foreground">
              Package manager for installation
            </p>
          </div>
        </div>
      </section>

      {/* Installation Steps */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-foreground">Installation</h2>
        
        {/* Step 1 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              1
            </div>
            <h3 className="text-lg font-semibold text-foreground">Install the Package</h3>
          </div>
          <CodeBlock
            title="Terminal"
            language="bash"
            code={`# Using pip
pip install rl-a2a

# Or using poetry
poetry add rl-a2a`}
          />
        </div>

        {/* Step 2 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              2
            </div>
            <h3 className="text-lg font-semibold text-foreground">Set Up Environment Variables</h3>
          </div>
          <p className="text-muted-foreground">
            Create a <code className="px-1 py-0.5 rounded bg-muted text-foreground">.env</code> file in your project root:
          </p>
          <CodeBlock
            title=".env"
            language="bash"
            code={`# AI Provider API Keys (at least one required for AI features)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

# Server Configuration
A2A_HOST=localhost
A2A_PORT=8000

# Optional: Security
SECRET_KEY=your_secret_key_here`}
          />
        </div>

        {/* Step 3 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              3
            </div>
            <h3 className="text-lg font-semibold text-foreground">Create Your First Agent</h3>
          </div>
          <p className="text-muted-foreground">
            Create a new file <code className="px-1 py-0.5 rounded bg-muted text-foreground">main.py</code>:
          </p>
          <CodeBlock
            title="main.py"
            language="python"
            code={`import asyncio
from rla2a import A2ASystem

async def main():
    # Initialize the system
    system = A2ASystem()
    
    # Create agents with different roles
    researcher = system.create_agent(
        name="Alice",
        role="researcher",
        ai_provider="openai"
    )
    
    analyst = system.create_agent(
        name="Bob",
        role="analyst",
        ai_provider="anthropic"
    )
    
    print(f"Created agents: {researcher}, {analyst}")
    
    # Start the server
    await system.start_server()

if __name__ == "__main__":
    asyncio.run(main())`}
          />
        </div>

        {/* Step 4 */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
              4
            </div>
            <h3 className="text-lg font-semibold text-foreground">Run Your Application</h3>
          </div>
          <CodeBlock
            title="Terminal"
            language="bash"
            code={`# Run directly
python main.py

# Or use the CLI
rla2a server --demo-agents 3`}
          />
          <div className="flex items-center gap-2 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
            <Zap className="w-5 h-5 text-green-500" />
            <p className="text-sm text-green-400">
              Your server is now running at <code className="px-1 py-0.5 rounded bg-green-500/20">http://localhost:8000</code>
            </p>
          </div>
        </div>
      </section>

      {/* Verify Installation */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Verify Installation</h2>
        <p className="text-muted-foreground">
          Open your browser and navigate to <code className="px-1 py-0.5 rounded bg-muted text-foreground">http://localhost:8000</code> to see the API documentation.
        </p>
        <p className="text-muted-foreground">
          You can also check the system status:
        </p>
        <CodeBlock
          title="Terminal"
          language="bash"
          code={`# Check system status
curl http://localhost:8000/status

# List all agents
curl http://localhost:8000/agents`}
          />
      </section>

      {/* Next Steps */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Next Steps</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">Learn Core Concepts</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Understand agents, messages, and the RL learning loop.
              </p>
              <Button variant="ghost" size="sm" className="text-primary">
                View Concepts <Play className="ml-2 w-4 h-4" />
              </Button>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">Build a Multi-Agent System</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Create a collaborative multi-agent application.
              </p>
              <Button variant="ghost" size="sm" className="text-primary">
                Start Tutorial <Play className="ml-2 w-4 h-4" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>
    </motion.div>
  )
}
