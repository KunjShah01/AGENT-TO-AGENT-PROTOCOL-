"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Copy, Check, Download, Package, Container } from "lucide-react"
import { useState } from "react"

function CopyableCode({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  return (
    <div className="relative">
      <pre className="p-4 bg-card/50 border border-border rounded-lg text-sm font-mono text-foreground overflow-x-auto">
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

export default function InstallationPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Getting Started</Badge>
        <h1 className="text-4xl font-bold text-foreground">Installation</h1>
        <p className="text-xl text-muted-foreground">
          Multiple ways to install and run RL-A2A. Choose the method that best
          fits your development workflow.
        </p>
      </div>

      {/* Installation Methods */}
      <Tabs defaultValue="pip" className="space-y-6">
        <TabsList className="bg-card/50 border border-border">
          <TabsTrigger value="pip" className="data-[state=active]:bg-primary/20">
            <Package className="w-4 h-4 mr-2" />
            pip
          </TabsTrigger>
          <TabsTrigger value="poetry" className="data-[state=active]:bg-primary/20">
            <Package className="w-4 h-4 mr-2" />
            Poetry
          </TabsTrigger>
          <TabsTrigger value="docker" className="data-[state=active]:bg-primary/20">
            <Container className="w-4 h-4 mr-2" />
            Docker
          </TabsTrigger>
          <TabsTrigger value="source" className="data-[state=active]:bg-primary/20">
            <Download className="w-4 h-4 mr-2" />
            From Source
          </TabsTrigger>
        </TabsList>

        {/* pip Installation */}
        <TabsContent value="pip" className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Using pip</h2>
          <p className="text-muted-foreground">
            The simplest way to install RL-A2A is using pip:
          </p>
          <CopyableCode code={`# Basic installation
pip install rl-a2a

# With all optional dependencies
pip install rl-a2a[all]

# With specific extras
pip install rl-a2a[openai,anthropic,google]

# Development dependencies
pip install rl-a2a[dev]`} />
          
          <h3 className="text-lg font-semibold text-foreground mt-6">Available Extras</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { name: "openai", desc: "OpenAI integration" },
              { name: "anthropic", desc: "Anthropic integration" },
              { name: "google", desc: "Google AI integration" },
              { name: "ollama", desc: "Ollama (local) integration" },
              { name: "security", desc: "JWT, bcrypt, security features" },
              { name: "viz", desc: "Visualization and dashboards" },
              { name: "mcp", desc: "MCP protocol support" },
              { name: "dev", desc: "Development tools" },
              { name: "all", desc: "All dependencies" },
            ].map((extra) => (
              <div
                key={extra.name}
                className="p-3 rounded-lg bg-card/30 border border-border"
              >
                <code className="text-sm text-primary">{extra.name}</code>
                <p className="text-xs text-muted-foreground mt-1">{extra.desc}</p>
              </div>
            ))}
          </div>
        </TabsContent>

        {/* Poetry Installation */}
        <TabsContent value="poetry" className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Using Poetry</h2>
          <p className="text-muted-foreground">
            Poetry provides better dependency management for Python projects:
          </p>
          <CopyableCode code={`# Add to your project
poetry add rl-a2a

# With extras
poetry add rl-a2a --extras "openai anthropic"

# Add all extras
poetry add rl-a2a --all-extras`} />
        </TabsContent>

        {/* Docker Installation */}
        <TabsContent value="docker" className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Using Docker</h2>
          <p className="text-muted-foreground">
            Run RL-A2A in a containerized environment:
          </p>
          <CopyableCode code={`# Pull the image
docker pull rl-a2a/latest

# Run with default settings
docker run -p 8000:8000 rl-a2a/latest

# Run with environment variables
docker run -p 8000:8000 \\
  -e OPENAI_API_KEY=your_key \\
  -e ANTHROPIC_API_KEY=your_key \\
  rl-a2a/latest

# Using docker-compose
docker-compose up -d`} />

          <h3 className="text-lg font-semibold text-foreground mt-6">docker-compose.yml</h3>
          <CopyableCode code={`version: '3.8'

services:
  rl-a2a:
    image: rl-a2a/latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=\${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=\${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=\${GOOGLE_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped`} />
        </TabsContent>

        {/* Source Installation */}
        <TabsContent value="source" className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Installing from Source</h2>
          <p className="text-muted-foreground">
            For development or to get the latest features:
          </p>
          <CopyableCode code={`# Clone the repository
git clone https://github.com/KunjShah01/RL-A2A.git
cd RL-A2A

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install in development mode
pip install -e ".[dev]"

# Or using poetry
poetry install --all-extras`} />
        </TabsContent>
      </Tabs>

      {/* System Requirements */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">System Requirements</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-4">Minimum Requirements</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Python 3.8 or higher</li>
                <li>• 2 GB RAM</li>
                <li>• 1 CPU core</li>
                <li>• 500 MB disk space</li>
              </ul>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-4">Recommended</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Python 3.10 or higher</li>
                <li>• 4 GB RAM</li>
                <li>• 2+ CPU cores</li>
                <li>• 2 GB disk space</li>
                <li>• GPU (for local AI models)</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Verify Installation */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Verify Installation</h2>
        <p className="text-muted-foreground">
          Run the following command to verify your installation:
        </p>
        <CopyableCode code={`# Check version
python -c "import rla2a; print(rla2a.__version__)"

# Or using CLI
rla2a info

# Expected output:
# RL-A2A Combined Enhanced
# Version: 2.0.0
# Security: Enhanced
# AI Providers: OpenAI, Anthropic, Google`} />
      </section>

      {/* Troubleshooting */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Troubleshooting</h2>
        <div className="space-y-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">Import Error: No module named 'rla2a'</h3>
              <p className="text-sm text-muted-foreground mb-3">
                Make sure you've activated your virtual environment and installed the package.
              </p>
              <CopyableCode code={`# Activate virtual environment
source venv/bin/activate

# Reinstall
pip install --force-reinstall rl-a2a`} />
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">SSL Certificate Error</h3>
              <p className="text-sm text-muted-foreground mb-3">
                If you encounter SSL errors during installation, try:
              </p>
              <CopyableCode code={`# Install certificates (macOS)
/Applications/Python\\ 3.x/Install\\ Certificates.command

# Or use trusted host
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org rl-a2a`} />
            </CardContent>
          </Card>
        </div>
      </section>
    </motion.div>
  )
}
