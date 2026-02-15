"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Copy, Check, Settings, Server, Shield, Database } from "lucide-react"
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

export default function ConfigurationPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Getting Started</Badge>
        <h1 className="text-4xl font-bold text-foreground">Configuration</h1>
        <p className="text-xl text-muted-foreground">
          Configure RL-A2A to fit your needs. From basic server settings to
          advanced security and AI provider configurations.
        </p>
      </div>

      {/* Environment Variables */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <Settings className="w-6 h-6 text-primary" />
          <h2 className="text-2xl font-bold text-foreground">Environment Variables</h2>
        </div>
        <p className="text-muted-foreground">
          RL-A2A uses environment variables for configuration. Create a <code className="px-1 py-0.5 rounded bg-muted text-foreground">.env</code> file
          in your project root:
        </p>
        <CopyableCode
          title=".env"
          code={`# =============================================================================
# RL-A2A Configuration
# =============================================================================

# -----------------------------------------------------------------------------
# AI Provider API Keys (configure at least one for AI features)
# -----------------------------------------------------------------------------
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# -----------------------------------------------------------------------------
# AI Model Configuration
# -----------------------------------------------------------------------------
DEFAULT_AI_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
GOOGLE_MODEL=gemini-1.5-flash
AI_TIMEOUT=30

# -----------------------------------------------------------------------------
# Server Configuration
# -----------------------------------------------------------------------------
A2A_HOST=localhost
A2A_PORT=8000
DASHBOARD_PORT=8501

# -----------------------------------------------------------------------------
# System Limits
# -----------------------------------------------------------------------------
MAX_AGENTS=100
MAX_CONNECTIONS=1000
MAX_MESSAGE_SIZE=1048576

# -----------------------------------------------------------------------------
# Security Configuration
# -----------------------------------------------------------------------------
SECRET_KEY=your-secret-key-min-32-characters-long
ACCESS_TOKEN_EXPIRE_HOURS=24
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
RATE_LIMIT_PER_MINUTE=60
SESSION_TIMEOUT=3600

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
LOG_FILE=rla2a.log
DEBUG=false`}
        />
      </section>

      {/* Configuration Reference */}
      <section className="space-y-6">
        <h2 className="text-2xl font-bold text-foreground">Configuration Reference</h2>

        {/* AI Providers */}
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Database className="w-5 h-5 text-purple-500" />
              <h3 className="text-lg font-semibold text-foreground">AI Providers</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 text-muted-foreground">Variable</th>
                    <th className="text-left py-2 text-muted-foreground">Description</th>
                    <th className="text-left py-2 text-muted-foreground">Default</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">OPENAI_API_KEY</td>
                    <td className="py-2 text-muted-foreground">OpenAI API key for GPT models</td>
                    <td className="py-2 text-muted-foreground">None</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">ANTHROPIC_API_KEY</td>
                    <td className="py-2 text-muted-foreground">Anthropic API key for Claude models</td>
                    <td className="py-2 text-muted-foreground">None</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">GOOGLE_API_KEY</td>
                    <td className="py-2 text-muted-foreground">Google AI API key for Gemini models</td>
                    <td className="py-2 text-muted-foreground">None</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">DEFAULT_AI_PROVIDER</td>
                    <td className="py-2 text-muted-foreground">Default AI provider to use</td>
                    <td className="py-2 text-muted-foreground">openai</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">AI_TIMEOUT</td>
                    <td className="py-2 text-muted-foreground">Timeout for AI requests (seconds)</td>
                    <td className="py-2 text-muted-foreground">30</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Server Settings */}
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Server className="w-5 h-5 text-blue-500" />
              <h3 className="text-lg font-semibold text-foreground">Server Settings</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 text-muted-foreground">Variable</th>
                    <th className="text-left py-2 text-muted-foreground">Description</th>
                    <th className="text-left py-2 text-muted-foreground">Default</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">A2A_HOST</td>
                    <td className="py-2 text-muted-foreground">Server host address</td>
                    <td className="py-2 text-muted-foreground">localhost</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">A2A_PORT</td>
                    <td className="py-2 text-muted-foreground">Server port</td>
                    <td className="py-2 text-muted-foreground">8000</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">MAX_AGENTS</td>
                    <td className="py-2 text-muted-foreground">Maximum number of agents</td>
                    <td className="py-2 text-muted-foreground">100</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">MAX_CONNECTIONS</td>
                    <td className="py-2 text-muted-foreground">Maximum concurrent connections</td>
                    <td className="py-2 text-muted-foreground">1000</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>

        {/* Security Settings */}
        <Card className="bg-card/50 border-border">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-5 h-5 text-green-500" />
              <h3 className="text-lg font-semibold text-foreground">Security Settings</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-2 text-muted-foreground">Variable</th>
                    <th className="text-left py-2 text-muted-foreground">Description</th>
                    <th className="text-left py-2 text-muted-foreground">Default</th>
                  </tr>
                </thead>
                <tbody className="text-foreground">
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">SECRET_KEY</td>
                    <td className="py-2 text-muted-foreground">Secret key for JWT signing (min 32 chars)</td>
                    <td className="py-2 text-muted-foreground">Auto-generated</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">ACCESS_TOKEN_EXPIRE_HOURS</td>
                    <td className="py-2 text-muted-foreground">Token expiration time (hours)</td>
                    <td className="py-2 text-muted-foreground">24</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">ALLOWED_ORIGINS</td>
                    <td className="py-2 text-muted-foreground">CORS allowed origins (comma-separated)</td>
                    <td className="py-2 text-muted-foreground">*</td>
                  </tr>
                  <tr className="border-b border-border/50">
                    <td className="py-2 font-mono text-primary">RATE_LIMIT_PER_MINUTE</td>
                    <td className="py-2 text-muted-foreground">API rate limit per minute</td>
                    <td className="py-2 text-muted-foreground">60</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Programmatic Configuration */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Programmatic Configuration</h2>
        <p className="text-muted-foreground">
          You can also configure RL-A2A programmatically in your code:
        </p>
        <CopyableCode
          title="config.py"
          code={`from rla2a import A2ASystem, Config

# Custom configuration
config = Config(
    # Server settings
    server_host="0.0.0.0",
    server_port=8000,
    
    # AI settings
    default_ai_provider="openai",
    openai_model="gpt-4o-mini",
    
    # Limits
    max_agents=50,
    max_connections=500,
    
    # Security
    secret_key="your-secret-key-here",
    access_token_expire_hours=12,
    
    # Debug
    debug=True,
    log_level="DEBUG"
)

# Initialize system with config
system = A2ASystem(config=config)`}
        />
      </section>

      {/* Configuration Best Practices */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Best Practices</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Security</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Never commit .env files to version control</li>
                <li>• Use strong, unique SECRET_KEY (32+ characters)</li>
                <li>• Restrict ALLOWED_ORIGINS in production</li>
                <li>• Rotate API keys regularly</li>
              </ul>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Performance</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Set appropriate MAX_AGENTS for your hardware</li>
                <li>• Use connection pooling for high traffic</li>
                <li>• Enable caching for repeated AI queries</li>
                <li>• Monitor memory usage with many agents</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>
    </motion.div>
  )
}
