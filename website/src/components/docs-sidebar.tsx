"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { ChevronDown, ChevronRight, Book, Rocket, Code, Layers, Database, Settings, Zap, Globe, Shield, Terminal } from "lucide-react"

const docsSections = [
  {
    title: "Getting Started",
    icon: Rocket,
    items: [
      { title: "Introduction", href: "/docs" },
      { title: "Quick Start", href: "/docs/quick-start" },
      { title: "Installation", href: "/docs/installation" },
      { title: "Configuration", href: "/docs/configuration" },
    ],
  },
  {
    title: "Core Concepts",
    icon: Layers,
    items: [
      { title: "Agents", href: "/docs/concepts/agents" },
      { title: "Messages", href: "/docs/concepts/messages" },
      { title: "Reinforcement Learning", href: "/docs/concepts/reinforcement-learning" },
      { title: "Communication Protocol", href: "/docs/concepts/protocol" },
    ],
  },
  {
    title: "API Reference",
    icon: Code,
    items: [
      { title: "Overview", href: "/docs/api/overview" },
      { title: "REST API", href: "/docs/api/rest" },
      { title: "WebSocket API", href: "/docs/api/websocket" },
      { title: "Python SDK", href: "/docs/api/python-sdk" },
      { title: "TypeScript SDK", href: "/docs/api/typescript-sdk" },
    ],
  },
  {
    title: "AI Providers",
    icon: Zap,
    items: [
      { title: "Overview", href: "/docs/providers/overview" },
      { title: "OpenAI", href: "/docs/providers/openai" },
      { title: "Anthropic", href: "/docs/providers/anthropic" },
      { title: "Google AI", href: "/docs/providers/google" },
      { title: "Ollama (Local)", href: "/docs/providers/ollama" },
    ],
  },
  {
    title: "Guides",
    icon: Book,
    items: [
      { title: "Building Your First Agent", href: "/docs/guides/first-agent" },
      { title: "Multi-Agent Systems", href: "/docs/guides/multi-agent" },
      { title: "Custom RL Algorithms", href: "/docs/guides/custom-rl" },
      { title: "Deployment", href: "/docs/guides/deployment" },
    ],
  },
  {
    title: "Advanced",
    icon: Settings,
    items: [
      { title: "Security", href: "/docs/advanced/security" },
      { title: "Performance Tuning", href: "/docs/advanced/performance" },
      { title: "MCP Integration", href: "/docs/advanced/mcp" },
      { title: "Custom Plugins", href: "/docs/advanced/plugins" },
    ],
  },
]

export function DocsSidebar() {
  const pathname = usePathname()
  const [expandedSections, setExpandedSections] = useState<string[]>(
    docsSections.map((s) => s.title)
  )

  const toggleSection = (title: string) => {
    setExpandedSections((prev) =>
      prev.includes(title)
        ? prev.filter((t) => t !== title)
        : [...prev, title]
    )
  }

  return (
    <aside className="w-64 flex-shrink-0 border-r border-border bg-card/30 h-[calc(100vh-4rem)] sticky top-16 overflow-y-auto">
      <nav className="p-4 space-y-2">
        {docsSections.map((section) => {
          const Icon = section.icon
          const isExpanded = expandedSections.includes(section.title)

          return (
            <div key={section.title} className="space-y-1">
              {/* Section Header */}
              <button
                onClick={() => toggleSection(section.title)}
                className="flex items-center justify-between w-full px-3 py-2 text-sm font-medium text-foreground hover:bg-muted rounded-lg transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Icon className="w-4 h-4 text-primary" />
                  <span>{section.title}</span>
                </div>
                {isExpanded ? (
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                ) : (
                  <ChevronRight className="w-4 h-4 text-muted-foreground" />
                )}
              </button>

              {/* Section Items */}
              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="pl-6 space-y-1">
                      {section.items.map((item) => (
                        <Link
                          key={item.href}
                          href={item.href}
                          className={cn(
                            "block px-3 py-1.5 text-sm rounded-lg transition-colors",
                            pathname === item.href
                              ? "bg-primary/10 text-primary font-medium"
                              : "text-muted-foreground hover:text-foreground hover:bg-muted"
                          )}
                        >
                          {item.title}
                        </Link>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )
        })}
      </nav>
    </aside>
  )
}
