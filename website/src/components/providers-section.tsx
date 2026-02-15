"use client"

import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

const providers = [
  {
    name: "OpenAI",
    description: "GPT-4 and GPT-4o models for advanced reasoning and generation",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    color: "from-green-500 to-emerald-500",
    logo: (
      <svg viewBox="0 0 24 24" className="w-8 h-8" fill="currentColor">
        <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.062L7.8048 19.45a4.5023 4.5023 0 0 1-6.2052-1.1463zm-1.264-10.3191a4.4847 4.4847 0 0 1 2.3655-1.9396v5.6784a.7664.7664 0 0 0 .3852.6765l5.8142 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3352 7.9847zm16.3825 3.8789l-5.8375-3.3863L14.9 5.3082a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6777a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0614l4.8303-2.7866a4.4992 4.4992 0 0 1 6.4804 4.6688zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z" />
      </svg>
    ),
  },
  {
    name: "Anthropic",
    description: "Claude models for safe, helpful, and honest AI interactions",
    models: ["claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku"],
    color: "from-orange-500 to-amber-500",
    logo: (
      <svg viewBox="0 0 24 24" className="w-8 h-8" fill="currentColor">
        <path d="M17.304 3.541l-5.296 16.918h-3.208L3.502 3.541h3.208l3.596 11.541 3.594-11.541h3.404zm3.194 0v16.918h-3.108V3.541h3.108z" />
      </svg>
    ),
  },
  {
    name: "Google AI",
    description: "Gemini models for multimodal understanding and generation",
    models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
    color: "from-blue-500 to-cyan-500",
    logo: (
      <svg viewBox="0 0 24 24" className="w-8 h-8" fill="currentColor">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" />
      </svg>
    ),
  },
]

export function ProvidersSection() {
  return (
    <section className="relative py-24 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-background via-card/30 to-background" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge variant="outline" className="mb-4">
            AI Providers
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Powered by{" "}
            <span className="gradient-text">Leading AI Models</span>
          </h2>
          <p className="max-w-2xl mx-auto text-muted-foreground text-lg">
            Seamlessly integrate with multiple AI providers. Switch between
            models without changing your code.
          </p>
        </motion.div>

        {/* Providers Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {providers.map((provider, index) => (
            <motion.div
              key={provider.name}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Card className="h-full bg-card/50 backdrop-blur-sm border-border hover:border-primary/30 transition-all duration-300 group">
                <CardContent className="p-6">
                  {/* Logo & Name */}
                  <div className="flex items-center gap-4 mb-4">
                    <div
                      className={`p-3 rounded-xl bg-gradient-to-br ${provider.color} text-white`}
                    >
                      {provider.logo}
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold text-foreground">
                        {provider.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        AI Provider
                      </p>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-muted-foreground mb-4">
                    {provider.description}
                  </p>

                  {/* Available Models */}
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">
                      Available Models
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {provider.models.map((model) => (
                        <Badge
                          key={model}
                          variant="secondary"
                          className="text-xs"
                        >
                          {model}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Integration Note */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-12 text-center"
        >
          <Card className="inline-block bg-primary/5 border-primary/20">
            <CardContent className="px-6 py-4">
              <p className="text-sm text-muted-foreground">
                <span className="text-primary font-medium">One API</span> —
                Switch providers with a single configuration change
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </section>
  )
}
