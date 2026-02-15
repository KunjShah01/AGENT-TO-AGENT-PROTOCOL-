"use client"

import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Eye, Brain, Zap, Trophy } from "lucide-react"

const steps = [
  {
    number: "01",
    icon: Eye,
    title: "Observe & Perceive",
    description:
      "Each agent observes its local state within the shared environment, including the states or actions of nearby agents.",
    color: "text-blue-400",
    bgColor: "bg-blue-400/10",
  },
  {
    number: "02",
    icon: Brain,
    title: "Formulate Policy",
    description:
      "Based on observations and long-term objectives, the agent's neural network formulates an optimal action policy.",
    color: "text-purple-400",
    bgColor: "bg-purple-400/10",
  },
  {
    number: "03",
    icon: Zap,
    title: "Act & Interact",
    description:
      "The agent takes an action that alters its own state and influences the environment for other agents.",
    color: "text-yellow-400",
    bgColor: "bg-yellow-400/10",
  },
  {
    number: "04",
    icon: Trophy,
    title: "Learn & Update",
    description:
      "The agent receives a reward signal used to update its policy network via backpropagation, refining future decisions.",
    color: "text-green-400",
    bgColor: "bg-green-400/10",
  },
]

export function HowItWorksSection() {
  return (
    <section id="how-it-works" className="relative py-24 overflow-hidden">
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
            How It Works
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            The Agent Decision Loop
          </h2>
          <p className="max-w-2xl mx-auto text-muted-foreground text-lg">
            A simplified look at the agent decision-making process within the
            A2A reinforcement learning framework.
          </p>
        </motion.div>

        {/* Timeline */}
        <div className="relative">
          {/* Connection Line */}
          <div className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-primary/50 via-primary to-primary/50" />

          {/* Steps */}
          <div className="space-y-12 lg:space-y-24">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isEven = index % 2 === 0

              return (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, x: isEven ? -50 : 50 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1 }}
                  className={`relative flex items-center ${
                    isEven ? "lg:flex-row" : "lg:flex-row-reverse"
                  } flex-col lg:gap-8`}
                >
                  {/* Content Card */}
                  <div
                    className={`w-full lg:w-1/2 ${
                      isEven ? "lg:pr-12 lg:text-right" : "lg:pl-12 lg:text-left"
                    }`}
                  >
                    <Card className="bg-card/50 backdrop-blur-sm border-border hover:border-primary/30 transition-colors">
                      <CardContent className="p-6">
                        <div
                          className={`inline-flex p-3 rounded-xl ${step.bgColor} mb-4`}
                        >
                          <Icon className={`w-6 h-6 ${step.color}`} />
                        </div>
                        <h3 className="text-xl font-semibold text-foreground mb-2">
                          {step.title}
                        </h3>
                        <p className="text-muted-foreground">
                          {step.description}
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Center Node */}
                  <div className="hidden lg:flex absolute left-1/2 -translate-x-1/2 w-12 h-12 rounded-full bg-card border-2 border-primary items-center justify-center z-10">
                    <span className="text-sm font-bold text-primary">
                      {step.number}
                    </span>
                  </div>

                  {/* Mobile Number */}
                  <div className="lg:hidden flex items-center gap-4 mb-4">
                    <div
                      className={`flex w-10 h-10 rounded-full ${step.bgColor} items-center justify-center`}
                    >
                      <span className={`text-sm font-bold ${step.color}`}>
                        {step.number}
                      </span>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Loop Indicator */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <div className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-primary/10 border border-primary/20">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm text-primary font-medium">
              Continuous Learning Loop
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
