"use client"

import { useRef, useState } from "react"
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { 
  Brain, 
  Network, 
  Zap, 
  Shield, 
  Code2, 
  LineChart,
  Cpu,
  Globe
} from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "Reinforcement Learning",
    description:
      "Advanced RL algorithms enable agents to learn optimal strategies through interaction and feedback, adapting to complex environments.",
    gradient: "from-blue-500 to-cyan-500",
  },
  {
    icon: Network,
    title: "Decentralized Architecture",
    description:
      "Agents operate independently while maintaining cohesive collaboration, ensuring robustness and eliminating single points of failure.",
    gradient: "from-purple-500 to-pink-500",
  },
  {
    icon: Zap,
    title: "Real-Time Communication",
    description:
      "WebSocket-based messaging with sub-millisecond latency. Support for 10,000+ messages per second with automatic load balancing.",
    gradient: "from-yellow-500 to-orange-500",
  },
  {
    icon: Shield,
    title: "Enterprise Security",
    description:
      "JWT authentication, rate limiting, and encrypted communications. Built-in security middleware for production deployments.",
    gradient: "from-green-500 to-emerald-500",
  },
  {
    icon: Code2,
    title: "Developer First",
    description:
      "Clean APIs, comprehensive SDKs, and extensive documentation. Get started in minutes with our intuitive developer experience.",
    gradient: "from-indigo-500 to-violet-500",
  },
  {
    icon: LineChart,
    title: "Performance Analytics",
    description:
      "Built-in monitoring dashboards with real-time metrics. Track agent performance, message throughput, and system health.",
    gradient: "from-rose-500 to-red-500",
  },
  {
    icon: Cpu,
    title: "Multi-Provider AI",
    description:
      "Seamless integration with OpenAI, Anthropic, and Google AI. Switch providers without changing your code.",
    gradient: "from-teal-500 to-cyan-500",
  },
  {
    icon: Globe,
    title: "Cloud Native",
    description:
      "Docker and Kubernetes ready. Deploy anywhere with our containerized architecture and horizontal scaling support.",
    gradient: "from-sky-500 to-blue-500",
  },
]

function FeatureCard({ feature, index }: { feature: typeof features[0]; index: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [isHovered, setIsHovered] = useState(false)

  const x = useMotionValue(0)
  const y = useMotionValue(0)

  const mouseXSpring = useSpring(x)
  const mouseYSpring = useSpring(y)

  const rotateX = useTransform(mouseYSpring, [-0.5, 0.5], ["7.5deg", "-7.5deg"])
  const rotateY = useTransform(mouseXSpring, [-0.5, 0.5], ["-7.5deg", "7.5deg"])

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return

    const rect = ref.current.getBoundingClientRect()
    const width = rect.width
    const height = rect.height
    const mouseX = e.clientX - rect.left
    const mouseY = e.clientY - rect.top
    const xPct = mouseX / width - 0.5
    const yPct = mouseY / height - 0.5

    x.set(xPct)
    y.set(yPct)
  }

  const handleMouseLeave = () => {
    x.set(0)
    y.set(0)
    setIsHovered(false)
  }

  const Icon = feature.icon

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={handleMouseLeave}
      style={{
        rotateX,
        rotateY,
        transformStyle: "preserve-3d",
      }}
      className="relative"
    >
      <Card
        className={`relative overflow-hidden bg-card/50 backdrop-blur-sm border-border transition-all duration-300 ${
          isHovered ? "border-primary/50 shadow-lg shadow-primary/5" : ""
        }`}
      >
        {/* Gradient overlay on hover */}
        <div
          className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 transition-opacity duration-300 ${
            isHovered ? "opacity-5" : ""
          }`}
        />

        <CardContent className="relative p-6" style={{ transform: "translateZ(50px)" }}>
          {/* Icon */}
          <div
            className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.gradient} mb-4`}
          >
            <Icon className="w-6 h-6 text-white" />
          </div>

          {/* Title */}
          <h3 className="text-xl font-semibold text-foreground mb-2">
            {feature.title}
          </h3>

          {/* Description */}
          <p className="text-muted-foreground text-sm leading-relaxed">
            {feature.description}
          </p>
        </CardContent>
      </Card>
    </motion.div>
  )
}

export function FeaturesSection() {
  return (
    <section id="features" className="relative py-24 overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 grid-pattern opacity-20" />
      <div className="absolute top-1/2 left-0 w-72 h-72 bg-primary/10 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Powerful Features for{" "}
            <span className="gradient-text">Intelligent Agents</span>
          </h2>
          <p className="max-w-2xl mx-auto text-muted-foreground text-lg">
            Everything you need to build, deploy, and scale multi-agent systems
            with reinforcement learning at their core.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <FeatureCard key={feature.title} feature={feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  )
}
