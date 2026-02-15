"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardHeader, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Check, Zap, Building, Rocket } from "lucide-react"

const plans = [
  {
    name: "Starter",
    description: "Perfect for experimentation and small projects",
    price: "Free",
    period: "forever",
    icon: Rocket,
    features: [
      "Up to 5 agents",
      "Basic RL algorithms",
      "Community support",
      "1,000 messages/month",
      "Single AI provider",
      "Local deployment",
    ],
    cta: "Get Started",
    popular: false,
  },
  {
    name: "Pro",
    description: "For teams building production multi-agent systems",
    price: "$49",
    period: "/month",
    icon: Zap,
    features: [
      "Up to 50 agents",
      "Advanced RL algorithms",
      "Priority support",
      "100,000 messages/month",
      "All AI providers",
      "Cloud deployment",
      "Performance analytics",
      "Custom agent roles",
    ],
    cta: "Start Free Trial",
    popular: true,
  },
  {
    name: "Enterprise",
    description: "For organizations with advanced requirements",
    price: "Custom",
    period: "contact us",
    icon: Building,
    features: [
      "Unlimited agents",
      "Custom RL algorithms",
      "24/7 dedicated support",
      "Unlimited messages",
      "All AI providers + custom",
      "Hybrid deployment",
      "Advanced analytics",
      "SLA guarantee",
      "Custom integrations",
      "On-premise option",
    ],
    cta: "Contact Sales",
    popular: false,
  },
]

export function PricingSection() {
  return (
    <section id="pricing" className="relative py-24 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 grid-pattern opacity-20" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 rounded-full blur-3xl" />

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <Badge variant="outline" className="mb-4">
            Pricing
          </Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-foreground mb-4">
            Simple, Transparent{" "}
            <span className="gradient-text">Pricing</span>
          </h2>
          <p className="max-w-2xl mx-auto text-muted-foreground text-lg">
            Start free and scale as you grow. No hidden fees, no surprises.
          </p>
        </motion.div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {plans.map((plan, index) => {
            const Icon = plan.icon
            return (
              <motion.div
                key={plan.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="relative"
              >
                {/* Popular Badge */}
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 z-10">
                    <Badge className="bg-primary text-primary-foreground px-4 py-1">
                      Most Popular
                    </Badge>
                  </div>
                )}

                <Card
                  className={`h-full bg-card/50 backdrop-blur-sm transition-all duration-300 ${
                    plan.popular
                      ? "border-primary shadow-lg shadow-primary/10 scale-105"
                      : "border-border hover:border-primary/30"
                  }`}
                >
                  <CardHeader className="text-center pb-4">
                    <div
                      className={`mx-auto w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
                        plan.popular
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="text-xl font-semibold text-foreground">
                      {plan.name}
                    </h3>
                    <p className="text-sm text-muted-foreground">
                      {plan.description}
                    </p>
                  </CardHeader>

                  <CardContent className="text-center pb-6">
                    <div className="mb-6">
                      <span className="text-4xl font-bold text-foreground">
                        {plan.price}
                      </span>
                      <span className="text-muted-foreground ml-1">
                        {plan.period}
                      </span>
                    </div>

                    <ul className="space-y-3 text-left">
                      {plan.features.map((feature) => (
                        <li
                          key={feature}
                          className="flex items-start gap-3 text-sm"
                        >
                          <Check className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                          <span className="text-muted-foreground">{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>

                  <CardFooter>
                    <Button
                      className={`w-full ${
                        plan.popular
                          ? "bg-primary hover:bg-primary/90"
                          : "bg-secondary hover:bg-secondary/80"
                      }`}
                      variant={plan.popular ? "default" : "secondary"}
                    >
                      {plan.cta}
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            )
          })}
        </div>

        {/* FAQ Teaser */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <p className="text-muted-foreground">
            Have questions?{" "}
            <a href="#faq" className="text-primary hover:underline">
              Check our FAQ
            </a>{" "}
            or{" "}
            <a href="#contact" className="text-primary hover:underline">
              contact us
            </a>
          </p>
        </motion.div>
      </div>
    </section>
  )
}
