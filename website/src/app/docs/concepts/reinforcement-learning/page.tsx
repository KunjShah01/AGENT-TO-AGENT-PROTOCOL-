"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Copy, Check, Brain, Trophy, Target, RotateCcw } from "lucide-react"
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

export default function ReinforcementLearningPage() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Core Concepts</Badge>
        <h1 className="text-4xl font-bold text-foreground">Reinforcement Learning</h1>
        <p className="text-xl text-muted-foreground">
          RL-A2A uses reinforcement learning to enable agents to learn optimal
          behaviors through interaction with their environment and feedback from
          other agents.
        </p>
      </div>

      {/* What is RL */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">What is Reinforcement Learning?</h2>
        <p className="text-muted-foreground">
          Reinforcement learning is a machine learning paradigm where an agent learns
          to make decisions by interacting with an environment. The agent receives
          rewards or penalties based on its actions, learning to maximize cumulative
          reward over time.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Brain className="w-5 h-5 text-purple-500" />
                <h3 className="font-semibold text-foreground">Agent</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                The learner that makes decisions and takes actions based on
                its current policy.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Target className="w-5 h-5 text-blue-500" />
                <h3 className="font-semibold text-foreground">Environment</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                The world the agent interacts with, providing states and
                rewards based on actions.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <Trophy className="w-5 h-5 text-yellow-500" />
                <h3 className="font-semibold text-foreground">Reward</h3>
              </div>
              <p className="text-sm text-muted-foreground">
                Feedback signal indicating how good an action was, guiding
                the learning process.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* RL Loop */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">The RL Learning Loop</h2>
        <div className="bg-card/50 border border-border rounded-lg p-6">
          <pre className="text-sm text-muted-foreground font-mono overflow-x-auto">
{`                    ┌─────────────────────────────────────┐
                    │           Environment               │
                    │  ┌───────────────────────────────┐  │
                    │  │         State (s)             │  │
                    │  └───────────────────────────────┘  │
                    └───────────────┬─────────────────────┘
                                    │
                                    │ Observation
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Policy    │───►│   Action    │───►│   Output    │      │
│  │   π(s)      │    │   a = π(s)  │    │             │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         ▲                                    │              │
│         │                                    │              │
│  ┌──────┴──────┐                             │              │
│  │   Update    │◄────────────────────────────┘              │
│  │   Policy    │         Reward (r)                         │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    │ Action (a)
                    ▼
                    ┌─────────────────────────────────────┐
                    │           Environment               │
                    │  ┌───────────────────────────────┐  │
                    │  │    Next State (s')            │  │
                    │  │    Reward (r)                 │  │
                    │  └───────────────────────────────┘  │
                    └─────────────────────────────────────┘`}
          </pre>
        </div>
      </section>

      {/* Key Components */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Key Components in RL-A2A</h2>
        <div className="space-y-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">State (Observation)</h3>
              <p className="text-sm text-muted-foreground mb-4">
                The agent's perception of the environment at a given time:
              </p>
              <CopyableCode
                code={`observation = {
    "agent_id": "agent_001",
    "position": {"x": 10.5, "y": 20.3, "z": 0},
    "velocity": {"x": 1.0, "y": 0.5, "z": 0},
    "energy": 85.0,
    "emotion": "curious",
    "nearby_agents": ["agent_002", "agent_003"],
    "timestamp": 1705312200.0
}`}
              />
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Action</h3>
              <p className="text-sm text-muted-foreground mb-4">
                The decision made by the agent based on its policy:
              </p>
              <CopyableCode
                code={`# Available actions
actions = [
    "move_forward",
    "turn_left",
    "turn_right",
    "communicate",
    "observe",
    "wait"
]

# Action response from server
action_response = {
    "command": "move_forward",
    "action_id": "action_abc123",
    "parameters": {"speed": 1.0},
    "timestamp": 1705312201.0
}`}
              />
            </CardContent>
          </Card>

          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Reward</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Feedback signal for learning:
              </p>
              <CopyableCode
                code={`# Reward calculation
def calculate_reward(action, result):
    reward = 0.0
    
    # Positive rewards
    if action == "communicate":
        reward += 0.2  # Encourage collaboration
    if action == "observe":
        reward += 0.15  # Encourage awareness
    
    # Negative rewards
    if result.get("collision"):
        reward -= 0.5  # Penalize collisions
    if result.get("energy") < 10:
        reward -= 0.2  # Penalize low energy
    
    # Goal achievement bonus
    if result.get("goal_reached"):
        reward += 1.0
    
    return reward

# Send feedback to server
feedback = {
    "agent_id": "agent_001",
    "action_id": "action_abc123",
    "reward": 0.35,
    "context": {
        "position": {"x": 11.5, "y": 20.3},
        "energy": 83.0
    }
}`}
              />
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Multi-Agent RL */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Multi-Agent Reinforcement Learning</h2>
        <p className="text-muted-foreground">
          RL-A2A extends traditional RL to multi-agent settings:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Decentralized Learning</h3>
              <p className="text-sm text-muted-foreground">
                Each agent learns its own policy independently, based on local
                observations and rewards. This improves scalability and robustness.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Emergent Cooperation</h3>
              <p className="text-sm text-muted-foreground">
                Agents learn to cooperate through shared rewards and communication,
                developing coordinated strategies without explicit programming.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Competitive Dynamics</h3>
              <p className="text-sm text-muted-foreground">
                Agents can also learn competitive strategies, optimizing their
                own rewards while considering opponent behaviors.
              </p>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-3">Communication Learning</h3>
              <p className="text-sm text-muted-foreground">
                Agents learn what and when to communicate, developing efficient
                communication protocols as part of their policy.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Implementing Custom RL */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Implementing Custom RL Algorithms</h2>
        <CopyableCode
          title="Python"
          code={`from rla2a import ReinforcementLearningSystem

class CustomRLSystem(ReinforcementLearningSystem):
    def __init__(self):
        super().__init__()
        self.learning_rate = 0.001
        self.discount_factor = 0.99
    
    def update_agent_performance(self, agent_id: str, reward: float):
        """Update agent's learning based on reward"""
        # Store reward history
        if agent_id not in self.reward_history:
            self.reward_history[agent_id] = []
        self.reward_history[agent_id].append(reward)
        
        # Calculate cumulative reward
        cumulative = sum(self.reward_history[agent_id][-100:])
        
        # Update policy (custom implementation)
        self._update_policy(agent_id, reward)
        
        return cumulative
    
    def _update_policy(self, agent_id: str, reward: float):
        """Custom policy update logic"""
        # Implement your RL algorithm here
        # e.g., Q-learning, Policy Gradient, PPO, etc.
        pass

# Use custom RL system
system = A2ASystem()
system.learning_system = CustomRLSystem()`}
        />
      </section>

      {/* Performance Metrics */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Tracking Performance</h2>
        <CopyableCode
          title="Python"
          code={`# Get agent performance metrics
agent = system.get_agent(agent_id)
metrics = agent.performance_metrics

print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Response Time: {metrics['response_time']:.2f}ms")
print(f"Learning Rate: {metrics['learning_rate']:.4f}")
print(f"Collaboration Score: {metrics['collaboration_score']:.2f}")

# Track learning progress
learning_data = system.learning_system.reward_history[agent_id]
avg_reward = sum(learning_data[-100:]) / len(learning_data[-100:])
print(f"Average Reward (last 100): {avg_reward:.3f}")`}
        />
      </section>
    </motion.div>
  )
}
