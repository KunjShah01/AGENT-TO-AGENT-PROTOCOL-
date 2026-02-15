"use client"

import { motion } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Copy, Check, Rocket, Code, Play, Trophy } from "lucide-react"
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

export default function FirstAgentGuide() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="space-y-4">
        <Badge variant="outline">Guide</Badge>
        <h1 className="text-4xl font-bold text-foreground">Building Your First Agent</h1>
        <p className="text-xl text-muted-foreground">
          A step-by-step tutorial to create, run, and interact with your first
          RL-A2A agent. By the end, you'll have a working agent that can
          communicate and learn.
        </p>
      </div>

      {/* Prerequisites */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Prerequisites</h2>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">Python 3.8+</Badge>
          <Badge variant="secondary">rl-a2a installed</Badge>
          <Badge variant="secondary">OpenAI API key (optional)</Badge>
        </div>
      </section>

      {/* Step 1 */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
            1
          </div>
          <h2 className="text-2xl font-bold text-foreground">Project Setup</h2>
        </div>
        <p className="text-muted-foreground">
          Create a new project directory and set up your environment:
        </p>
        <CopyableCode
          title="Terminal"
          code={`# Create project directory
mkdir my-first-agent
cd my-first-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install RL-A2A
pip install rl-a2a

# Create .env file
touch .env`}
        />
        <p className="text-muted-foreground">
          Add your API key to <code className="px-1 py-0.5 rounded bg-muted text-foreground">.env</code>:
        </p>
        <CopyableCode
          title=".env"
          code={`OPENAI_API_KEY=your_openai_api_key_here`}
        />
      </section>

      {/* Step 2 */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
            2
          </div>
          <h2 className="text-2xl font-bold text-foreground">Create the Server</h2>
        </div>
        <p className="text-muted-foreground">
          Create a server that will manage your agents:
        </p>
        <CopyableCode
          title="server.py"
          code={`import asyncio
from rla2a import A2ASystem

async def main():
    # Initialize the A2A system
    system = A2ASystem()
    
    # Create a demo agent
    agent_id = system.create_agent(
        name="MyFirstAgent",
        role="assistant",
        ai_provider="openai"
    )
    
    print(f"✅ Agent created: {agent_id}")
    print("🚀 Starting server on http://localhost:8000")
    
    # Start the server
    await system.start_server()

if __name__ == "__main__":
    asyncio.run(main())`}
        />
      </section>

      {/* Step 3 */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
            3
          </div>
          <h2 className="text-2xl font-bold text-foreground">Create the Client</h2>
        </div>
        <p className="text-muted-foreground">
          Create a client that will interact with your agent:
        </p>
        <CopyableCode
          title="client.py"
          code={`import asyncio
import websockets
import json
import requests

class MyAgent:
    def __init__(self, agent_id: str, server_url: str = "http://localhost:8000"):
        self.agent_id = agent_id
        self.server_url = server_url
        self.ws_url = server_url.replace("http", "ws")
        self.session_id = None
        self.websocket = None
    
    async def register(self):
        """Register with the server"""
        response = requests.post(
            f"{self.server_url}/register",
            params={"agent_id": self.agent_id}
        )
        data = response.json()
        self.session_id = data["session_id"]
        print(f"✅ Registered: {self.session_id}")
        return self.session_id
    
    async def connect(self):
        """Connect to WebSocket"""
        ws_url = f"{self.ws_url}/ws/{self.session_id}"
        self.websocket = await websockets.connect(ws_url)
        print("✅ WebSocket connected")
    
    async def send_observation(self, observation: dict):
        """Send observation and receive action"""
        await self.websocket.send(json.dumps(observation))
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def run(self, iterations: int = 5):
        """Run the agent loop"""
        await self.register()
        await self.connect()
        
        for i in range(iterations):
            # Send observation
            observation = {
                "agent_id": self.agent_id,
                "iteration": i + 1,
                "energy": 100 - (i * 5)
            }
            
            response = await self.send_observation(observation)
            print(f"📥 Iteration {i+1}: {response}")
            
            await asyncio.sleep(1)
        
        print("✅ Agent completed!")

async def main():
    agent = MyAgent("my_agent_001")
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())`}
        />
      </section>

      {/* Step 4 */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold">
            4
          </div>
          <h2 className="text-2xl font-bold text-foreground">Run Your Agents</h2>
        </div>
        <p className="text-muted-foreground">
          Open two terminals and run the server and client:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border">
            <CardContent className="p-4">
              <h3 className="font-semibold text-foreground mb-2">Terminal 1 - Server</h3>
              <CopyableCode code={`python server.py`} />
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border">
            <CardContent className="p-4">
              <h3 className="font-semibold text-foreground mb-2">Terminal 2 - Client</h3>
              <CopyableCode code={`python client.py`} />
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Expected Output */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Expected Output</h2>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-4">
            <h3 className="font-semibold text-foreground mb-2">Server Output</h3>
            <pre className="p-3 bg-muted rounded-lg text-sm font-mono text-foreground overflow-x-auto">
{`✅ Agent created: agent_abc123
🚀 Starting server on http://localhost:8000
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000`}
            </pre>
          </CardContent>
        </Card>
        <Card className="bg-card/50 border-border">
          <CardContent className="p-4">
            <h3 className="font-semibold text-foreground mb-2">Client Output</h3>
            <pre className="p-3 bg-muted rounded-lg text-sm font-mono text-foreground overflow-x-auto">
{`✅ Registered: agent_abc123
✅ WebSocket connected
📥 Iteration 1: {'command': 'observe', 'action_id': 'action_001'}
📥 Iteration 2: {'command': 'communicate', 'action_id': 'action_002'}
📥 Iteration 3: {'command': 'move_forward', 'action_id': 'action_003'}
📥 Iteration 4: {'command': 'observe', 'action_id': 'action_004'}
📥 Iteration 5: {'command': 'communicate', 'action_id': 'action_005'}
✅ Agent completed!`}
            </pre>
          </CardContent>
        </Card>
      </section>

      {/* Next Steps */}
      <section className="space-y-4">
        <h2 className="text-2xl font-bold text-foreground">Next Steps</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">Add More Agents</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Learn how to create multi-agent systems with collaboration.
              </p>
              <Button variant="ghost" size="sm" className="text-primary">
                Multi-Agent Guide <Play className="ml-2 w-4 h-4" />
              </Button>
            </CardContent>
          </Card>
          <Card className="bg-card/50 border-border hover:border-primary/30 transition-colors">
            <CardContent className="p-6">
              <h3 className="font-semibold text-foreground mb-2">Add RL Learning</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Implement reward feedback for reinforcement learning.
              </p>
              <Button variant="ghost" size="sm" className="text-primary">
                RL Tutorial <Trophy className="ml-2 w-4 h-4" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>
    </motion.div>
  )
}
