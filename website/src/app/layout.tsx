import type { Metadata } from "next"
import { Inter, JetBrains_Mono } from "next/font/google"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
})

export const metadata: Metadata = {
  title: "RL-A2A | Reinforcement Learning Agent-to-Agent Communication Platform",
  description:
    "Build intelligent multi-agent systems with reinforcement learning. Decentralized, adaptive, and scalable agent-to-agent communication for the next generation of AI applications.",
  keywords: [
    "reinforcement learning",
    "multi-agent systems",
    "AI",
    "machine learning",
    "agent communication",
    "OpenAI",
    "Anthropic",
    "Google AI",
  ],
  authors: [{ name: "Kunj Shah" }],
  openGraph: {
    title: "RL-A2A | Reinforcement Learning Agent-to-Agent Communication Platform",
    description:
      "Build intelligent multi-agent systems with reinforcement learning. Decentralized, adaptive, and scalable agent-to-agent communication.",
    type: "website",
    locale: "en_US",
    siteName: "RL-A2A",
  },
  twitter: {
    card: "summary_large_image",
    title: "RL-A2A | Reinforcement Learning Agent-to-Agent Communication Platform",
    description:
      "Build intelligent multi-agent systems with reinforcement learning.",
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}
      >
        {children}
      </body>
    </html>
  )
}
