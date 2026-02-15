"use client"

import { DocsSidebar } from "@/components/docs-sidebar"
import { Navigation } from "@/components/navigation"
import { Footer } from "@/components/footer"

export default function DocsLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <div className="flex pt-16">
        <DocsSidebar />
        <main className="flex-1 min-h-[calc(100vh-4rem)]">
          <div className="max-w-4xl mx-auto px-8 py-12">
            {children}
          </div>
        </main>
      </div>
    </div>
  )
}
