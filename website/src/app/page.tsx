import { Navigation } from "@/components/navigation"
import { HeroSection } from "@/components/hero-section"
import { FeaturesSection } from "@/components/features-section"
import { HowItWorksSection } from "@/components/how-it-works-section"
import { CodePreviewSection } from "@/components/code-preview-section"
import { ProvidersSection } from "@/components/providers-section"
import { PricingSection } from "@/components/pricing-section"
import { CTASection } from "@/components/cta-section"
import { Footer } from "@/components/footer"

export default function Home() {
  return (
    <main className="relative min-h-screen bg-background">
      <Navigation />
      <HeroSection />
      <FeaturesSection />
      <HowItWorksSection />
      <CodePreviewSection />
      <ProvidersSection />
      <PricingSection />
      <CTASection />
      <Footer />
    </main>
  )
}
