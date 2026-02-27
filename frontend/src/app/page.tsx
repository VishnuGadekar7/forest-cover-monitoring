"use client";

import Link from "next/link";
import { ArrowRight, Leaf, Shield, Cpu, Map } from "lucide-react";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0a0f1a] overflow-hidden flex flex-col">
      {/* ── Background Effects ── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-green-900/20 blur-[120px] rounded-full" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-blue-900/20 blur-[120px] rounded-full" />
      </div>

      {/* ── Navbar ── */}
      <nav className="relative z-10 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-500 to-blue-600 flex items-center justify-center shadow-lg">
            <Leaf className="w-4 h-4 text-white" />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-sm font-bold text-slate-100">ForestMonitor</span>
            <span className="text-[10px] text-slate-500 tracking-widest uppercase">Deep Learning · EO</span>
          </div>
        </div>
        <Link
          href="/upload"
          className="text-sm font-medium text-slate-300 hover:text-white transition-colors"
        >
          Launch App
        </Link>
      </nav>

      {/* ── Hero Section ── */}
      <main className="relative z-10 flex-grow flex flex-col items-center justify-center px-4 pt-10 pb-20 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-800/50 border border-slate-700 text-slate-300 text-xs font-medium mb-8 fade-in-up" style={{ animationDelay: '100ms' }}>
          <span className="w-2 h-2 rounded-full bg-blue-500 pulse-glow" />
          Team Axios 2026
        </div>

        <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight fade-in-up" style={{ animationDelay: '200ms' }}>
          Monitoring Earth's Forests <br className="hidden md:block" />
          using <span className="gradient-text">Deep Learning</span>
        </h1>

        <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-12 leading-relaxed fade-in-up" style={{ animationDelay: '300ms' }}>
          A research-grade Forest Cover Change Detection platform leveraging deep learning semantic segmentation to detect and monitor pixel-wise forest cover change across multi-temporal satellite imagery.
        </p>

        <div className="flex flex-col sm:flex-row items-center gap-4 fade-in-up" style={{ animationDelay: '400ms' }}>
          <Link
            href="/upload"
            className="group relative flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 shadow-lg shadow-green-900/20 hover:shadow-green-900/40 transition-all hover:scale-[1.02]"
          >
            Start Monitoring
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="px-8 py-4 rounded-xl font-semibold text-slate-300 bg-slate-800/50 border border-slate-700 hover:bg-slate-800 transition-colors"
          >
            View Documentation
          </a>
        </div>

        {/* ── Feature Grid ── */}
        <div className="grid md:grid-cols-3 gap-6 max-w-5xl w-full mt-24 fade-in-up" style={{ animationDelay: '600ms' }}>
          {[
            {
              icon: Cpu,
              title: "Advanced Segmentation",
              desc: "Powered by Attention U-Net and Vision Transformers for microscopic pixel accuracy.",
              color: "text-blue-400",
              bg: "bg-blue-500/10",
              border: "border-blue-500/20"
            },
            {
              icon: Map,
              title: "Geographic Integration",
              desc: "Dynamic change map generation with standard remote sensing color conventions.",
              color: "text-green-400",
              bg: "bg-green-500/10",
              border: "border-green-500/20"
            },
            {
              icon: Shield,
              title: "Research Grade",
              desc: "Designed to meet the rigorous presentation standards of academic reviewers.",
              color: "text-amber-400",
              bg: "bg-amber-500/10",
              border: "border-amber-500/20"
            }
          ].map((feat, idx) => (
            <div key={idx} className="glass-card p-6 text-left hover:-translate-y-1 transition-transform duration-300">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-5 border ${feat.bg} ${feat.border}`}>
                <feat.icon className={`w-6 h-6 ${feat.color}`} />
              </div>
              <h3 className="text-lg font-semibold text-slate-200 mb-2">{feat.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{feat.desc}</p>
            </div>
          ))}
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="relative z-10 border-t border-slate-800/60 py-6 text-center text-sm text-slate-500">
        <p>© 2026 Team Axios. All rights reserved.</p>
      </footer>
    </div>
  );
}
