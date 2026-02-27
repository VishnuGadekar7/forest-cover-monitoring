"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Zap, AlertCircle, Loader2 } from "lucide-react";
import Navbar from "@/components/Navbar";
import UploadCard from "@/components/UploadCard";
import { detectChange } from "@/lib/api";
import type { ChangeDetectionResult } from "@/lib/api";

// Store result in sessionStorage so the results page can read it
function storeResult(result: ChangeDetectionResult) {
  sessionStorage.setItem("detection_result", JSON.stringify(result));
}

export default function UploadPage() {
  const router = useRouter();
  const [imageT1, setImageT1] = useState<File | null>(null);
  const [imageT2, setImageT2] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const canDetect = imageT1 && imageT2 && !loading;

  const handleDetect = async () => {
    if (!imageT1 || !imageT2) return;
    setLoading(true);
    setProgress(0);
    setError(null);
    try {
      const result = await detectChange(imageT1, imageT2, setProgress);
      storeResult(result);
      router.push("/results");
    } catch (err: any) {
      const msg =
        err?.response?.data?.detail ??
        err?.message ??
        "An unexpected error occurred. Is the backend running?";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0f1a]">
      <Navbar />

      <main className="pt-24 pb-16 px-4 max-w-5xl mx-auto">
        {/* Hero */}
        <div className="text-center mb-12 fade-in-up">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-medium mb-6">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 pulse-glow" />
            Forest Change Detection
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4 leading-tight">
            Upload{" "}
            <span className="gradient-text">Satellite Images</span>
          </h1>
          <p className="text-slate-400 max-w-xl mx-auto text-base">
            Upload two multi-temporal satellite images. Our deep learning pipeline will segment
            forest cover and compute pixel-wise change statistics.
          </p>
        </div>

        {/* Upload panel */}
        <div className="glass-card glow-border p-8 mb-6">
          <div className="grid md:grid-cols-2 gap-8">
            <UploadCard
              label="Time Period 1 (Earlier)"
              badge="T₁"
              file={imageT1}
              onFile={setImageT1}
              onClear={() => setImageT1(null)}
            />
            <UploadCard
              label="Time Period 2 (Later)"
              badge="T₂"
              file={imageT2}
              onFile={setImageT2}
              onClear={() => setImageT2(null)}
            />
          </div>

          {/* Progress bar */}
          {loading && (
            <div className="mt-6">
              <div className="flex justify-between text-xs text-slate-500 mb-1.5">
                <span>Uploading & running inference…</span>
                <span>{progress}%</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-green-500 to-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-5 flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
              <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Action button */}
          <div className="mt-8 flex justify-center">
            <button
              id="detect-button"
              onClick={handleDetect}
              disabled={!canDetect}
              className={`relative flex items-center gap-3 px-10 py-4 rounded-xl font-semibold text-base transition-all duration-300 ${canDetect
                ? "bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white shadow-lg shadow-green-900/30 hover:shadow-green-900/50 hover:scale-[1.02]"
                : "bg-slate-800 text-slate-600 cursor-not-allowed"
                }`}
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 spin-slow" />
                  Analysing…
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Detect Forest Change
                </>
              )}
            </button>
          </div>
        </div>

        {/* Instruction cards */}
        <div className="grid md:grid-cols-3 gap-4 text-sm">
          {[
            { step: "01", title: "Upload Images", desc: "Select satellite imagery from two different time periods." },
            { step: "02", title: "Model Inference", desc: "Deep learning model segments forest cover in both images." },
            { step: "03", title: "Change Report", desc: "Pixel-wise comparison generates statistics and a visual change map." },
          ].map(({ step, title, desc }) => (
            <div key={step} className="glass-card p-5 hover:glow-border transition-all">
              <span className="text-xs font-mono text-blue-500 block mb-2">{step}</span>
              <h3 className="font-semibold text-slate-200 mb-1">{title}</h3>
              <p className="text-slate-500 text-xs leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
