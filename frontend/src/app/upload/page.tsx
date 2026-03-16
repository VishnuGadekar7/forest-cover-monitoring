"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Zap, AlertCircle, Loader2 } from "lucide-react";
import Navbar from "@/components/Navbar";
import UploadCard from "@/components/UploadCard";
import STACMap from "@/components/STACMap";
import { detectChange, detectChangeAutomated } from "@/lib/api";
import type { ChangeDetectionResult } from "@/lib/api";
import { motion, AnimatePresence } from "framer-motion";
import { Map, Upload as UploadIcon } from "lucide-react";

// Store result in sessionStorage so the results page can read it
function storeResult(result: ChangeDetectionResult) {
  sessionStorage.setItem("detection_result", JSON.stringify(result));
}

export default function UploadPage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<"manual" | "automated">("manual");
  const [imageT1, setImageT1] = useState<File | null>(null);
  const [imageT2, setImageT2] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState("attention_unet");
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
      const result = await detectChange(imageT1, imageT2, selectedModel, setProgress);
      storeResult(result);
      router.push("/results");
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAutomateQuery = async (bbox: [number, number, number, number], dateT1: string, dateT2: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await detectChangeAutomated({
        bbox,
        date_t1: dateT1,
        date_t2: dateT2,
        max_cloud_cover: 20,
        model_name: selectedModel
      });
      storeResult(result);
      router.push("/results");
    } catch (err: any) {
      handleApiError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleApiError = (err: any) => {
    const msg =
      err?.response?.data?.detail ??
      err?.message ??
      "An unexpected error occurred. Is the backend running?";
    setError(msg);
    // Auto-scroll to error
    setTimeout(() => {
      document.getElementById("error-box")?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
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

        {/* Tab Switcher */}
        <div className="flex justify-center mb-10">
          <div className="bg-slate-900/50 p-1.5 rounded-2xl border border-slate-800/50 flex items-center gap-1.5 backdrop-blur-md">
            <button
              onClick={() => setActiveTab("manual")}
              className={`flex items-center gap-2.5 px-7 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${activeTab === "manual" ? "bg-slate-800 text-white shadow-lg ring-1 ring-slate-700/50" : "text-slate-500 hover:text-slate-300"}`}
            >
              <UploadIcon className="w-4 h-4" />
              Manual Upload
            </button>
            <button
              onClick={() => setActiveTab("automated")}
              className={`flex items-center gap-2.5 px-7 py-2.5 rounded-xl text-sm font-bold transition-all duration-300 ${activeTab === "automated" ? "bg-slate-800 text-white shadow-lg ring-1 ring-slate-700/50" : "text-slate-500 hover:text-slate-300"}`}
            >
              <Map className="w-4 h-4" />
              Automated Ingestion
            </button>
          </div>
        </div>

        {/* Global Model Selection */}
        <div className="max-w-xl mx-auto mb-10">
          <div className="glass-card glow-border p-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
                <Zap className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-200">Processing Model</h3>
                <p className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">Segmentation Architecture</p>
              </div>
            </div>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full sm:w-auto bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-xl px-4 py-2.5 outline-none focus:ring-2 focus:ring-blue-500/40 transition-all cursor-pointer hover:border-slate-600"
            >
              <option value="attention_unet">Attention U-Net (Default)</option>
              <option value="resnet_unet">ResNet U-Net (Fast)</option>
              <option value="trans_unet">Trans-UNet (Advanced)</option>
            </select>
          </div>
        </div>

        {/* Dynamic Content */}
        <div className="relative">
          <AnimatePresence mode="wait">
            {activeTab === "manual" ? (
              <motion.div
                key="manual"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="glass-card glow-border p-8 mb-8"
              >
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
                  <div className="mt-8 animate-pulse">
                    <div className="flex justify-between text-[11px] text-slate-500 mb-2 uppercase tracking-tight font-medium">
                      <span>Uploading & Processing Data…</span>
                      <span>{progress}%</span>
                    </div>
                    <div className="h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="h-full bg-gradient-to-r from-green-500 via-blue-500 to-purple-500 rounded-full transition-all duration-500"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Action button */}
                <div className="mt-10 flex justify-center">
                  <button
                    id="detect-button"
                    onClick={handleDetect}
                    disabled={!canDetect}
                    className={`relative flex items-center gap-3 px-12 py-4 rounded-2xl font-bold text-base transition-all duration-300 ${canDetect
                      ? "bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 text-white shadow-xl shadow-green-900/30 hover:shadow-green-900/50 hover:scale-[1.03] active:scale-95"
                      : "bg-slate-800/50 text-slate-600 cursor-not-allowed border border-slate-800"
                      }`}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Processing…
                      </>
                    ) : (
                      <>
                        <Zap className="w-5 h-5" />
                        Detect Forest Change
                      </>
                    )}
                  </button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="automated"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="mb-8"
              >
                <STACMap onQuery={handleAutomateQuery} loading={loading} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Global Error box */}
          {error && (
            <motion.div
              id="error-box"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mb-8 flex items-start gap-4 p-5 rounded-2xl bg-red-500/5 border border-red-500/20 text-red-400/90 text-sm shadow-2xl backdrop-blur-sm"
            >
              <AlertCircle className="w-5 h-5 mt-0.5 shrink-0" />
              <div className="space-y-1">
                <p className="font-bold text-red-400">Analysis Failed</p>
                <p className="text-red-400/70 leading-relaxed font-medium">{error}</p>
              </div>
            </motion.div>
          )}
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
