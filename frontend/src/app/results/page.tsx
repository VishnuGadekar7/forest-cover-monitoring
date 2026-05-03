"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  TreePine, TrendingDown, TrendingUp, BarChart2, ArrowLeft, Map, Download, X
} from "lucide-react";
import dynamic from "next/dynamic";
import Navbar from "@/components/Navbar";
import StatCard from "@/components/StatCard";
import ForestChart from "@/components/ForestChart";
import { assetUrl, exportChangeMapTif } from "@/lib/api";
import type { ChangeDetectionResult } from "@/lib/api";

// ── Leaflet must be loaded client-side only (no SSR) ──────────────────────
const ChangeMap = dynamic(() => import("@/components/ChangeMap"), { ssr: false });

export default function ResultsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const targetId = searchParams.get("id"); // Get ID from URL e.g. /results?id=abc123xyz
  
  const [result, setResult] = useState<ChangeDetectionResult | null>(null);

  // Download Menu State
  const [isDownloadOpen, setIsDownloadOpen] = useState(false);
  const [downloadFormat, setDownloadFormat] = useState<"png" | "tif">("png");

  useEffect(() => {
    // Read the array from local storage
    const savedHistory = localStorage.getItem("prediction_history");
    const history = savedHistory ? JSON.parse(savedHistory) : [];
    
    if (history.length > 0) {
      if (targetId) {
        // Find the specific result the user clicked on
        const foundResult = history.find((r: ChangeDetectionResult) => r.id === targetId);
        setResult(foundResult || history[0]); // fallback to newest if ID not found
      } else {
        // Fallback: Just show the most recent one
        setResult(history[0]);
      }
    }
  }, [targetId]);

  const handleDownload = async () => {
    if (!result) return;

    if (downloadFormat === "png") {
      // Standard frontend PNG download
      const link = document.createElement("a");
      link.href = assetUrl(result.change_map_url);
	    link.target = '_blank';
      link.download = `change_map_${result.id}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      // TIF Download: Hit the backend API to generate the GeoTIFF
      try {
        const blob = await exportChangeMapTif(result.id, 4326);
        
        // Create a temporary object URL to trigger the browser download
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `change_map_${result.id}.tif`;
        
        document.body.appendChild(link);
        link.click();
        
        // Cleanup the DOM and memory
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      } catch (error) {
        console.error("Failed to export TIF:", error);
        alert("Failed to generate TIF export. Check backend logs.");
      }
    }     
    setIsDownloadOpen(false);
  };

  if (!result) {
    return (
      <div className="min-h-screen bg-[#0a0f1a] flex flex-col items-center justify-center gap-4">
        <Navbar />
        <div className="glass-card p-10 text-center max-w-sm mt-16">
          <BarChart2 className="w-10 h-10 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-300 font-semibold mb-2">No Results Yet</p>
          <p className="text-slate-600 text-sm mb-6">Upload two satellite images first to run change detection.</p>
          <button
            onClick={() => router.push("/upload")}
            className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors"
          >
            Go to Upload
          </button>
        </div>
      </div>
    );
  }

  const lossColor = result.percentage_change < 0 ? "red" : "green";

  return (
    <div className="min-h-screen bg-[#0a0f1a] relative">
      <Navbar />

      {/* --- Download Modal --- */}
      {isDownloadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
          <div className="glass-card w-full max-w-md p-6 relative animate-in fade-in zoom-in-95 duration-200">
            <button 
              onClick={() => setIsDownloadOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            
            <h3 className="text-xl font-bold text-white mb-1">Export Results</h3>
            <p className="text-sm text-slate-400 mb-6">Configure your output map parameters.</p>

            <div className="space-y-5">
              {/* Format Selection */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">File Format</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setDownloadFormat("png")}
                    className={`py-2 px-4 rounded-lg border text-sm font-medium transition-all ${
                      downloadFormat === "png" 
                      ? "bg-blue-600/20 border-blue-500 text-blue-400" 
                      : "bg-slate-800/50 border-slate-700 text-slate-400 hover:bg-slate-800"
                    }`}
                  >
                    PNG (Standard)
                  </button>
                  <button
                    onClick={() => setDownloadFormat("tif")}
                    className={`py-2 px-4 rounded-lg border text-sm font-medium transition-all ${
                      downloadFormat === "tif" 
                      ? "bg-blue-600/20 border-blue-500 text-blue-400" 
                      : "bg-slate-800/50 border-slate-700 text-slate-400 hover:bg-slate-800"
                    }`}
                  >
                    TIF (Geospatial)
                  </button>
                </div>
              </div>

              {/* Locked Geospatial Options */}
              <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-300">Coordinate Reference System</span>
                  <span className="text-xs font-mono bg-slate-800 px-2 py-1 rounded text-slate-400">EPSG:4326</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-300">Resolution</span>
                  <span className="text-xs font-mono bg-slate-800 px-2 py-1 rounded text-slate-400">Original Dimensions</span>
                </div>
                {downloadFormat === "tif" && (
                  <div className="flex items-center justify-between animate-in fade-in slide-in-from-top-2">
                    <span className="text-sm text-slate-300">Data Type</span>
                    <span className="text-xs font-mono bg-blue-900/30 text-blue-400 border border-blue-800 px-2 py-1 rounded">16-bit Integer</span>
                  </div>
                )}
              </div>

              <button
                onClick={handleDownload}
                className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold transition-colors flex items-center justify-center gap-2 mt-4"
              >
                <Download className="w-4 h-4" />
                Download Map
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="pt-24 pb-16 px-4 max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-10 fade-in-up">
          <div>
            <button
              onClick={() => router.push("/upload")}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors mb-3"
            >
              <ArrowLeft className="w-3.5 h-3.5" /> Back to Upload
            </button>
            <h1 className="text-3xl font-bold gradient-text">Change Detection Report</h1>
            <p className="text-slate-500 text-sm mt-1">
              Deep learning segmentation · Temporal pixel-wise comparison
            </p>
          </div>
          <div className="flex items-center gap-2 self-start md:self-auto">
            <span className={`text-2xl font-bold tabular-nums ${result.percentage_change < 0 ? "text-red-400" : "text-green-400"}`}>
              {result.percentage_change > 0 ? "+" : ""}
              {result.percentage_change.toFixed(2)}%
            </span>
            <span className="text-slate-500 text-sm">net change</span>
            {/* Download Button */}
            <button 
              onClick={() => setIsDownloadOpen(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm text-slate-200 transition-colors ml-4"
            >
              <Download className="w-4 h-4" />
              Export Map
            </button>
          </div>
        </div>

        {/* ── Stat cards ──────────────────────────────────────────────────── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatCard
            label="Forest Area T₁"
            value={result.forest_area_t1}
            icon={TreePine}
            color="green"
            subtitle="Earlier period"
            delay={0}
          />
          <StatCard
            label="Forest Area T₂"
            value={result.forest_area_t2}
            icon={TreePine}
            color="blue"
            subtitle="Later period"
            delay={80}
          />
          <StatCard
            label="Forest Loss"
            value={result.forest_loss}
            icon={TrendingDown}
            color="red"
            subtitle="T₁ → non-forest"
            delay={160}
          />
          <StatCard
            label="Forest Gain"
            value={result.forest_gain}
            icon={TrendingUp}
            color={result.forest_gain > 0 ? "green" : "amber"}
            subtitle="Non-forest → T₂"
            delay={240}
          />
        </div>

        {/* ── Image Row: Original Sources ────────────────────────────────────────── */}
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 fade-in-up">
          Source Imagery
        </h2>
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          {[
            { url: result.image_t1_url, label: "Original Satellite — T₁", badge: "Source T₁" },
            { url: result.image_t2_url, label: "Original Satellite — T₂", badge: "Source T₂" },
          ].map(({ url, label, badge }) => (
            <div key={badge} className="glass-card overflow-hidden fade-in-up">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
                <span className="text-xs font-medium text-slate-400">{label}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-500 font-mono">{badge}</span>
              </div>
              <img
                src={assetUrl(url)}
                alt={label}
                className="w-full object-contain bg-black/20"
                style={{ aspectRatio: "1/1", width: "100%" }}
                onError={(e) => {
                  (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='80'%3E%3Crect fill='%23111827' width='100' height='80'/%3E%3Ctext fill='%23334155' font-size='11' x='50' y='45' text-anchor='middle'%3EImage unavailable%3C/text%3E%3C/svg%3E";
                }}
              />
            </div>
          ))}
        </div>

        {/* ── Image Row: T1 | Change Map | T2 ─────────────────────────────── */}
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4 fade-in-up">
          Analysis Layers
        </h2>
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {[
            { url: result.mask_t1_url, label: "Predicted Mask — T₁", badge: "T₁" },
            { url: result.change_map_url, label: "Change Map", badge: "Δ" },
            { url: result.mask_t2_url, label: "Predicted Mask — T₂", badge: "T₂" },
          ].map(({ url, label, badge }) => (
            <div key={badge} className="glass-card overflow-hidden fade-in-up">
              <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
                <span className="text-xs font-medium text-slate-400">{label}</span>
                <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-500 font-mono">{badge}</span>
              </div>
              <img
                src={assetUrl(url)}
                alt={label}
                className="w-full object-contain bg-black/20"
                style={{ aspectRatio: "1/1", width: "100%" }}
                onError={(e) => {
                  (e.target as HTMLImageElement).src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='80'%3E%3Crect fill='%23111827' width='100' height='80'/%3E%3Ctext fill='%23334155' font-size='11' x='50' y='45' text-anchor='middle'%3EImage unavailable%3C/text%3E%3C/svg%3E";
                }}
              />
            </div>
          ))}
        </div>

        {/* ── Change colour legend ─────────────────────────────────────────── */}
        <div className="glass-card p-5 mb-8 fade-in-up">
          <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <Map className="w-4 h-4 text-blue-400" /> Change Map Legend
          </h3>
          <div className="flex flex-wrap gap-6 text-sm">
            {[
              { color: "bg-red-500", label: "Forest Loss", desc: "Was forest in T₁, non-forest in T₂" },
              { color: "bg-green-500", label: "Forest Gain", desc: "Was non-forest in T₁, forest in T₂" },
              { color: "bg-white", label: "Stable Forest", desc: "Forest in both T₁ and T₂" },
              { color: "bg-slate-900 border border-slate-700", label: "Non-Forest", desc: "Non-forest in both epochs" },
            ].map(({ color, label, desc }) => (
              <div key={label} className="flex items-start gap-3">
                <span className={`w-5 h-5 rounded mt-0.5 shrink-0 ${color}`} />
                <div>
                  <p className="font-medium text-slate-200">{label}</p>
                  <p className="text-xs text-slate-500">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── Charts ──────────────────────────────────────────────────────── */}
        <div className="mb-8 fade-in-up">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Statistical Analysis
          </h2>
          <ForestChart
            areaT1={result.forest_area_t1}
            areaT2={result.forest_area_t2}
            loss={result.forest_loss}
            gain={result.forest_gain}
          />
        </div>

        {/* ── Leaflet Map ──────────────────────────────────────────────────── */}
        <div className="fade-in-up">
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
            Geographic Visualisation
          </h2>
          <ChangeMap changeMapUrl={result.change_map_url} />
        </div>
      </main>
    </div>
  );
}