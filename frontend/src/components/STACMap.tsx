"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Calendar, Map as MapIcon, Loader2, Target } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface STACMapProps {
  onQuery: (bbox: [number, number, number, number], dateT1: string, dateT2: string) => void;
  loading: boolean;
}

export default function STACMap({ onQuery, loading }: STACMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);
  const [dateT1, setDateT1] = useState("2021-01-01/2021-12-31");
  const [dateT2, setDateT2] = useState("2023-01-01/2023-12-31");

  useEffect(() => {
    if (!mapRef.current) return;

    let isMounted = true;

    // Dynamically import Leaflet & Geoman
    import("leaflet").then((L) => {
      if (!isMounted || !mapRef.current || mapInstance.current) return;

      const leaflet = L.default;
      (window as any).L = leaflet; // Critical: Geoman needs global L

      import("@geoman-io/leaflet-geoman-free").then(() => {
        if (!isMounted || !mapRef.current || mapInstance.current) return;

        // Final check: ensure Leaflet didn't attach to this DOM already
        if ((mapRef.current as any)._leaflet_id) return;

        const map = leaflet.map(mapRef.current!, {
          center: [-9.0, -63.0], // Default to Amazon area (popular for forest change)
          zoom: 10,
          zoomControl: false,
        });

        leaflet.tileLayer(
          "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
          {
            attribution: '&copy; <a href="https://carto.com">CARTO</a>',
            maxZoom: 19,
          }
        ).addTo(map);

        // Add Zoom Control to a better position
        leaflet.control.zoom({ position: "bottomright" }).addTo(map);

        // Initialize Geoman
        (map as any).pm.addControls({
          position: 'topleft',
          drawMarker: false,
          drawCircleMarker: false,
          drawPolyline: false,
          drawPolygon: false,
          drawCircle: false,
          cutPolygon: false,
          rotateMode: false,
          drawRectangle: true,
        });

        // Customize Geoman Draw Colors
        (map as any).pm.setGlobalOptions({
          pathOptions: { color: '#22c55e', fillColor: '#22c55e', fillOpacity: 0.2 }
        });

        // Listen for Rectangle creation
        map.on('pm:create', (e: any) => {
          const layer = e.layer;
          const bounds = layer.getBounds();
          const west = bounds.getWest();
          const south = bounds.getSouth();
          const east = bounds.getEast();
          const north = bounds.getNorth();

          // Clean up previous layers if any
          map.eachLayer((l: any) => {
            if (l instanceof (leaflet as any).Rectangle && l !== layer) {
              map.removeLayer(l);
            }
          });

          setBbox([west, south, east, north]);
        });

        mapInstance.current = map;
      });
    });

    return () => {
      isMounted = false;
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, []);

  return (
    <div className="relative w-full h-[500px] rounded-2xl overflow-hidden border border-slate-800">
      <div ref={mapRef} className="w-full h-full z-0" />

      {/* Control Overlay */}
      <div className="absolute top-4 right-4 z-[1000] w-80 space-y-3">
        {/* Date Controls */}
        <div className="glass-card p-4 shadow-2xl border-slate-700/50">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Calendar className="w-3.5 h-3.5" /> Time Comparison
          </h4>

          <div className="space-y-3">
            <div className="space-y-1">
              <label className="text-[10px] text-slate-400">Baseline (T1)</label>
              <input
                type="text"
                value={dateT1}
                onChange={(e) => setDateT1(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-green-500/50 transition-colors"
                placeholder="YYYY-MM-DD or Range"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[10px] text-slate-400">Analysis (T2)</label>
              <input
                type="text"
                value={dateT2}
                onChange={(e) => setDateT2(e.target.value)}
                className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500/50 transition-colors"
                placeholder="YYYY-MM-DD or Range"
              />
            </div>
          </div>
        </div>

        {/* Action Button */}
        <AnimatePresence>
          {bbox && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
            >
              <button
                onClick={() => onQuery(bbox, dateT1, dateT2)}
                disabled={loading}
                className="w-full group relative flex items-center justify-center gap-3 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white py-3.5 rounded-xl font-bold text-sm shadow-xl shadow-green-900/40 transition-all hover:scale-[1.02] disabled:opacity-50 disabled:scale-100"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Target className="w-4 h-4 group-hover:scale-125 transition-transform" />
                )}
                Fetch & Analyze Area
              </button>
              <p className="text-[10px] text-center text-slate-400 mt-2">
                Coordinates: {bbox[0].toFixed(3)}, {bbox[1].toFixed(3)} to {bbox[2].toFixed(3)}, {bbox[3].toFixed(3)}
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {!bbox && (
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-3 text-[11px] text-blue-400 flex items-start gap-3">
            <MapIcon className="w-4 h-4 shrink-0 mt-0.5" />
            <span>Use the square tool on the left to draw your Area of Interest (AOI) for analysis.</span>
          </div>
        )}
      </div>

      {/* Floating Badge */}
      <div className="absolute bottom-4 left-4 z-[1000]">
        <div className="px-3 py-1.5 rounded-full glass-card border-slate-700/50 text-[10px] text-slate-400 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
          Sentinel-2 L2A Pipeline
        </div>
      </div>
    </div>
  );
}
