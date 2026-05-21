"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Calendar, Map as MapIcon, Loader2, Target, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import "./map.css";

interface STACMapProps {
  onQuery: (bbox: [number, number, number, number], dateT1: string, dateT2: string) => void;
  loading: boolean;
}

// Enforce strict pixel limit
const MAX_PIXEL_LIMIT = 50_000_000; 

// Helper function to calculate distance between coordinates in meters using Haversine
function getDistanceMeters(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6371e3; // Earth's radius in meters
  const phi1 = (lat1 * Math.PI) / 180;
  const phi2 = (lat2 * Math.PI) / 180;
  const deltaPhi = ((lat2 - lat1) * Math.PI) / 180;
  const deltaLambda = ((lon2 - lon1) * Math.PI) / 180;

  const a =
    Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
    Math.cos(phi1) * Math.cos(phi2) * Math.sin(deltaLambda / 2) * Math.sin(deltaLambda / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

  return R * c;
}

export default function STACMap({ onQuery, loading }: STACMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);
  const [isTooLarge, setIsTooLarge] = useState(false);
  const [dateT1, setDateT1] = useState("2025-01-01/2025-01-31");
  const [dateT2, setDateT2] = useState("2026-01-01/2026-01-31");

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
          center: [22.0, 78.0], // Default map center to India
          zoom: 5,
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

		      // Pixel distance calculations
    		  const widthMeters = getDistanceMeters(south, west, south, east);
          const heightMeters = getDistanceMeters(south, west, north, west);
          
          // Sentinel-2 has a spatial resolution of 10 meters per pixel
          const pixelWidth = widthMeters / 10;
          const pixelHeight = heightMeters / 10;
          const totalEstimatedPixels = pixelWidth * pixelHeight;

          if (totalEstimatedPixels > MAX_PIXEL_LIMIT) {
            setIsTooLarge(true);
            // Change the rectangle color to red dynamically to indicate validation failure
            layer.setStyle({ color: '#ef4444', fillColor: '#ef4444' });
          } else {
            setIsTooLarge(false);
            layer.setStyle({ color: '#22c55e', fillColor: '#22c55e' });
          }

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
        <AnimatePresence mode="wait">
          {bbox && (
            <motion.div
              initial={{ opacity: 0, y: 10, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 10, scale: 0.95 }}
            >
              {isTooLarge ? (
                // Over-Limit UI Layout State
                <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-3 text-[11px] text-red-400 flex items-start gap-3 shadow-2xl backdrop-blur-md">
                  <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                  <span>Selected area is too large! Please zoom in or select a smaller bounding box.</span>
                </div>
              ) : (
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
              )}
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
