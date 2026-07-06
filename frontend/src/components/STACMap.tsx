"use client";

import { useEffect, useRef, useState } from "react";
import { Search, Calendar, Map as MapIcon, Loader2, Target, AlertTriangle, CheckCircle2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import "./map.css";

interface STACMapProps {
  onQuery: (bbox: [number, number, number, number], dateT1: string, dateT2: string) => void;
  loading: boolean;
}

// Enforce strict pixel limit (50M pixels at 10m resolution)
const MAX_PIXEL_LIMIT = 50_000_000;

// Helper function to calculate distance between coordinates in meters using Haversine
function getDistanceMeters(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6371e3;
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
  const leafletRef = useRef<any>(null);

  // Coordinate Input State (String state allows easy typing of decimal points)
  const [west, setWest] = useState("");
  const [south, setSouth] = useState("");
  const [east, setEast] = useState("");
  const [north, setNorth] = useState("");

  const [bbox, setBbox] = useState<[number, number, number, number] | null>(null);
  const [isTooLarge, setIsTooLarge] = useState(false);
  const [dateT1, setDateT1] = useState("2025-01-01/2025-01-31");
  const [dateT2, setDateT2] = useState("2026-01-01/2026-01-31");

  // Validate area size
  const checkAreaLimit = (w: number, s: number, e: number, n: number) => {
    const widthMeters = getDistanceMeters(s, w, s, e);
    const heightMeters = getDistanceMeters(s, w, n, w);
    const totalEstimatedPixels = (widthMeters / 10) * (heightMeters / 10);
    return totalEstimatedPixels > MAX_PIXEL_LIMIT;
  };

  // 1. Initialize Map and Layers
  useEffect(() => {
    if (!mapRef.current) return;

    let isMounted = true;

    import("leaflet").then((L) => {
      if (!isMounted || !mapRef.current || mapInstance.current) return;

      const leaflet = L.default;
      leafletRef.current = leaflet;
      (window as any).L = leaflet;

      import("@geoman-io/leaflet-geoman-free").then(() => {
        if (!isMounted || !mapRef.current || mapInstance.current) return;
        if ((mapRef.current as any)._leaflet_id) return;

        const map = leaflet.map(mapRef.current!, {
          center: [22.0, 78.0], // Centered over India
          zoom: 5,
          zoomControl: false,
        });

        // Base Satellite Layer
        leaflet.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
          { maxZoom: 19 }
        ).addTo(map);

        // State Boundaries & Places Overlay
        leaflet.tileLayer(
          "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
          { maxZoom: 19 }
        ).addTo(map);

        leaflet.control.zoom({ position: "bottomright" }).addTo(map);

        (map as any).pm.addControls({
          position: "topleft",
          drawMarker: false,
          drawCircleMarker: false,
          drawPolyline: false,
          drawPolygon: false,
          drawCircle: false,
          cutPolygon: false,
          rotateMode: false,
          drawRectangle: true,
        });

        (map as any).pm.setGlobalOptions({
          pathOptions: { color: "#22c55e", fillColor: "#22c55e", fillOpacity: 0.2 },
        });

        map.on("pm:create", (e: any) => {
          const layer = e.layer;
          const bounds = layer.getBounds();
          const w = bounds.getWest();
          const s = bounds.getSouth();
          const ea = bounds.getEast();
          const n = bounds.getNorth();

          // Remove old drawn rectangles
          map.eachLayer((l: any) => {
            if (l instanceof (leaflet as any).Rectangle && l !== layer) {
              map.removeLayer(l);
            }
          });

          const tooLarge = checkAreaLimit(w, s, ea, n);
          setIsTooLarge(tooLarge);
          layer.setStyle({
            color: tooLarge ? "#ef4444" : "#22c55e",
            fillColor: tooLarge ? "#ef4444" : "#22c55e",
          });

          // Sync drawn coordinates with input boxes
          setWest(w.toFixed(5));
          setSouth(s.toFixed(5));
          setEast(ea.toFixed(5));
          setNorth(n.toFixed(5));
          setBbox([w, s, ea, n]);
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

  // Programmatically draw rectangle when user inputs coords manually
  const applyManualCoordinates = () => {
    if (!mapInstance.current || !leafletRef.current) return;

    const w = parseFloat(west);
    const s = parseFloat(south);
    const ea = parseFloat(east);
    const n = parseFloat(north);

    if (isNaN(w) || isNaN(s) || isNaN(ea) || isNaN(n) || w >= ea || s >= n) {
      alert("Please enter valid bounding box coordinates (West < East, South < North).");
      return;
    }

    const L = leafletRef.current;
    const map = mapInstance.current;

    // Clear existing rectangles
    map.eachLayer((l: any) => {
      if (l instanceof L.Rectangle) {
        map.removeLayer(l);
      }
    });

    const bounds = L.latLngBounds([s, w], [n, ea]);
    const tooLarge = checkAreaLimit(w, s, ea, n);
    setIsTooLarge(tooLarge);

    const rect = L.rectangle(bounds, {
      color: tooLarge ? "#ef4444" : "#22c55e",
      fillColor: tooLarge ? "#ef4444" : "#22c55e",
      fillOpacity: 0.2,
      weight: 2,
    }).addTo(map);

    map.fitBounds(bounds, { padding: [30, 30] });
    setBbox([w, s, ea, n]);
  };

  const areCoordsValid =
    west !== "" && south !== "" && east !== "" && north !== "" &&
    !isNaN(Number(west)) && !isNaN(Number(south)) &&
    !isNaN(Number(east)) && !isNaN(Number(north)) &&
    Number(west) < Number(east) && Number(south) < Number(north);

  return (
    <div className="space-y-6">
      {/* Instructional Help Banner (Outside Map) */}
      <div className="bg-blue-500/10 border border-blue-500/20 rounded-xl p-4 text-xs text-blue-300 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <MapIcon className="w-5 h-5 text-blue-400 shrink-0" />
          <span>
            Use the square drawing tool directly on the map, <strong>or enter coordinates manually below</strong> to define your Area of Interest (AOI).
          </span>
        </div>
        {bbox && !isTooLarge && (
          <span className="flex items-center gap-1.5 text-green-400 font-bold shrink-0 bg-green-500/10 px-2.5 py-1 rounded-full border border-green-500/20">
            <CheckCircle2 className="w-3.5 h-3.5" /> AOI Selected
          </span>
        )}
      </div>

      {/* Map Container */}
      <div className="relative w-full h-[450px] rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
        <div ref={mapRef} className="w-full h-full z-0" />
        <div className="absolute bottom-4 left-4 z-[1000] pointer-events-none">
          <div className="px-3 py-1.5 rounded-full bg-slate-900/80 backdrop-blur-md border border-slate-700/50 text-[10px] text-slate-300 flex items-center gap-2 shadow-lg">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
            Sentinel-2 L2A Pipeline
          </div>
        </div>
      </div>

      {/* Controls Panel (Outside Map) */}
      <div className="glass-card p-6 border-slate-800/80 shadow-xl space-y-6">
        {/* Section 1: Manual Coordinates */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-2">
              <Target className="w-3.5 h-3.5 text-green-400" /> Bounding Box Coordinates (WGS84 Decimal Degrees)
            </h4>
            {areCoordsValid && (
              <button
                type="button"
                onClick={applyManualCoordinates}
                className="text-xs bg-slate-800 hover:bg-slate-700 text-blue-400 hover:text-blue-300 px-3 py-1 rounded-lg border border-slate-700 font-semibold transition-all"
              >
                Preview Box on Map →
              </button>
            )}
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="space-y-1">
              <label className="text-[11px] text-slate-400 font-medium">Min Lon (West)</label>
              <input
                type="number"
                step="any"
                value={west}
                onChange={(e) => setWest(e.target.value)}
                placeholder="e.g. 84.215"
                className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-green-500/60 font-mono transition-colors"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] text-slate-400 font-medium">Min Lat (South)</label>
              <input
                type="number"
                step="any"
                value={south}
                onChange={(e) => setSouth(e.target.value)}
                placeholder="e.g. 23.530"
                className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-green-500/60 font-mono transition-colors"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] text-slate-400 font-medium">Max Lon (East)</label>
              <input
                type="number"
                step="any"
                value={east}
                onChange={(e) => setEast(e.target.value)}
                placeholder="e.g. 84.265"
                className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-green-500/60 font-mono transition-colors"
              />
            </div>
            <div className="space-y-1">
              <label className="text-[11px] text-slate-400 font-medium">Max Lat (North)</label>
              <input
                type="number"
                step="any"
                value={north}
                onChange={(e) => setNorth(e.target.value)}
                placeholder="e.g. 23.576"
                className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-green-500/60 font-mono transition-colors"
              />
            </div>
          </div>
        </div>

        <hr className="border-slate-800" />

        {/* Section 2: Time Period & Execution */}
        <div className="grid md:grid-cols-3 gap-6 items-end">
          <div className="space-y-1.5">
            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-blue-400" /> Baseline Date (T1)
            </label>
            <input
              type="text"
              value={dateT1}
              onChange={(e) => setDateT1(e.target.value)}
              className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500/60 font-mono transition-colors"
              placeholder="YYYY-MM-DD/YYYY-MM-DD"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-blue-400" /> Analysis Date (T2)
            </label>
            <input
              type="text"
              value={dateT2}
              onChange={(e) => setDateT2(e.target.value)}
              className="w-full bg-slate-900/80 border border-slate-800 rounded-xl px-3.5 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-blue-500/60 font-mono transition-colors"
              placeholder="YYYY-MM-DD/YYYY-MM-DD"
            />
          </div>

          <div>
            {isTooLarge ? (
              <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-2.5 text-[11px] text-red-400 flex items-center gap-2 font-medium">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>Selected area exceeds 50M pixels. Reduce box size.</span>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => {
                  if (!bbox && areCoordsValid) {
                    applyManualCoordinates();
                    onQuery([Number(west), Number(south), Number(east), Number(north)], dateT1, dateT2);
                  } else if (bbox) {
                    onQuery(bbox, dateT1, dateT2);
                  }
                }}
                disabled={loading || (!bbox && !areCoordsValid)}
                className="w-full flex items-center justify-center gap-2.5 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white py-2.5 rounded-xl font-bold text-sm shadow-xl shadow-green-900/30 transition-all hover:scale-[1.02] active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" /> Fetching & Processing...
                  </>
                ) : (
                  <>
                    <Target className="w-4 h-4" /> Fetch & Analyze Area
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}