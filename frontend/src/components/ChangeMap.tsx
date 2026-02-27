"use client";

import { useEffect, useRef } from "react";
import { assetUrl } from "@/lib/api";

interface ChangeMapProps {
  changeMapUrl: string;
}

export default function ChangeMap({ changeMapUrl }: ChangeMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstance.current) return;

    // Dynamically import Leaflet (SSR-safe)
    import("leaflet").then((L) => {
      const leaflet = L.default;

      // Fix default icon path for Next.js
      delete (leaflet.Icon.Default.prototype as any)._getIconUrl;
      leaflet.Icon.Default.mergeOptions({
        iconRetinaUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
        iconUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
        shadowUrl: "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
      });

      const map = leaflet.map(mapRef.current!, {
        center: [20.5937, 78.9629],  // Geographic center of India
        zoom: 5,
        zoomControl: true,
        attributionControl: true,
      });

      // Dark satellite basemap (CartoDB Dark)
      leaflet.tileLayer(
        "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        {
          attribution: '&copy; <a href="https://carto.com">CARTO</a>',
          maxZoom: 19,
        }
      ).addTo(map);

      // Overlay the change map image over India as a placeholder region
      // In production, use actual image bounds from rasterio metadata
      const imageBounds: [[number, number], [number, number]] = [
        [8.0, 68.0],
        [37.0, 97.5],
      ];
      const fullUrl = assetUrl(changeMapUrl);
      leaflet.imageOverlay(fullUrl, imageBounds, { opacity: 0.75 }).addTo(map);
      map.fitBounds(imageBounds);

      // Legend
      const legend = (leaflet.control as any)({ position: "bottomright" });
      legend.onAdd = () => {
        const div = leaflet.DomUtil.create("div");
        div.innerHTML = `
          <div style="background:rgba(10,15,26,0.9);border:1px solid #1f2d45;border-radius:8px;padding:12px;font-size:12px;color:#e2e8f0">
            <div style="font-weight:600;margin-bottom:8px">Change Legend</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px"><span style="width:14px;height:14px;background:#ef4444;border-radius:2px;display:inline-block"></span> Forest Loss</div>
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px"><span style="width:14px;height:14px;background:#22c55e;border-radius:2px;display:inline-block"></span> Forest Gain</div>
            <div style="display:flex;align-items:center;gap:8px"><span style="width:14px;height:14px;background:#ffffff;border-radius:2px;display:inline-block"></span> Stable Forest</div>
          </div>
        `;
        return div;
      };
      legend.addTo(map);

      mapInstance.current = map;
    });

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [changeMapUrl]);

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 pt-5 pb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-300">Geographic Change Map</h3>
        <div className="flex gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-red-500 inline-block" /> Loss
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-green-500 inline-block" /> Gain
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-white inline-block" /> Stable
          </span>
        </div>
      </div>
      <div ref={mapRef} className="w-full h-[420px]" />
    </div>
  );
}
