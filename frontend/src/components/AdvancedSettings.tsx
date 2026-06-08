"use client";

import { Settings, X, ChevronDown } from "lucide-react";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export interface InferenceSettings {
  contrast_stretch: boolean;
  percentile_2_98: boolean;
  esa_offset_fix: boolean;
  enable_ndvi_veto: boolean;
  ndvi_threshold: number;
  band_order: string;
}

export const defaultSettings: InferenceSettings = {
  contrast_stretch: true,
  percentile_2_98: true,
  esa_offset_fix: false,
  enable_ndvi_veto: true,
  ndvi_threshold: 0.25,
  band_order: "RGBN",
};

interface AdvancedSettingsProps {
  settings: InferenceSettings;
  setSettings: React.Dispatch<React.SetStateAction<InferenceSettings>>;
}

export default function AdvancedSettings({ settings, setSettings }: AdvancedSettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Lock background scrolling when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  const handleToggle = (key: keyof InferenceSettings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleChange = (key: keyof InferenceSettings, value: string | number) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <>
	{/* ── Minimal Inline Modal Trigger ── */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2.5 px-4 py-2 rounded-xl hover:bg-slate-800/80 text-sm font-medium text-slate-300 hover:text-white transition-all group"
      >
        <Settings className="w-4 h-4 text-slate-400 group-hover:text-blue-400 transition-colors" />
        Advanced Settings
        <ChevronDown className="w-4 h-4 text-slate-500 group-hover:text-slate-400 transition-colors" />
      </button>

      {/* ── Framer Motion Modal ── */}
      <AnimatePresence>
        {isOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center px-4">
            
            {/* Dark Blurred Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="absolute inset-0 bg-black/60 backdrop-blur-sm"
              onClick={() => setIsOpen(false)}
            />

            {/* Modal Container */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              className="relative w-full max-w-lg bg-[#0f1523] border border-slate-700/60 rounded-2xl shadow-2xl shadow-blue-900/10 overflow-hidden flex flex-col max-h-[85vh]"
            >
              {/* Header */}
              <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
                <div className="flex items-center gap-2.5">
                  <Settings className="w-5 h-5 text-blue-400" />
                  <h2 className="text-base font-bold text-slate-200">Processing Configuration</h2>
                </div>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-2 rounded-full text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Scrollable Body */}
              <div className="p-6 overflow-y-auto space-y-8 custom-scrollbar">
                
                {/* Contrast & Normalization */}
                <div className="space-y-4">
                  <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-4">Radiometric Enhancements</h4>
                  
                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-slate-200 group-hover:text-blue-400 transition-colors">Contrast Stretching</span>
                      <span className="text-[13px] text-slate-500 mt-0.5">Dynamically scale pixels to 0.0 - 1.0 bounds</span>
                    </div>
                    <input type="checkbox" checked={settings.contrast_stretch} onChange={() => handleToggle('contrast_stretch')} className="toggle-checkbox" />
                  </label>

                  <label className={`flex items-center justify-between cursor-pointer group ${!settings.contrast_stretch ? 'opacity-40 pointer-events-none' : ''}`}>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-slate-200 group-hover:text-blue-400 transition-colors">2% - 98% Percentile Clip</span>
                      <span className="text-[13px] text-slate-500 mt-0.5">Ignore extreme sensor outliers (glare/clouds)</span>
                    </div>
                    <input type="checkbox" checked={settings.percentile_2_98} onChange={() => handleToggle('percentile_2_98')} className="toggle-checkbox" />
                  </label>

                  <label className={`flex items-center justify-between cursor-pointer group ${settings.contrast_stretch ? 'opacity-40 pointer-events-none' : ''}`}>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-slate-200 group-hover:text-blue-400 transition-colors">ESA 2022 Offset Fix</span>
                      <span className="text-[13px] text-slate-500 mt-0.5">Remove +1000 baseline noise (If stretch is off)</span>
                    </div>
                    <input type="checkbox" checked={settings.esa_offset_fix} onChange={() => handleToggle('esa_offset_fix')} className="toggle-checkbox" />
                  </label>
                </div>

                <div className="h-px w-full bg-slate-800/80" />

                {/* Post-Processing Filters */}
                <div className="space-y-4">
                  <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-4">Spectral Veto Logic</h4>
                  
                  <label className="flex items-center justify-between cursor-pointer group">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-slate-200 group-hover:text-blue-400 transition-colors">Enable NDVI Veto</span>
                      <span className="text-[13px] text-slate-500 mt-0.5">Filter out false-positives using vegetation index</span>
                    </div>
                    <input type="checkbox" checked={settings.enable_ndvi_veto} onChange={() => handleToggle('enable_ndvi_veto')} className="toggle-checkbox" />
                  </label>

                  {settings.enable_ndvi_veto && (
                    <div className="flex flex-col gap-3 pt-2 bg-slate-900/30 p-4 rounded-xl border border-slate-800">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-slate-300">Veto Threshold</span>
                        <span className="text-xs font-mono bg-blue-500/10 border border-blue-500/20 px-2.5 py-1 rounded-md text-blue-400">
                          {settings.ndvi_threshold.toFixed(2)}
                        </span>
                      </div>
                      <input 
                        type="range" 
                        min="0" max="1" step="0.05" 
                        value={settings.ndvi_threshold} 
                        onChange={(e) => handleChange('ndvi_threshold', parseFloat(e.target.value))}
                        className="w-full accent-blue-500 cursor-pointer h-1.5 bg-slate-700 rounded-lg appearance-none" 
                      />
                      <div className="flex justify-between text-[10px] text-slate-500 font-medium px-1 uppercase">
                        <span>Permissive</span>
                        <span>Strict</span>
                      </div>
                    </div>
                  )}
                </div>

                <div className="h-px w-full bg-slate-800/80" />

                {/* Sensor Layout */}
                <div className="space-y-4 pb-2">
                  <h4 className="text-[11px] font-bold text-slate-500 uppercase tracking-widest mb-4">Sensor Configuration</h4>
                  
                  <div className="flex flex-col gap-2.5">
                    <span className="text-sm font-medium text-slate-200">Multispectral Band Order</span>
                    <select 
                      value={settings.band_order}
                      onChange={(e) => handleChange('band_order', e.target.value)}
                      className="w-full bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-xl focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 block p-3 outline-none cursor-pointer hover:border-slate-600 transition-colors"
                    >
                      <option value="RGBN">Standard Sentinel-2 (Red, Green, Blue, NIR)</option>
                      <option value="NRGB">PlanetScope (NIR, Red, Green, Blue)</option>
                      <option value="BGRN">OpenCV Native (Blue, Green, Red, NIR)</option>
                    </select>
                  </div>
                </div>

              </div>

              {/* Footer */}
              <div className="px-6 py-4 border-t border-slate-800 bg-slate-900/50 flex justify-end">
                <button
                  onClick={() => setIsOpen(false)}
                  className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold rounded-xl transition-colors shadow-lg shadow-blue-900/20"
                >
                  Apply Settings
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}