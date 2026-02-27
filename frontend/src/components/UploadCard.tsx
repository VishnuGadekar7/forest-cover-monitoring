"use client";

import { useCallback, useState } from "react";
import { UploadCloud, File as FileIcon, X, CheckCircle } from "lucide-react";

interface UploadCardProps {
  label: string;
  badge: string;
  file: File | null;
  onFile: (f: File) => void;
  onClear: () => void;
}

export default function UploadCard({ label, badge, file, onFile, onClear }: UploadCardProps) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) onFile(f);
    },
    [onFile]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  };

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-200">{label}</h3>
        <span className="text-xs px-2.5 py-1 rounded-full bg-blue-500/15 text-blue-400 border border-blue-500/30 font-mono">
          {badge}
        </span>
      </div>

      {/* Drop zone / Preview */}
      {!file ? (
        <label
          htmlFor={`upload-${badge}`}
          className={`drop-zone rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer min-h-[200px] ${dragging ? "drag-over" : ""
            }`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/30 flex items-center justify-center mb-4">
            <UploadCloud className="w-7 h-7 text-blue-400" />
          </div>
          <p className="text-sm text-slate-300 font-medium mb-1">Drop satellite image here</p>
          <p className="text-xs text-slate-600">PNG · JPG · TIFF · GeoTIFF</p>
          <input
            id={`upload-${badge}`}
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff"
            className="hidden"
            onChange={handleChange}
          />
        </label>
      ) : (
        <div className="relative rounded-xl overflow-hidden border border-green-500/30 group min-h-[200px]">
          {/* Image preview */}
          <img
            src={preview!}
            alt={label}
            className="w-full h-full object-cover min-h-[200px]"
          />
          {/* Overlay on hover */}
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col items-center justify-center gap-2">
            <CheckCircle className="w-8 h-8 text-green-400" />
            <p className="text-xs text-slate-300 font-medium truncate max-w-[80%] text-center">
              {file.name}
            </p>
          </div>
          {/* Clear button */}
          <button
            onClick={onClear}
            className="absolute top-2 right-2 w-7 h-7 rounded-full bg-red-500/80 hover:bg-red-500 flex items-center justify-center transition-colors z-10"
          >
            <X className="w-4 h-4 text-white" />
          </button>
          {/* File name bar */}
          <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-3 py-2 flex items-center gap-2">
            <FileIcon className="w-3 h-3 text-slate-400 shrink-0" />
            <span className="text-xs text-slate-300 truncate">{file.name}</span>
            <span className="text-xs text-slate-600 shrink-0">
              {(file.size / 1024).toFixed(0)} KB
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
