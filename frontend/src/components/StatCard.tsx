"use client";

import { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  icon: LucideIcon;
  color: "green" | "red" | "blue" | "amber";
  subtitle?: string;
  delay?: number;
}

const colorMap = {
  green: {
    icon: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/20",
    value: "text-green-400",
    glow: "shadow-green-500/10",
  },
  red: {
    icon: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
    value: "text-red-400",
    glow: "shadow-red-500/10",
  },
  blue: {
    icon: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
    value: "text-blue-400",
    glow: "shadow-blue-500/10",
  },
  amber: {
    icon: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
    value: "text-amber-400",
    glow: "shadow-amber-500/10",
  },
};

export default function StatCard({
  label,
  value,
  unit = "ha",
  icon: Icon,
  color,
  subtitle,
  delay = 0,
}: StatCardProps) {
  const c = colorMap[color];
  return (
    <div
      className={`stat-card p-6 fade-in-up shadow-lg ${c.glow}`}
      style={{ animationDelay: `${delay}ms` }}
    >
      {/* Icon */}
      <div className={`inline-flex p-2.5 rounded-lg ${c.bg} border ${c.border} mb-4`}>
        <Icon className={`w-5 h-5 ${c.icon}`} />
      </div>

      {/* Value */}
      <div className="flex items-end gap-1 mb-1">
        <span className={`text-3xl font-bold tabular-nums ${c.value}`}>
          {typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
        </span>
        <span className="text-sm text-slate-500 mb-1">{unit}</span>
      </div>

      {/* Label */}
      <p className="text-sm font-medium text-slate-300">{label}</p>
      {subtitle && <p className="text-xs text-slate-600 mt-0.5">{subtitle}</p>}
    </div>
  );
}
