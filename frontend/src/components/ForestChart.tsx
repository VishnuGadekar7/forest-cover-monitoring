"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface ForestChartProps {
  areaT1: number;
  areaT2: number;
  loss: number;
  gain: number;
}

// Custom tooltip
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card px-4 py-3 text-sm text-slate-200 shadow-xl">
        <p className="font-semibold mb-1">{label}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: <span className="font-bold">{p.value.toFixed(2)} ha</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function ForestChart({ areaT1, areaT2, loss, gain }: ForestChartProps) {
  const coverData = [
    { name: "Time 1 (T₁)", area: areaT1 },
    { name: "Time 2 (T₂)", area: areaT2 },
  ];

  const changeData = [
    { name: "Forest Loss", value: loss },
    { name: "Forest Gain", value: gain },
    { name: "Net Δ", value: gain - loss },
  ];

  return (
    <div className="grid md:grid-cols-2 gap-6">
      {/* Cover comparison */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          Forest Cover Comparison (ha)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={coverData} barSize={48}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2d45" />
            <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(59,130,246,0.05)" }} />
            <Bar dataKey="area" radius={[6, 6, 0, 0]} name="Forest Area">
              <Cell fill="#22c55e" />
              <Cell fill={areaT2 < areaT1 ? "#ef4444" : "#3b82f6"} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Change breakdown */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-semibold text-slate-300 mb-4">
          Change Breakdown (ha)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={changeData} barSize={36}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2d45" />
            <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#64748b", fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(59,130,246,0.05)" }} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]} name="Area (ha)">
              <Cell fill="#ef4444" />
              <Cell fill="#22c55e" />
              <Cell fill={gain - loss >= 0 ? "#3b82f6" : "#f97316"} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
