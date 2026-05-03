"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { 
  Trash2, AlertCircle, X, TrendingUp, TrendingDown, Map, Clock 
} from "lucide-react";
import Navbar from "@/components/Navbar";

export default function HistoryPage() {
  const router = useRouter();
  const [history, setHistory] = useState<any[]>([]);
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    setHistory(saved);
  }, []);

  // ── Delete Individual Entry ──
  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    const newHistory = history.filter((item) => item.id !== id);
    setHistory(newHistory);
    localStorage.setItem("prediction_history", JSON.stringify(newHistory));
  };

  // ── Clear All Entries ──
  const handleClearAll = () => {
    setHistory([]);
    localStorage.removeItem("prediction_history");
    setShowClearConfirm(false);
  };

  return (
    <div className="min-h-screen bg-[#0a0f1a] relative">
      <Navbar />

      {/* ── Clear All Confirmation Modal ── */}
      {showClearConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
          <div className="glass-card w-full max-w-sm p-6 relative animate-in fade-in zoom-in-95 duration-200">
            <button 
              onClick={() => setShowClearConfirm(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
            
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 rounded-full bg-red-500/20 text-red-400">
                <AlertCircle className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-white">Clear History</h3>
            </div>
            
            <p className="text-sm text-slate-400 mb-6 leading-relaxed">
              Are you sure you want to delete all saved analyses? This action cannot be undone.
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => setShowClearConfirm(false)}
                className="flex-1 py-2.5 rounded-lg border border-slate-700 bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleClearAll}
                className="flex-1 py-2.5 rounded-lg bg-red-600 hover:bg-red-500 text-white text-sm font-medium transition-colors"
              >
                Yes, Clear All
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="pt-24 pb-16 px-4 max-w-4xl mx-auto">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold gradient-text">Recent Analyses</h1>
            <p className="text-slate-500 text-sm mt-1">Your locally saved forest cover reports.</p>
          </div>
          
          {history.length > 0 && (
            <button
              onClick={() => setShowClearConfirm(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 hover:text-red-300 transition-colors text-sm font-medium self-start sm:self-auto"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          )}
        </div>
        
        {history.length === 0 ? (
          <div className="glass-card p-10 text-center">
            <div className="w-16 h-16 bg-slate-800/50 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-slate-700">
               <Map className="w-8 h-8 text-slate-500" />
            </div>
            <p className="text-slate-300 font-medium text-lg mb-2">No history found</p>
            <p className="text-slate-500 text-sm max-w-sm mx-auto">Your recent change detection analyses will appear here once you process them.</p>
            <button
              onClick={() => router.push("/upload")}
              className="mt-6 px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-colors"
            >
              Start a new analysis
            </button>
          </div>
        ) : (
          <div className="grid gap-4 fade-in-up">
            {history.map((item) => {
              const isLoss = item.percentage_change < 0;
              
              return (
                <div 
                  key={item.id}
                  onClick={() => router.push(`/results?id=${item.id}`)}
                  className="glass-card p-4 sm:p-5 cursor-pointer hover:border-slate-600 hover:bg-slate-800/30 transition-all flex flex-col sm:flex-row sm:items-center gap-4 group"
                >
                  {/* Left Side: Dynamic Icon & Details */}
                  <div className="flex items-center gap-4 flex-grow">
                    {/* Status Icon Box */}
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 border ${
                      isLoss 
                        ? 'bg-red-500/10 border-red-500/20 text-red-400' 
                        : 'bg-green-500/10 border-green-500/20 text-green-400'
                    }`}>
                      {isLoss ? <TrendingDown className="w-6 h-6" /> : <TrendingUp className="w-6 h-6" />}
                    </div>
                    
                    <div>
                      <h3 className="text-slate-200 font-semibold flex items-center gap-2 text-base">
                        <Map className="w-4 h-4 text-blue-400" />
                        Analysis #{item.id.substring(0, 8)}
                      </h3>
                      <p className="text-slate-500 text-xs mt-1 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5" />
                        {new Date(item.timestamp).toLocaleDateString(undefined, { 
                          month: 'short', day: 'numeric', year: 'numeric' 
                        })} 
                        <span className="opacity-50">•</span> 
                        {new Date(item.timestamp).toLocaleTimeString(undefined, { 
                          hour: '2-digit', minute: '2-digit' 
                        })}
                      </p>
                    </div>
                  </div>
                  
                  {/* Right Side: Stats & Action */}
                  <div className="flex items-center justify-between sm:justify-end gap-6 w-full sm:w-auto border-t border-slate-800/50 sm:border-t-0 pt-3 sm:pt-0">
                    <div className="text-left sm:text-right">
                      <p className={`text-lg font-bold ${isLoss ? 'text-red-400' : 'text-green-400'}`}>
                        {item.percentage_change > 0 ? "+" : ""}{item.percentage_change.toFixed(2)}%
                      </p>
                      <p className="text-slate-500 text-xs mt-0.5 uppercase tracking-wider font-semibold">Net Change</p>
                    </div>
                    
                    <button
                      onClick={(e) => handleDelete(e, item.id)}
                      className="p-2.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors opacity-100 sm:opacity-0 sm:group-hover:opacity-100 focus:opacity-100"
                      title="Delete this analysis"
                    >
                      <Trash2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}