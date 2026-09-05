"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Incident {
  id: number;
  article_id: string;
  title: string;
  source: string;
  date: string;
  incident_type: string;
  event_type: string;
  location: string[];
  coordinates: { lat: number; lon: number };
  matched_name: string | null;
  location_type: string | null;
  geo_confidence: string;
  geo_method: string | null;
  relevance_score: number;
  keywords: string[];
  precision_tier: string;
  aoi_buffer_km_approx: number;
  date_before: string;
  date_after: string;
  url: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const CACHE_DURATION = 1000 * 60 * 60 * 6; // 6 hours

const KEYWORD_COLORS: Record<string, string> = {
  wildfire:          "bg-red-500/20 text-red-300 border-red-500/40",
  "forest fire":     "bg-red-500/20 text-red-300 border-red-500/40",
  deforestation:     "bg-orange-500/20 text-orange-300 border-orange-500/40",
  "forest loss":     "bg-orange-500/20 text-orange-300 border-orange-500/40",
  "tree cover loss": "bg-orange-500/20 text-orange-300 border-orange-500/40",
  encroachment:      "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
  "forest encroachment": "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
  "tree felling":    "bg-green-500/20 text-green-300 border-green-500/40",
  "tree cutting":    "bg-green-500/20 text-green-300 border-green-500/40",
  "illegal logging": "bg-green-500/20 text-green-300 border-green-500/40",
  "timber smuggling":"bg-green-500/20 text-green-300 border-green-500/40",
  "forest diversion":"bg-blue-500/20 text-blue-300 border-blue-500/40",
  "forest clearance":"bg-blue-500/20 text-blue-300 border-blue-500/40",
  mining:            "bg-blue-500/20 text-blue-300 border-blue-500/40",
  "coal block":      "bg-blue-500/20 text-blue-300 border-blue-500/40",
  afforestation:     "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  "national park":   "bg-teal-500/20 text-teal-300 border-teal-500/40",
  "tiger reserve":   "bg-teal-500/20 text-teal-300 border-teal-500/40",
  "wildlife sanctuary": "bg-teal-500/20 text-teal-300 border-teal-500/40",
};

const INCIDENT_TYPE_COLORS: Record<string, { bg: string; text: string; dot: string }> = {
  WILDFIRE:         { bg: "bg-red-500/10 border-red-500/50",    text: "text-red-400",    dot: "bg-red-400" },
  DEFORESTATION:    { bg: "bg-orange-500/10 border-orange-500/50", text: "text-orange-400", dot: "bg-orange-400" },
  TREE_FELLING:     { bg: "bg-green-500/10 border-green-500/50",  text: "text-green-400",  dot: "bg-green-400" },
  ENCROACHMENT:     { bg: "bg-yellow-500/10 border-yellow-500/50",text: "text-yellow-400", dot: "bg-yellow-400" },
  FOREST_DIVERSION: { bg: "bg-blue-500/10 border-blue-500/50",   text: "text-blue-400",   dot: "bg-blue-400" },
  GENERAL:          { bg: "bg-cyan-500/10 border-cyan-500/50",    text: "text-cyan-400",   dot: "bg-cyan-400" },
};

function getKeywordColor(kw: string): string {
  const key = kw.toLowerCase();
  return KEYWORD_COLORS[key] || "bg-slate-700/50 text-slate-300 border-slate-600/40";
}

function getIncidentStyle(type: string) {
  return INCIDENT_TYPE_COLORS[type] || INCIDENT_TYPE_COLORS.GENERAL;
}

// ─── Keyword Chip ─────────────────────────────────────────────────────────────

function KeywordChip({
  label,
  active,
  count,
  onClick,
}: {
  label: string;
  active: boolean;
  count?: number;
  onClick: () => void;
}) {
  const colorClass = getKeywordColor(label);
  return (
    <button
      id={`keyword-chip-${label.toLowerCase().replace(/\s+/g, "-")}`}
      onClick={onClick}
      className={`
        inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium
        border transition-all duration-200 cursor-pointer
        ${active
          ? "ring-2 ring-cyan-400 ring-offset-1 ring-offset-[#020817] scale-105 shadow-lg shadow-cyan-500/20"
          : "hover:scale-105 hover:brightness-110"
        }
        ${colorClass}
      `}
    >
      {label}
      {count !== undefined && (
        <span className="bg-white/10 rounded-full px-1.5 py-0.5 text-[10px] font-bold">
          {count}
        </span>
      )}
    </button>
  );
}

// ─── Article Card ─────────────────────────────────────────────────────────────

function ArticleCard({
  item,
  activeKeywords,
  onKeywordClick,
}: {
  item: Incident;
  activeKeywords: Set<string>;
  onKeywordClick: (kw: string) => void;
}) {
  const style = getIncidentStyle(item.event_type);

  return (
    <div
      className={`
        relative rounded-2xl border p-6 transition-all duration-300
        hover:shadow-lg hover:shadow-cyan-500/10 hover:-translate-y-0.5
        bg-gradient-to-br from-[#081225] to-[#060f1e]
        ${style.bg}
      `}
    >
      {/* Top row: type badge + date */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 mt-0.5 ${style.dot}`} />
          <span className={`text-xs font-bold uppercase tracking-wider ${style.text}`}>
            {item.event_type.replace(/_/g, " ")}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {item.precision_tier === "precise" && (
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-medium">
              📍 Precise
            </span>
          )}
          <span className="text-xs text-slate-500">{item.date}</span>
        </div>
      </div>

      {/* Title */}
      <h2 className="text-lg font-bold text-white leading-snug mb-3 line-clamp-3">
        {item.url ? (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-cyan-300 transition-colors"
          >
            {item.title}
          </a>
        ) : (
          item.title
        )}
      </h2>

      {/* Meta row */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-slate-400 mb-4">
        <span className="flex items-center gap-1">
          <span className="text-slate-600">📰</span> {item.source}
        </span>
        {item.matched_name && (
          <span className="flex items-center gap-1">
            <span className="text-slate-600">📍</span> {item.matched_name}
          </span>
        )}
        {item.relevance_score !== undefined && (
          <span className="flex items-center gap-1">
            <span className="text-slate-600">⚡</span>
            <span className="text-cyan-400 font-medium">Score {item.relevance_score.toFixed(1)}</span>
          </span>
        )}
      </div>

      {/* Keywords */}
      {item.keywords && item.keywords.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {item.keywords.map((kw) => (
            <button
              key={kw}
              onClick={() => onKeywordClick(kw)}
              className={`
                inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-medium
                border transition-all duration-150 cursor-pointer
                ${activeKeywords.has(kw)
                  ? "ring-2 ring-cyan-400 ring-offset-1 ring-offset-[#081225]"
                  : "hover:brightness-110"
                }
                ${getKeywordColor(kw)}
              `}
            >
              {kw}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function NewsPage() {
  const router = useRouter();

  const [activeTab, setActiveTab] = useState<"live" | "historic">("live");
  const [liveNews, setLiveNews] = useState<Incident[]>([]);
  const [historicNews, setHistoricNews] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  // Filter state
  const [searchQuery, setSearchQuery] = useState("");
  const [activeKeywords, setActiveKeywords] = useState<Set<string>>(new Set());
  const [sortOrder, setSortOrder] = useState<"date_desc" | "date_asc" | "relevance">("date_desc");

  // ── Fetch live news ──────────────────────────────────────────────────────────
  useEffect(() => {
    const cached = localStorage.getItem("liveNewsV2");
    if (cached) {
      const parsed = JSON.parse(cached);
      if (Date.now() - parsed.timestamp < CACHE_DURATION) {
        setLiveNews(parsed.data);
        return;
      }
    }
    setLoading(true);
    fetch("http://127.0.0.1:8000/news?limit=100")
      .then((res) => res.json())
      .then((response) => {
        const newsData: Incident[] = response.data || response;
        setLiveNews(newsData);
        localStorage.setItem("liveNewsV2", JSON.stringify({ data: newsData, timestamp: Date.now() }));
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  // ── Fetch historic news ──────────────────────────────────────────────────────
  useEffect(() => {
    const cached = localStorage.getItem("historicNews");
    if (cached) {
      setHistoricNews(JSON.parse(cached));
      return;
    }
    fetch("http://127.0.0.1:8000/historic-news")
      .then((res) => res.json())
      .then((data) => {
        setHistoricNews(data);
        localStorage.setItem("historicNews", JSON.stringify(data));
      })
      .catch(console.error);
  }, []);

  // ── Derived data ─────────────────────────────────────────────────────────────

  const rawNews = activeTab === "live" ? liveNews : historicNews;

  // All unique keywords across ALL articles (for the chip panel)
  const allKeywords = useMemo(() => {
    const kwCount = new Map<string, number>();
    rawNews.forEach((item: any) => {
      (item.keywords || []).forEach((kw: string) => {
        kwCount.set(kw, (kwCount.get(kw) || 0) + 1);
      });
    });
    return Array.from(kwCount.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([kw, count]) => ({ kw, count }));
  }, [rawNews]);

  // Filtered + sorted articles
  const filteredNews = useMemo(() => {
    let items: any[] = [...rawNews];

    // Keyword filter (AND logic — article must have ALL active keywords)
    if (activeKeywords.size > 0) {
      items = items.filter((item) => {
        const itemKws = new Set<string>((item.keywords || []).map((k: string) => k.toLowerCase()));
        return Array.from(activeKeywords).every((kw) => itemKws.has(kw.toLowerCase()));
      });
    }

    // Free-text search
    if (searchQuery.trim()) {
      const q = searchQuery.trim().toLowerCase();
      items = items.filter(
        (item) =>
          (item.title || "").toLowerCase().includes(q) ||
          (item.source || "").toLowerCase().includes(q) ||
          (item.matched_name || "").toLowerCase().includes(q) ||
          (item.keywords || []).some((kw: string) => kw.toLowerCase().includes(q))
      );
    }

    // Sort
    if (sortOrder === "date_asc") {
      items.sort((a, b) => (a.date || "").localeCompare(b.date || ""));
    } else if (sortOrder === "relevance") {
      items.sort((a, b) => (b.relevance_score || 0) - (a.relevance_score || 0));
    } else {
      items.sort((a, b) => (b.date || "").localeCompare(a.date || ""));
    }

    return items;
  }, [rawNews, activeKeywords, searchQuery, sortOrder]);

  const toggleKeyword = useCallback((kw: string) => {
    setActiveKeywords((prev) => {
      const next = new Set(prev);
      if (next.has(kw)) next.delete(kw);
      else next.add(kw);
      return next;
    });
  }, []);

  const clearFilters = useCallback(() => {
    setActiveKeywords(new Set());
    setSearchQuery("");
  }, []);

  const hasFilters = activeKeywords.size > 0 || searchQuery.trim().length > 0;

  // ── Render ───────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#020817] text-white">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="border-b border-slate-800 bg-[#020817]/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button
            id="back-button"
            onClick={() => router.back()}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-[#081225] border border-cyan-500/40 text-cyan-400 hover:bg-cyan-500 hover:text-black transition-all text-sm font-semibold"
          >
            ← Back
          </button>

          <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-teal-300 bg-clip-text text-transparent">
            🌿 Forest Incident Monitor
          </h1>

          {/* Tabs */}
          <div className="flex gap-2">
            {(["live", "historic"] as const).map((tab) => (
              <button
                key={tab}
                id={`tab-${tab}`}
                onClick={() => { setActiveTab(tab); setActiveKeywords(new Set()); setSearchQuery(""); }}
                className={`px-5 py-2 rounded-xl text-sm font-semibold transition-all ${
                  activeTab === tab
                    ? "bg-cyan-500 text-black shadow-lg shadow-cyan-500/30"
                    : "bg-[#081225] border border-cyan-500/30 text-slate-300 hover:border-cyan-400"
                }`}
              >
                {tab === "live" ? "Live News" : "Historic Analysis"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* ── Search + Sort bar ──────────────────────────────────────────────── */}
        <div className="flex flex-col sm:flex-row gap-3 mb-6">
          {/* Search */}
          <div className="relative flex-1">
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 text-base select-none">🔍</span>
            <input
              id="news-search-input"
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by keyword, title, location, or source…"
              className="w-full pl-10 pr-4 py-3 rounded-xl bg-[#081225] border border-slate-700 text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 transition-all text-sm"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white transition-colors"
              >
                ✕
              </button>
            )}
          </div>

          {/* Sort */}
          <select
            id="news-sort-select"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as typeof sortOrder)}
            className="px-4 py-3 rounded-xl bg-[#081225] border border-slate-700 text-white text-sm focus:outline-none focus:border-cyan-500 transition-all cursor-pointer"
          >
            <option value="date_desc">📅 Newest First</option>
            <option value="date_asc">📅 Oldest First</option>
            <option value="relevance">⚡ Relevance Score</option>
          </select>
        </div>

        {/* ── Keyword chip panel ────────────────────────────────────────────── */}
        {activeTab === "live" && allKeywords.length > 0 && (
          <div className="bg-[#081225]/60 border border-slate-800 rounded-2xl p-5 mb-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">
                Filter by Keyword
              </h3>
              {hasFilters && (
                <button
                  id="clear-filters-btn"
                  onClick={clearFilters}
                  className="text-xs text-cyan-400 hover:text-cyan-300 font-medium transition-colors"
                >
                  Clear all ✕
                </button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {allKeywords.map(({ kw, count }) => (
                <KeywordChip
                  key={kw}
                  label={kw}
                  active={activeKeywords.has(kw)}
                  count={count}
                  onClick={() => toggleKeyword(kw)}
                />
              ))}
            </div>
          </div>
        )}

        {/* ── Results summary ───────────────────────────────────────────────── */}
        <div className="flex items-center justify-between mb-5">
          <p className="text-sm text-slate-400">
            {loading ? (
              <span className="animate-pulse">Loading articles…</span>
            ) : (
              <>
                Showing{" "}
                <span className="text-white font-semibold">{filteredNews.length}</span>
                {" "}of{" "}
                <span className="text-slate-300">{rawNews.length}</span>
                {" "}articles
                {hasFilters && (
                  <span className="ml-2 text-cyan-400">
                    ({activeKeywords.size > 0 ? `${activeKeywords.size} keyword${activeKeywords.size > 1 ? "s" : ""}` : ""}{activeKeywords.size > 0 && searchQuery ? " + " : ""}{searchQuery ? `"${searchQuery}"` : ""})
                  </span>
                )}
              </>
            )}
          </p>

          {/* Active keyword pills (compact) */}
          {activeKeywords.size > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {Array.from(activeKeywords).map((kw) => (
                <span
                  key={kw}
                  className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-medium border ${getKeywordColor(kw)}`}
                >
                  {kw}
                  <button
                    onClick={() => toggleKeyword(kw)}
                    className="ml-0.5 hover:opacity-70 transition-opacity"
                  >
                    ✕
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* ── Loading state ─────────────────────────────────────────────────── */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-4">
            <div className="w-12 h-12 rounded-full border-4 border-slate-700 border-t-cyan-400 animate-spin" />
            <p className="text-slate-400 text-sm">Fetching latest forest incidents…</p>
          </div>
        )}

        {/* ── Empty state ───────────────────────────────────────────────────── */}
        {!loading && filteredNews.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 gap-3 text-center">
            <span className="text-5xl">🌲</span>
            <p className="text-lg font-semibold text-slate-300">No articles match your filters</p>
            <p className="text-slate-500 text-sm">Try adjusting the keywords or clearing the search.</p>
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="mt-2 px-5 py-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/20 transition-all text-sm font-medium"
              >
                Clear all filters
              </button>
            )}
          </div>
        )}

        {/* ── Article grid ──────────────────────────────────────────────────── */}
        {!loading && filteredNews.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            {filteredNews.map((item: any, idx) =>
              activeTab === "live" ? (
                <ArticleCard
                  key={item.article_id || item.id || idx}
                  item={item}
                  activeKeywords={activeKeywords}
                  onKeywordClick={toggleKeyword}
                />
              ) : (
                /* Historic card (unchanged layout) */
                <div
                  key={item.id || idx}
                  className="bg-[#081225] border border-cyan-500/30 rounded-2xl p-6"
                >
                  <div className="flex justify-between items-start mb-4">
                    <h2 className="text-xl font-bold leading-snug">{item.title}</h2>
                    <span className="flex-shrink-0 ml-3 bg-green-500 text-black px-3 py-1 rounded-full text-xs font-bold">
                      {item.incident_type}
                    </span>
                  </div>
                  <p className="text-slate-400 text-sm mb-1">Source: {item.source}</p>
                  <p className="text-slate-400 text-sm mb-1">Date: {item.date}</p>
                  <p className="text-slate-400 text-sm mb-3">
                    Location: {item.location?.join(", ")}
                  </p>
                  {item.images && (
                    <div className="grid grid-cols-2 gap-3 mb-3">
                      {["before", "after"].map((k) => (
                        <div key={k}>
                          <h3 className="text-center text-sm font-semibold mb-1 capitalize">{k}</h3>
                          <img
                            src={`http://127.0.0.1:8000${item.images[`${k}_rgb`]}`}
                            className="rounded-xl border border-slate-700 w-full h-48 object-cover"
                            alt={k}
                            onError={() => {}}
                          />
                        </div>
                      ))}
                    </div>
                  )}
                  {item.forest_loss_percent && (
                    <p className="text-red-400 font-bold text-sm mb-2">
                      Forest Loss: {item.forest_loss_percent}%
                    </p>
                  )}
                  {item.reason && <p className="text-slate-300 text-sm mt-2">{item.reason}</p>}
                </div>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
