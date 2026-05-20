"use client";

import { useEffect, useState } from "react";

export default function NewsPage() {

  const [activeTab, setActiveTab] = useState("live");

  const [liveNews, setLiveNews] = useState<any[]>([]);
  const [historicNews, setHistoricNews] = useState<any[]>([]);

  // =====================================================
  // FETCH LIVE NEWS
  // =====================================================

  useEffect(() => {

    fetch("http://127.0.0.1:8000/news")
      .then((res) => res.json())
      .then((data) => setLiveNews(data))
      .catch((err) => console.log(err));

  }, []);

  // =====================================================
  // FETCH HISTORIC NEWS
  // =====================================================

  useEffect(() => {

    fetch("http://127.0.0.1:8000/historic-news")
      .then((res) => res.json())
      .then((data) => setHistoricNews(data))
      .catch((err) => console.log(err));

  }, []);

  // =====================================================
  // SELECT DATA BASED ON TAB
  // =====================================================

  const currentNews =
    activeTab === "live"
      ? liveNews
      : historicNews;

  return (

    <div className="min-h-screen bg-[#020817] text-white p-8">

      {/* ================================================= */}
      {/* TITLE */}
      {/* ================================================= */}

      <h1 className="text-5xl font-bold text-center mb-10">

        Forest Incident Monitoring

      </h1>

      {/* ================================================= */}
      {/* TABS */}
      {/* ================================================= */}

      <div className="flex gap-4 justify-center mb-10">

        <button

          onClick={() => setActiveTab("live")}

          className={`px-6 py-2 rounded-xl font-semibold transition-all

          ${activeTab === "live"

            ? "bg-cyan-500 text-black"

            : "bg-[#081225] border border-cyan-500 text-white"
          }`}
        >

          Live News

        </button>

        <button

          onClick={() => setActiveTab("historic")}

          className={`px-6 py-2 rounded-xl font-semibold transition-all

          ${activeTab === "historic"

            ? "bg-cyan-500 text-black"

            : "bg-[#081225] border border-cyan-500 text-white"
          }`}
        >

          Historic Analysis

        </button>

      </div>

      {/* ================================================= */}
      {/* NEWS GRID */}
      {/* ================================================= */}

      <div className="grid grid-cols-1 gap-8">

        {currentNews.slice(0, 10).map((item: any) => (

          <div

            key={item.id}

            className="
              bg-[#081225]
              border border-cyan-500
              rounded-2xl
              p-6
            "
          >

            {/* =========================================== */}
            {/* HEADER */}
            {/* =========================================== */}

            <div className="flex justify-between items-center mb-4">

              <h2 className="text-2xl font-bold">

                {item.title}

              </h2>

              <span className="
                bg-green-500
                text-black
                px-3 py-1
                rounded-full
                text-sm
              ">

                {item.incident_type}

              </span>

            </div>

            {/* =========================================== */}
            {/* DETAILS */}
            {/* =========================================== */}

            <p className="text-gray-400 mb-2">

              Source: {item.source}

            </p>

            <p className="text-gray-400 mb-2">

              Date: {item.date}

            </p>

            <p className="text-gray-400 mb-2">

              Location:

              {" "}

              {item.location?.join(", ")}

            </p>

            <p className="text-gray-400 mb-4">

              Coordinates:

              {" "}

              {item.coordinates?.lat},

              {" "}

              {item.coordinates?.lon}

            </p>

            {/* =========================================== */}
            {/* HISTORIC IMAGES */}
            {/* =========================================== */}

            {activeTab === "historic" && item.images && (

              <div className="grid grid-cols-2 gap-4 mb-4">

                {/* BEFORE IMAGE */}

                <div>

                  <h3 className="text-center mb-2 font-semibold">

                    Before

                  </h3>

                  <img

                    src={`http://127.0.0.1:8000${item.images.before_rgb}`}

                    className="
                        rounded-xl
                        border
                        border-gray-700
                        w-full
                        h-[300px]
                        object-cover
                    "

                    alt="Before"

                    onError={(e) => {
                        console.log("Before image failed");
                    }}
                />

                </div>

                {/* AFTER IMAGE */}

                <div>

                  <h3 className="text-center mb-2 font-semibold">

                    After

                  </h3>

                  <img

                    src={`http://127.0.0.1:8000${item.images.after_rgb}`}

                    className="
                        rounded-xl
                        border
                        border-gray-700
                        w-full
                        h-[300px]
                        object-cover
                    "

                    alt="After"

                    onError={(e) => {
                        console.log("After image failed");
                    }}
                />

                </div>

              </div>
            )}

            {/* =========================================== */}
            {/* FOREST LOSS */}
            {/* =========================================== */}

            {activeTab === "historic" &&
              item.forest_loss_percent && (

              <p className="text-red-400 font-bold mb-4">

                Forest Loss:

                {" "}

                {item.forest_loss_percent}%

              </p>
            )}

            {/* =========================================== */}
            {/* SUMMARY */}
            {/* =========================================== */}

            <p className="text-gray-300 mt-3">

              {item.reason}

            </p>

          </div>
        ))}
      </div>
    </div>
  );
}