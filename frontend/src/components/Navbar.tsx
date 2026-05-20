"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  Leaf,
  BarChart2,
  Upload,
  History,
  Newspaper,
} from "lucide-react";

const navItems = [
  {
    href: "/upload",
    label: "Upload",
    icon: Upload,
  },
  {
    href: "/results",
    label: "Results",
    icon: BarChart2,
  },
  {
    href: "/news",
    label: "News",
    icon: Newspaper,
  },
];

export default function Navbar() {

  const path = usePathname();

  // Hide navbar on landing page
  if (path === "/") return null;

  return (

    <nav
      className="
        fixed
        top-0
        left-0
        right-0
        z-50
        glass-card
        border-x-0
        border-t-0
        px-6
        py-3
        flex
        items-center
        justify-between
      "
    >

      {/* ===================================== */}
      {/* LEFT : LOGO */}
      {/* ===================================== */}

      <Link
        href="/"
        className="flex items-center gap-2.5 group shrink-0"
      >

        <div
          className="
            w-8
            h-8
            rounded-lg
            bg-gradient-to-br
            from-green-500
            to-blue-600
            flex
            items-center
            justify-center
            shadow-lg
            group-hover:scale-110
            transition-transform
          "
        >

          <Leaf className="w-4 h-4 text-white" />

        </div>

        <div className="flex flex-col leading-none">

          <span className="text-sm font-bold text-slate-100">

            ForestMonitor

          </span>

          <span
            className="
              text-[10px]
              text-slate-500
              tracking-widest
              uppercase
            "
          >

            Deep Learning · EO

          </span>

        </div>

      </Link>

      {/* ===================================== */}
      {/* CENTER : NAVIGATION */}
      {/* ===================================== */}

      <div className="flex-1 flex justify-center">

        <div
          className="
            hidden
            sm:flex
            items-center
            gap-2
            bg-slate-900/60
            rounded-xl
            p-1
          "
        >

          {navItems.map(({ href, label, icon: Icon }) => {

            const active = path.startsWith(href);

            return (

              <Link
                key={href}
                href={href}
                className={`
                  flex
                  items-center
                  gap-2
                  px-4
                  py-2
                  rounded-lg
                  text-sm
                  font-medium
                  whitespace-nowrap
                  transition-all
                  duration-200

                  ${
                    active
                      ? "bg-gradient-to-r from-green-600/30 to-blue-600/20 text-green-400 shadow-inner"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                  }
                `}
              >

                <Icon className="w-4 h-4" />

                {label}

              </Link>
            );
          })}

        </div>

      </div>

      {/* ===================================== */}
      {/* RIGHT : HISTORY */}
      {/* ===================================== */}

      <div className="flex items-center gap-4 shrink-0">

        <Link
          href="/history"
          className="
            flex
            items-center
            gap-2
            px-3
            py-2
            text-sm
            text-slate-400
            hover:text-slate-200
            transition-colors
          "
        >

          <History className="w-4 h-4" />

          <span className="hidden md:inline">

            Recent Analyses

          </span>

        </Link>

        {/* STATUS */}

        <div
          className="
            hidden
            lg:flex
            items-center
            gap-2
            border-l
            border-slate-800
            pl-4
          "
        >

          <span
            className="
              w-2
              h-2
              rounded-full
              bg-green-500
            "
          />

          <span className="text-xs text-slate-500">

            Engine Ready

          </span>

        </div>

      </div>

    </nav>
  );
}
