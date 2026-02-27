import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow image optimization for backend static assets
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/static/**",
      },
    ],
  },
};

export default nextConfig;
