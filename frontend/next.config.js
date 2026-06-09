/** @type {import('next').NextConfig} */
const nextConfig = {
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

module.exports = nextConfig;
