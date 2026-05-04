import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: { ignoreBuildErrors: true },
  async rewrites() {
    return [
      {
        source: "/api/langgraph/:path*",
        destination: `${process.env.LANGGRAPH_INTERNAL_URL ?? "http://localhost:8123"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
