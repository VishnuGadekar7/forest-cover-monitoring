import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Forest Cover Monitor | AI-Powered Earth Observation",
  description:
    "Research-grade forest cover change detection using deep learning semantic segmentation. Built for ISRO faculty presentation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} font-sans bg-[#0a0f1a] text-slate-100 antialiased`}>
        {children}
      </body>
    </html>
  );
}
