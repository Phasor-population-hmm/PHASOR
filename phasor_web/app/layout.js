import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import TopNav from "@/components/TopNav";
import JobDock from "@/components/JobDock";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata = {
  title: "PHASOR console",
  description: "Population HMM analysis of stimulus-locked output in the retina",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen bg-neutral-950 font-sans text-[15px] text-neutral-200 antialiased">
        <TopNav />
        <main className="mx-auto max-w-[1400px] px-8 py-12 pb-28">{children}</main>
        <JobDock />
      </body>
    </html>
  );
}
