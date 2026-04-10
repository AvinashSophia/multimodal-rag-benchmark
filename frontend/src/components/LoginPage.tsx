import { useState } from "react";
import type { User } from "../types";

interface Props {
  onLogin: (user: User) => void;
}

export default function LoginPage({ onLogin }: Props) {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) { setError("Please enter your name."); return; }
    if (!email.trim() || !email.includes("@")) { setError("Please enter a valid email."); return; }
    onLogin({ name: name.trim(), email: email.trim() });
  }

  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4"
      style={{ background: "#09090F" }}
    >
      {/* Logo */}
      <div className="mb-8 flex flex-col items-center gap-4">
        <img
          src="https://sophiaspatialai.com/wp-content/uploads/logo.webp"
          alt="SophiaSpatial AI"
          className="h-14 w-auto object-contain"
        />
        <div className="text-center">
          <p className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.6)" }}>
            Wisdom You Can See™
          </p>
        </div>
      </div>

      {/* Card */}
      <div
        className="w-full max-w-sm overflow-hidden rounded-2xl"
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(237,203,105,0.15)",
          backdropFilter: "blur(10px)",
        }}
      >
        <div
          className="h-px w-full"
          style={{ background: "linear-gradient(90deg, transparent, #EDCB69, transparent)" }}
        />
        <div className="p-8">
          <h1 className="mb-1 text-xl font-bold text-white">Welcome</h1>
          <p className="mb-6 text-sm" style={{ color: "rgba(255,255,255,0.4)" }}>
            Sign in to access DocuMind · FLVM Intelligence
          </p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.6)" }}>
                Full Name
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => { setName(e.target.value); setError(""); }}
                placeholder="Jane Smith"
                className="rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(237,203,105,0.2)",
                  color: "rgba(255,255,255,0.9)",
                }}
                autoFocus
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.6)" }}>
                Email Address
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => { setEmail(e.target.value); setError(""); }}
                placeholder="jane@company.com"
                className="rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(237,203,105,0.2)",
                  color: "rgba(255,255,255,0.9)",
                }}
              />
            </div>

            {error && (
              <p className="text-xs" style={{ color: "#FCA5A5" }}>{error}</p>
            )}

            <button
              type="submit"
              className="mt-2 w-full rounded-xl py-2.5 text-sm font-semibold transition"
              style={{ background: "linear-gradient(135deg, #EDCB69, #C9A83C)", color: "#09090F" }}
            >
              Continue →
            </button>
          </form>

          <p className="mt-5 text-center text-[11px]" style={{ color: "rgba(255,255,255,0.2)" }}>
            Internal tool · SophiaSpatial AI · Private & On-Premises
          </p>
        </div>
      </div>
    </div>
  );
}
