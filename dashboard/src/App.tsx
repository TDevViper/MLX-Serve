import { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";

const BASE = "http://localhost:8000";
const MAX_POINTS = 30;

interface Point {
  t: string;
  tps: number;
  latency: number;
  queue: number;
  memory: number;
}

function usePoll(url: string, interval = 2000) {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    const tick = async () => {
      try { setData(await (await fetch(url)).json()); } catch {}
    };
    tick();
    const id = setInterval(tick, interval);
    return () => clearInterval(id);
  }, [url, interval]);
  return data;
}

export default function App() {
  const health  = usePoll(`${BASE}/health`, 2000);
  const kv      = usePoll(`${BASE}/v1/kv_cache`, 3000);
  const batcher = usePoll(`${BASE}/v1/batcher`, 3000);
  const prefix  = usePoll(`${BASE}/v1/prefix_cache`, 3000);
  const [series, setSeries] = useState<Point[]>([]);

  useEffect(() => {
    if (!health) return;
    const m = health.metrics ?? {};
    const now = new Date().toLocaleTimeString();
    setSeries(prev => [...prev.slice(-MAX_POINTS + 1), {
      t: now,
      tps:     parseFloat((m.tokens_per_second ?? 0).toFixed(1)),
      latency: parseFloat(((m.avg_latency_ms ?? 0) / 1000).toFixed(3)),
      queue:   health.queue?.waiting ?? 0,
      memory:  parseFloat(((kv?.used_blocks ?? 0) / Math.max(kv?.total_blocks ?? 1, 1) * 100).toFixed(1)),
    }]);
  }, [health, kv]);

  const card = (title: string, value: string, sub?: string) => (
    <div style={{ background: "var(--color-background-secondary)", border: "1px solid var(--color-border-tertiary)", borderRadius: 12, padding: "16px 20px", minWidth: 140 }}>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 26, fontWeight: 500, color: "var(--color-text-primary)", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 4 }}>{sub}</div>}
    </div>
  );

  const m = health?.metrics ?? {};
  const q = health?.queue ?? {};
  const CHART_H = 180;
  const grid = "var(--color-border-tertiary)";
  const muted = "var(--color-text-secondary)";
  const panelStyle = { background: "var(--color-background-secondary)", border: "1px solid var(--color-border-tertiary)", borderRadius: 12, padding: 16 };
  const labelStyle = { fontSize: 12, color: muted, marginBottom: 12 };
  const tooltipStyle = { contentStyle: { background: "var(--color-background-primary)", border: "1px solid var(--color-border-secondary)", fontSize: 12 } };

  const online = health?.status === "ok";

  return (
    <div style={{ fontFamily: "system-ui, sans-serif", minHeight: "100vh", background: "#0d1117", color: "#e6edf3", padding: "24px 32px" }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 500, margin: 0 }}>MLX-Serve</h1>
        <span style={{
          fontSize: 12, padding: "2px 10px", borderRadius: 99,
          color: online ? "#3fb950" : "#f85149",
          background: online ? "#0f2d1a" : "#2d1414",
        }}>
          {online ? "● live" : "○ offline"}
        </span>
        <span style={{ fontSize: 12, color: "#8b949e", marginLeft: "auto" }}>{health?.model ?? "—"}</span>
      </div>

      {/* Stat cards */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 24 }}>
        {card("Tokens / sec",  `${(m.tokens_per_second ?? 0).toFixed(1)}`, "throughput")}
        {card("Avg latency",   `${(m.avg_latency_ms ?? 0).toFixed(0)}ms`,  "per request")}
        {card("Total requests",`${m.total_requests ?? 0}`,                  "served")}
        {card("Queue depth",   `${q.waiting ?? 0}`,                         "waiting")}
        {card("KV blocks",      kv     ? `${kv.used_blocks}/${kv.total_blocks}`            : "—", "used / total")}
        {card("Prefix hit rate",prefix ? `${(prefix.hit_rate * 100).toFixed(1)}%`          : "—", "cache")}
      </div>

      {/* Charts grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>

        <div style={panelStyle}>
          <div style={labelStyle}>Tokens / second</div>
          <ResponsiveContainer width="100%" height={CHART_H}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: muted }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: muted }} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="tps" stroke="#238636" strokeWidth={2} dot={false} name="tok/s" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={panelStyle}>
          <div style={labelStyle}>Latency (seconds)</div>
          <ResponsiveContainer width="100%" height={CHART_H}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: muted }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: muted }} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="latency" stroke="#1f6feb" strokeWidth={2} dot={false} name="latency" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={panelStyle}>
          <div style={labelStyle}>Queue depth over time</div>
          <ResponsiveContainer width="100%" height={CHART_H}>
            <BarChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: muted }} interval="preserveStartEnd" />
              <YAxis tick={{ fontSize: 10, fill: muted }} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="queue" fill="#8957e5" name="waiting" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div style={panelStyle}>
          <div style={labelStyle}>KV cache utilisation %</div>
          <ResponsiveContainer width="100%" height={CHART_H}>
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke={grid} />
              <XAxis dataKey="t" tick={{ fontSize: 10, fill: muted }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: muted }} />
              <Tooltip {...tooltipStyle} />
              <Line type="monotone" dataKey="memory" stroke="#e3b341" strokeWidth={2} dot={false} name="kv %" />
            </LineChart>
          </ResponsiveContainer>
        </div>

      </div>

      {/* Detail panels */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
        {[
          ["Continuous batcher", batcher],
          ["KV cache blocks",    kv],
          ["Prefix cache",       prefix],
        ].map(([title, obj]) => (
          <div key={title as string} style={panelStyle}>
            <div style={labelStyle}>{title as string}</div>
            {obj
              ? Object.entries(obj as object).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, padding: "4px 0", borderBottom: "1px solid #21262d" }}>
                    <span style={{ color: muted }}>{k}</span>
                    <span style={{ fontFamily: "monospace" }}>{String(v)}</span>
                  </div>
                ))
              : <div style={{ fontSize: 12, color: muted }}>waiting for data...</div>
            }
          </div>
        ))}
      </div>

    </div>
  );
}