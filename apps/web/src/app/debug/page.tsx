"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type ModelProfileDebug = {
  provider?: string;
  model_name?: string;
  context_window?: number;
  max_output_tokens?: number;
  compression_strategy?: string;
  tokenizer_hint?: string;
};

type RuntimeDebug = {
  worker?: { status?: string; service?: string };
  model?: { provider?: string | null; connection_mode?: string | null; name?: string | null; base_url?: string | null; profile?: ModelProfileDebug };
  permissions?: {
    write_mode?: string;
    mode?: string;
    sandbox?: Record<string, string | boolean>;
    approval?: Record<string, boolean>;
    tools?: Record<string, string>;
  };
  repository?: {
    source?: string | null;
    project_path?: string | null;
    branch?: string | null;
    configured_default_source?: string | null;
    configured_sources?: Array<{ provider?: string | null; baseUrl?: string | null; tokenConfigured?: boolean; owner?: string }>;
    effective_sources?: Array<{ provider?: string | null; baseUrl?: string | null; tokenConfigured?: boolean; owner?: string }>;
  };
  agent?: { orchestration_mode?: string };
  active_runs?: Array<Record<string, unknown>>;
  recent_runs?: Array<Record<string, unknown>>;
};

type GraphNode = { id: string; label: string; type: string };
type GraphEdge = { source: string; target: string; label: string };

type MemoryDebug = {
  items: Array<{ id: string; type: string; content: string; source: string; repo?: string | null; created_at: string; tags?: string[] }>;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
};

type SkillsDebug = {
  skills: Array<{ name: string; source_path: string; scope: string; description: string; enabled: boolean }>;
};

type IntegrationsDebug = {
  integrations: Array<{
    provider: string;
    status: string;
    configured?: boolean;
    accounts?: Array<{ id?: string; status?: string; toolkit?: string; user_id?: string }>;
    toolkits?: Array<{ id?: string; slug?: string; name?: string; tools_count?: number }>;
    toolkits_count?: number;
    tools_count: number;
    message?: string;
  }>;
};

type ToolsDebug = {
  tools: Array<{ name: string; installed: boolean; path?: string | null; version?: string | null; auth_status?: string | null }>;
  permissions: Record<string, string>;
};

type ContextPackDebug = {
  run_id?: string;
  timestamp?: string;
  pack_kind?: string;
  trigger_node?: string;
  compression_ratio?: number;
  estimated_input_tokens?: number;
  estimated_output_reserve?: number;
  included_sections?: string[];
  excluded_sections?: string[];
  retained_files?: string[];
  omitted_files?: Array<{ path?: string; reason?: string; chars?: number; included_chars?: number }>;
  retained_web_sources?: Array<{ title?: string | null; url?: string | null; source?: string | null; fetched_at?: string | null }>;
  warnings?: string[];
};

type ContextDebug = {
  model_profile?: ModelProfileDebug;
  latest_pack?: ContextPackDebug | null;
  recent_packs?: ContextPackDebug[];
};

const tabs = ["Dashboard", "Agents", "Context", "Memory", "Skills", "Integrations", "Tools", "Events / Runs", "Settings"] as const;
type DebugTab = typeof tabs[number];

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return (await response.json()) as T;
}

export default function DebugPage() {
  const [activeTab, setActiveTab] = useState<DebugTab>("Dashboard");
  const [runtime, setRuntime] = useState<RuntimeDebug | null>(null);
  const [context, setContext] = useState<ContextDebug | null>(null);
  const [memory, setMemory] = useState<MemoryDebug | null>(null);
  const [skills, setSkills] = useState<SkillsDebug | null>(null);
  const [integrations, setIntegrations] = useState<IntegrationsDebug | null>(null);
  const [tools, setTools] = useState<ToolsDebug | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function load() {
      try {
        setError(null);
        const [runtimePayload, contextPayload, memoryPayload, skillsPayload, integrationsPayload, toolsPayload] = await Promise.all([
          loadJson<RuntimeDebug>("/api/worker/debug/runtime"),
          loadJson<ContextDebug>("/api/worker/debug/context"),
          loadJson<MemoryDebug>("/api/worker/debug/memory"),
          loadJson<SkillsDebug>("/api/worker/debug/skills"),
          loadJson<IntegrationsDebug>("/api/worker/debug/integrations"),
          loadJson<ToolsDebug>("/api/worker/tools"),
        ]);
        setRuntime(runtimePayload);
        setContext(contextPayload);
        setMemory(memoryPayload);
        setSkills(skillsPayload);
        setIntegrations(integrationsPayload);
        setTools(toolsPayload);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load debug data.");
      }
  }

  useEffect(() => {
    void load();
  }, []);

  return (
    <div className="debug-shell">
      <aside className="debug-sidebar">
        <div className="debug-brand">RepoOperator Debug</div>
        {tabs.map((tab) => (
          <button
            key={tab}
            className={`debug-tab${activeTab === tab ? " debug-tab-active" : ""}`}
            type="button"
            onClick={() => setActiveTab(tab)}
          >
            {tab}
          </button>
        ))}
        <Link className="debug-back-link" href="/app">Back to app</Link>
      </aside>
      <main className="debug-main">
        <header className="debug-header">
          <h1>{activeTab}</h1>
          <div className="debug-header-actions">
            <button className="debug-secondary-button" type="button" onClick={() => void load()}>Reload</button>
            <span className={`debug-status${runtime?.worker?.status === "ok" ? " debug-status-ok" : ""}`}>
              {runtime?.worker?.status === "ok" ? "Worker online" : "Worker unavailable"}
            </span>
          </div>
        </header>
        {error && <div className="debug-error">{error}</div>}
        {activeTab === "Dashboard" && <Dashboard runtime={runtime} />}
        {activeTab === "Agents" && <Agents runtime={runtime} />}
        {activeTab === "Context" && <ContextPanel context={context} />}
        {activeTab === "Memory" && <MemoryPanel memory={memory} />}
        {activeTab === "Skills" && <SkillsPanel skills={skills} />}
        {activeTab === "Integrations" && <IntegrationsPanel integrations={integrations} />}
        {activeTab === "Tools" && <ToolsPanel tools={tools} />}
        {activeTab === "Events / Runs" && <RunsPanel runtime={runtime} />}
        {activeTab === "Settings" && <SettingsPanel runtime={runtime} />}
      </main>
    </div>
  );
}

function Dashboard({ runtime }: { runtime: RuntimeDebug | null }) {
  return (
    <div className="debug-grid">
      <Card title="Worker">
        <Row label="Status" value={runtime?.worker?.status ?? "-"} />
        <Row label="Service" value={runtime?.worker?.service ?? "-"} />
        <Row label="Active runs" value={String(runtime?.active_runs?.length ?? 0)} />
      </Card>
      <Card title="Model">
        <Row label="Provider" value={runtime?.model?.provider ?? "-"} />
        <Row label="Model" value={runtime?.model?.name ?? "-"} />
        <Row label="Base URL" value={runtime?.model?.base_url ?? "-"} />
      </Card>
      <Card title="Repository">
        <Row label="Source" value={runtime?.repository?.source ?? "-"} />
        <Row label="Project" value={runtime?.repository?.project_path ?? "-"} />
        <Row label="Branch" value={runtime?.repository?.branch ?? "-"} />
        <Row label="Default source" value={runtime?.repository?.configured_default_source ?? "-"} />
        <Row
          label="Configured sources"
          value={
            runtime?.repository?.configured_sources?.length
              ? runtime.repository.configured_sources.map(formatSource).join(", ")
              : "-"
          }
        />
        <Row
          label="Effective sources"
          value={
            runtime?.repository?.effective_sources?.length
              ? runtime.repository.effective_sources.map(formatSource).join(", ")
              : "-"
          }
        />
      </Card>
      <Card title="Permissions">
        <Row label="Mode" value={runtime?.permissions?.mode ?? "-"} />
        <Row label="Sandbox scope" value={String(runtime?.permissions?.sandbox?.scope ?? "-")} />
        <Row label="Approval policy" value={runtime?.permissions?.approval ? "Elevated actions require review" : "-"} />
      </Card>
    </div>
  );
}

function formatSource(source: { provider?: string | null; baseUrl?: string | null; tokenConfigured?: boolean; owner?: string }): string {
  const bits = [source.provider || "unknown"];
  if (source.baseUrl) bits.push(source.baseUrl);
  if (source.owner) bits.push(`owner: ${source.owner}`);
  bits.push(source.tokenConfigured ? "token configured" : "no token");
  return bits.join(" · ");
}

function Agents({ runtime }: { runtime: RuntimeDebug | null }) {
  return (
    <Card title="Agent Orchestration">
      <Row label="Mode" value={runtime?.agent?.orchestration_mode ?? "LangGraph"} />
      <Row label="Write router" value="LangGraph intent and proposal flow" />
    </Card>
  );
}

function ContextPanel({ context }: { context: ContextDebug | null }) {
  const latest = context?.latest_pack;
  return (
    <>
      <Card title="Model Profile">
        <Row label="Model" value={context?.model_profile?.model_name ?? "-"} />
        <Row label="Provider" value={context?.model_profile?.provider ?? "-"} />
        <Row label="Context window" value={String(context?.model_profile?.context_window ?? "-")} />
        <Row label="Output reserve" value={String(context?.model_profile?.max_output_tokens ?? "-")} />
        <Row label="Compression" value={context?.model_profile?.compression_strategy ?? "-"} />
      </Card>
      <Card title="Latest Context Pack">
        {latest ? (
          <>
            <Row label="Kind" value={latest.pack_kind ?? "-"} />
            <Row label="Trigger" value={latest.trigger_node ?? "-"} />
            <Row label="Compression ratio" value={latest.compression_ratio != null ? latest.compression_ratio.toFixed(4) : "-"} />
            <Row label="Input tokens" value={String(latest.estimated_input_tokens ?? "-")} />
            <Row label="Output reserve" value={String(latest.estimated_output_reserve ?? "-")} />
            <Row label="Included" value={latest.included_sections?.join(", ") || "-"} />
            <Row label="Excluded" value={latest.excluded_sections?.join(", ") || "-"} />
            <Row label="Warnings" value={latest.warnings?.join(" · ") || "-"} />
          </>
        ) : <div className="debug-placeholder">No context pack reports recorded yet.</div>}
      </Card>
      <Card title="Included Evidence">
        <Row label="Retained files" value={latest?.retained_files?.join(", ") || "-"} />
        <Row label="Omitted files" value={latest?.omitted_files?.map((file) => `${file.path ?? "unknown"} (${file.reason ?? "omitted"})`).join(", ") || "-"} />
        <Row label="Web sources" value={latest?.retained_web_sources?.map((source) => source.url || source.title || source.source || "source").join(", ") || "-"} />
      </Card>
    </>
  );
}

function MemoryPanel({ memory }: { memory: MemoryDebug | null }) {
  const [view, setView] = useState<"table" | "graph">("table");
  const [filters, setFilters] = useState<Record<string, boolean>>({
    memory: true,
    file: true,
    symbol: true,
    run: true,
    skill: true,
    thread: true,
    repository: true,
    proposal: true,
    edit: true,
    command: true,
  });
  const graph = memory?.graph;
  const visibleNodes = graph?.nodes.filter((node) => filters[node.type] ?? true) ?? [];
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = graph?.edges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)) ?? [];
  return (
    <>
      <div className="debug-toggle-row">
        <button className={`debug-secondary-button${view === "table" ? " debug-button-active" : ""}`} type="button" onClick={() => setView("table")}>Table</button>
        <button className={`debug-secondary-button${view === "graph" ? " debug-button-active" : ""}`} type="button" onClick={() => setView("graph")}>Graph</button>
      </div>
      {view === "table" ? (
      <Card title="Memory Table">
        <table className="debug-table">
          <thead><tr><th>id</th><th>type</th><th>content</th><th>source</th><th>repo</th><th>tags</th><th>created_at</th></tr></thead>
          <tbody>
            {memory?.items.length ? memory.items.map((item) => (
              <tr key={item.id}><td>{item.id}</td><td>{item.type}</td><td>{item.content}</td><td>{item.source}</td><td>{item.repo ?? "-"}</td><td>{item.tags?.join(", ") || "-"}</td><td>{item.created_at}</td></tr>
            )) : <tr><td colSpan={7}>No memory records yet.</td></tr>}
          </tbody>
        </table>
      </Card>
      ) : (
      <Card title="Memory Graph">
        <div className="debug-filter-row">
          {Object.keys(filters).map((key) => (
            <label key={key}>
              <input type="checkbox" checked={filters[key]} onChange={(event) => setFilters((current) => ({ ...current, [key]: event.target.checked }))} />
              {key}
            </label>
          ))}
        </div>
        {visibleNodes.length ? <MemoryGraph nodes={visibleNodes} edges={visibleEdges} /> : <div className="debug-placeholder">No graph data yet.</div>}
      </Card>
      )}
    </>
  );
}

function MemoryGraph({ nodes, edges }: { nodes: GraphNode[]; edges: GraphEdge[] }) {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const width = 920;
  const height = 520;
  const clusterOrder = ["repository", "thread", "run", "file", "symbol", "proposal", "edit", "command", "skill", "memory"];
  const clusters = clusterOrder
    .map((type) => ({ type, nodes: nodes.filter((node) => node.type === type) }))
    .filter((cluster) => cluster.nodes.length);
  const positioned = clusters.flatMap((cluster, clusterIndex) => {
    const columns = 3;
    const clusterWidth = width / columns;
    const clusterHeight = height / Math.ceil(Math.max(clusters.length, 1) / columns);
    const clusterX = (clusterIndex % columns) * clusterWidth;
    const clusterY = Math.floor(clusterIndex / columns) * clusterHeight;
    const centerX = clusterX + clusterWidth / 2;
    const centerY = clusterY + clusterHeight / 2 + 10;
    return cluster.nodes.map((node, nodeIndex) => {
      const angle = (Math.PI * 2 * nodeIndex) / Math.max(cluster.nodes.length, 1);
      const radius = Math.min(clusterWidth, clusterHeight) * 0.24;
      return {
        ...node,
        cluster: cluster.type,
        clusterX,
        clusterY,
        clusterWidth,
        clusterHeight,
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
      };
    });
  });
  const selected = positioned.find((node) => node.id === selectedNodeId);
  const byId = new Map(positioned.map((node) => [node.id, node]));
  return (
    <div className="memory-graph-layout">
      <svg className="memory-graph" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Memory relationship graph">
        {clusters.map((cluster, index) => {
          const columns = 3;
          const clusterWidth = width / columns;
          const clusterHeight = height / Math.ceil(Math.max(clusters.length, 1) / columns);
          const x = (index % columns) * clusterWidth + 10;
          const y = Math.floor(index / columns) * clusterHeight + 10;
          return (
            <g key={cluster.type}>
              <rect x={x} y={y} width={clusterWidth - 20} height={clusterHeight - 20} rx={16} className="memory-graph-cluster" />
              <text x={x + 14} y={y + 24} className="memory-graph-cluster-label">
                {clusterLabel(cluster.type)} ({cluster.nodes.length})
              </text>
            </g>
          );
        })}
        {edges.map((edge, index) => {
          const source = byId.get(edge.source);
          const target = byId.get(edge.target);
          if (!source || !target) return null;
          return <line key={`${edge.source}-${edge.target}-${index}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} className="memory-graph-edge" />;
        })}
        {positioned.map((node) => (
          <g key={node.id} role="button" tabIndex={0} onClick={() => setSelectedNodeId(node.id)}>
            <circle cx={node.x} cy={node.y} r={node.type === "repository" ? 24 : 18} className={`memory-graph-node memory-graph-node-${node.type}${node.id === selectedNodeId ? " memory-graph-node-selected" : ""}`} />
            <text x={node.x} y={node.y + 32} textAnchor="middle" className="memory-graph-label">{node.label.slice(0, 22)}</text>
          </g>
        ))}
      </svg>
      <aside className="memory-graph-details">
        {selected ? (
          <>
            <strong>{selected.label}</strong>
            <span>{clusterLabel(selected.type)}</span>
            <code>{selected.id}</code>
            <span>Related edges: {edges.filter((edge) => edge.source === selected.id || edge.target === selected.id).length}</span>
          </>
        ) : (
          <span>Select a node to inspect relationships.</span>
        )}
      </aside>
    </div>
  );
}

function clusterLabel(type: string): string {
  const labels: Record<string, string> = {
    repository: "Repositories",
    thread: "Threads",
    run: "Runs",
    file: "Files",
    symbol: "Symbols",
    skill: "Skills",
    proposal: "Proposals",
    edit: "Edits",
    command: "Commands",
    memory: "Memories",
  };
  return labels[type] || type;
}

function SkillsPanel({ skills }: { skills: SkillsDebug | null }) {
  return (
    <Card title="Discovered Skills">
      {skills?.skills.length ? skills.skills.map((skill) => (
        <div className="debug-list-item" key={`${skill.source_path}:${skill.name}`}>
          <strong>{skill.name}</strong>
          <span>{skill.scope} · {skill.enabled ? "enabled" : "disabled"}</span>
          <span>{skill.description || "No description"}</span>
          <code>{skill.source_path}</code>
        </div>
      )) : <div className="debug-placeholder">No skills.md files discovered.</div>}
    </Card>
  );
}

function IntegrationsPanel({ integrations }: { integrations: IntegrationsDebug | null }) {
  return (
    <Card title="Integration Status">
      {integrations?.integrations.map((integration) => (
        <div className="debug-list-item" key={integration.provider}>
          <strong>{integration.provider}</strong>
          <span>{integration.status} · toolkits: {integration.toolkits_count ?? integration.toolkits?.length ?? 0} · tools: {integration.tools_count}</span>
          {integration.message && <span>{integration.message}</span>}
          {integration.accounts?.length ? (
            <span>Connected accounts: {integration.accounts.map((account) => `${account.toolkit ?? "unknown"}:${account.status ?? "unknown"}`).join(", ")}</span>
          ) : <span>No connected accounts reported.</span>}
          <button className="debug-secondary-button" type="button" onClick={() => window.open("https://docs.composio.dev/docs/authenticating-tools", "_blank", "noopener,noreferrer")}>
            Open setup docs
          </button>
        </div>
      ))}
    </Card>
  );
}

function ToolsPanel({ tools }: { tools: ToolsDebug | null }) {
  return (
    <>
      <Card title="Local Tools">
        {tools?.tools.map((tool) => (
          <div className="debug-list-item" key={tool.name}>
            <strong>{tool.name}</strong>
            <span>{tool.installed ? "installed" : "missing"} · {tool.version ?? "no version detected"}</span>
            <span>Auth status: {tool.auth_status ?? "unknown"}</span>
            {tool.path && <code>{tool.path}</code>}
          </div>
        ))}
      </Card>
      <Card title="Tool Permission Layers">
        {tools?.permissions ? Object.entries(tools.permissions).map(([key, value]) => (
          <Row key={key} label={key.replaceAll("_", " ")} value={value} />
        )) : <div className="debug-placeholder">Tool permissions are not loaded yet.</div>}
      </Card>
    </>
  );
}

function RunsPanel({ runtime }: { runtime: RuntimeDebug | null }) {
  return (
    <Card title="Recent Runs">
      {runtime?.recent_runs?.length ? (
        <table className="debug-table">
          <thead><tr><th>run</th><th>time</th><th>repo</th><th>branch</th><th>intent</th><th>status</th><th>latency</th></tr></thead>
          <tbody>
            {runtime.recent_runs.map((run) => (
              <tr key={String(run.id)}>
                <td>{String(run.id ?? "-")}</td>
                <td>{String(run.timestamp ?? "-")}</td>
                <td>{String(run.repo ?? "-")}</td>
                <td>{String(run.branch ?? "-")}</td>
                <td>{String(run.intent ?? "-")}</td>
                <td>{String(run.status ?? "-")}</td>
                <td>{String(run.latency_ms ?? "-")} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <div className="debug-placeholder">No recent runs recorded yet.</div>}
    </Card>
  );
}

function SettingsPanel({ runtime }: { runtime: RuntimeDebug | null }) {
  return (
    <Card title="Settings Snapshot">
      <Row label="Connection mode" value={runtime?.model?.connection_mode ?? "-"} />
      <Row label="Permission mode" value={runtime?.permissions?.mode ?? "-"} />
      <Row label="Tool permissions" value={runtime?.permissions?.tools ? Object.entries(runtime.permissions.tools).map(([key, value]) => `${key}: ${value}`).join(" · ") : "-"} />
    </Card>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="debug-card"><h2>{title}</h2>{children}</section>;
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="debug-row"><span>{label}</span><strong>{value}</strong></div>;
}
