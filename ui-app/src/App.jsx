import { useEffect, useState } from "react";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_URL || "";

const defaultControls = {
  sector: "vector",
  season: "latest",
  top_n: 5,
  pool_size: 30,
};

function App() {
  const [controls, setControls] = useState(defaultControls);
  const [status, setStatus] = useState("idle");
  const [log, setLog] = useState([]);
  const [candidates, setCandidates] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [loadingCandidates, setLoadingCandidates] = useState(false);

  useEffect(() => {
    fetchCandidates();
    fetchMetrics();
  }, []);

  async function fetchCandidates(params = {}) {
    setLoadingCandidates(true);
    const query = new URLSearchParams({
      top_n: params.top_n || controls.top_n,
      pool_size: params.pool_size || controls.pool_size,
      season: params.season || controls.season,
    });
    const response = await fetch(`${API_BASE}/candidate_pool?${query}`);
    const payload = await response.json();
    setCandidates(payload.results || []);
    setMetrics((prev) => prev);
    setLoadingCandidates(false);
  }

  async function fetchMetrics() {
    const response = await fetch(`${API_BASE}/monitoring/metrics`);
    const payload = await response.json();
    setMetrics(payload.metrics || []);
  }

  async function runVectorBuilder() {
    setStatus("running vector builder");
    const response = await fetch(`${API_BASE}/pipeline/vector-builder`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: "json/sample_player_features.json", output: "output/runtime_vectors.json" }),
    });
    const payload = await response.json();
    setLog((prev) => [{ type: "vector", payload }, ...prev].slice(0, 6));
    setStatus(payload.returncode === 0 ? "vector success" : "vector error");
    fetchMetrics();
  }

  async function runCandidatePool() {
    setStatus("running candidate pool");
    const response = await fetch(`${API_BASE}/pipeline/export-candidate-pool`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        season: controls.season === "latest" ? null : Number(controls.season),
        pool_size: controls.pool_size,
        top_n: controls.top_n,
      }),
    });
    const payload = await response.json();
    setLog((prev) => [{ type: "pool", payload }, ...prev].slice(0, 6));
    setStatus(payload.returncode === 0 ? "pool success" : "pool error");
    fetchCandidates();
    fetchMetrics();
  }

  return (
    <div className="app-shell">
      <header>
        <h1>NBA MVP Control Center</h1>
        <p>Trigger scripts, inspect rankings, and monitor pipeline health.</p>
      </header>
      <section className="controls">
        <div className="control-card">
          <h2>Pipeline Controls</h2>
          <div className="control-row">
            <label>
              Season
              <input
                type="text"
                value={controls.season}
                onChange={(event) => setControls({ ...controls, season: event.target.value })}
              />
            </label>
            <label>
              Pool size
              <input
                type="number"
                value={controls.pool_size}
                min={1}
                onChange={(event) => setControls({ ...controls, pool_size: Number(event.target.value) })}
              />
            </label>
            <label>
              Top N
              <input
                type="number"
                value={controls.top_n}
                min={1}
                onChange={(event) => setControls({ ...controls, top_n: Number(event.target.value) })}
              />
            </label>
          </div>
          <div className="button-row">
            <button onClick={runVectorBuilder}>Run Vector Builder</button>
            <button onClick={runCandidatePool}>Export Candidate Pool</button>
          </div>
          <div className="status">Status: {status}</div>
        </div>
        <div className="metrics-card">
          <h2>Metrics</h2>
          <ul>
            {metrics.map((metric) => (
              <li key={metric.name + metric.timestamp}>
                <strong>{metric.name}</strong>: {metric.value} <span>{metric.timestamp}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
      <section className="main-grid">
        <div className="candidates">
          <h2>Top {controls.top_n} MVP Candidates</h2>
          {loadingCandidates ? (
            <div className="placeholder">Loading...</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Player</th>
                  <th>MVP%</th>
                  <th>Team</th>
                  <th>Preview</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((candidate) => (
                  <tr key={candidate.player_id}>
                    <td>{candidate.mvp_rank}</td>
                    <td>{candidate.player_name}</td>
                    <td>{(candidate.mvp_probability * 100).toFixed(1)}%</td>
                    <td>{candidate.metadata?.team || ""}</td>
                    <td>
                      <details>
                        <summary>Features</summary>
                        <pre>{JSON.stringify(candidate.vector.slice(0, 12), null, 2)}</pre>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="logs">
          <h2>Recent Actions</h2>
          <ul>
            {log.map((entry, index) => (
              <li key={index}>
                <code>{entry.type}</code>
                <pre>{JSON.stringify(entry.payload, null, 2)}</pre>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}

export default App;
