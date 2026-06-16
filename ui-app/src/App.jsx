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
  const [candidateError, setCandidateError] = useState("");

  useEffect(() => {
    fetchCandidates();
    fetchMetrics();
  }, []);

  async function fetchCandidates(params = {}) {
    setLoadingCandidates(true);
    setCandidateError("");
    try {
      const query = new URLSearchParams({
        top_n: String(params.top_n ?? controls.top_n),
        pool_size: String(params.pool_size ?? controls.pool_size),
        season: String(params.season ?? controls.season),
      });
      const response = await fetch(`${API_BASE}/candidate_pool?${query}`);
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `candidate_pool request failed (${response.status})`);
      }
      setCandidates(Array.isArray(payload.results) ? payload.results : []);
    } catch (error) {
      console.error("Failed to fetch candidates", error);
      setCandidates([]);
      setCandidateError(error.message || "Failed to load candidate table");
    } finally {
      setLoadingCandidates(false);
    }
  }

  async function fetchMetrics() {
    try {
      const response = await fetch(`${API_BASE}/monitoring/metrics`);
      const payload = await response.json();
      setMetrics(Array.isArray(payload.metrics) ? payload.metrics : []);
    } catch (error) {
      console.error("Failed to fetch metrics", error);
    }
  }

  async function runVectorBuilder() {
    setStatus("running vector builder");
    try {
      const response = await fetch(`${API_BASE}/pipeline/vector-builder`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: "json/sample_player_features.json", output: "output/runtime_vectors.json" }),
      });
      const payload = await response.json();
      setLog((prev) => [{ type: "vector", payload }, ...prev].slice(0, 6));
      setStatus(payload.returncode === 0 ? "vector success" : "vector error");
    } catch (error) {
      console.error("Vector builder failed", error);
      setStatus("vector error");
    } finally {
      fetchMetrics();
    }
  }

  async function runCandidatePool() {
    setStatus("running candidate pool");
    try {
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
      await fetchCandidates({
        season: controls.season,
        pool_size: controls.pool_size,
        top_n: controls.top_n,
      });
    } catch (error) {
      console.error("Candidate pool export failed", error);
      setStatus("pool error");
    } finally {
      fetchMetrics();
    }
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
          ) : candidateError ? (
            <div className="placeholder">{candidateError}</div>
          ) : candidates.length === 0 ? (
            <div className="placeholder">No candidates returned yet.</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Player</th>
                  <th>MVP Share</th>
                  <th>Team</th>
                  <th>Preview</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((candidate, index) => (
                  <tr key={candidate.player_id || `${candidate.player_name}-${index}`}>
                    <td>{candidate.mvp_rank}</td>
                    <td>{candidate.player_name}</td>
                    <td>
                      <details>
                        <summary>{((candidate.mvp_share_of_top_n ?? candidate.mvp_probability ?? 0) * 100).toFixed(1)}%</summary>
                        <div>Raw model probability: {((candidate.mvp_probability_raw ?? candidate.mvp_probability ?? 0) * 100).toFixed(2)}%</div>
                        <div>Share of displayed top N: {((candidate.mvp_share_of_top_n ?? 0) * 100).toFixed(2)}%</div>
                      </details>
                    </td>
                    <td>{candidate.metadata?.team || ""}</td>
                    <td>
                      <details>
                        <summary>Features</summary>
                        <pre>{JSON.stringify((candidate.vector || []).slice(0, 12), null, 2)}</pre>
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
