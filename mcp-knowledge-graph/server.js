// Minimal HTTP wrapper for MCP Knowledge Graph per docs/plan/mcp_http_api.md
// Runtime: Node >= 18 (no external deps)
// Persistence: JSONL under MEMORY_PATH (default /app/.aim)

const http = require('http');
const url = require('url');
const fs = require('fs');
const fsp = require('fs').promises;
const path = require('path');

const PORT = parseInt(process.env.PORT || '8080', 10);
let BASE_DIR = process.env.MEMORY_PATH || '/app/.aim';
const FILE_MARKER = { type: "_aim", source: "mcp-knowledge-graph" };
const START_TIME = Date.now();

/**
 * Probe whether a directory is writable by the current (non-root) user.
 * If not writable, we will dynamically fall back to a known-writable location.
 */
async function isWritable(dir) {
  try {
    await fsp.mkdir(dir, { recursive: true });
    const probe = path.join(dir, '.probe');
    await fsp.writeFile(probe, 'ok');
    await fsp.unlink(probe);
    return true;
  } catch (e) {
    return false;
  }
}

/**
 * Choose the first writable base directory from:
 *   1) MEMORY_PATH (if provided)
 *   2) /home/node/.aim
 *   3) /tmp/.aim
 * Updates the global BASE_DIR to the selected path.
 */
async function selectBaseDir() {
  const candidates = [];
  if (process.env.MEMORY_PATH && process.env.MEMORY_PATH.trim()) {
    candidates.push(process.env.MEMORY_PATH.trim());
  }
  candidates.push('/home/node/.aim');
  candidates.push('/tmp/.aim');

  for (const dir of candidates) {
    if (await isWritable(dir)) {
      BASE_DIR = dir;
      return dir;
    }
  }
  // Fallback to original (may still be read-only)
  return BASE_DIR;
}

function memoryFilePath(context) {
  const fname = context ? `memory-${sanitize(context)}.jsonl` : 'memory.jsonl';
  return path.join(BASE_DIR, fname);
}

function sanitize(s) {
  return String(s || '').replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 64);
}

async function ensureBaseDir() {
  const chosen = await selectBaseDir();
  await fsp.mkdir(chosen, { recursive: true });
}

async function loadGraph(context) {
  await ensureBaseDir();
  const file = memoryFilePath(context);
  try {
    const data = await fsp.readFile(file, 'utf-8');
    const lines = data.split('\n').filter(l => l.trim() !== '');
    if (lines.length === 0) return { entities: [], relations: [] };

    // optional marker check
    try {
      const first = JSON.parse(lines[0]);
      if (!(first && first.type === '_aim' && first.source === 'mcp-knowledge-graph')) {
        // If not our file, treat as empty to avoid corrupting unrelated files
        return { entities: [], relations: [] };
      }
    } catch {
      // no-op
    }

    const graph = { entities: [], relations: [] };
    for (let i = 1; i < lines.length; i++) {
      const obj = JSON.parse(lines[i]);
      if (obj.type === 'entity') graph.entities.push({ name: obj.name, entityType: obj.entityType, observations: obj.observations || [] });
      if (obj.type === 'relation') graph.relations.push({ from: obj.from, to: obj.to, relationType: obj.relationType });
    }
    return graph;
  } catch (err) {
    if (err && err.code === 'ENOENT') {
      return { entities: [], relations: [] };
    }
    throw err;
  }
}

async function saveGraph(graph, context) {
  await ensureBaseDir();
  const file = memoryFilePath(context);
  const lines = [
    JSON.stringify(FILE_MARKER),
    ...graph.entities.map(e => JSON.stringify({ type: 'entity', ...e })),
    ...graph.relations.map(r => JSON.stringify({ type: 'relation', ...r })),
  ];
  await fsp.writeFile(file, lines.join('\n'), 'utf-8');
}

async function upsertEntities(graph, entities) {
  let created = 0;
  for (const e of (entities || [])) {
    if (!graph.entities.some(x => x.name === e.name)) {
      graph.entities.push({ name: e.name, entityType: e.entityType || 'unknown', observations: Array.isArray(e.observations) ? e.observations.slice() : [] });
      created++;
    }
  }
  return created;
}

async function upsertRelations(graph, relations) {
  let created = 0;
  for (const r of (relations || [])) {
    const exists = graph.relations.some(x => x.from === r.from && x.to === r.to && x.relationType === r.relationType);
    if (!exists) {
      graph.relations.push({ from: r.from, to: r.to, relationType: r.relationType });
      created++;
    }
  }
  return created;
}

async function addObservations(graph, observations) {
  let added = 0;
  for (const o of (observations || [])) {
    const name = o.entityName;
    const contents = Array.isArray(o.contents) ? o.contents : [];
    let ent = graph.entities.find(x => x.name === name);
    if (!ent) {
      ent = { name, entityType: 'unknown', observations: [] };
      graph.entities.push(ent);
    }
    for (const c of contents) {
      if (!ent.observations.includes(c)) {
        ent.observations.push(c);
        added++;
      }
    }
  }
  return added;
}

function scoreEntity(ent, terms) {
  if (!terms || terms.length === 0) return 0;
  const hay = [
    ent.name || '',
    ent.entityType || '',
    ...(Array.isArray(ent.observations) ? ent.observations : []),
  ].join(' ').toLowerCase();
  let score = 0;
  for (const t of terms) {
    const q = String(t || '').toLowerCase();
    if (!q) continue;
    // naive scoring by occurrences
    score += hay.split(q).length - 1;
  }
  return score;
}

function scoreRelation(rel, terms) {
  if (!terms || terms.length === 0) return 0;
  const hay = `${rel.from} ${rel.to} ${rel.relationType}`.toLowerCase();
  let score = 0;
  for (const t of terms) {
    const q = String(t || '').toLowerCase();
    if (!q) continue;
    score += hay.split(q).length - 1;
  }
  return score;
}

async function handleHealth(req, res) {
  try {
    await ensureBaseDir();
    const dbPath = memoryFilePath(null);
    const exists = fs.existsSync(dbPath);
    respondJSON(res, 200, {
      status: 'ok',
      uptime_s: Math.round((Date.now() - START_TIME) / 1000),
      db: { exists, path: BASE_DIR },
      api_version: 1,
    });
  } catch (e) {
    respondJSON(res, 500, { error: { code: 'EHEALTH', message: String(e?.message || e) } });
  }
}

async function handleUpsert(req, res, body) {
  const started = Date.now();
  try {
    const { context, entities, relations, observations } = body || {};
    const graph = await loadGraph(context);
    const c1 = await upsertEntities(graph, entities);
    const c2 = await upsertRelations(graph, relations);
    const c3 = await addObservations(graph, observations);
    await saveGraph(graph, context);
    respondJSON(res, 200, { created: { entities: c1, relations: c2, observations: c3 }, took_ms: Date.now() - started, api_version: 1 });
  } catch (e) {
    respondJSON(res, 503, { error: { code: 'EUPSERT', message: String(e?.message || e) } });
  }
}

async function handleQuery(req, res, body) {
  const started = Date.now();
  try {
    const { context_hint, terms, retrieve_limit } = body || {};
    const limit = Math.max(0, Math.min(100, retrieve_limit || 5));
    const graph = await loadGraph(context_hint);

    const items = [];
    for (const ent of graph.entities) {
      const s = scoreEntity(ent, terms || []);
      if (s > 0) items.push({ type: 'entity', payload: ent, score: s });
      // observations as standalone items (optional)
      for (const obs of (ent.observations || [])) {
        const text = String(obs || '');
        const obsScore = (terms || []).reduce((acc, t) => acc + (text.toLowerCase().includes(String(t).toLowerCase()) ? 1 : 0), 0);
        if (obsScore > 0) items.push({ type: 'observation', payload: { entityName: ent.name, content: text }, score: obsScore });
      }
    }
    for (const rel of graph.relations) {
      const s = scoreRelation(rel, terms || []);
      if (s > 0) items.push({ type: 'relation', payload: rel, score: s });
    }

    items.sort((a, b) => (b.score || 0) - (a.score || 0));
    const used = items.slice(0, limit);

    respondJSON(res, 200, { items: used, total: items.length, took_ms: Date.now() - started, api_version: 1 });
  } catch (e) {
    respondJSON(res, 503, { error: { code: 'EQUERY', message: String(e?.message || e) } });
  }
}

async function handleRead(req, res, query) {
  try {
    const context = query.get('context') || null;
    const graph = await loadGraph(context);
    respondJSON(res, 200, { entities: graph.entities, relations: graph.relations, api_version: 1 });
  } catch (e) {
    respondJSON(res, 503, { error: { code: 'EREAD', message: String(e?.message || e) } });
  }
}

function respondJSON(res, status, obj) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('mcp-api-version', '1');
  res.end(JSON.stringify(obj));
}

function parseJSON(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => { data += chunk; if (data.length > 5 * 1024 * 1024) { reject(new Error('Payload too large')); req.destroy(); } });
    req.on('end', () => {
      if (!data) return resolve({});
      try {
        resolve(JSON.parse(data));
      } catch (e) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  const parsed = url.parse(req.url, true);
  const method = req.method || 'GET';
  const pathname = parsed.pathname || '/';

  // CORS for local testing (internal network only in compose)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (method === 'OPTIONS') { res.statusCode = 204; return res.end(); }

  try {
    if (method === 'GET' && pathname === '/health') {
      return await handleHealth(req, res);
    }
    if (method === 'POST' && pathname === '/v1/memory/upsert') {
      const body = await parseJSON(req);
      return await handleUpsert(req, res, body);
    }
    if (method === 'POST' && pathname === '/v1/memory/query') {
      const body = await parseJSON(req);
      return await handleQuery(req, res, body);
    }
    if (method === 'GET' && pathname === '/v1/memory/read') {
      const q = new URLSearchParams(parsed.query);
      return await handleRead(req, res, q);
    }

    respondJSON(res, 404, { error: { code: 'ENOTFOUND', message: 'Route not found' } });
  } catch (e) {
    respondJSON(res, 500, { error: { code: 'EUNEXPECTED', message: String(e?.message || e) } });
  }
});

server.listen(PORT, async () => {
  try { await ensureBaseDir(); } catch {}
  // eslint-disable-next-line no-console
  console.log(`mcp-knowledge-graph HTTP server listening on :${PORT}, base=${BASE_DIR}`);
});