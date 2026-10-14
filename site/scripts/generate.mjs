// Generates the model, evaluation, and repo-docs pages from repository data.
// Run before `astro build` / `astro dev`. Output directories are git-ignored.
//
// Sources (all read-only):
//   packages/onnx-runner/src/catalog.ts                          curated models and pinned revisions
//   packages/onnx-runner/templates/<owner>--<name>/<rev>/*.json  artifact manifests and templates
//   release/onnx-eval-receipt.json                               evaluation receipt
//   benchmark-results/2026-10-14-onnx-runtime-parity-v1/*.json   node-vs-pytorch parity reports
//   src/gliner_runner/model_inventory.py                         native PyTorch drift characterization
//   docs/*.md                                                    copied with links rewritten

import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, rmSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, posix, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO_URL = "https://github.com/stuffbucket/gliner-runner";
export const BASE = "/gliner-runner";
const PARITY_DIR = "benchmark-results/2026-10-14-onnx-runtime-parity-v1";
const RECEIPT_PATH = "release/onnx-eval-receipt.json";
const MODES = ["singleton", "repeated", "batch", "concurrent-group", "concurrent-all"];
const DOC_ORDER = [
  "onnx-runtime",
  "operations",
  "protocol",
  "architecture",
  "backends",
  "benchmarks",
  "registries",
  "releases",
  "development",
];

// ---------------------------------------------------------------- formatting

export function fmtBytes(n) {
  const units = [
    ["GiB", 1024 ** 3],
    ["MiB", 1024 ** 2],
    ["KiB", 1024],
  ];
  for (const [name, size] of units) {
    if (n >= size) return `${(n / size).toFixed(2)} ${name}`;
  }
  return `${n} B`;
}

export const fmtInt = (n) => n.toLocaleString("en-US");

export function fmtDrift(x) {
  if (x === null || x === undefined) return "n/a";
  if (x === 0) return "0";
  return x.toExponential(3);
}

const code = (s) => `\`${s}\``;
const yesNo = (b) => (b ? "pass" : "FAIL");

export function table(headers, rows, align = []) {
  const sep = headers.map((_, i) => (align[i] === "r" ? "---:" : "---"));
  const line = (cells) => `| ${cells.join(" | ")} |`;
  return [line(headers), line(sep), ...rows.map(line)].join("\n");
}

const frontmatter = (fields) =>
  `---\n${Object.entries(fields)
    .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : JSON.stringify(v)}`)
    .join("\n")}\n---\n\n`;

// ---------------------------------------------------------------- loading

const readJson = (root, rel) => JSON.parse(readFileSync(join(root, rel), "utf8"));

export function parseCatalog(source) {
  const start = source.indexOf("KNOWN_MODELS");
  const end = source.indexOf("\n];", start);
  const body = source.slice(start, end < 0 ? undefined : end);
  const models = [];
  for (const block of body.matchAll(/\{\s*modelId:[\s\S]*?\n {2}\}/g)) {
    const field = (key) => {
      const m = block[0].match(new RegExp(`${key}:\\s*"([^"]*)"`));
      if (!m) throw new Error(`catalog.ts: model block is missing ${key}`);
      return m[1];
    };
    const count = block[0].match(/parameterCount:\s*([\d_]+)/);
    if (!count) throw new Error("catalog.ts: model block is missing parameterCount");
    models.push({
      modelId: field("modelId"),
      repository: field("repository"),
      pinnedRevision: field("pinnedRevision"),
      description: field("description"),
      parameterCount: Number(count[1].replaceAll("_", "")),
    });
  }
  if (models.length === 0) throw new Error("catalog.ts: no curated models found");
  return models;
}

export function parseNativeEvidence(source) {
  const rows = [];
  const re =
    /\("(decide-[a-z0-9-]+)",\s*"([a-z]+\/[a-z0-9]+)"\):\s*\(\s*([0-9.e+-]+),\s*"([^"]*)",\s*((?:"[^"]*"\s*)+),?\s*\)/g;
  for (const m of source.matchAll(re)) {
    const scope = [...m[5].matchAll(/"([^"]*)"/g)].map((s) => s[1]).join("");
    rows.push({ modelId: m[1], profile: m[2], drift: Number(m[3]), order: m[4], scope });
  }
  return rows;
}

function loadTemplates(root) {
  const base = join(root, "packages/onnx-runner/templates");
  const byModel = new Map();
  for (const repoDir of readdirSync(base)) {
    for (const rev of readdirSync(join(base, repoDir))) {
      const dir = join(base, repoDir, rev);
      if (!existsSync(join(dir, "manifest.json"))) continue;
      const manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
      const template = JSON.parse(readFileSync(join(dir, "template.json"), "utf8"));
      byModel.set(manifest.model_id, { manifest, template, relDir: posix.join("packages/onnx-runner/templates", repoDir, rev) });
    }
  }
  return byModel;
}

function loadReports(root) {
  const dir = join(root, PARITY_DIR);
  const reports = [];
  for (const name of readdirSync(dir).sort()) {
    if (!/^node-vs-pytorch-.*\.json$/.test(name)) continue;
    reports.push({ file: name, ...JSON.parse(readFileSync(join(dir, name), "utf8")) });
  }
  return reports;
}

// ---------------------------------------------------------------- model pages

const hfTree = (repo, rev) => `https://huggingface.co/${repo}/tree/${rev}`;
const repoBlob = (path) => `${REPO_URL}/blob/main/${path}`;

function modelPage({ model, entry, receipt, reports, native, order }) {
  const { manifest, template } = entry;
  const sourceBytes = template.source_files.reduce((s, f) => s + f.size, 0);
  const artifactBytes = manifest.files.reduce((s, f) => s + f.size, 0);
  const results = receipt.results.filter((r) => r.model_id === model.modelId);
  const myReports = reports.filter((r) => r.model_id === model.modelId);
  const receiptModel = receipt.models.find((m) => m.model_id === model.modelId);

  const lines = [];
  lines.push(
    `${model.description}. Upstream model: [${model.repository}](${hfTree(model.repository, model.pinnedRevision)}), ` +
      `served by GLiNER Runner at one pinned revision. This page is generated from the repository's manifest, ` +
      `template, evaluation receipt, and benchmark results.`,
    "",
    "## Model details",
    "",
    table(
      ["Field", "Value"],
      [
        ["Runner model id", code(model.modelId)],
        ["Upstream repository", `[${model.repository}](https://huggingface.co/${model.repository})`],
        ["Pinned revision", `[${code(model.pinnedRevision)}](${hfTree(model.repository, model.pinnedRevision)})`],
        ["Parameters", `${fmtInt(model.parameterCount)} (${(model.parameterCount / 1e6).toFixed(0)}M)`],
        ["License", manifest.license],
        ["Operations", manifest.operations.map(code).join(", ")],
        ["Backend", `${code("onnx")} on ${code(manifest.provider)}`],
        ["Precision", `${code(manifest.precision)} (quantization: ${manifest.quantization})`],
        ["ONNX opset", String(manifest.opset)],
        ["Download (upstream source files)", `${fmtBytes(sourceBytes)}`],
        ["Installed artifact", `${fmtBytes(artifactBytes)}`],
        ["Source snapshot SHA-256", code(manifest.source_snapshot_sha256)],
        ["Manifest", `[${code("manifest.json")}](${repoBlob(`${entry.relDir}/manifest.json`)})`],
      ],
    ),
    "",
    "## Supported operations and profiles",
    "",
    table(
      ["Backend", "Profile", "Operation", "Status"],
      [
        [code("onnx"), "cpu/fp32", code("classify"), "Supported; parity-verified (see Evaluation)"],
        [code("pytorch"), "cpu/fp32, mps/fp32, mps/fp16", code("classify"), "Supported by the Python server; the correctness oracle"],
      ],
    ),
    "",
    "Anything else (other devices or precisions, quantization, other operations, other revisions) is rejected " +
      "with a structured error; there is no fallback. See [ONNX runtime](" +
      `${BASE}/docs/onnx-runtime/#supported-scope) for the exact scope.`,
    "",
    "## Install and use",
    "",
    "```sh",
    `gliner-runner-onnx models install ${model.modelId} --yes`,
    "gliner-runner-onnx serve",
    "```",
    "",
    `Request it with ${code(`"model": "${model.modelId}"`)}, ${code('"backend": "onnx"')}, and ${code('"precision": "fp32"')}. ` +
      `See the [protocol](${BASE}/docs/protocol/).`,
    "",
    "## Files",
    "",
    `Installed artifact files for revision ${code(model.pinnedRevision.slice(0, 12))}. ` +
      "Every file is verified by size and SHA-256 before the graph is loaded.",
    "",
    table(
      ["File", "Size", "SHA-256"],
      manifest.files.map((f) => [code(f.path), `${fmtBytes(f.size)} (${fmtInt(f.size)} B)`, code(f.sha256)]),
      [null, "r", null],
    ),
    "",
    "Upstream source files the artifact is assembled from:",
    "",
    table(
      ["File", "Role", "Size", "SHA-256"],
      template.source_files.map((f) => [code(f.path), f.role, `${fmtBytes(f.size)} (${fmtInt(f.size)} B)`, code(f.sha256)]),
      [null, null, "r", null],
    ),
    "",
    "## Evaluation",
    "",
    `ONNX Runtime responses compared with golden outputs from the PyTorch backend (cpu/fp32), replayed ` +
      `through the real HTTP server. Gating tolerance: absolute ${fmtDrift(receipt.tolerance)}. ` +
      `Raw max drift is the authoritative number; "synthetic zero" counts nonzero deltas below the tolerance.`,
    "",
    table(
      ["Corpus", "Requests", "Exact semantic matches", "Values compared", "Synthetic zero", "Raw max abs drift", "Result"],
      results.map((r) => [
        code(r.corpus),
        String(r.requests),
        `${r.exact_semantic_matches}/${r.requests}`,
        String(r.compared_values),
        String(r.synthetic_zero_diffs),
        fmtDrift(r.max_raw_abs_diff),
        yesNo(r.passed),
      ]),
      [null, "r", "r", "r", "r", "r", null],
    ),
    "",
  );

  for (const r of myReports) {
    lines.push(
      `### ${r.corpus_id} by execution mode`,
      "",
      table(
        ["Mode", "Requests", "Exact matches", "Mismatches", "Values", "Over tolerance", "Raw max abs drift"],
        MODES.filter((m) => r.modes[m]).map((m) => {
          const x = r.modes[m];
          return [
            m,
            String(x.requests),
            String(x.exact_semantic_matches),
            String(x.mismatches),
            String(x.numeric.compared_values),
            String(x.numeric.over_tolerance_diffs),
            fmtDrift(x.numeric.max_raw_abs_diff),
          ];
        }),
        [null, "r", "r", "r", "r", "r", "r"],
      ),
      "",
      `Repeat runs byte-identical: ${r.repeat_identical}. Corpus SHA-256: ${code(r.corpus_sha256)}. ` +
        `Full report: [${r.file.replace(/\.json$/, ".md")}](${repoBlob(`${PARITY_DIR}/${r.file.replace(/\.json$/, ".md")}`)}).`,
      "",
    );
  }

  const p = manifest.parity;
  lines.push(
    "### Build-time graph parity",
    "",
    `Checked when the ONNX graph was exported, on fixture suite ${code(p.fixture_suite)} ` +
      `(${p.rows} rows, ${p.valid_logits_compared} logits).`,
    "",
    table(
      ["Check", "Value"],
      [
        ["Tolerance", fmtDrift(p.absolute_tolerance)],
        ["Max abs logit difference, ONNX vs PyTorch", fmtDrift(p.max_abs_logit_difference)],
        ["Max abs logit difference, official API vs runner PyTorch path", fmtDrift(p.official_max_abs_logit_difference)],
        ["Max abs logit difference, ONNX vs official API", fmtDrift(p.onnx_official_max_abs_logit_difference)],
        ["Max abs logit difference, batch invariance", fmtDrift(p.batch_invariance_max_abs_logit_difference)],
        ["Selected labels matching", `${p.selected_index_matches}/${p.rows}`],
      ],
    ),
    "",
  );

  const nativeRows = native.filter((n) => n.modelId === model.modelId);
  if (nativeRows.length > 0) {
    lines.push(
      "### Native PyTorch profiles",
      "",
      "Score drift of each PyTorch profile against the cpu/fp32 direct official oracle, as reported by " +
        `${code("GET /v1/models")} on the Python server. See [Benchmarks](${BASE}/docs/benchmarks/) for the underlying runs.`,
      "",
      table(
        ["Profile", "Observed max abs score drift", "Order of magnitude", "Scope"],
        nativeRows.map((n) => [n.profile, fmtDrift(n.drift), n.order, n.scope]),
      ),
      "",
    );
  }

  lines.push(
    "Overall results across all models are on the [Evaluation](" + `${BASE}/evaluation/) page.`,
    "",
    "## Limitations",
    "",
    `- Evidence covers ${code("classify")} only, on ${code("CPUExecutionProvider")} in FP32, at revision ` +
      `${code(model.pinnedRevision.slice(0, 12))}, on one host: ${receipt.host.cpu_model} (${receipt.host.platform}-${receipt.host.arch}). ` +
      "It does not imply MPS, CUDA, FP16, BF16, quantized, other-operation, other-revision, or other-host support.",
    "- Raw confidences differ from PyTorch within the tolerance above; they are not bitwise equal. " +
      "Labels, ordering, cardinality, errors, and token usage are compared exactly.",
    "- Corpora are small, synthetic parity probes. They are not accuracy benchmarks and say nothing about " +
      "task quality on your data.",
  );
  if (/multi/.test(model.modelId)) {
    lines.push(
      "- Applying the English corpus to this model (and the multilingual corpus to English models) is a parity " +
        "probe, not a claim about language quality.",
    );
  } else {
    lines.push(
      "- This model is English-oriented. The multilingual corpus is applied as a parity probe only, not as a " +
        "claim about language quality.",
    );
  }
  lines.push(
    "- Only the pinned revision can be installed and run; a manifest for any other revision is rejected.",
    "",
    "## Template",
    "",
    `Artifacts are assembled in Node.js from the pinned safetensors using the bundled template ` +
      `[${code("template.json")}](${repoBlob(`${entry.relDir}/template.json`)}) ` +
      `(SHA-256 ${code(receiptModel?.template_sha256 ?? "unknown")}).`,
    "",
  );

  return (
    frontmatter({
      title: model.modelId,
      description: `${model.description}: ${model.repository} at ${model.pinnedRevision.slice(0, 12)}.`,
      sidebar: { order },
    }) + lines.join("\n")
  );
}

function modelsIndex({ models, templates, receipt }) {
  const rows = models.map((m) => {
    const entry = templates.get(m.modelId);
    const worst = Math.max(...receipt.results.filter((r) => r.model_id === m.modelId).map((r) => r.max_raw_abs_diff));
    return [
      `[${code(m.modelId)}](${BASE}/models/${m.modelId}/)`,
      `[${m.repository}](https://huggingface.co/${m.repository})`,
      `${(m.parameterCount / 1e6).toFixed(0)}M`,
      entry.manifest.license,
      fmtBytes(entry.manifest.files.reduce((s, f) => s + f.size, 0)),
      fmtDrift(worst),
    ];
  });
  return (
    frontmatter({
      title: "Models",
      description: "Curated GLiNER2.5 Decide models served by GLiNER Runner, each pinned to an exact upstream revision.",
      sidebar: { order: 0 },
    }) +
    "GLiNER Runner serves three curated models, each pinned to an immutable upstream revision. Each card lists " +
      "the pinned revision, artifact files with checksums, and parity evaluation results.\n\n" +
      table(["Model", "Upstream", "Parameters", "License", "Artifact size", "Max raw drift (ONNX vs PyTorch)"], rows, [
        null,
        null,
        "r",
        null,
        "r",
        "r",
      ]) +
      "\n\nEvaluation covers ONNX Runtime on CPU in FP32 for the `classify` operation. See " +
      `[Evaluation](${BASE}/evaluation/) for the full matrix.\n`
  );
}

// ---------------------------------------------------------------- evaluation

function evaluationPage({ root, models, receipt, reports, native }) {
  const receiptBytes = readFileSync(join(root, RECEIPT_PATH));
  const receiptSha = createHash("sha256").update(receiptBytes).digest("hex");
  const total = receipt.results.reduce((s, r) => s + r.requests, 0);
  const matched = receipt.results.reduce((s, r) => s + r.exact_semantic_matches, 0);
  const worst = Math.max(...receipt.results.map((r) => r.max_raw_abs_diff));
  const ref = reports[0].reference;
  const runtime = reports[0].runtime;
  const lines = [];

  lines.push(
    "The Python-free ONNX runner is accepted only against PyTorch. Two fixed request corpora are replayed " +
      "through the runner's real HTTP server for every curated model and compared with golden outputs produced by " +
      `the PyTorch backend (${code(ref.path)}, Fastino GLiNER2 ${ref.versions.gliner2}). ` +
      "This page is generated from the checked-in receipt and parity reports.",
    "",
    "## Summary",
    "",
    `- Overall: **${receipt.all_passed ? "all combinations passed" : "FAILED"}**`,
    `- Requests: ${matched}/${total} exact semantic matches across ${receipt.results.length} model/corpus combinations`,
    `- Raw max absolute drift: ${fmtDrift(worst)} (gating tolerance ${fmtDrift(receipt.tolerance)})`,
    `- Scope: ONNX Runtime ${code("CPUExecutionProvider")}, FP32, ${code("classify")} only`,
    "",
    "## Matrix: models by corpora",
    "",
    table(
      ["Model", "Corpus", "Requests", "Exact matches", "Values compared", "Synthetic zero", "Raw max abs drift", "Result"],
      receipt.results.map((r) => [
        `[${code(r.model_id)}](${BASE}/models/${r.model_id}/)`,
        code(r.corpus),
        String(r.requests),
        `${r.exact_semantic_matches}/${r.requests}`,
        String(r.compared_values),
        String(r.synthetic_zero_diffs),
        fmtDrift(r.max_raw_abs_diff),
        yesNo(r.passed),
      ]),
      [null, null, "r", "r", "r", "r", "r", null],
    ),
    "",
    "## Matrix: models by corpora by mode",
    "",
    `Max raw absolute drift per execution mode. Each cell is ${code("max drift (exact matches/requests)")}.`,
    "",
    table(
      ["Model", "Corpus", ...MODES],
      reports.map((r) => [
        code(r.model_id),
        code(r.corpus_id),
        ...MODES.map((m) => {
          const x = r.modes[m];
          return x ? `${fmtDrift(x.numeric.max_raw_abs_diff)} (${x.exact_semantic_matches}/${x.requests})` : "n/a";
        }),
      ]),
    ),
    "",
    "Modes: `singleton` (each case once), `repeated` (each case twice), `batch` (compatible requests in one " +
      "`/v1/batch`), `concurrent-group` (concurrent compatible requests), `concurrent-all` (all cases concurrently).",
    "",
    "## Receipt",
    "",
    `The receipt [${code(RECEIPT_PATH)}](${repoBlob(RECEIPT_PATH)}) records one real evaluation run. ` +
      `CI checks that it is coherent and not replayed; it never runs models. See the ` +
      `[evaluation receipt section](${BASE}/docs/onnx-runtime/#evaluation-receipt) for how it is produced and verified.`,
    "",
    table(
      ["Field", "Value"],
      [
        ["Receipt file SHA-256", code(receiptSha)],
        ["Run id", code(receipt.run_id)],
        ["Created", receipt.created_at],
        ["Runner version", receipt.runner.version],
        ["Runner identity SHA-256", code(receipt.runner.identity_sha256)],
        ["Previous receipt SHA-256", receipt.previous_receipt_sha256 ? code(receipt.previous_receipt_sha256) : "none (first receipt)"],
        ["Tolerance", fmtDrift(receipt.tolerance)],
        ["Host", `${receipt.host.cpu_model}, ${receipt.host.cpu_cores} cores, ${receipt.host.platform}-${receipt.host.arch}`],
        ["Node.js", receipt.host.node],
        ["onnxruntime-node", receipt.host.onnxruntime_node],
        ["Tokenizers", runtime["@huggingface/tokenizers"]],
        [
          "PyTorch reference",
          `Python ${ref.versions.python}, gliner2 ${ref.versions.gliner2}, torch ${ref.versions.torch}, transformers ${ref.versions.transformers} (${ref.device}/${ref.precision})`,
        ],
      ],
    ),
    "",
    "### Models in the receipt",
    "",
    table(
      ["Model", "Repository", "Revision", "Manifest SHA-256", "Template SHA-256"],
      receipt.models.map((m) => [
        `[${code(m.model_id)}](${BASE}/models/${m.model_id}/)`,
        m.repository,
        code(m.revision.slice(0, 12)),
        code(m.manifest_sha256),
        code(m.template_sha256),
      ]),
    ),
    "",
    "### Corpora and goldens",
    "",
    table(
      ["Kind", "Path", "SHA-256"],
      [
        ...receipt.corpora.map((c) => ["corpus", `[${code(c.path)}](${repoBlob(c.path)})`, code(c.sha256)]),
        ...receipt.goldens.map((g) => ["golden", `[${code(g.path)}](${repoBlob(g.path)})`, code(g.sha256)]),
      ],
    ),
    "",
    "## Native PyTorch profiles",
    "",
    "The Python server also reports measured score drift for its PyTorch profiles against the cpu/fp32 direct " +
      "official oracle. These are small characterizations, not universal accuracy guarantees.",
    "",
    table(
      ["Model", "Profile", "Max abs score drift", "Order", "Scope"],
      native.map((n) => [code(n.modelId), n.profile, fmtDrift(n.drift), n.order, n.scope]),
    ),
    "",
    `Raw runs and methodology: [Benchmarks](${BASE}/docs/benchmarks/) and the ` +
      `[parity report](${repoBlob(`${PARITY_DIR}/REPORT.md`)}).`,
    "",
    "## Method",
    "",
    "- Corpora `english-v1` and `multilingual-v1`: 14 cases each (10 successful classifications and 4 expected " +
      "structured `422` errors), synthetic text only. Each model runs against both corpora; cross-language " +
      "application is a parity probe, not a language-quality claim.",
    "- Authoritative comparison: exact equality of the semantic result (labels, ordering, cardinality, empty " +
      "results, decoder metadata, error code, message, retryable flag, HTTP status) and of token usage.",
    "- Numeric comparison: raw confidence and probability values compared without rounding against an absolute " +
      "tolerance that gates acceptance. \"Synthetic zero\" is a diagnostic view only.",
    "",
  );

  return (
    frontmatter({
      title: "Evaluation",
      description: "ONNX Runtime versus PyTorch parity matrix and the evaluation receipt.",
    }) + lines.join("\n")
  );
}

// ---------------------------------------------------------------- repo docs

const SCHEME = /^[a-z][a-z0-9+.-]*:/i;

/** Rewrites one link target from a doc under docs/ so it works on the site. */
export function rewriteTarget(target, { root, docNames, sourceDir = "docs" }) {
  if (target === "" || target.startsWith("#") || target.startsWith("/") || SCHEME.test(target)) return target;
  const hashAt = target.indexOf("#");
  const path = hashAt < 0 ? target : target.slice(0, hashAt);
  const hash = hashAt < 0 ? "" : target.slice(hashAt);
  const resolved = posix.normalize(posix.join(sourceDir, path));
  if (resolved.startsWith("..")) return target;
  const m = resolved.match(/^docs\/([^/]+)\.md$/);
  if (m && docNames.has(m[1])) return `${BASE}/docs/${m[1]}/${hash}`;
  const abs = join(root, resolved);
  const isDir = existsSync(abs) && statSync(abs).isDirectory();
  return `${REPO_URL}/${isDir ? "tree" : "blob"}/main/${resolved.replace(/\/$/, "")}${hash}`;
}

export function rewriteLinks(markdown, ctx) {
  let fenced = false;
  return markdown
    .split("\n")
    .map((line) => {
      if (/^\s*(```|~~~)/.test(line)) {
        fenced = !fenced;
        return line;
      }
      if (fenced) return line;
      const inline = line.replace(/(\]\()([^)\s]*)((?:\s+"[^"]*")?\))/g, (_, a, target, c) => `${a}${rewriteTarget(target, ctx)}${c}`);
      return inline.replace(/^(\s{0,3}\[[^\]]+\]:\s*)(\S+)/, (_, a, target) => `${a}${rewriteTarget(target, ctx)}`);
    })
    .join("\n");
}

function docPages(root) {
  const dir = join(root, "docs");
  const names = readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .map((f) => f.replace(/\.md$/, ""));
  const docNames = new Set(names);
  const ordered = [...names].sort((a, b) => {
    const ia = DOC_ORDER.indexOf(a);
    const ib = DOC_ORDER.indexOf(b);
    return (ia < 0 ? 999 : ia) - (ib < 0 ? 999 : ib) || a.localeCompare(b);
  });
  return ordered.map((name, i) => {
    const source = readFileSync(join(dir, `${name}.md`), "utf8");
    const heading = source.match(/^#\s+(.+?)\s*$/m);
    const title = heading ? heading[1] : name;
    const body = heading ? source.replace(heading[0], "").replace(/^\s*\n/, "") : source;
    const text =
      frontmatter({
        title,
        editUrl: `${REPO_URL}/edit/main/docs/${name}.md`,
        sidebar: { order: i },
      }) + rewriteLinks(body, { root, docNames });
    return { name, text };
  });
}

// ---------------------------------------------------------------- main

export function generate(root, outDir) {
  const catalog = parseCatalog(readFileSync(join(root, "packages/onnx-runner/src/catalog.ts"), "utf8"));
  const native = parseNativeEvidence(readFileSync(join(root, "src/gliner_runner/model_inventory.py"), "utf8"));
  const templates = loadTemplates(root);
  const receipt = readJson(root, RECEIPT_PATH);
  const reports = loadReports(root);

  const order = new Map(catalog.map((m, i) => [m.modelId, i + 1]));
  for (const model of catalog) {
    const entry = templates.get(model.modelId);
    const receiptModel = receipt.models.find((m) => m.model_id === model.modelId);
    if (!entry || !receiptModel) throw new Error(`no template or receipt entry for ${model.modelId}`);
    for (const [label, rev, repo, count] of [
      ["manifest", entry.manifest.revision, entry.manifest.repository, entry.manifest.parameter_count],
      ["receipt", receiptModel.revision, receiptModel.repository, model.parameterCount],
    ]) {
      if (rev !== model.pinnedRevision || repo !== model.repository || count !== model.parameterCount) {
        throw new Error(`${model.modelId}: ${label} disagrees with catalog.ts (revision, repository, or parameters)`);
      }
    }
    if (!reports.some((r) => r.model_id === model.modelId)) throw new Error(`no parity reports for ${model.modelId}`);
  }
  for (const r of reports) {
    const row = receipt.results.find((x) => x.model_id === r.model_id && x.corpus === r.corpus_id);
    if (!row || row.max_raw_abs_diff !== r.overall.max_raw_abs_diff) {
      console.warn(`warning: receipt and report disagree for ${r.model_id}/${r.corpus_id}`);
    }
  }

  for (const sub of ["models", "docs", "evaluation"]) rmSync(join(outDir, sub), { recursive: true, force: true });
  const write = (rel, text) => {
    const file = join(outDir, rel);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, text);
  };

  write("models/index.md", modelsIndex({ models: catalog, templates, receipt }));
  for (const model of catalog) {
    write(
      `models/${model.modelId}.md`,
      modelPage({ model, entry: templates.get(model.modelId), receipt, reports, native, order: order.get(model.modelId) }),
    );
  }
  write("evaluation/index.md", evaluationPage({ root, models: catalog, receipt, reports, native }));
  const docs = docPages(root);
  for (const { name, text } of docs) write(`docs/${name}.md`, text);
  return { models: catalog.length, docs: docs.length };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const here = dirname(fileURLToPath(import.meta.url));
  const root = resolve(here, "../..");
  const result = generate(root, resolve(here, "../src/content/docs"));
  console.log(`generated ${result.models} model pages, the evaluation page, and ${result.docs} docs pages`);
}
