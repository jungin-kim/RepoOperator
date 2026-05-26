#!/usr/bin/env node
/**
 * Prepack script: bundles apps/local-worker and apps/web into packages/cli/runtime/
 * so npm-published users get the full runtime sources inside the package.
 *
 * Excluded: node_modules, .venv, .next, __pycache__, .git, *.egg-info, *.tgz,
 * test results, Playwright reports, and conflict/safe-write duplicate files.
 */

const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");

const PACKAGE_ROOT = path.resolve(__dirname, "..");
const MONO_ROOT = path.resolve(PACKAGE_ROOT, "../..");
const RUNTIME_DEST = path.join(PACKAGE_ROOT, "runtime");
const HYGIENE_ROOTS = [
  path.join(MONO_ROOT, "apps"),
  path.join(MONO_ROOT, "packages"),
  path.join(MONO_ROOT, "docs"),
  path.join(MONO_ROOT, "scripts"),
];

const SOURCES = [
  {
    src: path.join(MONO_ROOT, "apps", "local-worker"),
    dest: path.join(RUNTIME_DEST, "local-worker"),
    name: "local-worker",
  },
  {
    src: path.join(MONO_ROOT, "apps", "web"),
    dest: path.join(RUNTIME_DEST, "web"),
    name: "web",
  },
];

const EXCLUDE_DIRS = new Set([
  "node_modules",
  ".venv",
  ".git",
  "__pycache__",
  ".next",
  ".mypy_cache",
  ".pytest_cache",
  "dist",
  "build",
  "test-results",
  "playwright-report",
]);

const EXCLUDE_PATTERNS = [
  /\.egg-info$/,
  /\.tgz$/,
  /\.pyc$/,
  /\.pyo$/,
  /\.DS_Store$/,
  /^tsconfig\.tsbuildinfo$/,
  /(?:\s+\d+|\s+copy)(?=\.[^.]+$)/i,
  /\.(?:bak|orig)$/i,
];

const CONFLICT_PATTERNS = [
  /(?:\s+\d+|\s+copy)(?=\.[^.]+$)/i,
  /\.(?:bak|orig)$/i,
];

async function copyFiltered(src, dest) {
  const stat = await fsp.stat(src);

  if (stat.isDirectory()) {
    const base = path.basename(src);
    if (EXCLUDE_DIRS.has(base) || EXCLUDE_PATTERNS.some((re) => re.test(base))) {
      return;
    }
    await fsp.mkdir(dest, { recursive: true });
    const entries = await fsp.readdir(src);
    for (const entry of entries) {
      await copyFiltered(path.join(src, entry), path.join(dest, entry));
    }
  } else {
    const base = path.basename(src);
    if (EXCLUDE_PATTERNS.some((re) => re.test(base))) {
      return;
    }
    await fsp.mkdir(path.dirname(dest), { recursive: true });
    await fsp.copyFile(src, dest);
  }
}

async function findConflictArtifacts(src, found = []) {
  const stat = await fsp.stat(src);
  const base = path.basename(src);
  if (EXCLUDE_DIRS.has(base)) {
    return found;
  }
  if (stat.isDirectory()) {
    const entries = await fsp.readdir(src);
    for (const entry of entries) {
      await findConflictArtifacts(path.join(src, entry), found);
    }
    return found;
  }
  if (isConflictArtifact(base)) {
    found.push(path.relative(MONO_ROOT, src));
  }
  return found;
}

function isConflictArtifact(base) {
  return CONFLICT_PATTERNS.some((re) => re.test(base));
}

async function assertWorkspaceHygiene() {
  const conflicts = [];
  for (const root of HYGIENE_ROOTS) {
    if (!fs.existsSync(root)) continue;
    await findConflictArtifacts(root, conflicts);
  }
  if (!conflicts.length) return;
  console.error("ERROR: duplicate/conflict artifacts were found and will not be packed:");
  for (const item of conflicts.sort()) {
    console.error(`  ${item}`);
  }
  console.error("Remove these files or run scripts/check-workspace-hygiene.py for diagnostics.");
  process.exit(1);
}

async function main() {
  console.log("prepare-runtime: bundling apps into packages/cli/runtime/");

  await assertWorkspaceHygiene();
  await fsp.rm(RUNTIME_DEST, { recursive: true, force: true });
  await fsp.mkdir(RUNTIME_DEST, { recursive: true });

  for (const { src, dest, name } of SOURCES) {
    if (!fs.existsSync(src)) {
      console.error(`ERROR: source not found: ${src}`);
      console.error(`  Make sure you run 'npm pack' from the monorepo root (not a standalone checkout).`);
      process.exit(1);
    }
    await copyFiltered(src, dest);
    console.log(`  bundled ${name} -> ${path.relative(PACKAGE_ROOT, dest)}`);
  }

  console.log("prepare-runtime: done");
}

main().catch((err) => {
  console.error("prepare-runtime failed:", err.message);
  process.exit(1);
});
