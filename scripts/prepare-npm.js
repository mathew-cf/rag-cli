#!/usr/bin/env node
/**
 * Prepare the npm packages for publishing.
 *
 * Responsibilities:
 *   1. Read the canonical version from Cargo.toml.
 *   2. Replace `0.0.0-PLACEHOLDER` in every npm/**\/package.json with
 *      that version, including the meta package's optionalDependencies.
 *   3. If --artifacts=<dir> is passed, copy each platform's binary from
 *      <dir>/rag-<target>/rag into npm/rag-cli-<platform>-<arch>/bin/rag
 *      and `chmod +x` it.
 *
 * Pure-ish: apart from `--artifacts`, everything is a deterministic
 * rewrite of tracked files. Safe to run locally for dry-runs.
 *
 * Tested by test/prepare-npm.test.js; the pure rewriter is factored into
 * `applyVersion()` and exported for that purpose.
 */

"use strict";

const fs = require("node:fs");
const path = require("node:path");

const REPO_ROOT = path.resolve(__dirname, "..");
const NPM_ROOT = path.join(REPO_ROOT, "npm");
const CARGO_TOML = path.join(REPO_ROOT, "Cargo.toml");

const PLACEHOLDER = "0.0.0-PLACEHOLDER";

/**
 * Platform-key → Rust target triple. The release workflow uses these
 * triples when invoking cargo; keep this map in sync with the build
 * matrix and with npm/rag-cli/src/platform.js SUPPORTED_PLATFORMS.
 */
const PLATFORM_TARGETS = Object.freeze({
  "darwin-arm64": "aarch64-apple-darwin",
  "darwin-x64": "x86_64-apple-darwin",
  "linux-x64": "x86_64-unknown-linux-gnu",
  "linux-arm64": "aarch64-unknown-linux-gnu",
});

/**
 * Extract the first `version = "..."` from a Cargo.toml string.
 * Exported so tests can exercise the parser without touching disk.
 */
function parseCargoVersion(toml) {
  const match = toml.match(/^version\s*=\s*"([^"]+)"/m);
  if (!match) throw new Error("Could not find `version = \"...\"` in Cargo.toml");
  return match[1];
}

/**
 * Replace every PLACEHOLDER with `version` in a package.json string.
 * Returns a new string; the input is unchanged. Tests assert this is
 * a complete replacement (no placeholder remains).
 */
function applyVersion(packageJsonText, version) {
  return packageJsonText.split(PLACEHOLDER).join(version);
}

/**
 * List all package.json paths under `npm/` — the meta package plus
 * each platform stub. Returned in no particular order.
 */
function listPackageJsons(root = NPM_ROOT) {
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((e) => e.isDirectory())
    .map((e) => path.join(root, e.name, "package.json"))
    .filter((p) => fs.existsSync(p));
}

/**
 * Rewrite every package.json under `npm/` to the given version.
 * Returns the list of files touched.
 */
function writeVersion(version, root = NPM_ROOT) {
  const files = listPackageJsons(root);
  for (const file of files) {
    const before = fs.readFileSync(file, "utf8");
    const after = applyVersion(before, version);
    if (before !== after) fs.writeFileSync(file, after);
  }
  return files;
}

/**
 * Copy platform binaries into the matching npm/rag-cli-<plat>/bin/rag.
 *
 * `layout` describes where the binaries live within the artifacts
 * directory — see the release workflow's artifact naming. The default
 * follows the `rag-<target>/rag` convention we emit from the CI build.
 */
function stageBinaries(
  artifactsDir,
  {
    layout = (target) => `rag-${target}/rag`,
    platforms = PLATFORM_TARGETS,
    npmRoot = NPM_ROOT,
  } = {},
) {
  const staged = [];
  for (const [platKey, target] of Object.entries(platforms)) {
    const src = path.join(artifactsDir, layout(target));
    const dst = path.join(npmRoot, `rag-cli-${platKey}`, "bin", "rag");

    if (!fs.existsSync(src)) {
      throw new Error(
        `Binary not found for ${platKey} at ${src}. ` +
          `Did the build-${target} job succeed?`,
      );
    }

    fs.mkdirSync(path.dirname(dst), { recursive: true });
    fs.copyFileSync(src, dst);
    fs.chmodSync(dst, 0o755);
    staged.push({ platform: platKey, src, dst });
  }
  return staged;
}

// -----------------------------------------------------------------
// CLI entry point — only runs when invoked directly (not on require).
// -----------------------------------------------------------------

function parseArgs(argv) {
  const args = { artifacts: null, version: null };
  for (const arg of argv) {
    if (arg.startsWith("--artifacts=")) args.artifacts = arg.slice(12);
    else if (arg.startsWith("--version=")) args.version = arg.slice(10);
  }
  return args;
}

function main(argv) {
  const args = parseArgs(argv);
  const version =
    args.version ?? parseCargoVersion(fs.readFileSync(CARGO_TOML, "utf8"));

  const touched = writeVersion(version);
  process.stdout.write(
    `Set version ${version} in ${touched.length} package.json file(s).\n`,
  );

  if (args.artifacts) {
    const staged = stageBinaries(args.artifacts);
    for (const entry of staged) {
      process.stdout.write(`  staged ${entry.platform} ← ${entry.src}\n`);
    }
  } else {
    process.stdout.write(
      "  (skipping binary staging — pass --artifacts=<dir> to stage)\n",
    );
  }
}

module.exports = {
  PLACEHOLDER,
  PLATFORM_TARGETS,
  applyVersion,
  listPackageJsons,
  parseArgs,
  parseCargoVersion,
  stageBinaries,
  writeVersion,
};

if (require.main === module) {
  try {
    main(process.argv.slice(2));
  } catch (err) {
    process.stderr.write(`prepare-npm.js: ${err.message}\n`);
    process.exit(1);
  }
}
