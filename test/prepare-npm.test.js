/**
 * Tests for scripts/prepare-npm.js.
 *
 * Each test operates on a temporary directory so the real `npm/` tree
 * never gets mutated. Uses node:test (built into Node 18+).
 */

"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  PLATFORM_TARGETS,
  applyVersion,
  listPackageJsons,
  parseArgs,
  parseCargoVersion,
  stageBinaries,
  writeVersion,
} = require("../scripts/prepare-npm");

function makeTempDir(prefix = "rag-npm-test-") {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix));
}

function writeJson(filePath, obj) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(obj, null, 2) + "\n");
}

test("parseCargoVersion extracts the first version from Cargo.toml", () => {
  const toml = `[package]
name = "rag-cli"
version = "1.2.3"
edition = "2021"
`;
  assert.equal(parseCargoVersion(toml), "1.2.3");
});

test("parseCargoVersion prefers the first declaration when there are multiple", () => {
  // `rag-cli-cuda/Cargo.toml` repeats the version; the root Cargo.toml
  // is authoritative — we assert we pick the first line, as the real
  // prepare script only reads the top-level file.
  const toml = `[package]
version = "0.5.0"
[other]
version = "9.9.9"
`;
  assert.equal(parseCargoVersion(toml), "0.5.0");
});

test("parseCargoVersion throws when no version is present", () => {
  assert.throws(() => parseCargoVersion("[package]\nname = \"rag-cli\"\n"));
});

test("applyVersion replaces every placeholder occurrence", () => {
  const input = `{"version":"0.0.0-PLACEHOLDER","peer":"0.0.0-PLACEHOLDER"}`;
  const output = applyVersion(input, "1.2.3");
  assert.equal(output, `{"version":"1.2.3","peer":"1.2.3"}`);
  assert.ok(!output.includes("PLACEHOLDER"));
});

test("applyVersion is a no-op when there are no placeholders", () => {
  const input = `{"version":"0.1.0"}`;
  assert.equal(applyVersion(input, "9.9.9"), input);
});

test("listPackageJsons finds direct children only", () => {
  const root = makeTempDir();
  writeJson(path.join(root, "meta", "package.json"), { name: "meta" });
  writeJson(path.join(root, "plat-a", "package.json"), { name: "plat-a" });
  // Nested directory — should NOT be included.
  writeJson(
    path.join(root, "plat-b", "sub", "package.json"),
    { name: "plat-b-sub" },
  );
  // Directory without a package.json — should be skipped.
  fs.mkdirSync(path.join(root, "empty"));

  const found = listPackageJsons(root).sort();
  assert.deepEqual(
    found,
    [
      path.join(root, "meta", "package.json"),
      path.join(root, "plat-a", "package.json"),
    ].sort(),
  );
});

test("writeVersion rewrites every discovered package.json", () => {
  const root = makeTempDir();
  writeJson(path.join(root, "meta", "package.json"), {
    name: "meta",
    version: "0.0.0-PLACEHOLDER",
    optionalDependencies: { "plat-a": "0.0.0-PLACEHOLDER" },
  });
  writeJson(path.join(root, "plat-a", "package.json"), {
    name: "plat-a",
    version: "0.0.0-PLACEHOLDER",
  });

  writeVersion("1.2.3", root);

  const meta = JSON.parse(
    fs.readFileSync(path.join(root, "meta", "package.json"), "utf8"),
  );
  const plat = JSON.parse(
    fs.readFileSync(path.join(root, "plat-a", "package.json"), "utf8"),
  );
  assert.equal(meta.version, "1.2.3");
  assert.equal(meta.optionalDependencies["plat-a"], "1.2.3");
  assert.equal(plat.version, "1.2.3");
});

test("stageBinaries copies the binary into every platform's bin dir", () => {
  const root = makeTempDir();
  const artifacts = makeTempDir();

  // Seed a package.json for each platform so stageBinaries has a
  // destination structure to populate.
  for (const platKey of Object.keys(PLATFORM_TARGETS)) {
    writeJson(path.join(root, `rag-cli-${platKey}`, "package.json"), {
      name: `plat-${platKey}`,
    });
  }

  // Seed synthetic binaries at the expected layout.
  for (const target of Object.values(PLATFORM_TARGETS)) {
    const src = path.join(artifacts, `rag-${target}`, "rag");
    fs.mkdirSync(path.dirname(src), { recursive: true });
    fs.writeFileSync(src, `#!/bin/sh\necho ${target}\n`);
  }

  const staged = stageBinaries(artifacts, { npmRoot: root });
  assert.equal(staged.length, Object.keys(PLATFORM_TARGETS).length);

  for (const platKey of Object.keys(PLATFORM_TARGETS)) {
    const dst = path.join(root, `rag-cli-${platKey}`, "bin", "rag");
    assert.ok(fs.existsSync(dst), `${platKey} binary should exist at ${dst}`);
    const mode = fs.statSync(dst).mode & 0o777;
    assert.ok(
      (mode & 0o100) !== 0,
      `${platKey} binary should be executable (got mode ${mode.toString(8)})`,
    );
  }
});

test("stageBinaries throws if a binary is missing", () => {
  const root = makeTempDir();
  const artifacts = makeTempDir();
  writeJson(path.join(root, "rag-cli-darwin-arm64", "package.json"), {
    name: "p",
  });
  // No binary seeded → expect an explanatory error.
  assert.throws(
    () =>
      stageBinaries(artifacts, {
        npmRoot: root,
        platforms: { "darwin-arm64": "aarch64-apple-darwin" },
      }),
    /Binary not found for darwin-arm64/,
  );
});

test("parseArgs handles the supported flags", () => {
  assert.deepEqual(
    parseArgs(["--version=1.2.3", "--artifacts=/tmp/a"]),
    { version: "1.2.3", artifacts: "/tmp/a" },
  );
  assert.deepEqual(parseArgs([]), { version: null, artifacts: null });
});
