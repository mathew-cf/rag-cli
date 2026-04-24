/**
 * Tests for npm/rag-cli/src/platform.js — the shim's platform logic.
 *
 * Uses node:test (built into Node 18+). Run with `node --test`.
 */

"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");

const {
  SUPPORTED_PLATFORMS,
  binaryFilename,
  isSupported,
  platformKey,
  subpackageName,
  unsupportedMessage,
} = require("../npm/rag-cli/src/platform");

test("platformKey composes platform and arch with a hyphen", () => {
  assert.equal(platformKey("darwin", "arm64"), "darwin-arm64");
  assert.equal(platformKey("linux", "x64"), "linux-x64");
});

test("isSupported recognises every published platform", () => {
  for (const key of SUPPORTED_PLATFORMS) {
    assert.equal(isSupported(key), true, `expected ${key} to be supported`);
  }
});

test("isSupported rejects unknown platforms", () => {
  assert.equal(isSupported("freebsd-x64"), false);
  assert.equal(isSupported("linux-riscv64"), false);
  assert.equal(isSupported("win32-x64"), false); // not yet published
  assert.equal(isSupported(""), false);
});

test("SUPPORTED_PLATFORMS matches the current publish matrix", () => {
  // Pin the expected list so a silent change to platform.js surfaces in
  // code review. Update both together.
  assert.deepEqual([...SUPPORTED_PLATFORMS], [
    "darwin-arm64",
    "darwin-x64",
    "linux-x64",
    "linux-arm64",
  ]);
});

test("SUPPORTED_PLATFORMS is frozen (tamper-proof)", () => {
  assert.throws(() => {
    /** @type {any} */ (SUPPORTED_PLATFORMS).push("freebsd-x64");
  });
});

test("subpackageName uses the @mathew-cf scope", () => {
  assert.equal(
    subpackageName("darwin-arm64"),
    "@mathew-cf/rag-cli-darwin-arm64",
  );
  assert.equal(subpackageName("linux-x64"), "@mathew-cf/rag-cli-linux-x64");
});

test("binaryFilename picks .exe for win32 and rag for Unix", () => {
  assert.equal(binaryFilename("darwin"), "rag");
  assert.equal(binaryFilename("linux"), "rag");
  assert.equal(binaryFilename("win32"), "rag.exe");
});

test("unsupportedMessage lists every published platform", () => {
  const msg = unsupportedMessage("freebsd-x64");
  assert.ok(msg.includes("freebsd-x64"), "should name the unsupported platform");
  for (const key of SUPPORTED_PLATFORMS) {
    assert.ok(msg.includes(key), `should list ${key} among supported platforms`);
  }
  assert.ok(
    msg.includes("cargo install rag-cli"),
    "should suggest the cargo fallback",
  );
});
