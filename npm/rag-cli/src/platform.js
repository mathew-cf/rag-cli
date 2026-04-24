/**
 * Platform → subpackage mapping.
 *
 * Kept in its own module (and written as plain CommonJS) so:
 *   - the shim at `bin/rag.js` can `require()` it at runtime
 *   - unit tests can import it and call `platformKey()` with mocked
 *     values without spawning a shell or a child process.
 *
 * Update SUPPORTED_PLATFORMS in lockstep with `npm/rag-cli/package.json`
 * `optionalDependencies` and the release workflow's build matrix.
 */

"use strict";

/**
 * Compose the platform key used in subpackage names.
 * Mirrors Node's `process.platform` + `process.arch` convention.
 * Exported so tests can feed synthetic platform/arch values without
 * mutating process globals.
 */
function platformKey(platform, arch) {
  return `${platform}-${arch}`;
}

/**
 * The set of platforms we publish prebuilt binaries for. Anything
 * outside this set falls back to `cargo install rag-cli`.
 */
const SUPPORTED_PLATFORMS = Object.freeze([
  "darwin-arm64",
  "darwin-x64",
  "linux-x64",
  "linux-arm64",
]);

/**
 * @param {string} key — result of platformKey(platform, arch)
 * @returns {boolean}
 */
function isSupported(key) {
  return SUPPORTED_PLATFORMS.includes(key);
}

/**
 * npm package name for a given platform key.
 * The scope matches the meta package.
 */
function subpackageName(key) {
  return `@mathew-cf/rag-cli-${key}`;
}

/**
 * Name of the binary file inside a subpackage's `bin/` directory.
 * Windows binaries would be `rag.exe`; Unix platforms use `rag`.
 * Currently all supported platforms are Unix — this helper is the
 * single place to update when/if win32 support lands.
 */
function binaryFilename(platform) {
  return platform === "win32" ? "rag.exe" : "rag";
}

/**
 * Build a human-readable diagnostic for unsupported platforms.
 * Kept as a pure function so tests can snapshot the message.
 */
function unsupportedMessage(key) {
  return [
    `@mathew-cf/rag-cli: no prebuilt binary for ${key}.`,
    `Supported platforms: ${SUPPORTED_PLATFORMS.join(", ")}.`,
    ``,
    `Fallback: install from source via Rust:`,
    `  cargo install rag-cli`,
  ].join("\n");
}

module.exports = {
  platformKey,
  isSupported,
  subpackageName,
  binaryFilename,
  unsupportedMessage,
  SUPPORTED_PLATFORMS,
};
