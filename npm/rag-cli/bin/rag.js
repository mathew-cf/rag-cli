#!/usr/bin/env node
/**
 * Meta-package shim for @mathew-cf/rag-cli.
 *
 * Resolves the platform-specific subpackage at runtime and execs its
 * `rag` binary. Signals are forwarded so Ctrl-C works; exit codes are
 * propagated so shell scripts can branch on success/failure.
 *
 * The platform detection logic lives in ../src/platform.js so it's
 * testable without spawning a process.
 */

"use strict";

const { spawn } = require("node:child_process");
const {
  binaryFilename,
  isSupported,
  platformKey,
  subpackageName,
  unsupportedMessage,
} = require("../src/platform");

const key = platformKey(process.platform, process.arch);

if (!isSupported(key)) {
  process.stderr.write(`${unsupportedMessage(key)}\n`);
  process.exit(1);
}

let binaryPath;
try {
  binaryPath = require.resolve(
    `${subpackageName(key)}/bin/${binaryFilename(process.platform)}`,
  );
} catch {
  process.stderr.write(
    `@mathew-cf/rag-cli: platform package ${subpackageName(key)} is not installed.\n` +
      `This usually means npm skipped optionalDependencies. Try:\n` +
      `  npm install @mathew-cf/rag-cli --force\n`,
  );
  process.exit(1);
}

const child = spawn(binaryPath, process.argv.slice(2), {
  stdio: "inherit",
  windowsHide: false,
});

// Forward terminal signals to the child. Without this, ^C gets eaten by
// the Node wrapper and the spawned binary keeps running.
const forwardedSignals = ["SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT"];
for (const sig of forwardedSignals) {
  process.on(sig, () => {
    if (!child.killed) child.kill(sig);
  });
}

child.on("error", (err) => {
  process.stderr.write(`@mathew-cf/rag-cli: failed to exec binary: ${err.message}\n`);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (signal) {
    // Re-raise so the shell sees the canonical exit-via-signal (128+n).
    process.kill(process.pid, signal);
  } else {
    process.exit(code ?? 1);
  }
});
