# Publishing rag-cli to npm

Binary distribution uses the standard "Rust CLI on npm" pattern: a meta
package with a JS shim resolves one of several platform-specific
subpackages at runtime. See `npm/rag-cli/src/platform.js` for the list
of supported platforms and `scripts/prepare-npm.js` for how the release
workflow stitches everything together.

## Package layout

```
@mathew-cf/rag-cli                      meta — user-facing; has the shim
├─ optionalDependencies
│  ├─ @mathew-cf/rag-cli-darwin-arm64   prebuilt, os+cpu restricted
│  ├─ @mathew-cf/rag-cli-darwin-x64
│  ├─ @mathew-cf/rag-cli-linux-x64
│  └─ @mathew-cf/rag-cli-linux-arm64
```

On install, npm evaluates `os` and `cpu` and only fetches the matching
platform package; the shim in `@mathew-cf/rag-cli/bin/rag.js` then
`require.resolve()`s its binary and execs it.

## One-time bootstrap (each new package)

npm's **Trusted Publishing** requires the package to exist on npm
before you can attach a trusted publisher. So the very first publish of
each package must be done manually from a local machine with a
long-lived `npm publish` credential. Afterwards, CI takes over via
OIDC and the token can be revoked.

### Step 1: local bootstrap publish

Run once for each of the 5 packages:

```bash
# Fill placeholders with the current version. Use a pre-release tag so
# the bootstrap commit doesn't show up as "latest" on npm.
node scripts/prepare-npm.js --version=0.3.2-bootstrap

# Stage real binaries into each platform package. `cargo build --release`
# only produces the host triple; the other three platforms can use a
# dummy binary for the bootstrap publish because no one will ever
# install 0.3.2-bootstrap anyway — they're just placeholder artefacts.
for plat in darwin-arm64 darwin-x64 linux-x64 linux-arm64; do
  mkdir -p "npm/rag-cli-$plat/bin"
  echo '#!/bin/sh' > "npm/rag-cli-$plat/bin/rag"
  echo 'echo bootstrap-only placeholder; exit 1' >> "npm/rag-cli-$plat/bin/rag"
  chmod +x "npm/rag-cli-$plat/bin/rag"
done

# Publish each package. `--tag bootstrap` keeps these versions from
# becoming the default "latest" dist-tag.
npm login # one-time
for pkg in rag-cli-darwin-arm64 rag-cli-darwin-x64 rag-cli-linux-x64 rag-cli-linux-arm64 rag-cli; do
  (cd "npm/$pkg" && npm publish --access public --tag bootstrap)
done

# Revert local file tampering.
git restore npm/
```

### Step 2: configure Trusted Publishing on npm

For each of the 5 packages (meta + 4 platforms) on https://www.npmjs.com :

1. Open the package page → **Settings** → **Publishing access**
2. Click **Add trusted publisher** → **GitHub Actions**
3. Fill in:
   - **Organization or user**: `mathew-cf`
   - **Repository**: `rag-cli`
   - **Workflow filename**: `release.yml`
   - **Environment name**: `release` (must match the `environment:` key in the workflow)
4. Save.

After this is done for all five, subsequent `npm publish` calls from
`.github/workflows/release.yml` will use OIDC — no credential needed.

### Step 3: revoke the bootstrap token

Delete the npm CLI token you used for the manual publish from
https://www.npmjs.com/settings/USERNAME/tokens. From this point
forward, all publishing happens via trusted publishing.

## Ongoing releases

Trigger the `CI / Release` workflow with a version bump choice (patch /
minor / major). The workflow will:

1. Build the Rust binary for each of 4 platforms
2. Upload each binary as a GitHub Actions artifact
3. Run `scripts/prepare-npm.js` to version and stage the packages
4. `npm publish --access public --provenance` each one via OIDC

The `--provenance` flag attaches a signed attestation linking the
published tarball to the exact GitHub Actions run that built it,
visible as a badge on each package's npm page.

## Local dry-run

You can exercise most of the pipeline without touching npm:

```bash
# Build a host binary
cargo build --release

# Stage it as all four platforms (fine for a smoke test)
mkdir -p /tmp/rag-artifacts
for t in aarch64-apple-darwin x86_64-apple-darwin \
         x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu; do
  mkdir -p "/tmp/rag-artifacts/rag-$t"
  cp target/release/rag "/tmp/rag-artifacts/rag-$t/rag"
done

# Fill placeholders + stage binaries
node scripts/prepare-npm.js --version=0.0.0-dryrun \
  --artifacts=/tmp/rag-artifacts

# Inspect what would be published
(cd npm/rag-cli && npm pack --dry-run)
(cd npm/rag-cli-darwin-arm64 && npm pack --dry-run)

# Reset
git restore npm/
```

## Running the test suite

```bash
npm test         # node --test test/npm-*.test.js test/prepare-*.test.js
actionlint .github/workflows/release.yml
cargo test       # Rust tests
```
