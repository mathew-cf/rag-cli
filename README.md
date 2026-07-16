# rag-cli

Local-first semantic search over your files. Index a directory and query it
with natural language — all embeddings are computed on your machine using
[ONNX Runtime](https://onnxruntime.ai/), with no external API calls.

## Quick start

Install (choose one):

```bash
# Via npm (prebuilt binary, no Rust toolchain needed)
npm install -g @mathew-cf/rag-cli

# Via Cargo (builds from source)
cargo install rag-cli

# From the repo
cargo install --path .
```

Use:

```bash
# Index a directory
rag index ./my-project

# Search it
rag search "how does authentication work"

# Pre-cache the embedding model (optional — makes first use faster)
rag download
```

### Supported npm platforms

Prebuilt binaries are published for macOS ARM64, macOS x86_64, Linux x86_64, Linux ARM64, and Windows x86_64. On any other platform, fall back to `cargo install rag-cli`.

## Commands

### `rag index [path]`

With a `<path>`, recursively discovers text files under it, chunks them,
computes embeddings, and writes the index to disk. With **no path**, reads a
[config file](#config-file-ragtoml) and builds every index it declares.

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output <dir>` | `.rag` | Where to store the index |
| `-m, --model <id>` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `--chunk-size <n>` | `512` | Chunk size in characters |
| `--chunk-overlap <n>` | `64` | Overlap between consecutive chunks |
| `--ext <list>` | — | Extra file extensions to index, beyond the built-in allowlist (comma-separated or repeated), e.g. `--ext mdx,rst` |
| `--exclude <list>` | — | Directory/file specs to skip (comma-separated or repeated) |
| `--include <list>` | — | Normally-skipped directories to index anyway, e.g. `--include dist` |
| `-c, --config <file>` | `rag.toml` | Config file to build from when no path is given |
| `--only <list>` | — | In config mode, build only these named indexes (skip slow ones you didn't change) |

An `--exclude`/`--include` spec without a `/` matches any path component by
name (e.g. `changelog` skips every `changelog/` directory). A spec containing a
`/` is treated as a relative-path prefix (e.g. `src/content/changelog` skips
only that one). `--include` re-enables directories that are skipped by default
(`node_modules`, `dist`, `build`, `vendor`, `target`, hidden dirs, …).

Re-running `rag index` on the same directory performs **incremental indexing** —
only changed or new files are re-embedded. File changes are detected using
[blake3](https://github.com/BLAKE3-team/BLAKE3) content hashes. If you change
the model or chunk settings the entire index is rebuilt automatically.

#### Config file (`rag.toml`)

To build a whole set of indexes with one command — instead of a shell script
that calls `rag index` once per directory — declare them in a `rag.toml` and run
`rag index` with no path. Global keys at the top are defaults; each `[[index]]`
may override them. Paths and output dirs are resolved relative to the config
file.

```toml
# Global defaults (all optional)
model = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 512
chunk_overlap = 64

[[index]]
name = "docs"                  # output defaults to .rag/<name>
path = "docs/src/content"      # relative to this config file
extensions = ["mdx"]           # extra extensions beyond the built-in allowlist
exclude = ["changelog"]        # skip these dirs/files
# include = ["dist"]           # re-include normally-skipped dirs
# output = ".rag/docs"         # override the default output dir

[[index]]
name = "reference"
path = "reference/md"
```

`rag index` looks for `rag.toml` then `.rag.toml` in the current directory, or
use `--config <file>`. To rebuild just some of the declared indexes, use
`--only`: `rag index --only docs`.

### `rag search <query>`

Embeds your query and returns the most similar chunks by cosine similarity.

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --index <dir>` | `.rag` | Index directory to search |
| `-k, --top-k <n>` | `5` | Number of results |
| `-m, --model <id>` | *(from index)* | Override embedding model |
| `--full` | off | Show full chunk text instead of truncated preview |
| `--json` | off | Output compact JSON (for piping to LLMs or other tools) |

### `rag info`

Prints index metadata: model, chunk count, source file count, index size, etc.

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --index <dir>` | `.rag` | Index directory to inspect |

## Hardware acceleration

Inference runs on [ONNX Runtime](https://onnxruntime.ai/) using the CPU
execution provider with an int8-quantized model. For a model this small,
int8-on-CPU is the fastest option available — roughly 2x the throughput of
fp32 at ~99.7% of the quality, and faster than GPU execution (small models are
dominated by per-kernel dispatch overhead, and int8 has no GPU fast path). No
GPU, CUDA toolkit, or extra flags are needed.

The quantized weights are architecture-specific and selected automatically:
arm64 (Apple Silicon, ARM servers) and x86-64 (AVX2) each get a tuned build.
The ONNX Runtime library is statically linked into the binary, so there's
nothing to install separately.

## Model management

rag-cli downloads model weights directly from HuggingFace over HTTPS on first
use, then caches them locally in the standard HuggingFace Hub layout
(`~/.cache/huggingface/hub`). We do this instead of using the `hf-hub` crate because `hf-hub` uses a bundled
certificate store and does not respect the system root CA certificates. That
makes it fail in environments with custom CA roots (corporate proxies, internal
TLS inspection, etc.). rag-cli uses `native-tls`, which delegates to the OS certificate store,
so it works in those environments without extra configuration.

You can control the cache location:

```bash
# Via flag
rag --cache-dir /path/to/cache index ./docs

# Via environment variable
export RAG_CACHE_DIR=/path/to/cache
rag index ./docs

# Or use HF_HOME (standard HuggingFace convention)
export HF_HOME=/path/to/hf
rag index ./docs
```

## Supported file types

rag-cli is aimed at prose, docs, and config — not source code. It indexes:

- **Docs / prose**: `.md`, `.txt`, `.tex`, `.org`, `.rst`
- **Data**: `.csv`, `.tsv`, `.log`
- **Config**: `.toml`, `.conf`, `.cfg`, `.ini`, `.env`, `.tf`, `.hcl`, `.nix`
- **Markup**: `.xml`, `.html`
- **Schemas**: `.sql`, `.proto`, `.graphql`
- **Build**: `Dockerfile`, `Makefile`, `.cmake`

Files like `Makefile`, `Dockerfile`, `LICENSE`, `README`, and `.gitignore` are
recognized by name.

Programming language sources (`.rs`, `.py`, `.ts`, `.go`, `.js`, shell scripts,
etc.), `.json`, `.yaml`/`.yml`, and `.css`/`.scss` are intentionally **not**
indexed — they tend to drown out useful matches with boilerplate. Index your
code with a code-aware tool instead.

To index an extension that isn't in the allowlist (for example `.mdx`), add it
with `--ext` (or an entry's `extensions` in `rag.toml`): `rag index ./docs --ext mdx`.

Hidden directories, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`,
and `build` are skipped automatically. Skip more with `--exclude`, or force a
skipped directory back in with `--include`.

## How it works

1. **Discover** text files recursively, skipping binary and vendored content
2. **Chunk** each file into overlapping segments (~512 chars), breaking at
   paragraph or line boundaries when possible
3. **Embed** chunks in batches of 64 using a BERT model
   (`all-MiniLM-L6-v2`, 384-dimensional vectors) with mean pooling and L2
   normalization
4. **Store** the index as a bincode-serialized file (`.rag/index.bin`) with
   JSON metadata (`.rag/meta.json`)
5. **Search** by embedding the query with the same model and ranking chunks
   by cosine similarity (dot product on normalized vectors)

## License

Apache-2.0
