# rag-cli

Local-first semantic search over your files. Index a directory and query it
with natural language — all embeddings are computed on your machine using
native CoreML or [ONNX Runtime](https://onnxruntime.ai/), with no external API calls.

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
the model or chunk settings — or when a rag-cli upgrade changes the embedding
backend/precision or on-disk format — the entire index is rebuilt automatically.
A no-change run checks the small metadata file and does not load or rewrite the
full index. Changed files reuse embeddings for chunk text already present in the
index.

#### Config file (`rag.toml`)

To build a whole set of indexes with one command — instead of a shell script
that calls `rag index` once per directory — declare them in a `rag.toml` and run
`rag index` with no path. Global keys at the top are defaults; each `[[index]]`
may override them. Paths and output dirs are resolved relative to the config
file. Relative source paths remain relative in `meta.json` (and therefore in
JSON search results), so committed indexes do not contain machine-specific
absolute paths.

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

Indexes can live in the root repository while their source directories are Git
submodules. This keeps generated data out of the submodules and lets each corpus
update independently:

```toml
[[index]]
name = "vendor-docs"
path = "vendor/docs"          # submodule
output = ".rag/vendor-docs"   # root-repository index
```

The same config can be searched as a federation with one query embedding:

```bash
rag search "cache behavior" --config rag.toml
rag search "cache behavior" --config rag.toml --only docs,reference
```

### `rag search <query>`

Embeds your query and returns the most similar chunks by cosine similarity.

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --index <dir>` | `.rag` | Index directory to search; repeat to federate several indexes |
| `-c, --config <file>` | — | Search indexes declared by a `rag.toml` instead of `--index` |
| `--only <list>` | — | With `--config`, search only these named indexes |
| `-k, --top-k <n>` | `5` | Number of results across all indexes |
| `-m, --model <id>` | *(from index)* | Override embedding model; must match the indexes |
| `--full` | off | Show full chunk text instead of truncated preview |
| `--json` | off | Output compact JSON (for piping to LLMs or other tools) |

Repeat `--index` to search existing indexes without merging or rebuilding them:

```bash
rag search "cache behavior" \
  -i .rag/product-docs \
  -i .rag/api-reference \
  -i .rag/examples
```

Federated indexes must use the same model and embedding dimensions. They are
loaded and searched sequentially, keeping peak memory near the largest index
rather than the sum of all indexes. Each index contributes its local top-k, then
rag-cli computes the global top-k. Exact-text deduplication remains per-index;
identical text stored in different indexes may appear more than once.

Federated JSON results retain `source`, `score`, `byte_offset`, and `text`, and
also include `index`, `root_dir`, and (for config entries) `index_name`. These
fields let callers resolve a relative source path against the correct corpus.
When an index was built from a relative CLI or `rag.toml` path, `root_dir`
preserves that relative path rather than exposing the builder's absolute path.

### `rag info`

Prints index metadata: format, model, chunk and unique-text counts, duplicate
count, source file count, index size, etc.

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --index <dir>` | `.rag` | Index directory to inspect |

## Hardware acceleration

Apple Silicon builds use a native FP16 CoreML model with pooling and normalization
fused into the compiled graph. Other platforms use [ONNX Runtime](https://onnxruntime.ai/)
on CPU with architecture-tuned int8 weights. The backend is selected at build
time; no GPU toolkit or runtime flags are required.

On Apple Silicon, native CoreML substantially reduces indexing time and memory
compared with the int8 ONNX path. The downloaded CoreML artifact is pinned to an
immutable Hugging Face revision. The ONNX Runtime library remains statically
linked for non-Apple builds, so there is nothing to install separately.

Persisted vectors use F16 independently of int8 model inference. Exact duplicate
chunk text is embedded and stored once, while compact occurrence records retain
every source and byte offset. Search decodes F16 values for exact cosine scoring.

## Model management

rag-cli downloads model weights directly from HuggingFace over HTTPS on first
use, then caches them locally in the standard HuggingFace Hub layout
(`~/.cache/huggingface/hub`). We do this instead of using the `hf-hub` crate because `hf-hub` uses a bundled
certificate store and does not respect the system root CA certificates. That
makes it fail in environments with custom CA roots (corporate proxies, internal
TLS inspection, etc.). rag-cli uses `native-tls`, which delegates to the OS certificate store,
so it works in those environments without extra configuration.

The native CoreML backend on Apple Silicon currently supports the default MiniLM model only.
Non-Apple ONNX builds retain model overrides when the requested repository
publishes the expected architecture-specific ONNX artifact.

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
3. **Deduplicate and embed** each unique chunk body once, in batches of 128,
   using `all-MiniLM-L6-v2` with mean pooling and L2 normalization
4. **Store** one F16 vector and one text body per unique chunk, plus compact
   source/offset occurrence records, in `.rag/index.bin`; inference remains
   int8 with f32 output, so F16 applies only to persisted vectors
5. **Search** by embedding the query with the same model and ranking unique
   chunk bodies by exact cosine similarity

## License

Apache-2.0
