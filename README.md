# rag-cli

Local-first semantic search over your files. Index a directory and query it
with natural language — all embeddings are computed on your machine using
[candle](https://github.com/huggingface/candle), with no external API calls.

## Quick start

```
cargo install --path .

# Index a directory
rag index ./my-project

# Search it
rag search "how does authentication work"
```

## Commands

### `rag index <path>`

Recursively discovers text files under `<path>`, chunks them, computes
embeddings, and writes the index to disk.

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output <dir>` | `.rag` | Where to store the index |
| `-m, --model <id>` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `--chunk-size <n>` | `512` | Chunk size in characters |
| `--chunk-overlap <n>` | `64` | Overlap between consecutive chunks |

Re-running `rag index` on the same directory performs **incremental indexing** —
only changed or new files are re-embedded. File changes are detected using
[blake3](https://github.com/BLAKE3-team/BLAKE3) content hashes. If you change
the model or chunk settings the entire index is rebuilt automatically.

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

On macOS, Metal GPU and Accelerate BLAS are enabled automatically.

For NVIDIA GPUs, install the CUDA build:

```
cargo install rag-cli-cuda
```

This requires the CUDA toolkit to be installed on your system. The binary name
is the same (`rag`), so it's a drop-in replacement.

On other platforms the default build falls back to CPU — no extra flags are
needed.

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

rag-cli indexes 60+ text file extensions including common source code (`.rs`,
`.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, etc.), markup (`.md`,
`.html`, `.tex`, `.rst`), config (`.toml`, `.yaml`, `.json`, `.ini`), and
infrastructure files (`.tf`, `.hcl`, `.dockerfile`, `.nix`). Files like
`Makefile`, `Dockerfile`, `LICENSE`, and `.gitignore` are recognized by name.

Hidden directories, `node_modules`, `target`, `__pycache__`, `vendor`, `dist`,
and `build` are skipped automatically.

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
