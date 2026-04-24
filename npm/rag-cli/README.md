# @mathew-cf/rag-cli

Local-first RAG CLI — semantic search over your files using local embeddings. All inference runs on your machine; no external API calls.

```bash
npm install -g @mathew-cf/rag-cli

rag index ./my-project
rag search "how does authentication work"
```

On install, npm fetches the prebuilt binary for your platform via `optionalDependencies`. Supported platforms: macOS ARM64, macOS x86_64, Linux x86_64, Linux ARM64.

On any other platform, install from source instead:

```bash
cargo install rag-cli
```

For the full command reference, see the [project README](https://github.com/mathew-cf/rag-cli#readme).

## License

Apache-2.0
