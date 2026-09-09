mod config;
mod embed;
mod index;
mod ingest;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::config::RagConfig;
use crate::embed::{
    download_model, embedding_backend, model_file_list, model_files_present, resolve_hf_cache,
    EmbeddingEngine, DEFAULT_MODEL,
};
use crate::index::{
    search_top_k, ChunkOccurrence, Index, IndexMeta, SourceRecord, TextRecord, INDEX_FORMAT_VERSION,
};
use crate::ingest::{chunk_file, discover_files, hash_files, DiscoveryConfig};

const DEFAULT_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_OVERLAP: usize = 64;
const DEFAULT_TOP_K: usize = 5;
/// Maximum number of semantic candidates considered per requested result when
/// grouping by source. This keeps source diversification bounded for large
/// indexes while allowing lower-ranked files past a run of same-file chunks.
const SOURCE_DIVERSE_OVERSAMPLE_FACTOR: usize = 10;

struct IndexSource<'a> {
    path: &'a Path,
    metadata_root: &'a Path,
}

struct IndexBuildSettings<'a> {
    model_id: &'a str,
    chunk_size: usize,
    chunk_overlap: usize,
    cache_dir: Option<&'a Path>,
}

struct SearchSettings<'a> {
    query: &'a str,
    index_dirs: Vec<PathBuf>,
    config_path: Option<&'a Path>,
    only: &'a [String],
    top_k: usize,
    model_override: Option<&'a str>,
    group_by_source: bool,
    full: bool,
    json: bool,
    cache_dir: Option<&'a Path>,
}

#[derive(Parser)]
#[command(name = "rag")]
#[command(about = "Local RAG — index and semantic search your files using local embeddings")]
#[command(version)]
struct Cli {
    /// Override HuggingFace model cache directory.
    /// Default: $HF_HOME/hub or ~/.cache/huggingface/hub
    #[arg(long, global = true, env = "RAG_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Index text files for semantic search.
    ///
    /// With a PATH, indexes that single directory using the flags below. With
    /// no PATH, reads a config file (`rag.toml` / `.rag.toml`, or `--config`)
    /// and builds every `[[index]]` it declares.
    Index {
        /// Directory to index. Omit to build every index in the config file.
        #[arg(conflicts_with_all = ["config", "only"])]
        path: Option<PathBuf>,

        /// Config file to build from when no PATH is given
        /// (default: ./rag.toml or ./.rag.toml).
        #[arg(short = 'c', long)]
        config: Option<PathBuf>,

        /// In config mode, build only these named indexes (comma-separated or
        /// repeated). Useful to skip slow indexes you didn't change.
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,

        /// Where to store the index (default: .rag in current directory).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// HuggingFace model ID for embeddings (default: all-MiniLM-L6-v2).
        #[arg(short, long)]
        model: Option<String>,

        /// Chunk size in characters (default: 512).
        #[arg(long)]
        chunk_size: Option<usize>,

        /// Chunk overlap in characters (default: 64).
        #[arg(long)]
        chunk_overlap: Option<usize>,

        /// Extra file extensions to index, beyond the built-in allowlist
        /// (comma-separated or repeated), e.g. `--ext mdx,rst`.
        #[arg(long, value_delimiter = ',')]
        ext: Vec<String>,

        /// Directory/file specs to skip (comma-separated or repeated). A bare
        /// name matches any path component; a spec with `/` is a path prefix.
        #[arg(long, value_delimiter = ',')]
        exclude: Vec<String>,

        /// Normally-skipped directories to index anyway (comma-separated or
        /// repeated), e.g. `--include dist`.
        #[arg(long, value_delimiter = ',')]
        include: Vec<String>,
    },

    /// Search the index with a natural language query.
    Search {
        /// The search query.
        query: String,

        /// Index directories to search (repeatable). When neither --index nor
        /// --config is given, uses rag.toml/.rag.toml if present, then .rag.
        #[arg(short, long, conflicts_with = "config")]
        index: Vec<PathBuf>,

        /// Search every index declared by this config file. When omitted,
        /// rag.toml/.rag.toml is discovered automatically.
        #[arg(short = 'c', long, conflicts_with = "index")]
        config: Option<PathBuf>,

        /// With an explicit or discovered config, search only these named indexes
        /// (comma-separated or repeated).
        #[arg(long, value_delimiter = ',', conflicts_with = "index")]
        only: Vec<String>,

        /// Number of results to return.
        #[arg(short = 'k', long, default_value_t = DEFAULT_TOP_K)]
        top_k: usize,

        /// HuggingFace model ID (must match the one used for indexing).
        #[arg(short, long)]
        model: Option<String>,

        /// Return at most one result from each source file.
        #[arg(long)]
        group_by_source: bool,

        /// Show full chunk text instead of truncated preview.
        #[arg(long)]
        full: bool,

        /// Output results as compact JSON (for piping to LLMs or other tools).
        #[arg(long)]
        json: bool,
    },

    /// Show index metadata and statistics.
    Info {
        /// Index directory. When neither --index nor --config is given, uses
        /// rag.toml/.rag.toml if present, then .rag.
        #[arg(short, long, conflicts_with = "config")]
        index: Option<PathBuf>,

        /// Show every index declared by this config file. When omitted,
        /// rag.toml/.rag.toml is discovered automatically.
        #[arg(short = 'c', long, conflicts_with = "index")]
        config: Option<PathBuf>,

        /// With an explicit or discovered config, show only these named indexes.
        #[arg(long, value_delimiter = ',', conflicts_with = "index")]
        only: Vec<String>,
    },

    /// Pre-download the embedding model weights + tokenizer into the cache.
    ///
    /// Useful on fresh installs: makes the first `rag index`/`rag search`
    /// fast instead of stalling on a ~90MB network fetch. Safe to re-run —
    /// if every file is already cached it exits immediately.
    Download {
        /// HuggingFace model ID for embeddings. When omitted, uses every model
        /// in rag.toml/.rag.toml if present, then the built-in default.
        #[arg(short, long, conflicts_with = "config")]
        model: Option<String>,

        /// Download models used by this config file. When omitted,
        /// rag.toml/.rag.toml is discovered automatically.
        #[arg(short = 'c', long, conflicts_with = "model")]
        config: Option<PathBuf>,

        /// With an explicit or discovered config, include only these indexes.
        #[arg(long, value_delimiter = ',', conflicts_with = "model")]
        only: Vec<String>,

        /// Verify each model loads successfully after downloading.
        #[arg(long)]
        verify: bool,
    },
}

/// Entry point for the CLI. Call this from `main()`.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let cache_dir = cli.cache_dir.as_deref();

    match cli.command {
        Commands::Index {
            path,
            config,
            only,
            output,
            model,
            chunk_size,
            chunk_overlap,
            ext,
            exclude,
            include,
        } => match path {
            Some(path) => {
                let discovery = DiscoveryConfig {
                    extra_extensions: ext,
                    exclude,
                    include,
                };
                cmd_index(
                    IndexSource {
                        path: &path,
                        metadata_root: &path,
                    },
                    output.as_deref(),
                    model.as_deref().unwrap_or(DEFAULT_MODEL),
                    chunk_size.unwrap_or(DEFAULT_CHUNK_SIZE),
                    chunk_overlap.unwrap_or(DEFAULT_CHUNK_OVERLAP),
                    &discovery,
                    cache_dir,
                )
            }
            None => {
                // Config mode: each `[[index]]` entry is self-describing, so the
                // ad-hoc single-index flags don't apply. Warn rather than
                // silently ignore them.
                if let Some(warning) = ignored_config_options_warning(&[
                    ("--output", output.is_some()),
                    ("--model", model.is_some()),
                    ("--chunk-size", chunk_size.is_some()),
                    ("--chunk-overlap", chunk_overlap.is_some()),
                    ("--ext", !ext.is_empty()),
                    ("--exclude", !exclude.is_empty()),
                    ("--include", !include.is_empty()),
                ]) {
                    eprintln!("{warning}");
                }
                cmd_index_from_config(config.as_deref(), &only, cache_dir)
            }
        },
        Commands::Search {
            query,
            index,
            config,
            only,
            top_k,
            model,
            group_by_source,
            full,
            json,
        } => cmd_search(SearchSettings {
            query: &query,
            index_dirs: index,
            config_path: config.as_deref(),
            only: &only,
            top_k,
            model_override: model.as_deref(),
            group_by_source,
            full,
            json,
            cache_dir,
        }),
        Commands::Info {
            index,
            config,
            only,
        } => cmd_info(index, config.as_deref(), &only),
        Commands::Download {
            model,
            config,
            only,
            verify,
        } => cmd_download(
            model.as_deref(),
            config.as_deref(),
            &only,
            verify,
            cache_dir,
        ),
    }
}

fn ignored_config_options_warning(options: &[(&str, bool)]) -> Option<String> {
    let ignored: Vec<&str> = options
        .iter()
        .filter_map(|(name, present)| present.then_some(*name))
        .collect();
    (!ignored.is_empty()).then(|| {
        format!(
            "warning: {} are ignored when building from a config file; set them per-[[index]] \
             in the config instead",
            ignored.join("/")
        )
    })
}

fn resolve_download_models_from_dir(
    model_override: Option<&str>,
    config_path: Option<&Path>,
    only: &[String],
    cwd: &Path,
) -> Result<Vec<String>> {
    if let Some(model) = model_override {
        if !only.is_empty() {
            anyhow::bail!("--only requires a config file, but --model was provided");
        }
        return Ok(vec![model.to_string()]);
    }

    let config_path = config_path
        .map(Path::to_path_buf)
        .or_else(|| RagConfig::locate(None, cwd));
    let Some(config_path) = config_path else {
        if !only.is_empty() {
            anyhow::bail!(
                "--only requires --config or a {} file in the current directory",
                config::DEFAULT_CONFIG_NAMES.join("/")
            );
        }
        return Ok(vec![DEFAULT_MODEL.to_string()]);
    };

    let config = RagConfig::load(&config_path)?;
    if let Some(missing) = only
        .iter()
        .find(|name| !config.indexes.iter().any(|entry| &entry.name == *name))
    {
        anyhow::bail!(
            "--only names an index not in {}: {:?}",
            config_path.display(),
            missing
        );
    }

    let mut seen = HashSet::new();
    let models = config
        .indexes
        .iter()
        .filter(|entry| only.is_empty() || only.iter().any(|name| name == &entry.name))
        .map(|entry| entry.model_id(&config, DEFAULT_MODEL))
        .filter(|model| seen.insert(model.clone()))
        .collect();
    Ok(models)
}

fn cmd_download(
    model_override: Option<&str>,
    config_path: Option<&Path>,
    only: &[String],
    verify: bool,
    cache_dir: Option<&Path>,
) -> Result<()> {
    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    let models = resolve_download_models_from_dir(model_override, config_path, only, &cwd)?;
    let hub_root = resolve_hf_cache(cache_dir)?;

    for (index, model_id) in models.iter().enumerate() {
        if index > 0 {
            eprintln!();
        }
        let start = Instant::now();
        eprintln!("Model: {model_id}");
        eprintln!("Cache: {}", hub_root.display());

        if model_files_present(&hub_root, model_id) {
            eprintln!(
                "All {} model file(s) already cached.",
                model_file_list().len()
            );
        } else if download_model(model_id, cache_dir)? {
            eprintln!(
                "Downloaded {} file(s) in {:.1}s",
                model_file_list().len(),
                start.elapsed().as_secs_f64()
            );
        }

        if verify {
            eprintln!("Verifying model loads and produces embeddings...");
            let mut engine = EmbeddingEngine::load(Some(model_id), cache_dir)?;
            let vec = engine.embed_one("hello world")?;
            if vec.is_empty() {
                anyhow::bail!("Model produced an empty embedding — installation may be corrupt");
            }
            eprintln!(
                "  Verified (hidden_size={}, elapsed={:.1}s)",
                vec.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    Ok(())
}

fn validate_chunk_settings(chunk_size: usize, chunk_overlap: usize) -> Result<()> {
    if chunk_size == 0 {
        anyhow::bail!("chunk size must be greater than zero");
    }
    if chunk_overlap >= chunk_size {
        anyhow::bail!("chunk overlap must be smaller than chunk size");
    }
    Ok(())
}

fn cmd_index(
    source: IndexSource<'_>,
    output: Option<&std::path::Path>,
    model_id: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    discovery: &DiscoveryConfig,
    cache_dir: Option<&std::path::Path>,
) -> Result<()> {
    validate_chunk_settings(chunk_size, chunk_overlap)?;
    let start = Instant::now();

    let root = source
        .path
        .canonicalize()
        .with_context(|| format!("Directory not found: {}", source.path.display()))?;

    if !root.is_dir() {
        anyhow::bail!("{} is not a directory", root.display());
    }

    let index_dir = output.map(PathBuf::from).unwrap_or_else(Index::default_dir);
    let metadata_root = normalized_metadata_path(source.metadata_root);

    // 1. Discover files and hash them
    eprintln!("Indexing: {}", root.display());
    let files = discover_files(&root, discovery)?;
    eprintln!("Found {} text files", files.len());

    if files.is_empty() {
        anyhow::bail!("No text files found in {}", root.display());
    }

    let current_hashes = hash_files(&files, &root)?;

    // 2. Try incremental indexing against an existing index. Read the small
    // metadata file first so a no-op run never materializes the full vector
    // index (hundreds of MiB for a large corpus).
    let had_previous = index_dir.join("index.bin").is_file();
    let prev_meta = Index::load_meta(&index_dir).ok();
    let can_reuse = had_previous
        && prev_meta.as_ref().is_some_and(|meta| {
            meta.root_dir == metadata_root
                && meta.reusable_for(model_id, embedding_backend(), chunk_size, chunk_overlap)
        });

    if can_reuse
        && prev_meta
            .as_ref()
            .is_some_and(|m| m.file_hashes == current_hashes)
    {
        let meta = prev_meta
            .as_ref()
            .context("Reusable index metadata disappeared during the no-op check")?;
        eprintln!(
            "Incremental: {} unchanged, 0 changed/new, 0 deleted",
            files.len()
        );
        eprintln!("Everything up to date, nothing to embed");
        eprintln!(
            "Index unchanged at {} ({} chunks, {:.1}s)",
            index_dir.display(),
            meta.num_chunks,
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    let settings = IndexBuildSettings {
        model_id,
        chunk_size,
        chunk_overlap,
        cache_dir,
    };
    let (sources, texts, occurrences, hidden_size) = if can_reuse {
        let prev = Index::load(&index_dir)?;
        incremental_index(&root, &files, &current_hashes, prev, &settings)?
    } else {
        if had_previous {
            eprintln!("Settings changed, performing full re-index");
        }
        full_index(&root, &files, &settings)?
    };

    if occurrences.is_empty() {
        anyhow::bail!("No text chunks produced. Check the directory contents.");
    }

    // 3. Save index
    let meta = IndexMeta {
        format_version: INDEX_FORMAT_VERSION,
        model_id: model_id.to_string(),
        embedding_backend: embedding_backend().to_string(),
        hidden_size,
        num_chunks: occurrences.len(),
        num_unique_texts: texts.len(),
        root_dir: metadata_root,
        created_at: chrono_now(),
        chunk_size,
        chunk_overlap,
        file_hashes: current_hashes.clone(),
    };

    let index = Index::new(meta, sources, texts, occurrences);
    index.save(&index_dir)?;

    let elapsed = start.elapsed();
    eprintln!(
        "Index saved to {} ({} chunks, {:.1}s)",
        index_dir.display(),
        index.meta.num_chunks,
        elapsed.as_secs_f64()
    );

    Ok(())
}

/// Build every index declared in a `rag.toml` config file.
///
/// Paths and output dirs in the config are resolved relative to the config
/// file's own directory, so `rag index` works from anywhere as long as
/// `--config` points at the file.
fn cmd_index_from_config(
    config_path: Option<&std::path::Path>,
    only: &[String],
    cache_dir: Option<&std::path::Path>,
) -> Result<()> {
    let cwd = std::env::current_dir().context("Failed to determine current directory")?;

    let config_path = RagConfig::locate(config_path, &cwd).ok_or_else(|| {
        anyhow::anyhow!(
            "No path given and no config file found. Pass a directory to index, or create a \
             {} file (or pass --config <file>).",
            config::DEFAULT_CONFIG_NAMES[0]
        )
    })?;

    let base = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let config = RagConfig::load(&config_path)?;

    // Optionally restrict to a subset by name (`--only akamai`).
    if let Some(missing) = only
        .iter()
        .find(|name| !config.indexes.iter().any(|e| &e.name == *name))
    {
        anyhow::bail!(
            "--only names an index not in {}: {:?}",
            config_path.display(),
            missing
        );
    }
    let selected: Vec<&config::IndexEntry> = config
        .indexes
        .iter()
        .filter(|e| only.is_empty() || only.iter().any(|n| n == &e.name))
        .collect();

    eprintln!(
        "Building {} index(es) from {}",
        selected.len(),
        config_path.display()
    );

    let total = selected.len();
    for (i, entry) in selected.iter().enumerate() {
        let index_path = base.join(&entry.path);
        let output = entry.output_path(&base);

        let model = entry.model_id(&config, DEFAULT_MODEL);
        let chunk_size = entry
            .chunk_size
            .or(config.chunk_size)
            .unwrap_or(DEFAULT_CHUNK_SIZE);
        let chunk_overlap = entry
            .chunk_overlap
            .or(config.chunk_overlap)
            .unwrap_or(DEFAULT_CHUNK_OVERLAP);

        let discovery = DiscoveryConfig {
            extra_extensions: entry.extensions.clone(),
            exclude: entry.exclude.clone(),
            include: entry.include.clone(),
        };

        eprintln!();
        eprintln!(
            "[{}/{}] {} → {}",
            i + 1,
            total,
            entry.name,
            output.display()
        );

        cmd_index(
            IndexSource {
                path: &index_path,
                metadata_root: &entry.path,
            },
            Some(&output),
            &model,
            chunk_size,
            chunk_overlap,
            &discovery,
            cache_dir,
        )
        .with_context(|| format!("Failed to build index {:?}", entry.name))?;
    }

    Ok(())
}

/// Strip redundant `.` components without canonicalizing, so relative paths
/// stay relative while meaningful `..` components and absolute roots remain.
/// Windows separators are serialized as `/` for portable metadata.
fn normalized_metadata_path(path: &Path) -> String {
    let normalized: PathBuf = path
        .components()
        .filter(|component| !matches!(component, std::path::Component::CurDir))
        .collect();
    if normalized.as_os_str().is_empty() {
        ".".to_string()
    } else {
        let value = normalized.to_string_lossy();
        if cfg!(windows) {
            value.replace('\\', "/")
        } else {
            value.into_owned()
        }
    }
}

/// Build a stable list of unique texts and, for each input, the corresponding
/// unique-text index. Exact duplicate chunks produce identical embeddings, so
/// embedding them once is lossless.
fn unique_text_plan<'a>(texts: impl Iterator<Item = &'a str>) -> (Vec<&'a str>, Vec<usize>) {
    let mut lookup = HashMap::new();
    let mut unique = Vec::new();
    let mut ids = Vec::new();

    for text in texts {
        let id = if let Some(&id) = lookup.get(text) {
            id
        } else {
            let id = unique.len();
            lookup.insert(text, id);
            unique.push(text);
            id
        };
        ids.push(id);
    }

    (unique, ids)
}

type IndexRecords = (
    Vec<SourceRecord>,
    Vec<TextRecord>,
    Vec<ChunkOccurrence>,
    usize,
);

/// Full re-index: chunk every file, embed each unique body once, and store
/// compact source/text IDs for every occurrence.
fn full_index(
    root: &Path,
    files: &[PathBuf],
    settings: &IndexBuildSettings<'_>,
) -> Result<IndexRecords> {
    let mut all_chunks = Vec::new();
    for file in files {
        match chunk_file(file, root, settings.chunk_size, settings.chunk_overlap) {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(e) => eprintln!("  Skipping {}: {e}", file.display()),
        }
    }

    let (unique_texts, text_ids) = unique_text_plan(all_chunks.iter().map(|c| c.text.as_str()));
    eprintln!(
        "Embedding {} unique chunks ({} duplicates reused)...",
        unique_texts.len(),
        all_chunks.len() - unique_texts.len()
    );
    let mut engine = EmbeddingEngine::load(Some(settings.model_id), settings.cache_dir)?;
    let hidden_size = engine.hidden_size();
    let embeddings = engine.embed_batch_progress(&unique_texts)?;
    let mut owned_texts: Vec<Option<String>> = vec![None; unique_texts.len()];
    drop(unique_texts);

    let mut sources = Vec::new();
    let mut source_ids = HashMap::new();
    let mut occurrences = Vec::with_capacity(all_chunks.len());

    for (chunk, text_id) in all_chunks.into_iter().zip(text_ids) {
        let source_id = if let Some(&id) = source_ids.get(&chunk.source) {
            id
        } else {
            let id = u32::try_from(sources.len()).context("Too many source files")?;
            source_ids.insert(chunk.source.clone(), id);
            sources.push(SourceRecord { path: chunk.source });
            id
        };
        let owned_text = owned_texts
            .get_mut(text_id)
            .context("Unique-text plan produced an invalid text ID")?;
        if owned_text.is_none() {
            *owned_text = Some(chunk.text);
        }
        occurrences.push(ChunkOccurrence {
            source_id,
            text_id: u32::try_from(text_id).context("Too many unique chunks")?,
            byte_offset: chunk.byte_offset,
        });
    }

    let texts = owned_texts
        .into_iter()
        .zip(embeddings)
        .map(|(text, embedding)| {
            TextRecord::new(text.expect("every text ID has an occurrence"), embedding)
        })
        .collect();

    Ok((sources, texts, occurrences, hidden_size))
}

/// Incremental re-index: preserve unchanged occurrences and stored F16 vectors,
/// embedding only new unique text introduced by changed files.
fn incremental_index(
    root: &Path,
    files: &[PathBuf],
    current_hashes: &BTreeMap<String, String>,
    prev: Index,
    settings: &IndexBuildSettings<'_>,
) -> Result<IndexRecords> {
    let mut unchanged = HashSet::new();
    let mut dirty_files = Vec::new();

    for file in files {
        let relative = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();
        if current_hashes.get(&relative) == prev.meta.file_hashes.get(&relative) {
            unchanged.insert(relative);
        } else {
            dirty_files.push(file);
        }
    }

    let deleted = prev
        .meta
        .file_hashes
        .keys()
        .filter(|path| !current_hashes.contains_key(path.as_str()))
        .count();
    eprintln!(
        "Incremental: {} unchanged, {} changed/new, {} deleted",
        unchanged.len(),
        dirty_files.len(),
        deleted
    );

    let hidden_size = prev.meta.hidden_size;
    let mut sources = prev.sources;
    let mut texts = prev.texts;
    let mut occurrences = Vec::new();
    for occurrence in prev.occurrences {
        let source = sources
            .get(occurrence.source_id as usize)
            .context("Index occurrence references an invalid source ID")?;
        texts
            .get(occurrence.text_id as usize)
            .context("Index occurrence references an invalid text ID")?;
        if unchanged.contains(&source.path) {
            occurrences.push(occurrence);
        }
    }

    let mut source_ids: HashMap<String, u32> = sources
        .iter()
        .enumerate()
        .map(|(id, source)| (source.path.clone(), id as u32))
        .collect();
    // Hash all retained text so changed files can reuse vectors from unchanged
    // files. Persisting a 32-byte hash per text would enlarge every index;
    // rebuilding this map is cheap relative to loading index.bin (the measured
    // 204k-text no-op path remains under 0.4s and bypasses this function).
    let mut text_ids: HashMap<blake3::Hash, u32> = texts
        .iter()
        .enumerate()
        .map(|(id, text)| (blake3::hash(text.text.as_bytes()), id as u32))
        .collect();
    let mut new_text_ids = Vec::new();

    for file in dirty_files {
        let new_chunks = match chunk_file(file, root, settings.chunk_size, settings.chunk_overlap) {
            Ok(chunks) => chunks,
            Err(error) => {
                eprintln!("  Skipping {}: {error}", file.display());
                continue;
            }
        };

        for chunk in new_chunks {
            let source_id = if let Some(&id) = source_ids.get(&chunk.source) {
                id
            } else {
                let id = u32::try_from(sources.len()).context("Too many source files")?;
                source_ids.insert(chunk.source.clone(), id);
                sources.push(SourceRecord { path: chunk.source });
                id
            };

            let text_hash = blake3::hash(chunk.text.as_bytes());
            let text_id = if let Some(&id) = text_ids.get(&text_hash) {
                let existing = texts
                    .get(id as usize)
                    .context("Text lookup references an invalid text ID")?;
                if existing.text != chunk.text {
                    anyhow::bail!("Blake3 collision while deduplicating chunk text");
                }
                id
            } else {
                let id = u32::try_from(texts.len()).context("Too many unique chunks")?;
                text_ids.insert(text_hash, id);
                texts.push(TextRecord::without_embedding(chunk.text));
                new_text_ids.push(id);
                id
            };

            occurrences.push(ChunkOccurrence {
                source_id,
                text_id,
                byte_offset: chunk.byte_offset,
            });
        }
    }

    if !new_text_ids.is_empty() {
        eprintln!("Embedding {} new unique chunks...", new_text_ids.len());
        let new_texts: Vec<&str> = new_text_ids
            .iter()
            .map(|&id| {
                texts
                    .get(id as usize)
                    .map(|text| text.text.as_str())
                    .context("New-text list references an invalid text ID")
            })
            .collect::<Result<_>>()?;
        let mut engine = EmbeddingEngine::load(Some(settings.model_id), settings.cache_dir)?;
        let embeddings = engine.embed_batch_progress(&new_texts)?;
        drop(new_texts);
        if embeddings.len() != new_text_ids.len() {
            anyhow::bail!(
                "Embedding backend returned {} vectors for {} texts",
                embeddings.len(),
                new_text_ids.len()
            );
        }
        for (text_id, embedding) in new_text_ids.into_iter().zip(embeddings) {
            texts
                .get_mut(text_id as usize)
                .context("New-text list references an invalid text ID")?
                .set_embedding(embedding);
        }
    } else {
        eprintln!("No new unique text to embed");
    }

    compact_records(&mut sources, &mut texts, &mut occurrences)?;
    Ok((sources, texts, occurrences, hidden_size))
}

fn compact_records(
    sources: &mut Vec<SourceRecord>,
    texts: &mut Vec<TextRecord>,
    occurrences: &mut [ChunkOccurrence],
) -> Result<()> {
    let mut active_sources = vec![false; sources.len()];
    let mut active_texts = vec![false; texts.len()];
    for occurrence in occurrences.iter() {
        *active_sources
            .get_mut(occurrence.source_id as usize)
            .context("Index occurrence references an invalid source ID")? = true;
        *active_texts
            .get_mut(occurrence.text_id as usize)
            .context("Index occurrence references an invalid text ID")? = true;
    }

    let mut source_remap = vec![None; sources.len()];
    let old_sources = std::mem::take(sources);
    for (old_id, source) in old_sources.into_iter().enumerate() {
        if active_sources.get(old_id).copied().unwrap_or(false) {
            let new_id = u32::try_from(sources.len()).context("Too many source files")?;
            *source_remap
                .get_mut(old_id)
                .context("Source remap is shorter than the source table")? = Some(new_id);
            sources.push(source);
        }
    }

    let mut text_remap = vec![None; texts.len()];
    let old_texts = std::mem::take(texts);
    for (old_id, text) in old_texts.into_iter().enumerate() {
        if active_texts.get(old_id).copied().unwrap_or(false) {
            let new_id = u32::try_from(texts.len()).context("Too many unique chunks")?;
            *text_remap
                .get_mut(old_id)
                .context("Text remap is shorter than the text table")? = Some(new_id);
            texts.push(text);
        }
    }

    for occurrence in occurrences {
        occurrence.source_id = source_remap
            .get(occurrence.source_id as usize)
            .and_then(|id| *id)
            .context("Active occurrence has no compacted source ID")?;
        occurrence.text_id = text_remap
            .get(occurrence.text_id as usize)
            .and_then(|id| *id)
            .context("Active occurrence has no compacted text ID")?;
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct SearchIndexSpec {
    name: Option<String>,
    path: PathBuf,
}

impl SearchIndexSpec {
    fn label(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| self.path.display().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FederatedSourceIdentity {
    /// Absolute metadata roots can be compared across independently stored
    /// indexes. Canonicalization also collapses symlink aliases when possible.
    Canonical(PathBuf),
    /// Relative metadata roots do not record what directory they were relative
    /// to when the index was built. Scope them to the index rather than risk
    /// merging unrelated files that happen to have the same relative name.
    IndexScoped {
        index_path: PathBuf,
        root_dir: PathBuf,
        source: PathBuf,
    },
}

#[derive(Debug, Clone)]
struct FederatedSearchResult {
    index_name: Option<String>,
    index_path: PathBuf,
    root_dir: String,
    source: String,
    score: f32,
    byte_offset: usize,
    text: String,
    index_order: usize,
    text_id: usize,
}

impl FederatedSearchResult {
    fn source_identity(&self) -> FederatedSourceIdentity {
        let root = Path::new(&self.root_dir);
        let source = Path::new(&self.source);
        if root.is_absolute() {
            let canonical_root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
            let canonical_source = canonical_root.join(source);
            FederatedSourceIdentity::Canonical(
                canonical_source.canonicalize().unwrap_or(canonical_source),
            )
        } else {
            // There is no reliable corpus anchor in old index metadata for a
            // relative root (notably `.`). Make the index path absolute and
            // canonical when possible so aliases of the same index still
            // group, but separate index files remain separate namespaces.
            let index_path = self.index_path.canonicalize().unwrap_or_else(|_| {
                std::env::current_dir()
                    .map(|cwd| cwd.join(&self.index_path))
                    .unwrap_or_else(|_| self.index_path.clone())
            });
            FederatedSourceIdentity::IndexScoped {
                index_path,
                root_dir: root.to_path_buf(),
                source: source.to_path_buf(),
            }
        }
    }
}

/// A single search result for JSON output.
#[derive(serde::Serialize)]
struct JsonResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    index_name: Option<String>,
    index: String,
    root_dir: String,
    source: String,
    score: f32,
    byte_offset: usize,
    text: String,
}

fn configured_search_indexes(config_path: &Path, only: &[String]) -> Result<Vec<SearchIndexSpec>> {
    let base = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let config = RagConfig::load(config_path)?;

    if let Some(missing) = only
        .iter()
        .find(|name| !config.indexes.iter().any(|entry| &entry.name == *name))
    {
        anyhow::bail!(
            "--only names an index not in {}: {:?}",
            config_path.display(),
            missing
        );
    }

    Ok(config
        .indexes
        .into_iter()
        .filter(|entry| only.is_empty() || only.iter().any(|name| name == &entry.name))
        .map(|entry| SearchIndexSpec {
            path: entry.output_path(&base),
            name: Some(entry.name),
        })
        .collect())
}

fn resolve_search_indexes(
    index_dirs: Vec<PathBuf>,
    config_path: Option<&Path>,
    only: &[String],
) -> Result<Vec<SearchIndexSpec>> {
    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    resolve_search_indexes_from_dir(index_dirs, config_path, only, &cwd)
}

fn resolve_search_indexes_from_dir(
    index_dirs: Vec<PathBuf>,
    config_path: Option<&Path>,
    only: &[String],
    cwd: &Path,
) -> Result<Vec<SearchIndexSpec>> {
    if let Some(config_path) = config_path {
        return configured_search_indexes(config_path, only);
    }
    if !index_dirs.is_empty() {
        if !only.is_empty() {
            anyhow::bail!("--only requires a config file, but --index was provided");
        }
        return Ok(index_dirs
            .into_iter()
            .map(|path| SearchIndexSpec { name: None, path })
            .collect());
    }
    if let Some(config_path) = RagConfig::locate(None, cwd) {
        return configured_search_indexes(&config_path, only);
    }
    if !only.is_empty() {
        anyhow::bail!(
            "--only requires --config or a {} file in the current directory",
            config::DEFAULT_CONFIG_NAMES.join("/")
        );
    }
    Ok(vec![SearchIndexSpec {
        name: None,
        path: Index::default_dir(),
    }])
}

fn validate_search_metadata(
    specs: &[SearchIndexSpec],
    metas: &[IndexMeta],
    model_override: Option<&str>,
) -> Result<String> {
    let first = metas.first().context("No indexes selected for search")?;
    let first_spec = specs.first().context("No indexes selected for search")?;

    for (spec, meta) in specs.iter().zip(metas) {
        if meta.format_version != INDEX_FORMAT_VERSION {
            anyhow::bail!(
                "Index {} uses format v{}, but this binary requires v{}; rebuild it",
                spec.path.display(),
                meta.format_version,
                INDEX_FORMAT_VERSION
            );
        }
        if meta.model_id != first.model_id {
            anyhow::bail!(
                "Cannot federate indexes with different models: {} uses {:?}, while {} uses {:?}",
                first_spec.path.display(),
                first.model_id,
                spec.path.display(),
                meta.model_id
            );
        }
        if meta.hidden_size != first.hidden_size {
            anyhow::bail!(
                "Cannot federate indexes with different embedding dimensions: {} uses {}, while {} uses {}",
                first_spec.path.display(),
                first.hidden_size,
                spec.path.display(),
                meta.hidden_size
            );
        }
    }

    if let Some(model_override) = model_override {
        if model_override != first.model_id {
            anyhow::bail!(
                "Search model {:?} does not match index model {:?}",
                model_override,
                first.model_id
            );
        }
    }

    let mut backends: Vec<&str> = metas
        .iter()
        .map(|meta| meta.embedding_backend.as_str())
        .collect();
    backends.sort_unstable();
    backends.dedup();
    if backends.len() > 1 {
        eprintln!(
            "warning: federated indexes were built by different compatible backends: {}",
            backends.join(", ")
        );
    }

    Ok(first.model_id.clone())
}

fn search_one_index(
    spec: &SearchIndexSpec,
    index_order: usize,
    index: &Index,
    query_embedding: &[f32],
    top_k: usize,
    group_by_source: bool,
) -> Result<Vec<FederatedSearchResult>> {
    if !group_by_source {
        // Keep the original unique-text search and first-occurrence resolution
        // untouched when source grouping was not requested.
        let results = search_top_k(query_embedding, &index.texts, top_k);
        let mut representative = vec![None; index.texts.len()];
        for occurrence in &index.occurrences {
            index
                .sources
                .get(occurrence.source_id as usize)
                .context("Index occurrence references an invalid source ID")?;
            representative
                .get_mut(occurrence.text_id as usize)
                .context("Index occurrence references an invalid text ID")?
                .get_or_insert(occurrence);
        }

        return results
            .into_iter()
            .map(|result| {
                let occurrence = representative
                    .get(result.text_id)
                    .and_then(|occurrence| *occurrence)
                    .context("Active text record has no source occurrence")?;
                federated_result(spec, index_order, index, &result, occurrence)
            })
            .collect();
    }

    if top_k == 0 {
        return Ok(Vec::new());
    }

    let candidate_limit = top_k.saturating_mul(SOURCE_DIVERSE_OVERSAMPLE_FACTOR);
    let results = search_top_k(
        query_embedding,
        &index.texts,
        candidate_limit.min(index.texts.len()),
    );

    // Map only the bounded semantic window, then retain one occurrence per
    // source. In particular, do not build a list containing every duplicate
    // occurrence of a candidate text: that can be much larger than the index's
    // source set and candidate window.
    let candidate_rank: HashMap<usize, usize> = results
        .iter()
        .enumerate()
        .map(|(rank, result)| (result.text_id, rank))
        .collect();
    let mut best_by_source: HashMap<&str, (usize, &ChunkOccurrence)> = HashMap::new();
    let compare_candidate =
        |a_source: &str,
         &(a_rank, a_occurrence): &(usize, &ChunkOccurrence),
         b_source: &str,
         &(b_rank, b_occurrence): &(usize, &ChunkOccurrence)| {
            results[b_rank]
                .score
                .total_cmp(&results[a_rank].score)
                .then_with(|| a_source.cmp(b_source))
                .then_with(|| a_occurrence.byte_offset.cmp(&b_occurrence.byte_offset))
                .then_with(|| results[a_rank].text.text.cmp(&results[b_rank].text.text))
                .then_with(|| results[a_rank].text_id.cmp(&results[b_rank].text_id))
        };
    for occurrence in &index.occurrences {
        let source = index
            .sources
            .get(occurrence.source_id as usize)
            .context("Index occurrence references an invalid source ID")?;
        index
            .texts
            .get(occurrence.text_id as usize)
            .context("Index occurrence references an invalid text ID")?;

        let Some(&rank) = candidate_rank.get(&(occurrence.text_id as usize)) else {
            continue;
        };
        let source_path = source.path.as_str();
        let candidate = (rank, occurrence);
        if let Some(best) = best_by_source.get(source_path) {
            if compare_candidate(source_path, &candidate, source_path, best).is_lt() {
                best_by_source.insert(source_path, candidate);
            }
            continue;
        }
        if best_by_source.len() < candidate_limit {
            best_by_source.insert(source_path, candidate);
            continue;
        }
        let worst_source = best_by_source
            .iter()
            .max_by(|(a_source, a), (b_source, b)| compare_candidate(a_source, a, b_source, b))
            .map(|(path, _)| (*path).to_string());
        if let Some(worst_source) = worst_source {
            let worst = best_by_source
                .get(worst_source.as_str())
                .expect("selected source must remain present");
            if compare_candidate(source_path, &candidate, worst_source.as_str(), worst).is_lt() {
                best_by_source.remove(worst_source.as_str());
                best_by_source.insert(source_path, candidate);
            }
        }
    }

    // Return an oversampled source window, rather than only k local sources,
    // so the global merge can refill slots removed by cross-index overlap.
    let grouped = best_by_source
        .into_values()
        .map(|(rank, occurrence)| {
            federated_result(spec, index_order, index, &results[rank], occurrence)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(limit_federated_results(grouped, candidate_limit))
}

fn federated_result(
    spec: &SearchIndexSpec,
    index_order: usize,
    index: &Index,
    result: &crate::index::SearchResult<'_>,
    occurrence: &ChunkOccurrence,
) -> Result<FederatedSearchResult> {
    let source = index
        .sources
        .get(occurrence.source_id as usize)
        .context("Index occurrence references an invalid source ID")?;
    Ok(FederatedSearchResult {
        index_name: spec.name.clone(),
        index_path: spec.path.clone(),
        root_dir: index.meta.root_dir.clone(),
        source: source.path.clone(),
        score: result.score,
        byte_offset: occurrence.byte_offset,
        text: result.text.text.clone(),
        index_order,
        text_id: result.text_id,
    })
}

fn compare_federated_results(
    a: &FederatedSearchResult,
    b: &FederatedSearchResult,
) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        // Score ties must not depend on hash iteration, occurrence order, or
        // select_nth_unstable's partitioning at the cutoff.
        .then_with(|| a.root_dir.cmp(&b.root_dir))
        .then_with(|| a.source.cmp(&b.source))
        .then_with(|| a.byte_offset.cmp(&b.byte_offset))
        .then_with(|| a.text.cmp(&b.text))
        .then_with(|| a.index_path.cmp(&b.index_path))
        .then_with(|| a.index_name.cmp(&b.index_name))
        .then_with(|| a.index_order.cmp(&b.index_order))
        .then_with(|| a.text_id.cmp(&b.text_id))
}

fn limit_federated_results(
    mut results: Vec<FederatedSearchResult>,
    limit: usize,
) -> Vec<FederatedSearchResult> {
    if results.len() > limit {
        results.select_nth_unstable_by(limit, compare_federated_results);
        results.truncate(limit);
    }
    results.sort_by(compare_federated_results);
    results
}

fn merge_federated_results(
    mut results: Vec<FederatedSearchResult>,
    top_k: usize,
    group_by_source: bool,
) -> Vec<FederatedSearchResult> {
    if group_by_source {
        results.sort_by(compare_federated_results);
        let mut seen_sources = HashSet::new();
        results.retain(|result| seen_sources.insert(result.source_identity()));
    }
    limit_federated_results(results, top_k)
}

fn cmd_search(settings: SearchSettings<'_>) -> Result<()> {
    let SearchSettings {
        query,
        index_dirs,
        config_path,
        only,
        top_k,
        model_override,
        group_by_source,
        full,
        json,
        cache_dir,
    } = settings;
    let start = Instant::now();
    let specs = resolve_search_indexes(index_dirs, config_path, only)?;
    let metas: Vec<IndexMeta> = specs
        .iter()
        .map(|spec| {
            Index::load_meta(&spec.path).with_context(|| {
                format!(
                    "No index found at {}. Build it with `rag index` first.",
                    spec.path.display()
                )
            })
        })
        .collect::<Result<_>>()?;
    let model_id = validate_search_metadata(&specs, &metas, model_override)?;

    // Load the model and embed the query once for every compatible index.
    let mut engine = EmbeddingEngine::load(Some(&model_id), cache_dir)?;
    let query_embedding = engine.embed_one(query)?;
    let expected_dimension = metas
        .first()
        .context("No indexes selected for search")?
        .hidden_size;
    if query_embedding.len() != expected_dimension {
        anyhow::bail!(
            "Query embedding dimension {} does not match index dimension {}",
            query_embedding.len(),
            expected_dimension
        );
    }
    let embed_time = start.elapsed();

    // Search sequentially so peak memory is bounded by the largest index rather
    // than the sum of every federated index.
    let mut candidates = Vec::with_capacity(specs.len().saturating_mul(top_k));
    for (index_order, (spec, expected_meta)) in specs.iter().zip(&metas).enumerate() {
        let index = Index::load(&spec.path)
            .with_context(|| format!("Failed to load federated index {}", spec.path.display()))?;
        let loaded_meta = &index.meta;
        if loaded_meta.model_id != expected_meta.model_id
            || loaded_meta.embedding_backend != expected_meta.embedding_backend
            || loaded_meta.hidden_size != expected_meta.hidden_size
            || loaded_meta.format_version != expected_meta.format_version
        {
            anyhow::bail!(
                "Index metadata changed while loading {}",
                spec.path.display()
            );
        }
        candidates.extend(search_one_index(
            spec,
            index_order,
            &index,
            &query_embedding,
            top_k,
            group_by_source,
        )?);
    }
    let results = merge_federated_results(candidates, top_k, group_by_source);
    let search_time = start.elapsed();

    if json {
        let json_results: Vec<JsonResult> = results
            .iter()
            .map(|result| JsonResult {
                index_name: result.index_name.clone(),
                index: result.index_path.display().to_string(),
                root_dir: result.root_dir.clone(),
                source: result.source.clone(),
                score: result.score,
                byte_offset: result.byte_offset,
                text: result.text.clone(),
            })
            .collect();
        println!("{}", serde_json::to_string(&json_results)?);
    } else {
        println!();
        println!("Query: {query}");
        println!("─────────────────────────────────────────");

        if results.is_empty() {
            println!("No results found.");
        } else {
            let federated = specs.len() > 1;
            for (i, result) in results.iter().enumerate() {
                let source = if federated {
                    let spec = specs
                        .get(result.index_order)
                        .context("Search result references an invalid index")?;
                    format!("[{}] {}", spec.label(), result.source)
                } else {
                    result.source.clone()
                };
                let preview = if full {
                    result.text.clone()
                } else {
                    truncate_text(&result.text, 200)
                };

                println!();
                println!(
                    "  [{rank}] {source} (score: {score:.4})",
                    rank = i + 1,
                    score = result.score
                );
                println!("      offset: {} bytes", result.byte_offset);
                println!();
                for line in preview.lines() {
                    println!("      {line}");
                }
            }
        }

        println!();
        println!("─────────────────────────────────────────");
        println!(
            "  {} results from {} index(es) in {:.1}ms (embed: {:.1}ms)",
            results.len(),
            specs.len(),
            search_time.as_secs_f64() * 1000.0,
            embed_time.as_secs_f64() * 1000.0,
        );
    }

    Ok(())
}

fn cmd_info(index_dir: Option<PathBuf>, config_path: Option<&Path>, only: &[String]) -> Result<()> {
    let specs = resolve_search_indexes(index_dir.into_iter().collect(), config_path, only)?;
    for (position, spec) in specs.iter().enumerate() {
        if position > 0 {
            println!();
        }
        print_index_info(spec)?;
    }
    Ok(())
}

fn print_index_info(spec: &SearchIndexSpec) -> Result<()> {
    let index = Index::load(&spec.path).with_context(|| {
        format!(
            "No index found at {}. Run `rag index` first.",
            spec.path.display()
        )
    })?;

    let m = &index.meta;
    let duplicate_chunks = index.occurrences.len().saturating_sub(index.texts.len());
    let index_path = spec.path.join("index.bin");
    let size = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);

    match &spec.name {
        Some(name) => println!("RAG Index Info ({name})"),
        None => println!("RAG Index Info"),
    }
    println!("─────────────────────────────────────────");
    println!("  Index path:    {}", spec.path.display());
    println!("  Root dir:      {}", m.root_dir);
    println!(
        "  Format:        v{} (F16 unique vectors)",
        m.format_version
    );
    println!("  Model:         {}", m.model_id);
    println!("  Hidden size:   {}", m.hidden_size);
    println!("  Chunks:        {}", m.num_chunks);
    println!("  Unique text:   {}", index.texts.len());
    println!("  Duplicates:    {}", duplicate_chunks);
    println!("  Source files:  {}", index.sources.len());
    println!("  Chunk size:    {} chars", m.chunk_size);
    println!("  Chunk overlap: {} chars", m.chunk_overlap);
    println!("  Created:       {}", m.created_at);
    println!("  Index size:    {}", format_bytes(size));

    Ok(())
}

fn truncate_text(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        let mut end = max_chars;
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }
        format!("{}...", &text[..end.min(text.len())])
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn chrono_now() -> String {
    use std::process::Command;
    Command::new("date")
        .arg("+%Y-%m-%dT%H:%M:%S%z")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        compact_records, ignored_config_options_warning, merge_federated_results,
        normalized_metadata_path, resolve_download_models_from_dir,
        resolve_search_indexes_from_dir, search_one_index, unique_text_plan,
        validate_chunk_settings, validate_search_metadata, Cli, Commands, FederatedSearchResult,
        SearchIndexSpec, DEFAULT_MODEL,
    };
    use crate::index::{
        ChunkOccurrence, Index, IndexMeta, SourceRecord, TextRecord, INDEX_FORMAT_VERSION,
    };
    use clap::Parser;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    #[test]
    fn chunk_settings_require_progress_and_smaller_overlap() {
        assert!(validate_chunk_settings(512, 64).is_ok());
        assert!(validate_chunk_settings(0, 0).is_err());
        assert!(validate_chunk_settings(64, 64).is_err());
        assert!(validate_chunk_settings(64, 65).is_err());
    }

    fn meta(model: &str, hidden_size: usize) -> IndexMeta {
        IndexMeta {
            format_version: INDEX_FORMAT_VERSION,
            model_id: model.into(),
            embedding_backend: "coreml-native-fp16".into(),
            hidden_size,
            num_chunks: 1,
            num_unique_texts: 1,
            root_dir: "/docs".into(),
            created_at: "now".into(),
            chunk_size: 512,
            chunk_overlap: 64,
            file_hashes: BTreeMap::new(),
        }
    }

    fn temp_dir(prefix: &str) -> PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("rag-cli-{prefix}-{}-{suffix}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temporary directory should be created");
        dir
    }

    fn result(score: f32, index_order: usize, text_id: usize) -> FederatedSearchResult {
        FederatedSearchResult {
            index_name: None,
            index_path: PathBuf::from(format!(".rag/{index_order}")),
            root_dir: format!("/root/{index_order}"),
            source: format!("{text_id}.md"),
            score,
            byte_offset: 0,
            text: format!("text {text_id}"),
            index_order,
            text_id,
        }
    }

    #[test]
    fn index_path_conflicts_with_config_options() {
        for args in [
            ["rag", "index", "docs", "--config", "rag.toml"].as_slice(),
            ["rag", "index", "docs", "--only", "one"].as_slice(),
        ] {
            let error = Cli::try_parse_from(args)
                .err()
                .expect("index path and config-only options should conflict");
            assert!(error.to_string().contains("cannot be used with"));
        }
    }

    #[test]
    fn config_mode_warning_lists_only_ignored_options() {
        let warning = ignored_config_options_warning(&[
            ("--output", false),
            ("--model", true),
            ("--chunk-size", true),
            ("--include", false),
        ])
        .expect("present options should produce a warning");

        assert!(warning.contains("--model/--chunk-size are ignored"));
        assert!(!warning.contains("--output"));
        assert!(!warning.contains("--include"));
        assert!(ignored_config_options_warning(&[("--model", false)]).is_none());
    }

    #[test]
    fn index_config_mode_retains_explicit_overrides_for_warning() {
        let cli = Cli::try_parse_from([
            "rag",
            "index",
            "--config",
            "rag.toml",
            "--model",
            "custom/model",
            "--chunk-size",
            "256",
            "--chunk-overlap",
            "32",
        ])
        .expect("config mode overrides should parse for a warning");

        let Commands::Index {
            model,
            chunk_size,
            chunk_overlap,
            ..
        } = cli.command
        else {
            panic!("expected index command");
        };
        assert_eq!(model.as_deref(), Some("custom/model"));
        assert_eq!(chunk_size, Some(256));
        assert_eq!(chunk_overlap, Some(32));
    }

    #[test]
    fn search_cli_accepts_repeated_indexes() {
        let cli =
            Cli::try_parse_from(["rag", "search", "query", "-i", ".rag/one", "-i", ".rag/two"])
                .expect("repeated indexes should parse");

        let Commands::Search {
            index,
            config,
            group_by_source,
            ..
        } = cli.command
        else {
            panic!("expected search command");
        };
        assert_eq!(
            index,
            vec![PathBuf::from(".rag/one"), PathBuf::from(".rag/two")]
        );
        assert!(config.is_none());
        assert!(!group_by_source, "source grouping must remain opt-in");
    }

    #[test]
    fn search_cli_accepts_group_by_source() {
        let cli = Cli::try_parse_from(["rag", "search", "query", "--group-by-source"])
            .expect("source grouping should parse");

        let Commands::Search {
            group_by_source, ..
        } = cli.command
        else {
            panic!("expected search command");
        };
        assert!(group_by_source);
    }

    #[test]
    fn search_cli_accepts_config_and_only() {
        let cli = Cli::try_parse_from([
            "rag", "search", "query", "--config", "rag.toml", "--only", "one,two",
        ])
        .expect("config search should parse");

        let Commands::Search {
            index,
            config,
            only,
            ..
        } = cli.command
        else {
            panic!("expected search command");
        };
        assert!(index.is_empty());
        assert_eq!(config, Some(PathBuf::from("rag.toml")));
        assert_eq!(only, vec!["one", "two"]);
    }

    #[test]
    fn search_cli_accepts_only_with_auto_discovered_config() {
        let cli = Cli::try_parse_from(["rag", "search", "query", "--only", "one,two"])
            .expect("auto-config --only should parse");

        let Commands::Search { config, only, .. } = cli.command else {
            panic!("expected search command");
        };
        assert!(config.is_none());
        assert_eq!(only, vec!["one", "two"]);
    }

    #[test]
    fn search_defaults_to_config_discovered_in_current_directory() {
        let dir = temp_dir("auto-config");
        std::fs::write(
            dir.join("rag.toml"),
            "[[index]]\nname = \"docs\"\npath = \"docs\"\noutput = \"indexes/docs\"\n",
        )
        .expect("test config should be written");

        let specs = resolve_search_indexes_from_dir(vec![], None, &[], &dir)
            .expect("default search should discover rag.toml");
        std::fs::remove_dir_all(&dir).expect("temporary directory should be removed");

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name.as_deref(), Some("docs"));
        assert_eq!(specs[0].path, dir.join("indexes/docs"));
    }

    #[test]
    fn download_discovers_and_deduplicates_config_models() {
        let dir = temp_dir("download-config");
        std::fs::write(
            dir.join("rag.toml"),
            "model = \"global/model\"\n\
             [[index]]\nname = \"one\"\npath = \"one\"\n\
             [[index]]\nname = \"two\"\npath = \"two\"\nmodel = \"other/model\"\n\
             [[index]]\nname = \"three\"\npath = \"three\"\nmodel = \"other/model\"\n",
        )
        .expect("test config should be written");

        let models = resolve_download_models_from_dir(None, None, &[], &dir)
            .expect("download should discover config models");
        let selected = resolve_download_models_from_dir(None, None, &["two".into()], &dir)
            .expect("download --only should select one model");
        std::fs::remove_dir_all(&dir).expect("temporary directory should be removed");

        assert_eq!(models, vec!["global/model", "other/model"]);
        assert_eq!(selected, vec!["other/model"]);
    }

    #[test]
    fn download_model_resolution_covers_override_fallback_and_invalid_only() {
        let dir = temp_dir("download-precedence");
        std::fs::write(
            dir.join("rag.toml"),
            "model = \"config/model\"\n[[index]]\nname = \"docs\"\npath = \"docs\"\n",
        )
        .expect("test config should be written");

        let explicit = resolve_download_models_from_dir(Some("explicit/model"), None, &[], &dir)
            .expect("explicit model should win over discovery");
        let missing = resolve_download_models_from_dir(None, None, &["missing".into()], &dir)
            .expect_err("unknown --only name should fail");

        std::fs::remove_file(dir.join("rag.toml")).expect("test config should be removed");
        let fallback = resolve_download_models_from_dir(None, None, &[], &dir)
            .expect("missing config should use built-in model");
        std::fs::remove_dir_all(&dir).expect("temporary directory should be removed");

        assert_eq!(explicit, vec!["explicit/model"]);
        assert!(missing.to_string().contains("missing"));
        assert_eq!(fallback, vec![DEFAULT_MODEL]);
    }

    #[test]
    fn config_only_selection_cli_contract() {
        for args in [
            ["rag", "info", "--only", "docs"].as_slice(),
            ["rag", "download", "--only", "docs"].as_slice(),
        ] {
            Cli::try_parse_from(args).expect("--only should allow auto-discovered config");
        }

        for args in [
            ["rag", "search", "q", "--index", "idx", "--only", "docs"].as_slice(),
            ["rag", "info", "--index", "idx", "--only", "docs"].as_slice(),
            ["rag", "download", "--model", "m", "--only", "docs"].as_slice(),
        ] {
            assert!(
                Cli::try_parse_from(args).is_err(),
                "--only should conflict with an explicit non-config source"
            );
        }
    }

    #[test]
    fn search_config_resolution_follows_precedence_matrix() {
        let dir = temp_dir("search-precedence");
        std::fs::write(
            dir.join("rag.toml"),
            "[[index]]\nname = \"regular\"\npath = \"docs\"\n",
        )
        .expect("regular config should be written");
        std::fs::write(
            dir.join(".rag.toml"),
            "[[index]]\nname = \"dotfile\"\npath = \"docs\"\n",
        )
        .expect("dotfile config should be written");

        let cases = [
            (
                Some(PathBuf::from("explicit-index")),
                None,
                "explicit-index",
            ),
            (None, Some(dir.join(".rag.toml")), ".rag/dotfile"),
            (None, None, ".rag/regular"),
        ];
        for (index, config, expected_suffix) in cases {
            let specs = resolve_search_indexes_from_dir(
                index.into_iter().collect(),
                config.as_deref(),
                &[],
                &dir,
            )
            .expect("search source should resolve");
            assert_eq!(specs.len(), 1);
            assert!(specs[0].path.ends_with(expected_suffix));
        }

        std::fs::remove_file(dir.join("rag.toml")).expect("regular config should be removed");
        let dotfile = resolve_search_indexes_from_dir(vec![], None, &[], &dir)
            .expect("dotfile config should be discovered");
        assert_eq!(dotfile[0].name.as_deref(), Some("dotfile"));

        std::fs::remove_file(dir.join(".rag.toml")).expect("dotfile config should be removed");
        let fallback = resolve_search_indexes_from_dir(vec![], None, &[], &dir)
            .expect("search should fall back to .rag");
        std::fs::remove_dir_all(&dir).expect("temporary directory should be removed");
        assert_eq!(fallback[0].path, PathBuf::from(".rag"));
    }

    #[test]
    fn explicit_config_resolves_output_from_config_not_current_directory() {
        let config_dir = temp_dir("explicit-config");
        let other_cwd = temp_dir("other-cwd");
        let config_path = config_dir.join("custom.toml");
        std::fs::write(
            &config_path,
            "[[index]]\nname = \"docs\"\npath = \"docs\"\noutput = \"indexes/docs\"\n",
        )
        .expect("explicit config should be written");

        let specs = resolve_search_indexes_from_dir(vec![], Some(&config_path), &[], &other_cwd)
            .expect("explicit config should resolve");
        std::fs::remove_dir_all(&config_dir).expect("config directory should be removed");
        std::fs::remove_dir_all(&other_cwd).expect("other cwd should be removed");

        assert_eq!(specs[0].path, config_dir.join("indexes/docs"));
    }

    #[test]
    fn federated_metadata_requires_one_model_and_dimension() {
        let specs = vec![
            SearchIndexSpec {
                name: Some("one".into()),
                path: PathBuf::from(".rag/one"),
            },
            SearchIndexSpec {
                name: Some("two".into()),
                path: PathBuf::from(".rag/two"),
            },
        ];

        assert_eq!(
            validate_search_metadata(&specs, &[meta("m", 2), meta("m", 2)], None)
                .expect("matching indexes should federate"),
            "m"
        );
        assert!(
            validate_search_metadata(&specs, &[meta("m", 2), meta("other", 2)], None)
                .expect_err("different models must fail")
                .to_string()
                .contains("different models")
        );
        assert!(
            validate_search_metadata(&specs, &[meta("m", 2), meta("m", 3)], None)
                .expect_err("different dimensions must fail")
                .to_string()
                .contains("different embedding dimensions")
        );
    }

    #[test]
    fn grouped_search_continues_past_same_source_chunks() {
        let index = Index::new(
            meta("m", 2),
            vec![
                SourceRecord {
                    path: "a.md".into(),
                },
                SourceRecord {
                    path: "b.md".into(),
                },
                SourceRecord {
                    path: "c.md".into(),
                },
            ],
            vec![
                TextRecord::new("a best".into(), vec![1.0, 0.0]),
                TextRecord::new("a second".into(), vec![0.99, 0.1]),
                TextRecord::new("a third".into(), vec![0.98, 0.2]),
                TextRecord::new("b best".into(), vec![0.8, 0.6]),
                TextRecord::new("c best".into(), vec![0.7, 0.714]),
            ],
            vec![
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 0,
                    byte_offset: 0,
                },
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 1,
                    byte_offset: 10,
                },
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 2,
                    byte_offset: 20,
                },
                ChunkOccurrence {
                    source_id: 1,
                    text_id: 3,
                    byte_offset: 30,
                },
                ChunkOccurrence {
                    source_id: 2,
                    text_id: 4,
                    byte_offset: 40,
                },
            ],
        );
        let spec = SearchIndexSpec {
            name: None,
            path: ".rag".into(),
        };

        let grouped = search_one_index(&spec, 0, &index, &[1.0, 0.0], 3, true)
            .expect("grouped search should succeed");

        assert_eq!(
            grouped
                .iter()
                .map(|item| item.source.as_str())
                .collect::<Vec<_>>(),
            vec!["a.md", "b.md", "c.md"]
        );
        assert_eq!(grouped[0].text, "a best");
    }

    #[test]
    fn grouped_search_expands_identical_text_to_each_source() {
        let index = Index::new(
            meta("m", 2),
            vec![
                SourceRecord {
                    path: "first.md".into(),
                },
                SourceRecord {
                    path: "second.md".into(),
                },
            ],
            vec![TextRecord::new("shared text".into(), vec![1.0, 0.0])],
            vec![
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 0,
                    byte_offset: 11,
                },
                ChunkOccurrence {
                    source_id: 1,
                    text_id: 0,
                    byte_offset: 22,
                },
            ],
        );
        let spec = SearchIndexSpec {
            name: None,
            path: ".rag".into(),
        };

        let default = search_one_index(&spec, 0, &index, &[1.0, 0.0], 2, false)
            .expect("default search should succeed");
        let grouped = search_one_index(&spec, 0, &index, &[1.0, 0.0], 2, true)
            .expect("grouped search should succeed");

        assert_eq!(default.len(), 1);
        assert_eq!(default[0].source, "first.md");
        assert_eq!(
            grouped
                .iter()
                .map(|item| (item.source.as_str(), item.byte_offset))
                .collect::<Vec<_>>(),
            vec![("first.md", 11), ("second.md", 22)]
        );
    }

    #[test]
    fn grouped_search_keeps_each_sources_highest_scoring_occurrence() {
        let index = Index::new(
            meta("m", 2),
            vec![SourceRecord {
                path: "only.md".into(),
            }],
            vec![
                TextRecord::new("lower".into(), vec![0.6, 0.8]),
                TextRecord::new("higher".into(), vec![1.0, 0.0]),
            ],
            vec![
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 0,
                    byte_offset: 5,
                },
                ChunkOccurrence {
                    source_id: 0,
                    text_id: 1,
                    byte_offset: 50,
                },
            ],
        );
        let spec = SearchIndexSpec {
            name: None,
            path: ".rag".into(),
        };

        let grouped = search_one_index(&spec, 0, &index, &[1.0, 0.0], 2, true)
            .expect("grouped search should succeed");

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].text, "higher");
        assert_eq!(grouped[0].byte_offset, 50);
    }

    #[test]
    fn grouped_federated_search_deduplicates_canonical_sources_and_refills() {
        let make_index = |shared_score: [f32; 2], unique: &str, unique_score: [f32; 2]| {
            let mut metadata = meta("m", 2);
            metadata.root_dir = "/workspace".into();
            Index::new(
                metadata,
                vec![
                    SourceRecord {
                        path: "shared.md".into(),
                    },
                    SourceRecord {
                        path: unique.into(),
                    },
                    SourceRecord {
                        path: format!("extra-{unique}"),
                    },
                ],
                vec![
                    TextRecord::new(format!("shared from {unique}"), shared_score.to_vec()),
                    TextRecord::new(format!("unique {unique}"), unique_score.to_vec()),
                    TextRecord::new(format!("extra {unique}"), vec![0.6, 0.8]),
                ],
                vec![
                    ChunkOccurrence {
                        source_id: 0,
                        text_id: 0,
                        byte_offset: 20,
                    },
                    ChunkOccurrence {
                        source_id: 1,
                        text_id: 1,
                        byte_offset: 10,
                    },
                    ChunkOccurrence {
                        source_id: 2,
                        text_id: 2,
                        byte_offset: 30,
                    },
                ],
            )
        };
        let specs = [
            SearchIndexSpec {
                name: Some("one".into()),
                path: ".rag/one".into(),
            },
            SearchIndexSpec {
                name: Some("two".into()),
                path: ".rag/two".into(),
            },
        ];
        let indexes = [
            make_index([0.99, 0.141], "alpha.md", [0.8, 0.6]),
            make_index([1.0, 0.0], "beta.md", [0.9, 0.436]),
        ];
        let mut candidates = Vec::new();
        for (order, (spec, index)) in specs.iter().zip(&indexes).enumerate() {
            let local = search_one_index(spec, order, index, &[1.0, 0.0], 2, true)
                .expect("grouped index search should succeed");
            assert_eq!(
                local.len(),
                3,
                "each index should overfetch source candidates"
            );
            candidates.extend(local);
        }

        let default = merge_federated_results(candidates.clone(), 2, false);
        assert_eq!(
            default
                .iter()
                .filter(|result| result.source == "shared.md")
                .count(),
            2,
            "the default merge must preserve overlapping index results"
        );

        let grouped = merge_federated_results(candidates, 2, true);
        assert_eq!(
            grouped
                .iter()
                .map(|result| (result.source.as_str(), result.text.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("shared.md", "shared from beta.md"),
                ("beta.md", "unique beta.md"),
            ]
        );
    }

    #[test]
    fn grouped_federated_search_scopes_relative_roots_to_each_index() {
        let relative = |index_path: &str, score: f32, index_order: usize| {
            let mut item = result(score, index_order, 0);
            item.index_path = index_path.into();
            item.root_dir = ".".into();
            item.source = "same-name.md".into();
            item
        };

        let grouped = merge_federated_results(
            vec![
                relative("/corpus-one/.rag", 1.0, 0),
                relative("/corpus-two/.rag", 0.9, 1),
            ],
            2,
            true,
        );

        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].index_path, PathBuf::from("/corpus-one/.rag"));
        assert_eq!(grouped[1].index_path, PathBuf::from("/corpus-two/.rag"));
    }

    #[test]
    fn grouped_search_does_not_materialize_duplicate_occurrences() {
        const DUPLICATES: usize = 100_000;
        let occurrences = (0..DUPLICATES)
            .rev()
            .map(|byte_offset| ChunkOccurrence {
                source_id: 0,
                text_id: 0,
                byte_offset,
            })
            .collect();
        let index = Index::new(
            meta("m", 1),
            vec![SourceRecord {
                path: "duplicate-heavy.md".into(),
            }],
            vec![TextRecord::new("repeated".into(), vec![1.0])],
            occurrences,
        );
        let spec = SearchIndexSpec {
            name: None,
            path: ".rag".into(),
        };

        let grouped = search_one_index(&spec, 0, &index, &[1.0], 5, true)
            .expect("duplicate-heavy grouped search should succeed");

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].byte_offset, 0);
    }

    #[test]
    fn grouped_cutoff_ties_are_deterministic_by_source_and_offset() {
        let tied = |root: &str, source: &str, offset: usize, index_order: usize| {
            let mut item = result(1.0, index_order, index_order);
            item.root_dir = root.into();
            item.source = source.into();
            item.byte_offset = offset;
            item
        };
        let candidates = vec![
            tied("/z", "a.md", 1, 0),
            tied("/a", "c.md", 1, 1),
            tied("/a", "a.md", 20, 2),
            tied("/a", "b.md", 50, 3),
            tied("/a", "a.md", 10, 4),
        ];
        let expected = vec![("/a", "a.md", 10), ("/a", "b.md", 50)];

        let forward = merge_federated_results(candidates.clone(), 2, true);
        let reverse = merge_federated_results(candidates.into_iter().rev().collect(), 2, true);
        for merged in [forward, reverse] {
            assert_eq!(
                merged
                    .iter()
                    .map(|item| {
                        (
                            item.root_dir.as_str(),
                            item.source.as_str(),
                            item.byte_offset,
                        )
                    })
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn federated_merge_returns_global_top_k_with_stable_ties() {
        let merged = merge_federated_results(
            vec![
                result(0.8, 1, 0),
                result(0.9, 1, 1),
                result(0.9, 0, 2),
                result(0.7, 0, 3),
            ],
            3,
            false,
        );

        assert_eq!(
            merged
                .iter()
                .map(|item| (item.score, item.index_order, item.text_id))
                .collect::<Vec<_>>(),
            vec![(0.9, 0, 2), (0.9, 1, 1), (0.8, 1, 0)]
        );
    }

    #[test]
    fn metadata_paths_remain_relative_and_drop_redundant_current_dir() {
        assert_eq!(
            normalized_metadata_path(Path::new("./docs/content")),
            "docs/content"
        );
        assert_eq!(
            normalized_metadata_path(Path::new("../shared/docs")),
            "../shared/docs"
        );
        assert_eq!(normalized_metadata_path(Path::new(".")), ".");
    }

    #[test]
    fn absolute_metadata_paths_remain_absolute() {
        let path = if cfg!(windows) {
            Path::new(r"C:\docs\content")
        } else {
            Path::new("/docs/content")
        };
        assert!(Path::new(&normalized_metadata_path(path)).is_absolute());
    }

    #[test]
    fn relative_metadata_round_trip_remains_reusable() {
        let metadata_root = normalized_metadata_path(Path::new("./docs/content"));
        let mut metadata = meta("m", 2);
        metadata.root_dir = metadata_root.clone();

        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "rag-cli-relative-metadata-{}-{suffix}",
            std::process::id()
        ));
        Index::new(metadata, vec![], vec![], vec![])
            .save(&dir)
            .expect("relative metadata should save");
        let loaded = Index::load_meta(&dir).expect("relative metadata should load");
        std::fs::remove_dir_all(&dir).expect("temporary index should be removed");

        let can_reuse = loaded.root_dir == metadata_root
            && loaded.reusable_for("m", "coreml-native-fp16", 512, 64);
        assert!(can_reuse, "relative metadata should permit index reuse");
    }

    #[test]
    fn unique_text_plan_reuses_duplicate_ids_in_input_order() {
        let input = ["alpha", "beta", "alpha", "gamma", "beta"];
        let (unique, ids) = unique_text_plan(input.into_iter());

        assert_eq!(unique, vec!["alpha", "beta", "gamma"]);
        assert_eq!(ids, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn compact_records_removes_orphans_and_remaps_occurrences() {
        let mut sources = vec![
            SourceRecord {
                path: "removed.md".into(),
            },
            SourceRecord {
                path: "kept.md".into(),
            },
        ];
        let mut texts = vec![
            TextRecord::new("removed".into(), vec![1.0, 0.0]),
            TextRecord::new("kept".into(), vec![0.0, 1.0]),
        ];
        let mut occurrences = vec![ChunkOccurrence {
            source_id: 1,
            text_id: 1,
            byte_offset: 42,
        }];

        compact_records(&mut sources, &mut texts, &mut occurrences)
            .expect("valid records should compact");

        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].path, "kept.md");
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0].text, "kept");
        assert_eq!(occurrences[0].source_id, 0);
        assert_eq!(occurrences[0].text_id, 0);
        assert_eq!(occurrences[0].byte_offset, 42);
    }

    #[test]
    fn compact_records_rejects_invalid_ids() {
        let mut sources = vec![];
        let mut texts = vec![];
        let mut occurrences = vec![ChunkOccurrence {
            source_id: 0,
            text_id: 0,
            byte_offset: 0,
        }];

        let error = compact_records(&mut sources, &mut texts, &mut occurrences)
            .expect_err("invalid IDs must not silently remap to zero");

        assert!(error.to_string().contains("invalid source ID"));
    }
}
