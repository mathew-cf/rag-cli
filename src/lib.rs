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

        /// HuggingFace model ID for embeddings.
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,

        /// Chunk size in characters.
        #[arg(long, default_value_t = DEFAULT_CHUNK_SIZE)]
        chunk_size: usize,

        /// Chunk overlap in characters.
        #[arg(long, default_value_t = DEFAULT_CHUNK_OVERLAP)]
        chunk_overlap: usize,

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

        /// Index directories to search (repeatable; default: .rag).
        #[arg(short, long, conflicts_with = "config")]
        index: Vec<PathBuf>,

        /// Search every index declared by this config file.
        #[arg(short = 'c', long, conflicts_with = "index")]
        config: Option<PathBuf>,

        /// With --config, search only these named indexes (comma-separated or repeated).
        #[arg(long, value_delimiter = ',', requires = "config")]
        only: Vec<String>,

        /// Number of results to return.
        #[arg(short = 'k', long, default_value_t = DEFAULT_TOP_K)]
        top_k: usize,

        /// HuggingFace model ID (must match the one used for indexing).
        #[arg(short, long)]
        model: Option<String>,

        /// Show full chunk text instead of truncated preview.
        #[arg(long)]
        full: bool,

        /// Output results as compact JSON (for piping to LLMs or other tools).
        #[arg(long)]
        json: bool,
    },

    /// Show index metadata and statistics.
    Info {
        /// Index directory (default: .rag).
        #[arg(short, long)]
        index: Option<PathBuf>,
    },

    /// Pre-download the embedding model weights + tokenizer into the cache.
    ///
    /// Useful on fresh installs: makes the first `rag index`/`rag search`
    /// fast instead of stalling on a ~90MB network fetch. Safe to re-run —
    /// if every file is already cached it exits immediately.
    Download {
        /// HuggingFace model ID for embeddings.
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: String,

        /// Verify the model loads successfully after downloading (runs a
        /// tiny inference to catch corrupt downloads).
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
                // `--only` selects among config entries; it has no meaning for a
                // single ad-hoc index.
                if !only.is_empty() {
                    eprintln!("warning: --only is ignored when a PATH is given");
                }
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
                    &model,
                    chunk_size,
                    chunk_overlap,
                    &discovery,
                    cache_dir,
                )
            }
            None => {
                // Config mode: each `[[index]]` entry is self-describing, so the
                // ad-hoc single-index flags don't apply. Warn rather than
                // silently ignore them.
                if output.is_some() || !ext.is_empty() || !exclude.is_empty() || !include.is_empty()
                {
                    eprintln!(
                        "warning: --output/--ext/--exclude/--include are ignored when building \
                         from a config file; set them per-[[index]] in the config instead"
                    );
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
            full,
            json,
        } => cmd_search(SearchSettings {
            query: &query,
            index_dirs: index,
            config_path: config.as_deref(),
            only: &only,
            top_k,
            model_override: model.as_deref(),
            full,
            json,
            cache_dir,
        }),
        Commands::Info { index } => cmd_info(index.as_deref()),
        Commands::Download { model, verify } => cmd_download(&model, verify, cache_dir),
    }
}

fn cmd_download(model_id: &str, verify: bool, cache_dir: Option<&std::path::Path>) -> Result<()> {
    let start = Instant::now();
    let hub_root = resolve_hf_cache(cache_dir)?;

    eprintln!("Model: {model_id}");
    eprintln!("Cache: {}", hub_root.display());

    if model_files_present(&hub_root, model_id) {
        eprintln!(
            "All {} model file(s) already cached.",
            model_file_list().len()
        );
    } else {
        let downloaded = download_model(model_id, cache_dir)?;
        if downloaded {
            eprintln!(
                "Downloaded {} file(s) in {:.1}s",
                model_file_list().len(),
                start.elapsed().as_secs_f64()
            );
        }
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
        let output = entry
            .output
            .clone()
            .unwrap_or_else(|| base.join(".rag").join(&entry.name));

        let model = entry
            .model
            .clone()
            .or_else(|| config.model.clone())
            .unwrap_or_else(|| DEFAULT_MODEL.to_string());
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

#[derive(Debug)]
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
    let cwd = std::env::current_dir().context("Failed to determine current directory")?;
    let config_path = RagConfig::locate(Some(config_path), &cwd)
        .context("Search config path could not be resolved")?;
    let base = config_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
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

    Ok(config
        .indexes
        .into_iter()
        .filter(|entry| only.is_empty() || only.iter().any(|name| name == &entry.name))
        .map(|entry| {
            let output = entry
                .output
                .unwrap_or_else(|| PathBuf::from(".rag").join(&entry.name));
            SearchIndexSpec {
                path: base.join(output),
                name: Some(entry.name),
            }
        })
        .collect())
}

fn resolve_search_indexes(
    index_dirs: Vec<PathBuf>,
    config_path: Option<&Path>,
    only: &[String],
) -> Result<Vec<SearchIndexSpec>> {
    if let Some(config_path) = config_path {
        return configured_search_indexes(config_path, only);
    }
    if index_dirs.is_empty() {
        return Ok(vec![SearchIndexSpec {
            name: None,
            path: Index::default_dir(),
        }]);
    }
    Ok(index_dirs
        .into_iter()
        .map(|path| SearchIndexSpec { name: None, path })
        .collect())
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
) -> Result<Vec<FederatedSearchResult>> {
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

    results
        .into_iter()
        .map(|result| {
            let occurrence = representative
                .get(result.text_id)
                .and_then(|occurrence| *occurrence)
                .context("Active text record has no source occurrence")?;
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
        })
        .collect()
}

fn merge_federated_results(
    mut results: Vec<FederatedSearchResult>,
    top_k: usize,
) -> Vec<FederatedSearchResult> {
    let by_score = |a: &FederatedSearchResult, b: &FederatedSearchResult| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.index_order.cmp(&b.index_order))
            .then_with(|| a.text_id.cmp(&b.text_id))
    };
    if results.len() > top_k {
        results.select_nth_unstable_by(top_k, by_score);
        results.truncate(top_k);
    }
    results.sort_by(by_score);
    results
}

fn cmd_search(settings: SearchSettings<'_>) -> Result<()> {
    let SearchSettings {
        query,
        index_dirs,
        config_path,
        only,
        top_k,
        model_override,
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
        )?);
    }
    let results = merge_federated_results(candidates, top_k);
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

fn cmd_info(index_dir: Option<&std::path::Path>) -> Result<()> {
    let index_dir = index_dir
        .map(PathBuf::from)
        .unwrap_or_else(Index::default_dir);

    let index = Index::load(&index_dir).with_context(|| {
        format!(
            "No index found at {}. Run `rag index <path>` first.",
            index_dir.display()
        )
    })?;

    let m = &index.meta;
    let duplicate_chunks = index.occurrences.len().saturating_sub(index.texts.len());

    let index_path = index_dir.join("index.bin");
    let size = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);

    println!("RAG Index Info");
    println!("─────────────────────────────────────────");
    println!("  Index path:    {}", index_dir.display());
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
        compact_records, merge_federated_results, normalized_metadata_path, unique_text_plan,
        validate_search_metadata, Cli, Commands, FederatedSearchResult, SearchIndexSpec,
    };
    use crate::index::{
        ChunkOccurrence, Index, IndexMeta, SourceRecord, TextRecord, INDEX_FORMAT_VERSION,
    };
    use clap::Parser;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

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
    fn search_cli_accepts_repeated_indexes() {
        let cli =
            Cli::try_parse_from(["rag", "search", "query", "-i", ".rag/one", "-i", ".rag/two"])
                .expect("repeated indexes should parse");

        let Commands::Search { index, config, .. } = cli.command else {
            panic!("expected search command");
        };
        assert_eq!(
            index,
            vec![PathBuf::from(".rag/one"), PathBuf::from(".rag/two")]
        );
        assert!(config.is_none());
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
    fn federated_merge_returns_global_top_k_with_stable_ties() {
        let merged = merge_federated_results(
            vec![
                result(0.8, 1, 0),
                result(0.9, 1, 1),
                result(0.9, 0, 2),
                result(0.7, 0, 3),
            ],
            3,
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
