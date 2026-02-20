mod embed;
mod index;
mod ingest;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

use std::collections::BTreeMap;

use crate::embed::{select_device, EmbeddingEngine, DEFAULT_MODEL};
use crate::index::{search_top_k, ChunkRecord, Index, IndexMeta};
use crate::ingest::{chunk_file, discover_files, hash_files};

const DEFAULT_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_OVERLAP: usize = 64;
const DEFAULT_TOP_K: usize = 5;

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
    /// Index a directory of text files for semantic search.
    Index {
        /// Directory to index.
        path: PathBuf,

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
    },

    /// Search the index with a natural language query.
    Search {
        /// The search query.
        query: String,

        /// Index directory (default: .rag).
        #[arg(short, long)]
        index: Option<PathBuf>,

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
}

/// Entry point for the CLI. Call this from `main()`.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let cache_dir = cli.cache_dir.as_deref();

    match cli.command {
        Commands::Index {
            path,
            output,
            model,
            chunk_size,
            chunk_overlap,
        } => cmd_index(
            &path,
            output.as_deref(),
            &model,
            chunk_size,
            chunk_overlap,
            cache_dir,
        ),
        Commands::Search {
            query,
            index,
            top_k,
            model,
            full,
            json,
        } => cmd_search(
            &query,
            index.as_deref(),
            top_k,
            model.as_deref(),
            full,
            json,
            cache_dir,
        ),
        Commands::Info { index } => cmd_info(index.as_deref()),
    }
}

fn cmd_index(
    path: &PathBuf,
    output: Option<&std::path::Path>,
    model_id: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    cache_dir: Option<&std::path::Path>,
) -> Result<()> {
    let start = Instant::now();

    let root = path
        .canonicalize()
        .with_context(|| format!("Directory not found: {}", path.display()))?;

    if !root.is_dir() {
        anyhow::bail!("{} is not a directory", root.display());
    }

    let index_dir = output.map(PathBuf::from).unwrap_or_else(Index::default_dir);

    // 1. Discover files and hash them
    eprintln!("Indexing: {}", root.display());
    let files = discover_files(&root)?;
    eprintln!("Found {} text files", files.len());

    if files.is_empty() {
        anyhow::bail!("No text files found in {}", root.display());
    }

    let current_hashes = hash_files(&files, &root)?;

    // 2. Try incremental indexing against an existing index
    let prev_index = Index::load(&index_dir).ok();
    let can_reuse = prev_index.as_ref().is_some_and(|prev| {
        prev.meta.model_id == model_id
            && prev.meta.chunk_size == chunk_size
            && prev.meta.chunk_overlap == chunk_overlap
    });

    let (chunks, file_hashes, hidden_size) = if can_reuse {
        let prev = prev_index.unwrap();
        incremental_index(
            &root,
            &files,
            &current_hashes,
            &prev,
            model_id,
            chunk_size,
            chunk_overlap,
            cache_dir,
        )?
    } else {
        if prev_index.is_some() {
            eprintln!("Settings changed, performing full re-index");
        }
        full_index(
            &root,
            &files,
            &current_hashes,
            model_id,
            chunk_size,
            chunk_overlap,
            cache_dir,
        )?
    };

    if chunks.is_empty() {
        anyhow::bail!("No text chunks produced. Check the directory contents.");
    }

    // 3. Save index
    let meta = IndexMeta {
        model_id: model_id.to_string(),
        hidden_size,
        num_chunks: chunks.len(),
        root_dir: root.to_string_lossy().to_string(),
        created_at: chrono_now(),
        chunk_size,
        chunk_overlap,
        file_hashes,
    };

    let index = Index::new(meta, chunks);
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

/// Full re-index: chunk and embed every file from scratch.
fn full_index(
    root: &std::path::Path,
    files: &[PathBuf],
    current_hashes: &BTreeMap<String, String>,
    model_id: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    cache_dir: Option<&std::path::Path>,
) -> Result<(Vec<ChunkRecord>, BTreeMap<String, String>, usize)> {
    let mut all_chunks = Vec::new();
    for file in files {
        match chunk_file(file, root, chunk_size, chunk_overlap) {
            Ok(chunks) => all_chunks.extend(chunks),
            Err(e) => eprintln!("  Skipping {}: {e}", file.display()),
        }
    }

    eprintln!("Embedding {} chunks...", all_chunks.len());
    let device = select_device()?;
    let engine = EmbeddingEngine::load(Some(model_id), &device, cache_dir)?;
    let hidden_size = engine.hidden_size();

    let texts: Vec<String> = all_chunks.iter().map(|c| c.text.clone()).collect();
    let embeddings = engine.embed_batch_progress(&texts)?;

    let chunks: Vec<ChunkRecord> = all_chunks
        .into_iter()
        .zip(embeddings)
        .map(|(tc, emb)| ChunkRecord {
            source: tc.source,
            byte_offset: tc.byte_offset,
            text: tc.text,
            embedding: emb,
        })
        .collect();

    Ok((chunks, current_hashes.clone(), hidden_size))
}

/// Incremental re-index: only re-chunk and re-embed changed/new files.
fn incremental_index(
    root: &std::path::Path,
    files: &[PathBuf],
    current_hashes: &BTreeMap<String, String>,
    prev: &Index,
    model_id: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    cache_dir: Option<&std::path::Path>,
) -> Result<(Vec<ChunkRecord>, BTreeMap<String, String>, usize)> {
    // Partition files into unchanged vs. changed/new
    let mut unchanged: Vec<&str> = Vec::new();
    let mut dirty_files: Vec<&PathBuf> = Vec::new();

    for file in files {
        let relative = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();

        let cur_hash = current_hashes.get(&relative);
        let prev_hash = prev.meta.file_hashes.get(&relative);

        if cur_hash.is_some() && cur_hash == prev_hash {
            unchanged.push(
                prev.meta
                    .file_hashes
                    .get_key_value(&relative)
                    .unwrap()
                    .0
                    .as_str(),
            );
        } else {
            dirty_files.push(file);
        }
    }

    let deleted: Vec<&str> = prev
        .meta
        .file_hashes
        .keys()
        .filter(|k| !current_hashes.contains_key(k.as_str()))
        .map(|k| k.as_str())
        .collect();

    eprintln!(
        "Incremental: {} unchanged, {} changed/new, {} deleted",
        unchanged.len(),
        dirty_files.len(),
        deleted.len(),
    );

    // Keep chunks from unchanged files
    let mut chunks: Vec<ChunkRecord> = prev
        .chunks
        .iter()
        .filter(|c| unchanged.contains(&c.source.as_str()))
        .cloned()
        .collect();

    // Chunk and embed dirty files
    let hidden_size = if !dirty_files.is_empty() {
        let mut new_text_chunks = Vec::new();
        for file in &dirty_files {
            match chunk_file(file, root, chunk_size, chunk_overlap) {
                Ok(cs) => new_text_chunks.extend(cs),
                Err(e) => eprintln!("  Skipping {}: {e}", file.display()),
            }
        }

        if !new_text_chunks.is_empty() {
            eprintln!("Embedding {} new/changed chunks...", new_text_chunks.len());
            let device = select_device()?;
            let engine = EmbeddingEngine::load(Some(model_id), &device, cache_dir)?;
            let hs = engine.hidden_size();

            let texts: Vec<String> = new_text_chunks.iter().map(|c| c.text.clone()).collect();
            let embeddings = engine.embed_batch_progress(&texts)?;

            let new_chunks: Vec<ChunkRecord> = new_text_chunks
                .into_iter()
                .zip(embeddings)
                .map(|(tc, emb)| ChunkRecord {
                    source: tc.source,
                    byte_offset: tc.byte_offset,
                    text: tc.text,
                    embedding: emb,
                })
                .collect();

            chunks.extend(new_chunks);
            hs
        } else {
            prev.meta.hidden_size
        }
    } else {
        eprintln!("Everything up to date, nothing to embed");
        prev.meta.hidden_size
    };

    Ok((chunks, current_hashes.clone(), hidden_size))
}

/// A single search result for JSON output.
#[derive(serde::Serialize)]
struct JsonResult {
    source: String,
    score: f32,
    byte_offset: usize,
    text: String,
}

fn cmd_search(
    query: &str,
    index_dir: Option<&std::path::Path>,
    top_k: usize,
    model_override: Option<&str>,
    full: bool,
    json: bool,
    cache_dir: Option<&std::path::Path>,
) -> Result<()> {
    let start = Instant::now();

    let index_dir = index_dir
        .map(PathBuf::from)
        .unwrap_or_else(Index::default_dir);

    let index = Index::load(&index_dir).with_context(|| {
        format!(
            "No index found at {}. Run `rag index <path>` first.",
            index_dir.display()
        )
    })?;

    let model_id = model_override.unwrap_or(&index.meta.model_id);

    // Load model and embed the query
    let device = select_device()?;
    let engine = EmbeddingEngine::load(Some(model_id), &device, cache_dir)?;
    let query_embedding = engine.embed_one(query)?;

    let embed_time = start.elapsed();

    // Search
    let results = search_top_k(&query_embedding, &index.chunks, top_k);

    let search_time = start.elapsed();

    if json {
        let json_results: Vec<JsonResult> = results
            .iter()
            .map(|r| JsonResult {
                source: r.chunk.source.clone(),
                score: r.score,
                byte_offset: r.chunk.byte_offset,
                text: r.chunk.text.clone(),
            })
            .collect();
        println!("{}", serde_json::to_string(&json_results)?);
    } else {
        // Print results
        println!();
        println!("Query: {query}");
        println!("─────────────────────────────────────────");

        if results.is_empty() {
            println!("No results found.");
        } else {
            for (i, result) in results.iter().enumerate() {
                let preview = if full {
                    result.chunk.text.clone()
                } else {
                    truncate_text(&result.chunk.text, 200)
                };

                println!();
                println!(
                    "  [{rank}] {source} (score: {score:.4})",
                    rank = i + 1,
                    source = result.chunk.source,
                    score = result.score
                );
                println!("      offset: {} bytes", result.chunk.byte_offset);
                println!();
                for line in preview.lines() {
                    println!("      {line}");
                }
            }
        }

        println!();
        println!("─────────────────────────────────────────");
        println!(
            "  {} results in {:.1}ms (embed: {:.1}ms)",
            results.len(),
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

    let mut sources: Vec<&str> = index.chunks.iter().map(|c| c.source.as_str()).collect();
    sources.sort();
    sources.dedup();

    let index_path = index_dir.join("index.bin");
    let size = std::fs::metadata(&index_path).map(|m| m.len()).unwrap_or(0);

    println!("RAG Index Info");
    println!("─────────────────────────────────────────");
    println!("  Index path:    {}", index_dir.display());
    println!("  Root dir:      {}", m.root_dir);
    println!("  Model:         {}", m.model_id);
    println!("  Hidden size:   {}", m.hidden_size);
    println!("  Chunks:        {}", m.num_chunks);
    println!("  Source files:  {}", sources.len());
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
