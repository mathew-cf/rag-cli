//! `rag.toml` config-file support.
//!
//! A config file declares one or more named indexes so a whole set can be
//! built with a single `rag index` (no path argument) instead of a shell
//! script that calls `rag index` once per directory. Global defaults at the
//! top of the file apply to every `[[index]]` entry unless the entry overrides
//! them.
//!
//! ```toml
//! # global defaults (all optional)
//! model = "sentence-transformers/all-MiniLM-L6-v2"
//! chunk_size = 512
//! chunk_overlap = 64
//!
//! [[index]]
//! name = "akamai"                       # output defaults to .rag/<name>
//! path = "akamai-property-mgr-docs/md"  # resolved relative to the config file
//! extensions = []
//! exclude = ["deprecated"]
//! include = []
//!
//! [[index]]
//! name = "cloudflare-docs"
//! path = "cloudflare-docs/src/content"
//! extensions = ["mdx"]
//! exclude = ["changelog", "partials"]
//! ```

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// Config-file names looked up (in order) when no `--config` is given.
pub const DEFAULT_CONFIG_NAMES: &[&str] = &["rag.toml", ".rag.toml"];

/// Top-level `rag.toml` document.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RagConfig {
    /// Default embedding model for entries that don't set their own.
    pub model: Option<String>,
    /// Default chunk size for entries that don't set their own.
    pub chunk_size: Option<usize>,
    /// Default chunk overlap for entries that don't set their own.
    pub chunk_overlap: Option<usize>,
    /// The indexes to build.
    #[serde(default, rename = "index")]
    pub indexes: Vec<IndexEntry>,
}

/// A single `[[index]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    /// Logical name; also the default output dir (`.rag/<name>`).
    pub name: String,
    /// Directory to index, resolved relative to the config file's directory.
    pub path: PathBuf,
    /// Output index directory. Defaults to `.rag/<name>` (relative to the
    /// config file's directory).
    pub output: Option<PathBuf>,
    /// Per-entry model override.
    pub model: Option<String>,
    /// Per-entry chunk size override.
    pub chunk_size: Option<usize>,
    /// Per-entry chunk overlap override.
    pub chunk_overlap: Option<usize>,
    /// Extra file extensions beyond the built-in allowlist.
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Directory/file specs to skip.
    #[serde(default)]
    pub exclude: Vec<String>,
    /// Normally-skipped directories to re-include.
    #[serde(default)]
    pub include: Vec<String>,
}

impl RagConfig {
    /// Parse a config file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: RagConfig = toml::from_str(&text)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        if config.indexes.is_empty() {
            bail!(
                "Config file {} declares no [[index]] entries",
                path.display()
            );
        }
        // Reject duplicate names — they'd otherwise clobber each other's default
        // output directory (.rag/<name>).
        for (i, entry) in config.indexes.iter().enumerate() {
            if let Some(dup) = config.indexes[..i].iter().find(|e| e.name == entry.name) {
                bail!("Duplicate index name in {}: {:?}", path.display(), dup.name);
            }
        }
        Ok(config)
    }

    /// Locate a config file: the explicit `--config` path if given, otherwise
    /// the first of [`DEFAULT_CONFIG_NAMES`] present in `dir`.
    pub fn locate(explicit: Option<&Path>, dir: &Path) -> Option<PathBuf> {
        if let Some(p) = explicit {
            return Some(p.to_path_buf());
        }
        DEFAULT_CONFIG_NAMES
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.is_file())
    }
}
