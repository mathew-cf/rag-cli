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
//! no_ignore = false
//! hidden = false
//! [search]
//! hybrid = true
//! semantic_weight = 1.0
//! keyword_weight = 2.0
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
    /// Default file-walk behavior for all entries.
    pub no_ignore: Option<bool>,
    pub hidden: Option<bool>,
    /// Defaults for `rag search`; CLI options take precedence.
    pub search: Option<SearchConfig>,
    /// The indexes to build.
    #[serde(default, rename = "index")]
    pub indexes: Vec<IndexEntry>,
}

/// Search behavior shared by all indexes in a config file.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchConfig {
    pub hybrid: Option<bool>,
    pub semantic_weight: Option<f32>,
    pub keyword_weight: Option<f32>,
}

/// A single `[[index]]` entry.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    /// Logical name; also the default output dir (`.rag/<name>`).
    pub name: String,
    /// Directory to index, resolved relative to the config file's directory.
    pub path: Option<PathBuf>,
    /// Multiple directories sharing one index, resolved relative to the config.
    pub paths: Option<Vec<PathBuf>>,
    /// Output index directory. Defaults to `.rag/<name>` (relative to the
    /// config file's directory).
    pub output: Option<PathBuf>,
    /// Per-entry model override.
    pub model: Option<String>,
    /// Per-entry chunk size override.
    pub chunk_size: Option<usize>,
    /// Per-entry chunk overlap override.
    pub chunk_overlap: Option<usize>,
    /// Per-entry file-walk overrides.
    pub no_ignore: Option<bool>,
    pub hidden: Option<bool>,
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

pub struct ResolvedSource {
    pub root: PathBuf,
    pub metadata_root: PathBuf,
    pub discovery: crate::ingest::DiscoveryConfig,
}

impl IndexEntry {
    /// Build the file selection rules shared by indexing and live keyword search.
    pub fn discovery(&self, config: &RagConfig) -> crate::ingest::DiscoveryConfig {
        crate::ingest::DiscoveryConfig {
            extra_extensions: self.extensions.clone(),
            exclude: self.exclude.clone(),
            include: self.include.clone(),
            include_paths: Vec::new(),
            no_ignore: self.no_ignore.or(config.no_ignore).unwrap_or(false),
            hidden: self.hidden.or(config.hidden).unwrap_or(false),
        }
    }

    /// Preserve legacy single-path metadata; multi-path sources use one canonical
    /// common root with a strict directory selection beneath it.
    pub fn resolve_source(&self, config: &RagConfig, base: &Path) -> Result<ResolvedSource> {
        let mut discovery = self.discovery(config);
        match (&self.path, &self.paths) {
            (Some(path), None) => Ok(ResolvedSource {
                root: base.join(path),
                metadata_root: path.clone(),
                discovery,
            }),
            (None, Some(paths)) if !paths.is_empty() => {
                let mut directories = Vec::new();
                for path in paths {
                    let resolved = base.join(path).canonicalize().with_context(|| {
                        format!("Directory not found: {}", base.join(path).display())
                    })?;
                    if !resolved.is_dir() {
                        bail!("Source is not a directory: {}", resolved.display());
                    }
                    directories.push(resolved);
                }
                directories.sort();
                directories.dedup();
                let mut root = directories[0].clone();
                while !directories.iter().all(|path| path.starts_with(&root)) {
                    if !root.pop() {
                        bail!("Sources do not share a filesystem root");
                    }
                }
                discovery.include_paths = directories
                    .iter()
                    .map(|path| path.strip_prefix(&root).unwrap().to_path_buf())
                    .collect();
                Ok(ResolvedSource {
                    metadata_root: root.clone(),
                    root,
                    discovery,
                })
            }
            _ => bail!(
                "Index {:?} requires exactly one of path or nonempty paths",
                self.name
            ),
        }
    }

    /// Resolve this entry's index output relative to the config directory.
    pub fn output_path(&self, config_dir: &Path) -> PathBuf {
        self.output
            .as_ref()
            .map(|output| config_dir.join(output))
            .unwrap_or_else(|| config_dir.join(".rag").join(&self.name))
    }

    /// Resolve this entry's model using the config-level default and then the
    /// built-in default.
    pub fn model_id(&self, config: &RagConfig, default_model: &str) -> String {
        self.model
            .clone()
            .or_else(|| config.model.clone())
            .unwrap_or_else(|| default_model.to_string())
    }
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
            if !matches!((&entry.path, &entry.paths), (Some(_), None))
                && !matches!((&entry.path, &entry.paths), (None, Some(paths)) if !paths.is_empty())
            {
                bail!(
                    "Index {:?} requires exactly one of path or nonempty paths",
                    entry.name
                );
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(output: Option<&str>) -> IndexEntry {
        IndexEntry {
            name: "docs".into(),
            path: Some("sources/docs".into()),
            paths: None,
            output: output.map(PathBuf::from),
            model: None,
            chunk_size: None,
            chunk_overlap: None,
            no_ignore: None,
            hidden: None,
            extensions: vec![],
            exclude: vec![],
            include: vec![],
        }
    }

    #[test]
    fn paths_resolves_common_root_and_preserves_legacy_metadata() {
        let root = std::env::temp_dir().join(format!("rag-config-paths-{}", std::process::id()));
        for dir in ["a/docs", "b/docs"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        let config: RagConfig =
            toml::from_str("[[index]]\nname='docs'\npaths=['a/docs','b/docs','a/docs']\n").unwrap();
        let source = config.indexes[0].resolve_source(&config, &root).unwrap();
        assert_eq!(source.root, root.canonicalize().unwrap());
        assert_eq!(
            source.discovery.include_paths,
            vec![PathBuf::from("a/docs"), PathBuf::from("b/docs")]
        );
        let config: RagConfig = toml::from_str("[[index]]\nname='docs'\npath='a/docs'\n").unwrap();
        let source = config.indexes[0].resolve_source(&config, &root).unwrap();
        assert_eq!(source.metadata_root, PathBuf::from("a/docs"));
        assert!(source.discovery.include_paths.is_empty());
        let config: RagConfig =
            toml::from_str("[[index]]\nname='docs'\npaths=['a/docs']\n").unwrap();
        let source = config.indexes[0].resolve_source(&config, &root).unwrap();
        assert_eq!(source.root, root.join("a/docs").canonicalize().unwrap());
        assert_eq!(source.discovery.include_paths, vec![PathBuf::new()]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn output_paths_resolve_from_config_directory() {
        let base = Path::new("repo/config");
        assert_eq!(entry(None).output_path(base), base.join(".rag/docs"));
        assert_eq!(
            entry(Some("indexes/docs")).output_path(base),
            base.join("indexes/docs")
        );

        let absolute = if cfg!(windows) {
            PathBuf::from(r"C:\indexes\docs")
        } else {
            PathBuf::from("/indexes/docs")
        };
        assert_eq!(
            entry(Some(absolute.to_str().expect("test path should be UTF-8"))).output_path(base),
            absolute
        );
    }

    #[test]
    fn search_defaults_parse_alongside_index_entries() {
        let config: RagConfig = toml::from_str(
            "no_ignore = true\nhidden = true\n\n[search]\nhybrid = true\nsemantic_weight = 2.0\nkeyword_weight = 0.5\n\n[[index]]\nname = 'docs'\npath = 'docs'\nno_ignore = false\n",
        )
        .unwrap();
        let discovery = config.indexes[0].discovery(&config);
        assert!(!discovery.no_ignore);
        assert!(discovery.hidden);
        let search = config.search.unwrap();
        assert_eq!(search.hybrid, Some(true));
        assert_eq!(search.semantic_weight, Some(2.0));
        assert_eq!(search.keyword_weight, Some(0.5));
    }
}
