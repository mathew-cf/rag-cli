use anyhow::{Context, Result};
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Text file extensions we'll index.
///
/// Programming language source files (e.g. .rs, .py, .ts, .go) and .json are
/// intentionally excluded — RAG indexing is aimed at prose/docs/config, not code.
const TEXT_EXTENSIONS: &[&str] = &[
    // Docs / prose
    "txt",
    "md",
    "tex",
    "org",
    "rst",
    // Data
    "csv",
    "tsv",
    "log",
    // Config
    "toml",
    "conf",
    "cfg",
    "ini",
    "env",
    "tf",
    "hcl",
    "nix",
    // Markup
    "xml",
    "html",
    // Schemas
    "sql",
    "proto",
    "graphql",
    // Build
    "dockerfile",
    "makefile",
    "cmake",
];

/// Directory names skipped by default during discovery. Callers can force any
/// of these back in via [`DiscoveryConfig::include`].
const DEFAULT_SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "__pycache__",
    ".git",
    "vendor",
    "dist",
    "build",
];

/// Files with no extension that are likely text.
const TEXT_FILENAMES: &[&str] = &[
    "Makefile",
    "Dockerfile",
    "Rakefile",
    "Gemfile",
    "LICENSE",
    "README",
    "CHANGELOG",
    "CONTRIBUTING",
    "Justfile",
    "Taskfile",
    "Vagrantfile",
    ".gitignore",
    ".dockerignore",
    ".editorconfig",
];

/// Normalize an extension token: trim surrounding whitespace, drop a single
/// leading `.`, and lowercase it. Empty tokens become `None` so callers can
/// filter them out (e.g. a trailing comma in `--ext md,mdx,`).
fn normalize_extension(ext: &str) -> Option<String> {
    let trimmed = ext.trim().trim_start_matches('.').to_lowercase();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Build the set of file extensions to index: the built-in [`TEXT_EXTENSIONS`]
/// allowlist plus any caller-supplied extras. Extras are normalized and merged,
/// so `--ext .MDX` and `--ext mdx` are equivalent, and duplicates are harmless.
pub fn build_extension_set(extra_extensions: &[String]) -> BTreeSet<String> {
    let mut set: BTreeSet<String> = TEXT_EXTENSIONS.iter().map(|e| e.to_string()).collect();
    set.extend(
        extra_extensions
            .iter()
            .filter_map(|e| normalize_extension(e)),
    );
    set
}

/// Controls which files [`discover_files`] picks up.
#[derive(Debug, Clone, Default)]
pub struct DiscoveryConfig {
    /// Extra file extensions to index, on top of the built-in allowlist.
    pub extra_extensions: Vec<String>,
    /// Directory/file specs to skip, on top of the built-in defaults.
    pub exclude: Vec<String>,
    /// Specs matching normally-skipped directories (defaults or hidden) that
    /// should be indexed anyway.
    pub include: Vec<String>,
    /// Search files ignored by .gitignore, .ignore, .rgignore, git excludes, and global ignores.
    pub no_ignore: bool,
    /// Include hidden directories as well as ordinary directories.
    pub hidden: bool,
}

/// Options for the shared source walker. A discovery config applies the same
/// corpus rules to indexing and config-based live keyword search.
pub struct FileWalkOptions<'a> {
    pub discovery: Option<&'a DiscoveryConfig>,
    pub globs: &'a [String],
    pub no_ignore: bool,
    pub hidden: bool,
    pub follow_links: bool,
}

/// Match a path spec against an entry. A spec containing `/` is treated as a
/// relative-path prefix (matches the path itself or anything beneath it); a
/// bare spec matches any single path component by name.
fn matches_spec(spec: &str, rel_path: &str, name: &str) -> bool {
    if spec.contains('/') {
        let spec = spec.trim_matches('/');
        rel_path == spec || rel_path.starts_with(&format!("{spec}/"))
    } else {
        name == spec
    }
}

fn discovery_entry_allowed(
    root: &Path,
    path: &Path,
    is_dir: bool,
    config: &DiscoveryConfig,
    hidden: bool,
) -> bool {
    let rel = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");
    if rel.is_empty() {
        return true;
    }
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    if config
        .include
        .iter()
        .any(|spec| matches_spec(spec, &rel, &name))
    {
        return true;
    }
    if config
        .exclude
        .iter()
        .any(|spec| matches_spec(spec, &rel, &name))
    {
        return false;
    }
    if !hidden && name.starts_with('.') {
        return false;
    }
    !is_dir || !DEFAULT_SKIP_DIRS.contains(&name.as_ref())
}

fn indexable_file(path: &Path, extensions: &BTreeSet<String>) -> bool {
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    path.extension()
        .is_some_and(|ext| extensions.contains(&ext.to_string_lossy().to_lowercase()))
        || TEXT_FILENAMES.iter().any(|name| filename.as_ref() == *name)
}

/// Walk live files using ripgrep's ignore and glob rules. When `discovery` is
/// present, also apply the index's extension and include/exclude rules.
pub fn walk_files(root: &Path, options: &FileWalkOptions<'_>) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        anyhow::bail!("Search path {} does not exist", root.display());
    }
    let discovery = options.discovery.cloned();
    let no_ignore = options.no_ignore || discovery.as_ref().is_some_and(|config| config.no_ignore);
    let hidden = options.hidden || discovery.as_ref().is_some_and(|config| config.hidden);
    let mut builder = WalkBuilder::new(root);
    if !no_ignore {
        builder.add_custom_ignore_filename(".rgignore");
    }
    builder.follow_links(options.follow_links);
    builder
        .parents(!no_ignore)
        .ignore(!no_ignore)
        .git_ignore(!no_ignore)
        .git_global(!no_ignore)
        .git_exclude(!no_ignore)
        .hidden(options.discovery.is_none() && !hidden);
    if !options.globs.is_empty() {
        let mut overrides = OverrideBuilder::new(root);
        for glob in options.globs {
            overrides
                .add(glob)
                .with_context(|| format!("Invalid glob {glob:?}"))?;
        }
        builder.overrides(overrides.build()?);
    }
    if let Some(config) = discovery.as_ref() {
        let config = config.clone();
        let root = root.to_path_buf();
        builder.filter_entry(move |entry| {
            discovery_entry_allowed(
                &root,
                entry.path(),
                entry.file_type().is_some_and(|kind| kind.is_dir()),
                &config,
                hidden,
            )
        });
    }
    let extensions = discovery
        .as_ref()
        .map(|config| build_extension_set(&config.extra_extensions));
    let mut files = Vec::new();
    for entry in builder.build() {
        let entry = entry.with_context(|| format!("Failed to walk {}", root.display()))?;
        if entry.file_type().is_some_and(|kind| kind.is_file())
            && extensions
                .as_ref()
                .is_none_or(|extensions| indexable_file(entry.path(), extensions))
        {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

/// Chunk of text with metadata about where it came from.
pub struct TextChunk {
    /// Source file path relative to the root directory.
    pub source: String,
    /// Byte offset of this chunk within the file.
    pub byte_offset: usize,
    /// The text content.
    pub text: String,
}

/// Discover all text files under the given root directory.
///
/// `extra_extensions` widens the built-in allowlist with additional file
/// extensions (e.g. `["mdx"]`); they are normalized (leading `.` stripped,
/// lowercased) before matching.
pub fn discover_files(root: &Path, config: &DiscoveryConfig) -> Result<Vec<std::path::PathBuf>> {
    walk_files(
        root,
        &FileWalkOptions {
            discovery: Some(config),
            globs: &[],
            no_ignore: false,
            hidden: false,
            follow_links: true,
        },
    )
}

/// Read a file and split it into overlapping chunks.
pub fn chunk_file(
    path: &Path,
    root: &Path,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Result<Vec<TextChunk>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    Ok(chunk_text(&content, &relative, chunk_size, chunk_overlap))
}

/// Split text into overlapping chunks of approximately `chunk_size` characters.
/// Tries to break at paragraph/line boundaries when possible.
fn chunk_text(text: &str, source: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<TextChunk> {
    let original = text;
    let text = original.trim();
    let trimmed_prefix = original.len() - original.trim_start().len();
    if text.is_empty() {
        return vec![];
    }

    // If the whole text fits in one chunk, return it directly
    if text.len() <= chunk_size {
        return vec![TextChunk {
            source: source.to_string(),
            byte_offset: trimmed_prefix,
            text: text.to_string(),
        }];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());

        // Don't split in the middle of a UTF-8 character
        while end < text.len() && !text.is_char_boundary(end) {
            end += 1;
        }

        // Try to break at a paragraph boundary (double newline)
        if end < text.len() {
            if let Some(pos) = text[start..end].rfind("\n\n") {
                if pos > chunk_size / 4 {
                    end = start + pos + 2; // include the double newline
                }
            } else if let Some(pos) = text[start..end].rfind('\n') {
                // Fall back to single newline
                if pos > chunk_size / 4 {
                    end = start + pos + 1;
                }
            }
        }

        let chunk_text = text[start..end].trim();
        if !chunk_text.is_empty() {
            let leading_space = text[start..end].len() - text[start..end].trim_start().len();
            chunks.push(TextChunk {
                source: source.to_string(),
                byte_offset: trimmed_prefix + start + leading_space,
                text: chunk_text.to_string(),
            });
        }

        // Advance with overlap
        let advance = end - start;
        if advance <= chunk_overlap {
            start = end; // avoid infinite loop on tiny chunks
        } else {
            start = end - chunk_overlap;
        }

        // Ensure we don't split mid-character on the overlap boundary
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
    }

    chunks
}

/// Compute the blake3 hash of a file's contents, returning the hex string.
pub fn hash_file(path: &Path) -> Result<String> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read file for hashing: {}", path.display()))?;
    Ok(blake3::hash(&data).to_hex().to_string())
}

/// Hash all discovered files, returning a map of relative path -> blake3 hex hash.
pub fn hash_files(files: &[std::path::PathBuf], root: &Path) -> Result<BTreeMap<String, String>> {
    let mut hashes = BTreeMap::new();
    for file in files {
        let relative = file
            .strip_prefix(root)
            .unwrap_or(file)
            .to_string_lossy()
            .to_string();
        let hash = hash_file(file)?;
        hashes.insert(relative, hash);
    }
    Ok(hashes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_small_text() {
        let chunks = chunk_text("hello world", "test.txt", 1000, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn test_chunk_large_text() {
        let text = "a".repeat(500) + "\n\n" + &"b".repeat(500);
        let chunks = chunk_text(&text, "test.txt", 600, 50);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn chunk_offsets_address_exact_original_file_bytes() {
        let original = "  αα\n\n  beta  \n\n gamma  ";
        let chunks = chunk_text(original, "test.txt", 10, 2);
        assert!(chunks.len() > 1);
        for chunk in chunks {
            let end = chunk.byte_offset + chunk.text.len();
            assert_eq!(
                original.get(chunk.byte_offset..end),
                Some(chunk.text.as_str())
            );
        }
    }

    #[test]
    fn test_chunk_empty() {
        let chunks = chunk_text("", "test.txt", 100, 10);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_normalize_extension() {
        assert_eq!(normalize_extension(".MDX"), Some("mdx".to_string()));
        assert_eq!(normalize_extension("  md "), Some("md".to_string()));
        assert_eq!(normalize_extension("mdx"), Some("mdx".to_string()));
        assert_eq!(normalize_extension(""), None);
        assert_eq!(normalize_extension("  "), None);
        assert_eq!(normalize_extension("."), None);
    }

    #[test]
    fn test_build_extension_set_merges_and_dedups() {
        let set = build_extension_set(&[".MDX".to_string(), "mdx".to_string(), "rst".to_string()]);
        // built-in default is still present
        assert!(set.contains("md"));
        // extra extension added, case-normalized and de-duplicated
        assert!(set.contains("mdx"));
        // already-default extension supplied as extra is harmless
        assert!(set.contains("rst"));
    }

    #[test]
    fn test_matches_spec_by_name() {
        // bare spec matches a path component by name, anywhere in the tree
        assert!(matches_spec(
            "deprecated",
            "reference/deprecated",
            "deprecated"
        ));
        assert!(matches_spec("dist", "dist", "dist"));
        assert!(!matches_spec("dist", "src/content", "content"));
    }

    #[test]
    fn test_matches_spec_by_path_prefix() {
        // spec with a slash is a relative-path prefix
        assert!(matches_spec(
            "src/content/changelog",
            "src/content/changelog",
            "changelog"
        ));
        assert!(matches_spec(
            "src/content/changelog",
            "src/content/changelog/2024.md",
            "2024.md"
        ));
        assert!(!matches_spec(
            "src/content/changelog",
            "src/content/other",
            "other"
        ));
        // leading/trailing slashes are tolerated
        assert!(matches_spec("/partials/", "partials", "partials"));
    }

    #[test]
    fn shared_discovery_applies_ignore_extension_and_source_rules() {
        let root = std::env::temp_dir().join(format!(
            "rag-discovery-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for dir in [".git", "dist", ".hidden"] {
            std::fs::create_dir_all(root.join(dir)).unwrap();
        }
        std::fs::write(root.join(".gitignore"), "ignored.md\n").unwrap();
        std::fs::write(root.join(".rgignore"), "rgignored.md\n").unwrap();
        for name in [
            "keep.md",
            "ignored.md",
            "rgignored.md",
            "skip.md",
            "extra.mdx",
            "source.rs",
            "dist/built.md",
            ".hidden/secret.md",
            ".secret.md",
        ] {
            std::fs::write(root.join(name), "cache\n").unwrap();
        }
        let mut config = DiscoveryConfig {
            extra_extensions: vec!["mdx".into()],
            exclude: vec!["skip.md".into()],
            include: vec!["dist".into()],
            ..Default::default()
        };
        let files = discover_files(&root, &config).unwrap();
        let relative: Vec<_> = files
            .iter()
            .map(|path| {
                path.strip_prefix(&root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect();
        assert!(relative.contains(&"keep.md".into()));
        assert!(relative.contains(&"extra.mdx".into()));
        assert!(relative.contains(&"dist/built.md".into()));
        for excluded in [
            "ignored.md",
            "rgignored.md",
            "skip.md",
            "source.rs",
            ".hidden/secret.md",
            ".secret.md",
        ] {
            assert!(
                !relative.contains(&excluded.to_string()),
                "{excluded} was included"
            );
        }

        let globbed = walk_files(
            &root,
            &FileWalkOptions {
                discovery: None,
                globs: &["ignored.md".into()],
                no_ignore: false,
                hidden: false,
                follow_links: false,
            },
        )
        .unwrap();
        assert!(globbed.contains(&root.join("ignored.md")));

        config.no_ignore = true;
        config.hidden = true;
        let expanded = discover_files(&root, &config).unwrap();
        assert!(expanded.contains(&root.join("ignored.md")));
        assert!(expanded.contains(&root.join("rgignored.md")));
        assert!(expanded.contains(&root.join(".hidden/secret.md")));
        assert!(expanded.contains(&root.join(".secret.md")));
        std::fs::remove_dir_all(root).unwrap();
    }
}
