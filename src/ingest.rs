use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::path::Path;
use walkdir::WalkDir;

/// Text file extensions we'll index.
const TEXT_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "rs",
    "py",
    "js",
    "ts",
    "go",
    "java",
    "c",
    "cpp",
    "h",
    "hpp",
    "rb",
    "sh",
    "bash",
    "zsh",
    "fish",
    "toml",
    "yaml",
    "yml",
    "json",
    "xml",
    "html",
    "css",
    "scss",
    "sql",
    "proto",
    "graphql",
    "dockerfile",
    "makefile",
    "cmake",
    "tex",
    "org",
    "rst",
    "csv",
    "tsv",
    "log",
    "conf",
    "cfg",
    "ini",
    "env",
    "tf",
    "hcl",
    "nix",
    "lua",
    "vim",
    "el",
    "lisp",
    "clj",
    "ex",
    "exs",
    "erl",
    "hs",
    "ml",
    "mli",
    "scala",
    "kt",
    "swift",
    "r",
    "jl",
    "php",
    "pl",
    "pm",
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
pub fn discover_files(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Skip hidden directories and common non-text directories
            if e.file_type().is_dir() {
                return !name.starts_with('.')
                    && name != "node_modules"
                    && name != "target"
                    && name != "__pycache__"
                    && name != ".git"
                    && name != "vendor"
                    && name != "dist"
                    && name != "build";
            }
            true
        })
    {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let path = entry.path();
        let filename = entry.file_name().to_string_lossy();

        // Check by extension
        let is_text = path
            .extension()
            .map(|ext| {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                TEXT_EXTENSIONS.contains(&ext_lower.as_str())
            })
            .unwrap_or(false);

        // Check by filename
        let is_known_text = TEXT_FILENAMES.iter().any(|name| filename.as_ref() == *name);

        if is_text || is_known_text {
            files.push(path.to_path_buf());
        }
    }

    files.sort();
    Ok(files)
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
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    // If the whole text fits in one chunk, return it directly
    if text.len() <= chunk_size {
        return vec![TextChunk {
            source: source.to_string(),
            byte_offset: 0,
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
            chunks.push(TextChunk {
                source: source.to_string(),
                byte_offset: start,
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
    fn test_chunk_empty() {
        let chunks = chunk_text("", "test.txt", 100, 10);
        assert!(chunks.is_empty());
    }
}
