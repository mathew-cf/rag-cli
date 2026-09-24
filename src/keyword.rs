//! Live file search built on ripgrep's matcher, searcher, and ignore crates.

use anyhow::{Context, Result};
use grep_matcher::Matcher;
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{SearcherBuilder, Sink, SinkMatch};
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct LineHit {
    pub byte_offset: usize,
    pub line_number: Option<u64>,
    pub line: String,
    pub spans: Vec<Range<usize>>,
}

pub fn matcher(patterns: &[String], fixed: bool, case_insensitive: bool) -> Result<RegexMatcher> {
    if patterns.is_empty() {
        anyhow::bail!("at least one -e/--regexp pattern is required");
    }
    let mut builder = RegexMatcherBuilder::new();
    builder.case_insensitive(case_insensitive);
    if fixed {
        let escaped: Vec<String> = patterns
            .iter()
            .map(|pattern| regex::escape(pattern))
            .collect();
        builder
            .build_many(&escaped)
            .context("Invalid keyword pattern")
    } else {
        builder
            .build_many(patterns)
            .context("Invalid keyword regex")
    }
}

pub fn walk_files(root: &Path, globs: &[String]) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        anyhow::bail!("Search path {} does not exist", root.display());
    }
    let mut builder = WalkBuilder::new(root);
    if !globs.is_empty() {
        let mut overrides = OverrideBuilder::new(root);
        for glob in globs {
            overrides
                .add(glob)
                .with_context(|| format!("Invalid glob {glob:?}"))?;
        }
        builder.overrides(overrides.build()?);
    }
    let mut paths = Vec::new();
    for entry in builder.build() {
        let entry = entry.with_context(|| format!("Failed to walk {}", root.display()))?;
        if entry.file_type().is_some_and(|kind| kind.is_file()) {
            paths.push(entry.into_path());
        }
    }
    paths.sort();
    Ok(paths)
}

struct CollectSink<'a> {
    matcher: &'a RegexMatcher,
    first_only: bool,
    hits: Vec<LineHit>,
}

impl Sink for CollectSink<'_> {
    type Error = io::Error;

    fn matched(
        &mut self,
        _searcher: &grep_searcher::Searcher,
        mat: &SinkMatch<'_>,
    ) -> Result<bool, io::Error> {
        let mut spans = Vec::new();
        self.matcher
            .find_iter(mat.bytes(), |found| {
                spans.push(found.start()..found.end());
                !self.first_only
            })
            .map_err(io::Error::other)?;
        if !spans.is_empty() {
            self.hits.push(LineHit {
                byte_offset: usize::try_from(mat.absolute_byte_offset())
                    .map_err(io::Error::other)?,
                line_number: mat.line_number(),
                line: String::from_utf8_lossy(mat.bytes())
                    .trim_end_matches(['\r', '\n'])
                    .into(),
                spans,
            });
        }
        Ok(!self.first_only)
    }
}

pub fn scan_file(matcher: &RegexMatcher, path: &Path, first_only: bool) -> Result<Vec<LineHit>> {
    let mut sink = CollectSink {
        matcher,
        first_only,
        hits: Vec::new(),
    };
    SearcherBuilder::new()
        .line_number(true)
        .build()
        .search_path(matcher, path, &mut sink)
        .with_context(|| format!("Failed to search {}", path.display()))?;
    Ok(sink.hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_and_fixed_string_modes_differ() {
        let root = std::env::temp_dir().join(format!("rag-keyword-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("sample.md");
        std::fs::write(&path, "foo.bar\nfooXbar\n").unwrap();
        let pattern = vec!["foo.bar".to_string()];
        assert_eq!(
            scan_file(&matcher(&pattern, false, false).unwrap(), &path, false)
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            scan_file(&matcher(&pattern, true, false).unwrap(), &path, false)
                .unwrap()
                .len(),
            1
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
