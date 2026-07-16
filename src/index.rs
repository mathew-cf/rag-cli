use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A single chunk of text with its embedding and source metadata.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChunkRecord {
    /// Source file path (relative to indexed directory).
    pub source: String,
    /// Byte offset of this chunk within the source file.
    pub byte_offset: usize,
    /// The chunk text.
    pub text: String,
    /// Precomputed embedding vector (L2 normalized).
    pub embedding: Vec<f32>,
}

/// Metadata stored alongside the index.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IndexMeta {
    /// Model used to generate embeddings.
    pub model_id: String,
    /// Inference backend/precision that produced the vectors (e.g.
    /// `onnx-qint8-arm64`). Vectors from different backends are not
    /// interchangeable, so a change here forces a full re-index. Defaults to
    /// empty for indexes built before this field existed, which likewise
    /// triggers a rebuild on next `rag index`.
    #[serde(default)]
    pub embedding_backend: String,
    /// Embedding dimensionality.
    pub hidden_size: usize,
    /// Number of chunks.
    pub num_chunks: usize,
    /// Root directory that was indexed.
    pub root_dir: String,
    /// Timestamp of index creation.
    pub created_at: String,
    /// Chunk size in characters used during indexing.
    pub chunk_size: usize,
    /// Chunk overlap in characters.
    pub chunk_overlap: usize,
    /// Blake3 content hash per source file (relative path -> hex hash).
    /// Used for incremental re-indexing.
    #[serde(default)]
    pub file_hashes: BTreeMap<String, String>,
}

impl IndexMeta {
    /// Whether an existing index can be reused for an incremental re-index
    /// instead of a full rebuild. All embedding-affecting settings must match:
    /// the model, the inference backend/precision (fp32 and int8 vectors are
    /// not interchangeable), and the chunking parameters.
    pub fn reusable_for(
        &self,
        model_id: &str,
        embedding_backend: &str,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> bool {
        self.model_id == model_id
            && self.embedding_backend == embedding_backend
            && self.chunk_size == chunk_size
            && self.chunk_overlap == chunk_overlap
    }
}

/// The full on-disk index: metadata + all chunk records.
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub meta: IndexMeta,
    pub chunks: Vec<ChunkRecord>,
}

impl Index {
    pub fn new(meta: IndexMeta, chunks: Vec<ChunkRecord>) -> Self {
        Self { meta, chunks }
    }

    /// Save index to a directory (creates `index.bin` and `meta.json`).
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create index directory: {}", dir.display()))?;

        let meta_path = dir.join("meta.json");
        let meta_json = serde_json::to_string_pretty(&self.meta)?;
        std::fs::write(&meta_path, meta_json)
            .with_context(|| format!("Failed to write {}", meta_path.display()))?;

        let index_path = dir.join("index.bin");
        let encoded = bincode::serialize(self).context("Failed to serialize index")?;
        std::fs::write(&index_path, encoded)
            .with_context(|| format!("Failed to write {}", index_path.display()))?;

        Ok(())
    }

    /// Load index from a directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let index_path = dir.join("index.bin");
        let data = std::fs::read(&index_path)
            .with_context(|| format!("Failed to read {}", index_path.display()))?;
        let index: Self = bincode::deserialize(&data)
            .context("Failed to deserialize index (corrupted or version mismatch?)")?;
        Ok(index)
    }

    /// Default index directory.
    pub fn default_dir() -> PathBuf {
        PathBuf::from(".rag")
    }
}

/// Compute cosine similarity between two L2-normalized vectors.
/// Since both are normalized, this is just the dot product.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Search result with score and chunk reference.
#[derive(Debug)]
pub struct SearchResult<'a> {
    pub score: f32,
    pub chunk: &'a ChunkRecord,
}

/// Find the top-k most similar chunks to the query embedding.
pub fn search_top_k<'a>(
    query_embedding: &[f32],
    chunks: &'a [ChunkRecord],
    k: usize,
) -> Vec<SearchResult<'a>> {
    let mut scored: Vec<SearchResult<'a>> = chunks
        .iter()
        .map(|chunk| SearchResult {
            score: cosine_similarity(query_embedding, &chunk.embedding),
            chunk,
        })
        .collect();

    // Sort descending by score
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> IndexMeta {
        IndexMeta {
            model_id: "m".to_string(),
            embedding_backend: "onnx-qint8-arm64".to_string(),
            hidden_size: 384,
            num_chunks: 0,
            root_dir: "/tmp".to_string(),
            created_at: "now".to_string(),
            chunk_size: 512,
            chunk_overlap: 64,
            file_hashes: BTreeMap::new(),
        }
    }

    #[test]
    fn reusable_when_everything_matches() {
        assert!(meta().reusable_for("m", "onnx-qint8-arm64", 512, 64));
    }

    #[test]
    fn not_reusable_when_backend_differs() {
        // The int8 -> different-backend switch must force a full re-index even
        // though the model_id and chunking are unchanged.
        assert!(!meta().reusable_for("m", "onnx-fp32", 512, 64));
    }

    #[test]
    fn not_reusable_when_model_or_chunking_differs() {
        let m = meta();
        assert!(!m.reusable_for("other", "onnx-qint8-arm64", 512, 64));
        assert!(!m.reusable_for("m", "onnx-qint8-arm64", 256, 64));
        assert!(!m.reusable_for("m", "onnx-qint8-arm64", 512, 32));
    }

    #[test]
    fn embedding_backend_defaults_to_empty_for_legacy_meta() {
        // An index written before `embedding_backend` existed must deserialize
        // with an empty backend, which then fails `reusable_for` against any
        // real backend -> automatic full rebuild on upgrade.
        let json = r#"{
            "model_id": "m",
            "hidden_size": 384,
            "num_chunks": 0,
            "root_dir": "/tmp",
            "created_at": "now",
            "chunk_size": 512,
            "chunk_overlap": 64
        }"#;
        let parsed: IndexMeta = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.embedding_backend, "");
        assert!(!parsed.reusable_for("m", "onnx-qint8-arm64", 512, 64));
    }
}
