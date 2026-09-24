use anyhow::{Context, Result};
use half::{f16, prelude::HalfFloatVecExt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

pub const INDEX_FORMAT_VERSION: u32 = 3;

/// A source file referenced by one or more chunk occurrences.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SourceRecord {
    pub path: String,
}

/// One unique chunk body and its embedding.
///
/// Embeddings are stored as IEEE-754 binary16 bits. Inference still produces
/// f32 vectors; only the persisted representation is quantized.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextRecord {
    /// Identifies the original bytes without retaining the chunk body on disk.
    pub text_hash: [u8; 32],
    /// Available while building an index; search loads text from source files.
    #[serde(skip)]
    pub text: String,
    pub embedding_f16: Vec<u16>,
    /// L2 norm after decoding the stored F16 values, used for exact cosine.
    pub embedding_norm: f32,
}

impl TextRecord {
    pub fn new(text: String, embedding: Vec<f32>) -> Self {
        let half_values: Vec<f16> = Vec::from_f32_slice(&embedding);
        let embedding_norm = half_values
            .iter()
            .map(|value| {
                let value = value.to_f32();
                value * value
            })
            .sum::<f32>()
            .sqrt();
        let embedding_f16 = half_values.reinterpret_into();
        Self {
            text_hash: *blake3::hash(text.as_bytes()).as_bytes(),
            text,
            embedding_f16,
            embedding_norm,
        }
    }

    pub fn without_embedding(text: String) -> Self {
        Self {
            text_hash: *blake3::hash(text.as_bytes()).as_bytes(),
            text,
            embedding_f16: Vec::new(),
            embedding_norm: 0.0,
        }
    }

    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        let replacement = Self::new(std::mem::take(&mut self.text), embedding);
        *self = replacement;
    }
}

/// A source location at which a unique chunk body occurs.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChunkOccurrence {
    pub source_id: u32,
    pub text_id: u32,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Metadata stored alongside the index.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct IndexMeta {
    /// On-disk schema version. A mismatch forces a full rebuild.
    #[serde(default)]
    pub format_version: u32,
    /// Model used to generate embeddings.
    pub model_id: String,
    /// Inference backend/precision that produced the vectors.
    #[serde(default)]
    pub embedding_backend: String,
    /// Embedding dimensionality.
    pub hidden_size: usize,
    /// Number of source occurrences, including duplicate chunk text.
    pub num_chunks: usize,
    /// Number of unique chunk bodies and stored vectors.
    #[serde(default)]
    pub num_unique_texts: usize,
    /// Root directory that was indexed. Relative CLI and config paths remain
    /// relative so committed metadata is portable across machines.
    pub root_dir: String,
    /// Source root relative to the index directory, so moved repositories work.
    #[serde(default)]
    pub source_root_from_index: String,
    /// Timestamp of index creation.
    pub created_at: String,
    /// Chunk size in characters used during indexing.
    pub chunk_size: usize,
    /// Chunk overlap in characters.
    pub chunk_overlap: usize,
    /// Blake3 content hash per source file (relative path -> hex hash).
    #[serde(default)]
    pub file_hashes: BTreeMap<String, String>,
}

impl IndexMeta {
    /// Whether an existing index can be reused for an incremental re-index.
    pub fn reusable_for(
        &self,
        model_id: &str,
        embedding_backend: &str,
        chunk_size: usize,
        chunk_overlap: usize,
    ) -> bool {
        self.format_version == INDEX_FORMAT_VERSION
            && self.model_id == model_id
            && self.embedding_backend == embedding_backend
            && self.chunk_size == chunk_size
            && self.chunk_overlap == chunk_overlap
    }
}

/// Storage-v2 index: unique text/vectors are separate from source occurrences.
#[derive(Serialize, Deserialize)]
pub struct Index {
    pub meta: IndexMeta,
    pub sources: Vec<SourceRecord>,
    pub texts: Vec<TextRecord>,
    pub occurrences: Vec<ChunkOccurrence>,
}

static SAVE_GENERATION: AtomicU64 = AtomicU64::new(0);

struct PublicationLock {
    path: PathBuf,
}

impl PublicationLock {
    fn acquire(dir: &Path) -> Result<Self> {
        let path = dir.join(".index-publish.lock");
        for _ in 0..100 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    writeln!(file, "{}", std::process::id())?;
                    file.sync_all()?;
                    return Ok(Self { path });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Never reclaim by age: deleting a path that another writer
                    // just recreated can admit two publishers. A lock left by a
                    // crashed process is rare and must be removed explicitly.
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => {
                    return Err(error).context("Failed to acquire index publication lock")
                }
            }
        }
        anyhow::bail!(
            "Another process is publishing this index; retry shortly. If its process crashed, remove {}",
            path.display()
        )
    }
}

impl Drop for PublicationLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl Index {
    pub fn new(
        meta: IndexMeta,
        sources: Vec<SourceRecord>,
        texts: Vec<TextRecord>,
        occurrences: Vec<ChunkOccurrence>,
    ) -> Self {
        Self {
            meta,
            sources,
            texts,
            occurrences,
        }
    }

    /// Save index to a directory (creates `index.bin` and `meta.json`).
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create index directory: {}", dir.display()))?;

        let generation = SAVE_GENERATION.fetch_add(1, Ordering::Relaxed);
        let suffix = format!("tmp-{}-{generation}", std::process::id());
        let meta_path = dir.join("meta.json");
        let index_path = dir.join("index.bin");
        let meta_temp = dir.join(format!("meta.json.{suffix}"));
        let index_temp = dir.join(format!("index.bin.{suffix}"));

        let result = (|| -> Result<()> {
            let meta_json = serde_json::to_string_pretty(&self.meta)?;
            let mut meta_file = std::fs::File::create(&meta_temp)
                .with_context(|| format!("Failed to create {}", meta_temp.display()))?;
            meta_file
                .write_all(meta_json.as_bytes())
                .with_context(|| format!("Failed to write {}", meta_temp.display()))?;
            meta_file
                .sync_all()
                .with_context(|| format!("Failed to sync {}", meta_temp.display()))?;

            let file = std::fs::File::create(&index_temp)
                .with_context(|| format!("Failed to create {}", index_temp.display()))?;
            let mut writer = BufWriter::new(file);
            bincode::serialize_into(&mut writer, self)
                .with_context(|| format!("Failed to write {}", index_temp.display()))?;
            writer
                .flush()
                .with_context(|| format!("Failed to flush {}", index_temp.display()))?;
            writer
                .get_ref()
                .sync_all()
                .with_context(|| format!("Failed to sync {}", index_temp.display()))?;

            let _publication_lock = PublicationLock::acquire(dir)?;
            // Each file replacement is atomic. Metadata is the commit marker;
            // readers also compare it with the metadata embedded in index.bin,
            // so the brief two-rename window fails closed instead of mixing generations.
            std::fs::rename(&index_temp, &index_path)
                .with_context(|| format!("Failed to replace {}", index_path.display()))?;
            std::fs::rename(&meta_temp, &meta_path)
                .with_context(|| format!("Failed to replace {}", meta_path.display()))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&meta_temp);
            let _ = std::fs::remove_file(&index_temp);
        }
        result
    }

    /// Load only index metadata, without materializing the vector index.
    /// The same metadata is the first bincode field, so this also cheaply
    /// validates that the two published files belong to one generation.
    pub fn load_meta(dir: &Path) -> Result<IndexMeta> {
        let meta_path = dir.join("meta.json");
        let file = std::fs::File::open(&meta_path)
            .with_context(|| format!("Failed to open {}", meta_path.display()))?;
        let metadata: IndexMeta = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("Failed to deserialize {}", meta_path.display()))?;
        if metadata.format_version != INDEX_FORMAT_VERSION {
            anyhow::bail!(
                "Index {} uses format v{}, but this binary requires v{}; rebuild it",
                dir.display(),
                metadata.format_version,
                INDEX_FORMAT_VERSION
            );
        }

        let index_path = dir.join("index.bin");
        let index_file = std::fs::File::open(&index_path)
            .with_context(|| format!("Failed to open {}", index_path.display()))?;
        let embedded: IndexMeta = bincode::deserialize_from(BufReader::new(index_file))
            .context("Failed to read embedded index metadata (corrupted or version mismatch?)")?;
        if metadata != embedded {
            anyhow::bail!(
                "Index generation mismatch between meta.json and index.bin; rebuild the index"
            );
        }
        Ok(metadata)
    }

    /// Load index from a directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let index_path = dir.join("index.bin");
        let file = std::fs::File::open(&index_path)
            .with_context(|| format!("Failed to open {}", index_path.display()))?;
        let index: Self = bincode::deserialize_from(BufReader::new(file))
            .context("Failed to deserialize index (corrupted or version mismatch?)")?;
        let metadata = Self::load_meta(dir)?;
        if metadata != index.meta {
            anyhow::bail!(
                "Index generation mismatch between meta.json and index.bin; rebuild the index"
            );
        }
        Ok(index)
    }

    pub fn default_dir() -> PathBuf {
        PathBuf::from(".rag")
    }
}

/// Exact cosine similarity between an f32 query and an F16-stored vector.
///
/// The query must already be L2-normalized. Embedding backends enforce this
/// before search, while `stored_norm` accounts for F16 conversion error.
pub fn cosine_similarity_f16(query: &[f32], stored: &[u16], stored_norm: f32) -> f32 {
    if stored_norm == 0.0 {
        return 0.0;
    }
    query
        .iter()
        .zip(stored)
        .map(|(&a, &bits)| a * f16::from_bits(bits).to_f32())
        .sum::<f32>()
        / stored_norm
}

#[derive(Debug)]
pub struct SearchResult<'a> {
    pub score: f32,
    pub text_id: usize,
    pub text: &'a TextRecord,
}

/// Find the top-k unique chunk bodies. Source occurrences are resolved by the
/// caller so repeated boilerplate cannot consume multiple result slots.
pub fn search_top_k<'a>(
    query_embedding: &[f32],
    texts: &'a [TextRecord],
    k: usize,
) -> Vec<SearchResult<'a>> {
    if k == 0 || texts.is_empty() {
        return Vec::new();
    }

    #[cfg(debug_assertions)]
    {
        let query_norm = query_embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        debug_assert!(
            (query_norm - 1.0).abs() < 1e-3,
            "search query must be L2-normalized, got norm {query_norm}"
        );
    }

    let mut scored: Vec<SearchResult<'a>> = texts
        .iter()
        .enumerate()
        .map(|(text_id, text)| SearchResult {
            score: cosine_similarity_f16(query_embedding, &text.embedding_f16, text.embedding_norm),
            text_id,
            text,
        })
        .collect();

    let by_score_desc = |a: &SearchResult<'_>, b: &SearchResult<'_>| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.text_id.cmp(&b.text_id))
    };

    if scored.len() > k {
        scored.select_nth_unstable_by(k, by_score_desc);
        scored.truncate(k);
    }
    scored.sort_by(by_score_desc);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> IndexMeta {
        IndexMeta {
            format_version: INDEX_FORMAT_VERSION,
            model_id: "m".to_string(),
            embedding_backend: "onnx-qint8-arm64".to_string(),
            hidden_size: 2,
            num_chunks: 0,
            num_unique_texts: 0,
            root_dir: "/tmp".to_string(),
            source_root_from_index: ".".to_string(),
            created_at: "now".to_string(),
            chunk_size: 512,
            chunk_overlap: 64,
            file_hashes: BTreeMap::new(),
        }
    }

    #[test]
    fn serialized_record_omits_chunk_text() {
        let record = TextRecord::new("private chunk body".into(), vec![1.0, 0.0]);
        let bytes = bincode::serialize(&record).unwrap();
        assert!(!bytes
            .windows(record.text.len())
            .any(|window| window == record.text.as_bytes()));
        let restored: TextRecord = bincode::deserialize(&bytes).unwrap();
        assert!(restored.text.is_empty());
        assert_eq!(restored.text_hash, record.text_hash);
    }

    #[test]
    fn save_replaces_complete_files_and_load_rejects_mixed_generations() {
        let dir = std::env::temp_dir().join(format!(
            "rag-index-save-{}-{}",
            std::process::id(),
            SAVE_GENERATION.fetch_add(1, Ordering::Relaxed),
        ));
        let index = Index::new(meta(), Vec::new(), Vec::new(), Vec::new());
        index.save(&dir).unwrap();
        assert_eq!(Index::load(&dir).unwrap().meta, index.meta);
        assert!(std::fs::read_dir(&dir).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));

        let mut mismatched = index.meta.clone();
        mismatched.created_at = "different generation".to_string();
        std::fs::write(
            dir.join("meta.json"),
            serde_json::to_string_pretty(&mismatched).unwrap(),
        )
        .unwrap();
        assert!(Index::load(&dir)
            .err()
            .unwrap()
            .to_string()
            .contains("generation mismatch"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reusable_when_everything_matches() {
        assert!(meta().reusable_for("m", "onnx-qint8-arm64", 512, 64));
    }

    #[test]
    fn not_reusable_when_format_backend_model_or_chunking_differs() {
        let mut m = meta();
        m.format_version = 1;
        assert!(!m.reusable_for("m", "onnx-qint8-arm64", 512, 64));

        let m = meta();
        assert!(!m.reusable_for("m", "onnx-fp32", 512, 64));
        assert!(!m.reusable_for("other", "onnx-qint8-arm64", 512, 64));
        assert!(!m.reusable_for("m", "onnx-qint8-arm64", 256, 64));
        assert!(!m.reusable_for("m", "onnx-qint8-arm64", 512, 32));
    }

    #[test]
    fn f16_record_preserves_cosine_order_and_norm() {
        let high = TextRecord::new("high".into(), vec![1.0, 0.0]);
        let low = TextRecord::new("low".into(), vec![0.0, 1.0]);
        let texts = vec![low, high];
        let results = search_top_k(&[1.0, 0.0], &texts, 2);

        assert_eq!(results[0].text.text, "high");
        assert!((results[0].score - 1.0).abs() < 1e-5);
        assert_eq!(results[1].text.text, "low");
    }

    #[test]
    fn top_k_handles_zero_and_k_larger_than_corpus() {
        let texts = vec![TextRecord::new("only".into(), vec![1.0])];
        assert!(search_top_k(&[1.0], &texts, 0).is_empty());
        assert_eq!(search_top_k(&[1.0], &texts, 10).len(), 1);
    }

    #[test]
    fn top_k_breaks_equal_score_ties_by_text_id() {
        let texts = vec![
            TextRecord::new("first".into(), vec![1.0]),
            TextRecord::new("second".into(), vec![1.0]),
        ];
        let results = search_top_k(&[1.0], &texts, 1);

        assert_eq!(results[0].text_id, 0);
        assert_eq!(results[0].text.text, "first");
    }

    #[test]
    fn legacy_metadata_forces_rebuild() {
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
        assert_eq!(parsed.format_version, 0);
        assert_eq!(parsed.embedding_backend, "");
        assert!(!parsed.reusable_for("m", "onnx-qint8-arm64", 512, 64));
    }
}
