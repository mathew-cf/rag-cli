use anyhow::{Context, Result};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use coreml_native::{AsMultiArray, BatchProvider, BorrowedTensor, ComputeUnits, Model};
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use ort::execution_providers::CPUExecutionProvider;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use ort::session::{builder::GraphOptimizationLevel, Session};
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use tokenizers::TruncationParams;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
// 128 is ~20% faster than 64 on the 90k-chunk Akamai corpus. Going to 256
// gained only another ~4% while pushing peak RSS close to 3 GiB.
const BATCH_SIZE: usize = 128;

// Apple builds use a native FP16 CoreML model with pooling and normalization
// fused into the graph. Other platforms use ONNX Runtime with an int8 model.
// Both recipes implement all-MiniLM-L6-v2, but backend changes still invalidate
// an index because the persisted vectors are not bit-identical.

/// Architecture-specific int8 ONNX weight file, published under `onnx/` in the
/// sentence-transformers repos. arm64 (Apple Silicon / ARM servers) and x86-64
/// have separately-tuned quantizations; anything else falls back to fp32.
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    target_arch = "aarch64"
))]
const ONNX_MODEL_FILE: &str = "onnx/model_qint8_arm64.onnx";
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    target_arch = "x86_64"
))]
const ONNX_MODEL_FILE: &str = "onnx/model_quint8_avx2.onnx";
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
const ONNX_MODEL_FILE: &str = "onnx/model.onnx";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MODEL_ARTIFACT_ID: &str = "haihengh/all-MiniLM-L6-v2-coreml";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MODEL_ARTIFACT_REVISION: &str = "c683f20435a63c9884472f5de9f56865e865fd99";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const COREML_MODEL_DIR: &str = "all-minilm-l6-v2.mlmodelc";
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const COREML_SEQUENCE_LENGTH: usize = 256;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const COREML_HIDDEN_SIZE: usize = 384;

/// Backend identifier stamped into the index metadata. Changing precision or
/// architecture changes this string, which invalidates the reuse check in
/// `lib.rs` and forces an automatic full re-index (fp32 and int8 vectors are
/// not numerically interchangeable across a whole index).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub const EMBEDDING_BACKEND: &str = "coreml-native-fp16";
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    target_arch = "aarch64"
))]
pub const EMBEDDING_BACKEND: &str = "onnx-qint8-arm64";
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    target_arch = "x86_64"
))]
pub const EMBEDDING_BACKEND: &str = "onnx-quint8-avx2";
#[cfg(all(
    not(all(target_os = "macos", target_arch = "aarch64")),
    not(any(target_arch = "aarch64", target_arch = "x86_64"))
))]
pub const EMBEDDING_BACKEND: &str = "onnx-fp32";

/// The backend/precision identifier for indexes built by this binary.
pub fn embedding_backend() -> &'static str {
    EMBEDDING_BACKEND
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MODEL_FILES: &[&str] = &[
    "all-minilm-l6-v2.mlmodelc/analytics/coremldata.bin",
    "all-minilm-l6-v2.mlmodelc/coremldata.bin",
    "all-minilm-l6-v2.mlmodelc/metadata.json",
    "all-minilm-l6-v2.mlmodelc/model.mil",
    "all-minilm-l6-v2.mlmodelc/weights/weight.bin",
    "tokenizer/tokenizer.json",
];
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", ONNX_MODEL_FILE];

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[derive(serde::Deserialize)]
struct HfConfig {
    hidden_size: usize,
}

pub struct EmbeddingEngine {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    model: Model,
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    session: Session,
    tokenizer: Tokenizer,
    hidden_size: usize,
}

// ---------------------------------------------------------------------------
// HuggingFace-compatible cache layout
// ---------------------------------------------------------------------------

/// Resolve the HF cache root: --cache-dir flag > $HF_HOME/hub > ~/.cache/huggingface/hub
pub fn resolve_hf_cache(override_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = override_dir {
        return Ok(dir.to_path_buf());
    }
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

/// Convert "sentence-transformers/all-MiniLM-L6-v2" -> "models--sentence-transformers--all-MiniLM-L6-v2"
fn model_repo_dir(hub_root: &Path, model_id: &str) -> PathBuf {
    let folder = format!("models--{}", model_id.replace('/', "--"));
    hub_root.join(folder)
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn snapshot_revision(model_id: &str) -> &str {
    if model_id == MODEL_ARTIFACT_ID {
        MODEL_ARTIFACT_REVISION
    } else {
        "main"
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn snapshot_revision(_model_id: &str) -> &str {
    "main"
}

/// Path to a specific file in a cached model snapshot.
fn snapshot_path(hub_root: &Path, model_id: &str, filename: &str) -> PathBuf {
    model_repo_dir(hub_root, model_id)
        .join("snapshots")
        .join(snapshot_revision(model_id))
        .join(filename)
}

// ---------------------------------------------------------------------------
// Download via ureq + native-tls (uses system cert store)
// ---------------------------------------------------------------------------

fn build_agent() -> Result<ureq::Agent> {
    let tls = native_tls::TlsConnector::new().context("Failed to create TLS connector")?;
    Ok(ureq::AgentBuilder::new()
        .tls_connector(Arc::new(tls))
        .build())
}

fn download_file(
    agent: &ureq::Agent,
    model_id: &str,
    revision: &str,
    filename: &str,
    dest: &Path,
) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model_id, revision, filename
    );

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let tmp = dest.with_extension("tmp");

    eprintln!("  Downloading {filename}...");
    let resp = agent
        .get(&url)
        .call()
        .with_context(|| format!("HTTP request failed for {url}"))?;

    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&tmp)
        .with_context(|| format!("Failed to create {}", tmp.display()))?;
    std::io::copy(&mut reader, &mut file)
        .with_context(|| format!("Failed to write {}", tmp.display()))?;

    std::fs::rename(&tmp, dest)
        .with_context(|| format!("Failed to move {} into place", dest.display()))?;

    Ok(())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn artifact_model_id(model_id: &str) -> Result<&str> {
    if model_id != DEFAULT_MODEL {
        anyhow::bail!(
            "The native CoreML backend currently supports only {DEFAULT_MODEL}, got {model_id}"
        );
    }
    Ok(MODEL_ARTIFACT_ID)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn artifact_model_id(model_id: &str) -> Result<&str> {
    Ok(model_id)
}

/// Ensure all model files are present locally, downloading any that are missing.
fn ensure_model_files(hub_root: &Path, model_id: &str) -> Result<()> {
    let artifact_id = artifact_model_id(model_id)?;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let revision = MODEL_ARTIFACT_REVISION;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let revision = "main";
    let mut need_download = false;
    for f in MODEL_FILES {
        if !snapshot_path(hub_root, artifact_id, f).exists() {
            need_download = true;
            break;
        }
    }

    if !need_download {
        return Ok(());
    }

    let agent = build_agent()?;
    for filename in MODEL_FILES {
        let dest = snapshot_path(hub_root, artifact_id, filename);
        download_file(&agent, artifact_id, revision, filename, &dest)
            .with_context(|| format!("Failed to fetch {filename} for {artifact_id}"))?;
    }

    Ok(())
}

/// Check whether every required model file is already present locally.
///
/// Cheap: just stats each file without hitting the network. Returned
/// separately from `download_model` so callers can distinguish "already
/// cached" from "downloaded this run" in their output.
pub fn model_files_present(hub_root: &Path, model_id: &str) -> bool {
    let Ok(artifact_id) = artifact_model_id(model_id) else {
        return false;
    };
    MODEL_FILES
        .iter()
        .all(|f| snapshot_path(hub_root, artifact_id, f).exists())
}

/// The list of files that make up a usable embedding model snapshot.
/// Exposed so the `rag download` command can print per-file status.
pub fn model_file_list() -> &'static [&'static str] {
    MODEL_FILES
}

/// Ensure the weights and tokenizer for `model_id` are present in the
/// HuggingFace-style cache rooted at `hub_root`, downloading any missing
/// files over HTTPS. Used by the `rag download` command so users can
/// warm the cache ahead of their first `rag index` or `rag search`.
///
/// Returns `true` if any file had to be downloaded (useful when the
/// caller wants to print "already cached" vs. "freshly downloaded").
pub fn download_model(model_id: &str, cache_dir: Option<&Path>) -> Result<bool> {
    let hub_root = resolve_hf_cache(cache_dir)?;
    let before = model_files_present(&hub_root, model_id);
    ensure_model_files(&hub_root, model_id)?;
    Ok(!before)
}

// ---------------------------------------------------------------------------
// Embedding engine
// ---------------------------------------------------------------------------

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

/// Mean-pool one sequence's token embeddings over its attention mask, then
/// L2-normalize — sentence-transformers' default pooling for MiniLM-style
/// models. `states` is a `[seq * hidden]` row-major slice for a single
/// sequence; `mask` is that sequence's `[seq]` mask (1 = real token, 0 = pad).
/// Padding tokens are excluded from the average. Returns a `hidden`-length
/// unit vector (all-zero input yields an all-zero vector rather than NaNs).
#[cfg_attr(all(target_os = "macos", target_arch = "aarch64"), allow(dead_code))]
fn pool_and_normalize(states: &[f32], mask: &[i64], seq: usize, hidden: usize) -> Vec<f32> {
    let mut pooled = vec![0f32; hidden];
    let mut msum = 0f32;
    for (s, &mask_val) in mask.iter().enumerate().take(seq) {
        let m = mask_val as f32;
        if m == 0.0 {
            continue;
        }
        msum += m;
        let base = s * hidden;
        let Some(row) = states.get(base..base.saturating_add(hidden)) else {
            break;
        };
        for (p, value) in pooled.iter_mut().zip(row) {
            *p += value * m;
        }
    }
    if msum > 0.0 {
        for p in pooled.iter_mut() {
            *p /= msum;
        }
    }
    let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for p in pooled.iter_mut() {
            *p /= norm;
        }
    }
    pooled
}

impl EmbeddingEngine {
    /// Load the embedding model.
    /// Downloads from HuggingFace on first use (via native-tls), cached in HF layout after that.
    pub fn load(model_id: Option<&str>, cache_dir: Option<&Path>) -> Result<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL);
        let hub_root = resolve_hf_cache(cache_dir)?;

        eprintln!("Loading model: {model_id}");
        eprintln!("  Cache: {}", hub_root.display());

        ensure_model_files(&hub_root, model_id)?;
        let artifact_id = artifact_model_id(model_id)?;

        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let tokenizer_path = snapshot_path(&hub_root, artifact_id, "tokenizer/tokenizer.json");
            let model_path = snapshot_path(&hub_root, artifact_id, COREML_MODEL_DIR);
            let mut tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(COREML_SEQUENCE_LENGTH),
                ..Default::default()
            }));
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length: COREML_SEQUENCE_LENGTH,
                    ..Default::default()
                }))
                .map_err(|e| anyhow::anyhow!("Failed to configure truncation: {e}"))?;
            let model = Model::load(&model_path, ComputeUnits::All).with_context(|| {
                format!(
                    "Failed to load native CoreML model {}",
                    model_path.display()
                )
            })?;
            let hidden_size = COREML_HIDDEN_SIZE;
            eprintln!("  Model loaded (hidden_size={hidden_size}, backend={EMBEDDING_BACKEND})");
            Ok(Self {
                model,
                tokenizer,
                hidden_size,
            })
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let config_path = snapshot_path(&hub_root, artifact_id, "config.json");
            let tokenizer_path = snapshot_path(&hub_root, artifact_id, "tokenizer.json");
            let weights_path = snapshot_path(&hub_root, artifact_id, ONNX_MODEL_FILE);
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: HfConfig = serde_json::from_str(&config_str)?;
            let hidden_size = config.hidden_size;
            let threads = intra_threads();
            let session = Session::builder()
                .context("Failed to create ONNX session builder")?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(threads)?
                .with_execution_providers([CPUExecutionProvider::default().build()])?
                .commit_from_file(&weights_path)
                .with_context(|| format!("Failed to load ONNX model {}", weights_path.display()))?;
            let tokenizer =
                Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!(
                "  Model loaded (hidden_size={hidden_size}, backend={EMBEDDING_BACKEND}, threads={threads})"
            );
            Ok(Self {
                session,
                tokenizer,
                hidden_size,
            })
        }
    }

    /// Embed a single text string. Returns a normalized f32 vector.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch_inner(&[text], None)?;
        results
            .into_iter()
            .next()
            .context("Embedding backend returned no result for one input")
    }

    /// Embed borrowed texts with a progress bar. Borrowing avoids cloning the
    /// complete corpus just to pass chunk text into the tokenizer.
    pub fn embed_batch_progress(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total_batches = texts.len().div_ceil(BATCH_SIZE);
        let pb = ProgressBar::new(total_batches as u64);
        let style = ProgressStyle::default_bar()
            .template("  Embedding [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
            .context("Invalid embedding progress-bar template")?
            .progress_chars("=>-");
        pb.set_style(style);

        let result = self.embed_batch_inner(texts, Some(&pb));
        pb.finish_and_clear();
        result
    }

    /// Core batching loop. Texts are grouped into batches by ascending length
    /// so each batch pads to a similar sequence length — with
    /// [`PaddingStrategy::BatchLongest`], a single long text in an otherwise
    /// short batch forces every row to that length, wasting compute on padding
    /// tokens. Sorting first keeps padding tight; results are scattered back to
    /// the caller's original order.
    fn embed_batch_inner(
        &mut self,
        texts: &[&str],
        pb: Option<&ProgressBar>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Sort by byte length (a cheap, stable proxy for token count — no need
        // to tokenize twice just to bucket) while retaining the original ID.
        let mut ordered: Vec<(usize, &str)> = texts.iter().copied().enumerate().collect();
        ordered.sort_by_key(|(_, text)| text.len());

        let mut results: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for batch_entries in ordered.chunks(BATCH_SIZE) {
            let batch: Vec<&str> = batch_entries.iter().map(|(_, text)| *text).collect();
            let embeddings = self.embed_chunk(&batch)?;
            for ((idx, _), embedding) in batch_entries.iter().zip(embeddings) {
                *results
                    .get_mut(*idx)
                    .context("Embedding result references an invalid input ID")? = embedding;
            }
            if let Some(pb) = pb {
                pb.inc(1);
            }
        }
        Ok(results)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn embed_chunk(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;
        let owned: Vec<(Vec<i32>, Vec<i32>, Vec<i32>)> = encodings
            .iter()
            .map(|encoding| {
                (
                    encoding.get_ids().iter().map(|&x| x as i32).collect(),
                    encoding
                        .get_attention_mask()
                        .iter()
                        .map(|&x| x as i32)
                        .collect(),
                    encoding.get_type_ids().iter().map(|&x| x as i32).collect(),
                )
            })
            .collect();
        let shape = [1, COREML_SEQUENCE_LENGTH];
        let ids: Vec<BorrowedTensor<'_>> = owned
            .iter()
            .map(|values| BorrowedTensor::from_i32(&values.0, &shape))
            .collect::<std::result::Result<_, _>>()?;
        let masks: Vec<BorrowedTensor<'_>> = owned
            .iter()
            .map(|values| BorrowedTensor::from_i32(&values.1, &shape))
            .collect::<std::result::Result<_, _>>()?;
        let types: Vec<BorrowedTensor<'_>> = owned
            .iter()
            .map(|values| BorrowedTensor::from_i32(&values.2, &shape))
            .collect::<std::result::Result<_, _>>()?;
        let inputs: Vec<Vec<(&str, &dyn AsMultiArray)>> = (0..texts.len())
            .map(|i| {
                vec![
                    ("input_ids", &ids[i] as &dyn AsMultiArray),
                    ("attention_mask", &masks[i] as &dyn AsMultiArray),
                    ("token_type_ids", &types[i] as &dyn AsMultiArray),
                ]
            })
            .collect();
        let input_refs: Vec<&[(&str, &dyn AsMultiArray)]> =
            inputs.iter().map(Vec::as_slice).collect();
        let batch = BatchProvider::new(&input_refs)?;
        let predictions = self.model.predict_batch(&batch)?;
        (0..texts.len())
            .map(|i| {
                let (embedding, shape) = predictions.get_f32(i, "div_1")?;
                if shape != [1, self.hidden_size] {
                    anyhow::bail!("Unexpected CoreML embedding shape: {shape:?}");
                }
                Ok(embedding)
            })
            .collect()
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    fn embed_chunk(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let bsz = encodings.len();
        let seq = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

        let mut ids = Vec::with_capacity(bsz * seq);
        let mut mask = Vec::with_capacity(bsz * seq);
        for e in &encodings {
            ids.extend(e.get_ids().iter().map(|&x| x as i64));
            mask.extend(e.get_attention_mask().iter().map(|&x| x as i64));
        }
        let types = vec![0i64; bsz * seq];

        let ids_t = Tensor::from_array(([bsz, seq], ids))?;
        let mask_t = Tensor::from_array(([bsz, seq], mask.clone()))?;
        let types_t = Tensor::from_array(([bsz, seq], types))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => ids_t,
            "attention_mask" => mask_t,
            "token_type_ids" => types_t,
        ])?;

        // last_hidden_state: [bsz, seq, hidden]. Mean-pool over unmasked tokens,
        // then L2-normalize — matching sentence-transformers' default pooling.
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let hidden = shape[2] as usize;

        let mut results = Vec::with_capacity(bsz);
        for b in 0..bsz {
            let states = &data[b * seq * hidden..(b + 1) * seq * hidden];
            let row_mask = &mask[b * seq..(b + 1) * seq];
            results.push(pool_and_normalize(states, row_mask, seq, hidden));
        }

        Ok(results)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "expected {b}, got {a}");
    }

    fn l2(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn backend_id_matches_target_arch() {
        // The stamped backend must track the arch-selected quantization so the
        // reuse check invalidates indexes built on a different backend.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert_eq!(embedding_backend(), "coreml-native-fp16");
        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            target_arch = "aarch64"
        ))]
        assert_eq!(embedding_backend(), "onnx-qint8-arm64");
        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            target_arch = "x86_64"
        ))]
        assert_eq!(embedding_backend(), "onnx-quint8-avx2");
        #[cfg(all(
            not(all(target_os = "macos", target_arch = "aarch64")),
            not(any(target_arch = "aarch64", target_arch = "x86_64"))
        ))]
        assert_eq!(embedding_backend(), "onnx-fp32");
    }

    #[test]
    fn model_files_cover_tokenizer_and_weights() {
        let files = model_file_list();
        assert!(
            files.iter().any(|f| f.ends_with("tokenizer.json")),
            "missing tokenizer.json"
        );
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        assert!(
            files.iter().any(|f| f.ends_with("model.mil")),
            "no compiled CoreML model selected: {files:?}"
        );
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            assert!(files.contains(&"config.json"), "missing config.json");
            assert!(
                files.iter().any(|f| f.ends_with(".onnx")),
                "no .onnx weight file selected: {files:?}"
            );
        }
    }

    #[test]
    fn pool_single_token_is_just_normalized() {
        // One token, no padding: pooled == the token, then L2-normalized.
        let out = pool_and_normalize(&[3.0, 4.0], &[1], 1, 2);
        approx(out[0], 0.6);
        approx(out[1], 0.8);
        approx(l2(&out), 1.0);
    }

    #[test]
    fn pool_excludes_padding_tokens() {
        // Second token is padding (mask 0) with huge values — it must not leak
        // into the pooled result, which should equal the first token normalized.
        let states = [3.0, 4.0, 1000.0, 1000.0];
        let out = pool_and_normalize(&states, &[1, 0], 2, 2);
        approx(out[0], 0.6);
        approx(out[1], 0.8);
        approx(l2(&out), 1.0);
    }

    #[test]
    fn pool_averages_multiple_real_tokens() {
        // Two real tokens [1,0] and [0,1] average to [0.5,0.5], normalized to
        // [1/√2, 1/√2].
        let states = [1.0, 0.0, 0.0, 1.0];
        let out = pool_and_normalize(&states, &[1, 1], 2, 2);
        let inv_sqrt2 = 1.0 / 2.0_f32.sqrt();
        approx(out[0], inv_sqrt2);
        approx(out[1], inv_sqrt2);
        approx(l2(&out), 1.0);
    }

    #[test]
    fn pool_all_padding_yields_zeros_not_nan() {
        // No real tokens: must return zeros, never NaN/inf (guards div-by-zero).
        let out = pool_and_normalize(&[5.0, 5.0], &[0], 1, 2);
        assert_eq!(out, vec![0.0, 0.0]);
        assert!(out.iter().all(|x| x.is_finite()));
    }
}
