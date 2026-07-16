use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
const BATCH_SIZE: usize = 64;

// Inference runs on ONNX Runtime (CPU execution provider) with an int8-quantized
// model. int8 model quantization is a CPU-only win (~2x throughput vs fp32) at
// ~99.7% quality retention; on GPU int8 is actually slower, which is one reason
// this build is CPU-only. The quantized ONNX variant is architecture-specific,
// so the exact file (and the backend tag stamped into the index) depends on the
// target CPU — see `ONNX_MODEL_FILE` / `embedding_backend`.

/// Architecture-specific int8 ONNX weight file, published under `onnx/` in the
/// sentence-transformers repos. arm64 (Apple Silicon / ARM servers) and x86-64
/// have separately-tuned quantizations; anything else falls back to fp32.
#[cfg(target_arch = "aarch64")]
const ONNX_MODEL_FILE: &str = "onnx/model_qint8_arm64.onnx";
#[cfg(target_arch = "x86_64")]
const ONNX_MODEL_FILE: &str = "onnx/model_quint8_avx2.onnx";
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
const ONNX_MODEL_FILE: &str = "onnx/model.onnx";

/// Backend identifier stamped into the index metadata. Changing precision or
/// architecture changes this string, which invalidates the reuse check in
/// `lib.rs` and forces an automatic full re-index (fp32 and int8 vectors are
/// not numerically interchangeable across a whole index).
#[cfg(target_arch = "aarch64")]
pub const EMBEDDING_BACKEND: &str = "onnx-qint8-arm64";
#[cfg(target_arch = "x86_64")]
pub const EMBEDDING_BACKEND: &str = "onnx-quint8-avx2";
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const EMBEDDING_BACKEND: &str = "onnx-fp32";

/// The backend/precision identifier for indexes built by this binary.
pub fn embedding_backend() -> &'static str {
    EMBEDDING_BACKEND
}

const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", ONNX_MODEL_FILE];

/// Minimal view of a HuggingFace `config.json` — we only need the hidden size.
#[derive(serde::Deserialize)]
struct HfConfig {
    hidden_size: usize,
}

pub struct EmbeddingEngine {
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

/// Path to a specific file in the snapshot: <repo>/snapshots/main/<filename>
fn snapshot_path(hub_root: &Path, model_id: &str, filename: &str) -> PathBuf {
    model_repo_dir(hub_root, model_id)
        .join("snapshots")
        .join("main")
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

fn download_file(agent: &ureq::Agent, model_id: &str, filename: &str, dest: &Path) -> Result<()> {
    if dest.exists() {
        return Ok(());
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        model_id, filename
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

/// Ensure all model files are present locally, downloading any that are missing.
fn ensure_model_files(hub_root: &Path, model_id: &str) -> Result<()> {
    let mut need_download = false;
    for f in MODEL_FILES {
        if !snapshot_path(hub_root, model_id, f).exists() {
            need_download = true;
            break;
        }
    }

    if !need_download {
        return Ok(());
    }

    let agent = build_agent()?;
    for filename in MODEL_FILES {
        let dest = snapshot_path(hub_root, model_id, filename);
        download_file(&agent, model_id, filename, &dest)
            .with_context(|| format!("Failed to fetch {filename} for {model_id}"))?;
    }

    Ok(())
}

/// Check whether every required model file is already present locally.
///
/// Cheap: just stats each file without hitting the network. Returned
/// separately from `download_model` so callers can distinguish "already
/// cached" from "downloaded this run" in their output.
pub fn model_files_present(hub_root: &Path, model_id: &str) -> bool {
    MODEL_FILES
        .iter()
        .all(|f| snapshot_path(hub_root, model_id, f).exists())
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
        for (h, p) in pooled.iter_mut().enumerate() {
            *p += states[base + h] * m;
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

        let config_path = snapshot_path(&hub_root, model_id, "config.json");
        let tokenizer_path = snapshot_path(&hub_root, model_id, "tokenizer.json");
        let weights_path = snapshot_path(&hub_root, model_id, ONNX_MODEL_FILE);

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

    /// Embed a single text string. Returns a normalized f32 vector.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed a batch of texts. Returns normalized f32 vectors.
    pub fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_batch_inner(texts, None)
    }

    /// Embed texts with a progress bar.
    pub fn embed_batch_progress(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total_batches = texts.len().div_ceil(BATCH_SIZE);
        let pb = ProgressBar::new(total_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Embedding [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );

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
        texts: &[String],
        pb: Option<&ProgressBar>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Sort indices by byte length (a cheap, stable proxy for token count —
        // no need to tokenize twice just to bucket).
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut results: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
        for batch_indices in order.chunks(BATCH_SIZE) {
            let batch: Vec<String> = batch_indices.iter().map(|&i| texts[i].clone()).collect();
            let embeddings = self.embed_chunk(&batch)?;
            for (&idx, emb) in batch_indices.iter().zip(embeddings) {
                results[idx] = emb;
            }
            if let Some(pb) = pb {
                pb.inc(1);
            }
        }
        Ok(results)
    }

    fn embed_chunk(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
        #[cfg(target_arch = "aarch64")]
        assert_eq!(embedding_backend(), "onnx-qint8-arm64");
        #[cfg(target_arch = "x86_64")]
        assert_eq!(embedding_backend(), "onnx-quint8-avx2");
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(embedding_backend(), "onnx-fp32");
    }

    #[test]
    fn model_files_cover_config_tokenizer_and_onnx() {
        let files = model_file_list();
        assert!(files.contains(&"config.json"), "missing config.json");
        assert!(files.contains(&"tokenizer.json"), "missing tokenizer.json");
        assert!(
            files.iter().any(|f| f.ends_with(".onnx")),
            "no .onnx weight file selected: {files:?}"
        );
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
