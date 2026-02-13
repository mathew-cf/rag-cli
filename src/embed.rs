use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

pub const DEFAULT_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";
const BATCH_SIZE: usize = 64;

const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

pub struct EmbeddingEngine {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
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

// ---------------------------------------------------------------------------
// Embedding engine
// ---------------------------------------------------------------------------

impl EmbeddingEngine {
    /// Load the embedding model.
    /// Downloads from HuggingFace on first use (via native-tls), cached in HF layout after that.
    pub fn load(model_id: Option<&str>, device: &Device, cache_dir: Option<&Path>) -> Result<Self> {
        let model_id = model_id.unwrap_or(DEFAULT_MODEL);
        let hub_root = resolve_hf_cache(cache_dir)?;

        eprintln!("Loading model: {model_id}");
        eprintln!("  Cache: {}", hub_root.display());

        ensure_model_files(&hub_root, model_id)?;

        let config_path = snapshot_path(&hub_root, model_id, "config.json");
        let tokenizer_path = snapshot_path(&hub_root, model_id, "tokenizer.json");
        let weights_path = snapshot_path(&hub_root, model_id, "model.safetensors");

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: Config = serde_json::from_str(&config_str)?;
        let hidden_size = config.hidden_size;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, device)
                .context("Failed to load model weights")?
        };
        let model = BertModel::load(vb, &config).context("Failed to build BERT model")?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;

        eprintln!("  Model loaded (hidden_size={hidden_size}, device={device:?})");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            hidden_size,
        })
    }

    /// Embed a single text string. Returns a normalized f32 vector.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Embed a batch of texts. Returns normalized f32 vectors.
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(BATCH_SIZE) {
            let batch_embeddings = self.embed_chunk(chunk)?;
            all_embeddings.extend(batch_embeddings);
        }
        Ok(all_embeddings)
    }

    /// Embed texts with a progress bar.
    pub fn embed_batch_progress(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let total_batches = (texts.len() + BATCH_SIZE - 1) / BATCH_SIZE;
        let pb = ProgressBar::new(total_batches as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  Embedding [{bar:40.cyan/blue}] {pos}/{len} batches ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );

        let mut all_embeddings = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(BATCH_SIZE) {
            let batch_embeddings = self.embed_chunk(chunk)?;
            all_embeddings.extend(batch_embeddings);
            pb.inc(1);
        }

        pb.finish_and_clear();
        Ok(all_embeddings)
    }

    fn embed_chunk(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut tokenizer = self.tokenizer.clone();
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let tokens = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|t| {
                let ids: Vec<u32> = t.get_ids().to_vec();
                Tensor::new(ids.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let attention_masks: Vec<Tensor> = tokens
            .iter()
            .map(|t| {
                let mask: Vec<u32> = t.get_attention_mask().to_vec();
                Tensor::new(mask.as_slice(), &self.device)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_masks, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        // Forward pass: [batch, seq_len, hidden_size]
        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling with attention mask
        let mask = attention_mask.to_dtype(DTYPE)?.unsqueeze(2)?;
        let masked = embeddings.broadcast_mul(&mask)?;
        let summed = masked.sum(1)?;
        let mask_sum = mask.sum(1)?;
        let mean_pooled = summed.broadcast_div(&mask_sum)?;

        // L2 normalize
        let norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean_pooled.broadcast_div(&norm)?;

        let batch_size = texts.len();
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let emb = normalized.get(i)?.to_vec1::<f32>()?;
            results.push(emb);
        }

        Ok(results)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Select the best available device.
pub fn select_device() -> Result<Device> {
    #[cfg(target_os = "macos")]
    {
        if let Ok(device) = Device::new_metal(0) {
            eprintln!("Using Metal GPU");
            return Ok(device);
        }
    }
    eprintln!("Using CPU");
    Ok(Device::Cpu)
}
