# Benchmark results

This document records the measurements used to choose rag-cli's indexing,
storage, chunking, and embedding defaults. It focuses on cold indexing because
search was already fast enough for interactive use.

## Summary

The selected quality-first configuration is:

- 512-character chunks with 64-character overlap
- exact chunk-text deduplication before inference
- inference batches of 128
- one F16 vector and one text body per unique chunk
- compact source and occurrence records
- native CoreML FP16 inference on Apple Silicon
- quantized ONNX Runtime CPU inference on other platforms

On the two large benchmark corpora, storage-v2 plus exact deduplication reduced
CPU indexing time by **27–57%** and index size by **46–64%**. Native CoreML then
reduced Apple Silicon indexing time by a further **61%** while substantially
reducing peak memory. The final native CoreML path was **2.5× faster** than the
optimized CPU-int8 path and **3.5–5.9× faster** than the original implementation.

The default model remains `sentence-transformers/all-MiniLM-L6-v2`. A larger
EmbeddingGemma configuration improved the retrieval proxy by 6 percentage
points, but was approximately 7.5× slower than native MiniLM. That makes it a
possible quality-oriented profile rather than a suitable default.

## Test corpora

The large-corpus tests used two English technical-document collections. Names
and paths are intentionally omitted; the labels below are used throughout this
document.

| Corpus | Eligible source | Files | Chunk occurrences at 512/64 | Unique chunk text |
|---|---:|---:|---:|---:|
| A | 31.6 MiB | 2,652 | 89,598 | 78,988 |
| B | 126.8 MiB | 10,517 | 346,779 | 204,075 |

Corpus A was also used for a 50-query retrieval proxy. Each query was derived
from a unique document heading, and a hit required the expected source to appear
in the first five results. The reported metrics are Recall@5 and MRR@5.

This proxy is useful for detecting regressions on the actual workload, but it is
not a formal relevance-judgment benchmark. Small differences should not be
overinterpreted; large and consistent changes are still informative.

Unless stated otherwise, benchmarks used:

- release builds
- 512-character chunks and 64-character overlap
- the same source snapshot for each comparison
- sequential runs without competing high-load processes
- Apple Silicon for macOS backend comparisons

## Baseline

The original CPU-int8 implementation stored an F32 vector and full source/text
metadata for every chunk occurrence.

| Corpus | Chunks | Cold build | Index size | Throughput |
|---|---:|---:|---:|---:|
| A | 89,598 | 156 s | 176.5 MB | ~575 chunks/s |
| B | 346,779 | 679 s | 681.1 MB | ~511 chunks/s |

Buffered bincode serialization and deserialization removed an index-sized
temporary allocation. Metadata-only no-op checks then avoided loading the vector
index when no source files had changed.

## Batch size

Increasing MiniLM's CPU-int8 batch size from 64 to 128 improved throughput by
about 20%. Increasing it again to 256 provided only about 4% more throughput
while adding roughly 1 GiB of peak memory.

| Chunk / overlap | Batch | Chunks | Cold build | Index size | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 512 / 64 | 64 | 89,598 | 156 s | 176 MB | 1.64 GiB |
| 512 / 64 | 128 | 89,598 | 124 s | 176 MB | 2.21 GiB |
| 1024 / 128 | 64 | 44,441 | 69 s | 106 MB | 1.08 GiB |
| 1024 / 128 | 128 | 44,441 | 66 s | 106 MB | 1.84 GiB |
| 1024 / 128 | 256 | 44,441 | 63 s | 106 MB | 2.88 GiB |

**Decision:** use batch 128. Batch size does not change embeddings, retrieval
quality, or index size.

## Chunk-size trade-offs

Larger chunks substantially reduce the number of embeddings, but they also
change document boundaries and reduce the chance that a result isolates the
relevant passage.

| Chunk / overlap | Chunks | Cold build | Index size | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| 512 / 64 | 89,598 | 124–156 s | 176 MB | 80% | .671 |
| 768 / 96 | 59,610 | 84 s | 130 MB | no improvement | no improvement |
| 1024 / 128 | 44,441 | 69 s | 106 MB | 72% | .577 |

The 1024/128 configuration was about 2.3× faster and 40% smaller than the
original 512/64 run, but lost 8 recall points and substantially reduced MRR.
The intermediate 768/96 setting did not improve the proxy.

**Decision:** retain 512/64 as the quality-first default. Larger chunks can be an
explicit speed/size profile, but should not be selected automatically from total
corpus size. Corpus-size-based selection would also cause unrelated new files to
change the chunking recipe and force existing content to be re-embedded.

## Exact-text deduplication

Many generated or templated documents contain identical chunk bodies. Embedding
identical text more than once produces the same vector, so exact deduplication is
lossless.

| Corpus | Occurrences | Unique text | Duplicate occurrences | Duplicate rate |
|---|---:|---:|---:|---:|
| A | 89,598 | 78,988 | 10,610 | 11.8% |
| B | 346,779 | 204,075 | 142,704 | 41.2% |

This optimization is especially valuable for Corpus B: inference work falls by
more than 40% before changing the model or chunking strategy.

Search scores each unique text once and resolves it to a deterministic source
occurrence. This also prevents repeated boilerplate from consuming several
positions in the top-k results.

## Storage-v2

Storage-v2 separates three concepts:

1. source paths
2. unique chunk text and its vector
3. source/text/offset occurrences

Each unique vector is persisted as F16 rather than F32. Inference remains at the
backend's native precision; only the stored output vector is converted. The
stored F16 norm is retained so search still computes cosine similarity against
the decoded vector.

### End-to-end CPU results

| Corpus | Storage-v1 | Storage-v2 | Build reduction | Size reduction |
|---|---:|---:|---:|---:|
| A | 156 s / 176.5 MB | 114.6 s / 94.6 MB | 27% | 46% |
| B | 679 s / 681.1 MB | 293.4 s / 244.6 MB | 57% | 64% |

Peak RSS for the clean storage-v2 CPU runs was 2.53 GB on Corpus A and 2.43 GB
on Corpus B. The larger corpus no longer requires memory proportional to a
second serialized copy of the index.

### Quality effect

On Corpus A, storage-v2 changed the retrieval proxy as follows:

| Format | Recall@5 | MRR@5 |
|---|---:|---:|
| F32 per occurrence | 80% | .671 |
| Unique F16 vectors | 78% | .670 |

The two-point recall difference is consistent with a combination of F16
rounding and collapsing duplicate result bodies. MRR was effectively unchanged.

### No-op indexing and search

For Corpus B:

| Operation | Time | Peak RSS |
|---|---:|---:|
| No-op index check | 0.38 s | 27.5 MB |
| Search | 0.28 s | 376.5 MB |

The no-op path reads metadata and source hashes without materializing the vector
index. Search remains a flat exact scan; unique F16 vectors made it smaller and
faster without introducing an approximate index.

### Federated search

A federated query over both storage-v2 indexes searched 283,063 unique vectors
(436,377 source occurrences) in **0.44 seconds** with **383.3 MB maximum RSS**.
The larger index alone took 0.28 seconds and 376.5 MB. Searching indexes
sequentially therefore kept peak memory close to the largest individual index,
while embedding the query only once and computing a global top-k across both.

## Inference backend comparison

### Earlier CPU, Metal, and ONNX measurements

An earlier apples-to-apples run over 14,679 MiniLM chunks showed why ONNX CPU
int8 became the initial default:

| Backend | Approximate throughput | Relative observation |
|---|---:|---|
| Candle CPU F32 | 93 chunks/s | Slowest CPU path |
| ONNX Runtime CPU F32 | 276 chunks/s | About 3× Candle CPU |
| Candle Metal F32 | 300–321 chunks/s | Dispatch-bound |
| ONNX Runtime CPU int8 | 595 chunks/s | About 1.85× Candle Metal |

Model int8 quantization is different from quantizing stored output embeddings.
The int8 model still emits F32 embeddings. Published Sentence Transformers
backend tests reported about 99.7% quality retention for ONNX model
quantization, whereas output-vector int8 has a materially larger retrieval cost.

### Native CoreML MiniLM

A native FP16 CoreML MLProgram was tested with fixed 1×256 inputs and fused
pooling and normalization. The model is loaded once and receives batches through
CoreML's batch-prediction API.

A 128-input microbenchmark sustained approximately **1,730 embeddings/s** at
about **150 MiB**. CPU-only, CPU+Neural Engine, and all-compute-unit modes were
similar, so these measurements do not support attributing the speedup
specifically to the Neural Engine.

End-to-end results:

| Corpus | CPU-int8 storage-v2 | Native CoreML | Speedup | CoreML peak RSS | Index size |
|---|---:|---:|---:|---:|---:|
| A | 114.6 s | 45.0 s | 2.55× | 402 MB | 94.6 MB |
| B | 293.4 s | 115.7 s | 2.54× | 878 MB | 244.6 MB |

Corpus A retrieval remained at 78% Recall@5. MRR changed from .670 on CPU-int8
to .678 on native CoreML.

**Decision:** use native CoreML MiniLM on Apple Silicon and retain quantized
ONNX Runtime CPU inference elsewhere. The backend identifier is part of index
metadata, so changing backend or precision automatically forces a clean rebuild.

## Alternative model evaluation

Published leaderboards were used only to nominate candidates. Final decisions
were based on end-to-end measurements on Corpus A with the real chunking,
storage, and retrieval pipeline.

| Model and backend | Output dimensions | Cold build | Peak RSS | Index size | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM native CoreML | 384 | **45.0 s** | **402 MB** | 94.6 MB | 78% | .678 |
| MiniLM CPU-int8 | 384 | 114.6 s | 2.53 GB | 94.6 MB | 78% | .670 |
| GIST MiniLM CPU-int8 | 384 | 123.6 s | 1.96 GB | 94.6 MB | 72% | .628 |
| EmbeddingGemma CoreML MRL-256 | 256 | 336.1 s | 1.71 GB* | **75.3 MB** | **84%** | **.727** |
| Granite Small English R2 CPU-int8 | 384 | 717.0 s | 15.4 GB | 94.6 MB | 78% | .712 |

\* CoreML memory reporting is not perfectly comparable with ordinary process
memory. The operating-system tool reported 1.71 GB maximum resident size and a
428.6 MB peak-memory-footprint metric for the same EmbeddingGemma run.

### GIST MiniLM

GIST MiniLM looked attractive from published results because it retains a
six-layer MiniLM architecture. On the actual corpus, however, it was slower than
the current CPU-int8 model and reduced both Recall@5 and MRR.

**Decision:** reject it. Published aggregate retrieval gains did not transfer to
this corpus.

### Granite Small English R2

Granite improved MRR on Corpus A but did not improve Recall@5. At batch 128 its
CPU-int8 path was about 6.3× slower than MiniLM CPU-int8 and consumed far more
memory. A smaller batch would lower memory but would not close the throughput
gap.

Granite's FP16 ONNX export also performed poorly through ONNX Runtime's CoreML
execution provider:

- dynamic dimensions initially prevented CoreML compilation
- fixed batch and sequence dimensions allowed compilation
- the graph was split into 26 CoreML subgraphs
- sustained throughput was only about 51–68 chunks/s
- end-to-end indexing was roughly 8–11× slower than MiniLM CPU-int8 on smaller
  real-document workloads

The fragmented graph creates many CPU/CoreML boundaries. This result applies to
Granite through ONNX Runtime's CoreML provider; it does not imply that native
CoreML embedding models are inherently slow.

**Decision:** reject Granite for the default path.

### EmbeddingGemma MRL-256

EmbeddingGemma used fixed 1×128 inputs, asymmetric query/document prefixes, and
the first 256 Matryoshka dimensions followed by re-normalization.

Its microbenchmark showed real Neural Engine benefit:

| Compute units | Throughput |
|---|---:|
| CPU only | ~143 embeddings/s |
| CPU + Neural Engine / all | ~243 embeddings/s |

Despite that acceleration, it remained approximately 7.5× slower than native
MiniLM end to end. It improved Recall@5 from 78% to 84%, improved MRR from .678
to .727, and reduced index size by about 21% through its 256-dimensional output.

**Decision:** do not make it the default. It is a defensible optional quality
profile when six additional recall points justify substantially longer indexing
and a much larger model artifact.

## Approaches not selected

### HNSW

HNSW accelerates search but does not reduce embedding time. It also adds graph
storage. Since exact flat search was already about 0.3 seconds on the larger
corpus, HNSW optimizes the wrong bottleneck for this workload.

### Stored int8 or binary vectors

Quantizing persisted output vectors would save more space, but published and
local evidence indicates a larger quality penalty than F16. F16 captures most
of the vector-size win while preserving exact flat cosine ranking closely.

### Automatic large-corpus chunking

Automatically increasing chunk size based on corpus size would improve build
time and storage, but it would lower measured recall and make unrelated corpus
growth change the embedding recipe. Explicit profiles are safer and more
predictable.

### ONNX Runtime CoreML for complex transformer graphs

Fixed shapes alone are insufficient. A model must also compile into a small
number of accelerator-friendly subgraphs. Native MLPrograms with bounded shapes,
FP16 dense operations, fused pooling/normalization, and long-lived model reuse
were much more effective than a fragmented ONNX graph.

## Final conclusions

1. **Avoid redundant inference first.** Exact text deduplication removed 12–41%
   of embedding work without changing semantics.
2. **Use a normalized index layout.** One F16 vector per unique text produced
   larger practical gains than adding an approximate search structure.
3. **Batch 128 is the CPU sweet spot.** Batch 256's small throughput gain did not
   justify its memory cost.
4. **Keep 512/64 as the default.** Larger chunks are faster and smaller, but the
   measured recall loss is too large for a quality-first default.
5. **Use native CoreML on Apple Silicon.** Native MiniLM was about 2.5× faster
   than the optimized CPU-int8 path with lower memory and comparable retrieval.
6. **Evaluate models on the target corpus.** GIST's published advantage did not
   transfer, Granite's quality gain was operationally too expensive, and
   EmbeddingGemma exposed a clear but costly speed/quality trade-off.
7. **Keep exact search until it is the bottleneck.** At roughly 0.3 seconds on
   the larger corpus, indexing throughput and resident index size matter more.
