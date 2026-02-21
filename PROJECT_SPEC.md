# 🧠 HNS — Hierarchical Narrative Synthesis
### Deep Generative AI · Latent Representation Learning · Hierarchical Reasoning

> **Comprehensive Architecture & Execution Plan**
> **Project Type:** Final Year Computer Science Project
> **Date:** February 2026

---

## 1. EXECUTIVE SUMMARY

HNS (Hierarchical Narrative Synthesis) addresses a fundamental limitation of existing document analysis systems: they operate on data from a single modality, at a single level of abstraction, and generate flat, unstructured responses.

Real-world enterprise data is heterogeneous — financial CSV records, system log files, PDF reports, and plain text coexist in the same analytical pipeline. Existing RAG (Retrieval-Augmented Generation) systems treat all of these uniformly: embed → search → respond. This collapses rich cross-source relationships and fails to distinguish noise from signal.

**Core Innovation:** HNS introduces a two-phase architecture that (1) learns general latent structure from large multi-domain text corpora using deep generative models offline, then (2) projects user-specific heterogeneous data into that learned space, runs a 4-layer hierarchical reasoning pipeline, and synthesizes a structured, evidence-backed narrative report.

**Key Differentiators:**
- Frozen pre-trained backbone + custom-trained deep generative head (DAE + VAE + K-Means)
- Cluster-aware retrieval that elevates topically coherent evidence
- Multi-layer reasoning with conflict detection before synthesis
- Automatic evaluation metrics grounded in the learned latent space

---

## 2. RESEARCH BACKGROUND & ACADEMIC FOUNDATION

### 2.1 Deep Generative Models for Representation Learning

**Key Academic References:**

**Kingma & Welling (2013) — "Auto-Encoding Variational Bayes"**
- Introduced the Variational Autoencoder (VAE) framework
- Showed that learning probabilistic latent variables produces smoother, more interpolable representations than standard autoencoders
- Foundational basis for our VAE-based representation layer

**Vincent et al. (2010) — "Stacked Denoising Autoencoders"**
- Demonstrated that training autoencoders to reconstruct from corrupted inputs produces more robust features
- Directly motivates our DenoisingAutoencoder (DAE) as a pre-filter before VAE compression

**Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Meta AI)**
- Established the RAG architecture as a foundation for grounding LLM generation in retrieved evidence
- Our system extends this with latent-space-aware retrieval and hierarchical post-processing

### 2.2 Hierarchical Reasoning in Language Systems

**Key Academic References:**

**Wei et al. (2022) — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Google Brain)**
- Demonstrated that breaking complex questions into sub-steps dramatically improves LLM accuracy
- Directly motivates our `decompose_query()` planning layer

**Zhong et al. (2022) — "Analytical Reasoning of Text" (ACL 2022)**
- Demonstrates structured evidence linking as a prerequisite for faithful analytical responses
- Informs our evidence linking and validation layers

**Izacard & Grave (2021) — "Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering"**
- Shows that multi-document retrieval with structured fusion significantly outperforms single-passage RAG
- Supports our cluster-aware multi-source retrieval design

### 2.3 Heterogeneous Data Analysis

**Key Academic References:**

**Reimers & Gurevych (2019) — "Sentence-BERT: Sentence Embeddings using Siamese BERT Networks"**
- Established sentence-level embeddings as a universal representation layer
- Justifies our use of `all-MiniLM-L6-v2` as the frozen embedding backbone

**Brown et al. (2020) — "Language Models are Few-Shot Learners" (OpenAI GPT-3)**
- Demonstrated that large instruction-tuned LLMs can generate high-quality structured text from context alone
- Basis for using Mistral-7B-Instruct as our narrative synthesis engine

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 — OFFLINE TRAINING                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Multi-Domain Training Corpus (src/training/data_loader.py) │ │
│  │  • Wikipedia-style encyclopedic text   (~1000 segments)    │ │
│  │  • arXiv-style scientific abstracts    (~500 segments)     │ │
│  │  • Tabular data as natural language    (~300 segments)     │ │
│  │  • Web article paragraphs              (~500 segments)     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ EmbeddingModel — all-MiniLM-L6-v2 (FROZEN, 384-d)         │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓  384-d vectors                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ DenoisingAutoencoder (DAE) — TRAINED FROM SCRATCH          │ │
│  │  Architecture: 384→256→128→256→384                         │ │
│  │  Loss: MSE(recon, clean) | Noise: Gaussian σ=0.3           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓  noise-robust 384-d               │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ VariationalAutoencoder (VAE) — TRAINED FROM SCRATCH        │ │
│  │  Architecture: 384→256→[μ,σ]→64→256→384                   │ │
│  │  Loss: MSE + β·KL Divergence (β=0.001)                     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓  64-d latent vectors              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ K-Means Clustering (k=5) — FIT FROM SCRATCH                │ │
│  │  Input: 64-d latent vectors                                 │ │
│  │  Output: 5 structural topic clusters + centroids           │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│            Artifacts saved to trained_models/                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2 — ONLINE INFERENCE                     │
│                                                                  │
│  User uploads CSV / PDF / TXT / LOG files                        │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Ingestion & Chunking (src/ingestion.py)                    │ │
│  │  CSV → pandas → string | PDF → pdfminer | TXT → UTF-8     │ │
│  │  → 200-word sliding window chunks                          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Representation Projection Pipeline                          │ │
│  │  (src/synthesis_engine.py + loaded trained_models/)        │ │
│  │                                                            │ │
│  │  embed (384-d) → DAE denoise → VAE encode (64-d)          │ │
│  │  → K-Means predict → cluster label per segment            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ FAISS Vector Store (IndexFlatL2, 384-d)                    │ │
│  │   raw embeddings + text stored for retrieval               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Hierarchical Reasoning Pipeline (src/reasoning.py)         │ │
│  │                                                            │ │
│  │  Layer 1 — Planning:   Query decomposition (LLM)           │ │
│  │  Layer 2 — Retrieval:  Cluster-aware FAISS search          │ │
│  │  Layer 3 — Evidence:   Cross-source linking                │ │
│  │  Layer 4 — Validation: Confidence + conflict detection     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Narrative Synthesis (src/synthesis_engine.py)              │ │
│  │   Mistral-7B-Instruct-v0.2 via HuggingFace Inference API  │ │
│  │   → Structured Markdown Report                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             ↓                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Evaluation Layer (src/evaluator.py)                        │ │
│  │  VAE Confidence | Anomaly Likelihood | Coverage | FIIT     │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. DETAILED MODULE SPECIFICATIONS

### 4.1 Module 1 — Offline Training Engine (`train.py`)

#### 4.1.1 Corpus Construction (`src/training/data_loader.py`)

| Function | Domain | Segments | Rationale |
|----------|--------|----------|-----------|
| `load_wikipedia_subset()` | Encyclopedic | ~1000 | General semantic diversity |
| `load_arxiv_subset()` | Scientific | ~500 | Technical language patterns |
| `load_tabular_subset()` | Structured data | ~300 | CSV/financial text patterns |
| `load_webtext_subset()` | Web articles | ~500 | Informal narrative structure |

All loaders include synthetic fallbacks to avoid dependency on large downloads, enabling fast training in any environment.

#### 4.1.2 Model Architectures

**Denoising Autoencoder (DAE)**
```python
class DenoisingAutoencoder(nn.Module):
    # Encoder: Linear(384,256) → ReLU → Dropout(0.2)
    #          Linear(256,128) → ReLU → Dropout(0.2)
    # Decoder: Linear(128,256) → ReLU → Dropout(0.2)
    #          Linear(256,384) → Sigmoid

    def forward(self, x):
        x_noisy = self.add_noise(x)          # Gaussian noise, σ=0.3
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        return decoded                        # Loss vs original clean x
```

**Variational Autoencoder (VAE)**
```python
class VariationalAutoencoder(nn.Module):
    # Encoder: Linear(384,256) → ReLU → [fc_mu(64), fc_logvar(64)]
    # Reparameterize: z = μ + ε·σ  (ε ~ N(0,1))
    # Decoder: Linear(64,256) → ReLU → Linear(256,384) → Sigmoid

    def loss(self, recon, x, mu, logvar):
        recon_loss = F.mse_loss(recon, x)
        kl_loss    = -0.5 * (1 + logvar - mu² - exp(logvar))
        return recon_loss + 0.001 * kl_loss.mean()
```

#### 4.1.3 Training Configuration

| Parameter | DAE | VAE |
|-----------|-----|-----|
| Epochs | 50 | 50 |
| Batch size | 64 | 64 |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) |
| Loss | MSE | MSE + β·KL |
| β (KL weight) | — | 0.001 |

---

### 4.2 Module 2 — Representation Projection Pipeline (`src/synthesis_engine.py`)

Transforms each user segment into a full latent representation at inference time:

```
Text segment
    ↓ EmbeddingModel.generate_embeddings()
384-d semantic vector
    ↓ DenoisingAutoencoder.denoise()
384-d noise-robust representation
    ↓ VariationalAutoencoder.get_latent()
64-d probabilistic latent vector (uses μ, discards σ at inference)
    ↓ LatentClusterer.predict()
Cluster label ∈ {0,1,2,3,4}
```

**VectorStore** uses FAISS `IndexFlatL2` to store the raw 384-d embeddings alongside text metadata for similarity retrieval.

---

### 4.3 Module 3 — Hierarchical Reasoning (`src/reasoning.py`)

The `HierarchicalReasoner` class runs a 4-layer sequential pipeline:

#### Layer 1 — Planning: `decompose_query()`
Uses the LLM to break complex queries into 2–4 targeted sub-questions:
```
"What caused the Q3 revenue drop and did server errors contribute?"
    → ["What were the Q3 revenue figures?",
       "What server errors occurred in Q3?",
       "Is there a correlation between server errors and revenue?"]
```

#### Layer 2 — Cluster-Aware Retrieval: `cluster_aware_retrieve()`
Extends FAISS similarity search with a cluster boost:
```python
# Documents from the same cluster as the query get a -0.3 distance bonus
adjusted_score = raw_l2_distance - (0.3 if same_cluster else 0.0)
```
Retrieves 2× candidates, re-ranks by adjusted score, returns top-N.

#### Layer 3 — Evidence Linking: `link_evidence()`
Groups retrieved documents by inferred source type using keyword detection:

| Bucket | Detection Keywords |
|--------|--------------------|
| `csv_data` | revenue, profit, cost, quarter, $, record |
| `pdf_docs` | section, chapter, figure, abstract, methodology |
| `log_entries` | error, warning, timestamp, exception, traceback |
| `text_docs` | everything else |

#### Layer 4 — Validation: `validate_evidence()`
Performs basic quality assurance before synthesis:
- **Confidence score**: `1.0` → `0.6` if score variance > 5.0 across retrieved docs
- **Conflict flags**: raised when evidence from different source types contradicts
- **Coverage report**: number of distinct source types represented

---

### 4.4 Module 4 — Narrative Synthesis (`src/synthesis_engine.py → generate_narrative()`)

Constructs a structured prompt including:
- The query and all retrieved context documents
- Validation metadata (confidence level, conflict flags)
- Explicit instructions for hierarchical Markdown output

Calls **Mistral-7B-Instruct-v0.2** via the HuggingFace `InferenceClient` chat completions endpoint.

**Prompt Structure:**
```
SYSTEM: You are an Expert Hierarchical Narrative Analyst.
        Synthesize heterogeneous data into structured Markdown reports.

USER:   QUERY: {user_query}
        CONTEXT DATA:
        {retrieved_documents}
        VALIDATION:
        Confidence: {score} | Conflicts: {flags}

        INSTRUCTIONS:
        1. Answer based ONLY on context provided
        2. Structure as: Executive Summary → Findings → Evidence → Anomalies
        3. Flag low-confidence sections explicitly
        4. Cross-reference sources where relationships exist
```

---

### 4.5 Module 5 — Evaluation Layer (`src/evaluator.py`)

Four automatic quality metrics computed post-synthesis:

| Metric | Method | Target |
|--------|--------|--------|
| **VAE Confidence** | `1 - avg_reconstruction_loss / 10` | > 0.7 |
| **Anomaly Likelihood** | Distance from nearest cluster centroid, normalised | < 0.3 |
| **Evidence Coverage** | % of context docs with key terms appearing in narrative | > 0.6 |
| **Faithfulness (FIIT)** | LLM auditor scores 0–1 against source context | > 0.7 |

---

## 5. DATASET & DATA PIPELINE

### 5.1 Offline Training Data

Synthetic multi-domain corpus generated by `src/training/data_loader.py`:

| Domain | Generation Method | Count |
|--------|-------------------|-------|
| Encyclopedic | Parameterised Wikipedia-style sentences | 1000 |
| Scientific | arXiv abstract templates + random fields | 500 |
| Tabular | Finance/Healthcare/Sales row descriptions | 300 |
| Web text | Article paragraph templates | 500 |

Cached to `trained_models/training_corpus.pkl` after first generation.

### 5.2 User Data (Online Phase)

Any combination of:

| Format | Parser | Typical Use Case |
|--------|--------|-----------------|
| `.csv` | pandas → `to_string()` | Financial records, sales data, metrics |
| `.pdf` | pdfminer.six | Research papers, annual reports |
| `.txt` | UTF-8 decode | Notes, documentation |
| `.log` | UTF-8 decode | Server logs, system events |

All formats are chunked uniformly into **200-word segments** before embedding, ensuring consistent input dimensionality across all models.

---

## 6. TECHNOLOGY STACK (100% FREE / OPEN-SOURCE)

### 6.1 Core Technologies

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.10+ | Primary language |
| **Deep Learning** | PyTorch | DAE, VAE model definitions and training |
| **Embeddings** | `sentence-transformers` | `all-MiniLM-L6-v2` (384-d) |
| **Vector Search** | `faiss-cpu` | IndexFlatL2 similarity search |
| **Clustering** | `scikit-learn` | K-Means (k=5) |
| **PDF Parsing** | `pdfminer.six` | Text extraction from PDF |
| **Data** | `pandas`, `numpy` | Tabular processing |
| **LLM API** | `huggingface_hub` | InferenceClient → Mistral-7B |
| **Frontend** | `Streamlit` | Web UI (3-tab dashboard) |
| **Visualization** | `plotly`, `scikit-learn PCA` | Latent space scatter plots |

### 6.2 Free Resource Usage

| Resource | Usage |
|----------|-------|
| HuggingFace Free Tier | Mistral-7B inference API (30 req/min) |
| HuggingFace Hub | `all-MiniLM-L6-v2` model download |
| GitHub | Code versioning |
| Local CPU | Training (DAE + VAE ~2–5 min on CPU) |

---

## 7. PROJECT DIRECTORY STRUCTURE

```
GenAI/
│
├── train.py                      # Offline training orchestrator (run once)
├── app.py                        # Streamlit web application
├── env_config.txt                # API keys (gitignored)
├── env_template.txt              # Template for contributors
├── requirements.txt              # All dependencies
├── README.md                     # Project overview
│
├── src/
│   ├── __init__.py
│   ├── models.py                 # EmbeddingModel, DAE, VAE, LatentClusterer
│   ├── ingestion.py              # File parsers + 200-word chunker
│   ├── synthesis_engine.py       # Projection pipeline + VectorStore + generate_narrative()
│   ├── reasoning.py              # HierarchicalReasoner (4-layer pipeline)
│   ├── evaluator.py              # Quality metrics (VAE confidence, anomaly, coverage, FIIT)
│   ├── llm_wrapper.py            # HuggingFace / OpenAI abstraction
│   ├── rag.py                    # Legacy RAG utilities
│   ├── visualize.py              # Plotting helpers
│   └── training/
│       ├── __init__.py
│       └── data_loader.py        # Multi-domain synthetic corpus builder
│
└── trained_models/               # Saved artifacts (gitignored)
    ├── dae_model.pth             # DAE weights (1 MB)
    ├── vae_model.pth             # VAE weights (1.1 MB)
    ├── clusterer.pkl             # K-Means model + centroids
    ├── training_embeddings.npy   # 384-d training corpus embeddings (2 MB)
    ├── latent_vectors.npy        # 64-d latent vectors (333 KB)
    └── training_metadata.pkl     # Training stats (losses, dims, sizes)
```

---

## 8. IMPLEMENTATION TIMELINE

### Phase 1 — Model Design & Training Pipeline (Week 1)
**Duration:** 5–7 days

- **Day 1–2:** Define model architectures (DAE, VAE, LatentClusterer in `src/models.py`)
- **Day 3–4:** Build data loader with multi-domain synthetic corpus (`src/training/data_loader.py`)
- **Day 5–6:** Write training orchestrator (`train.py`), validate loss curves
- **Day 7:** Save and verify all artifacts in `trained_models/`

**Deliverables:**
- All model classes implemented
- Training pipeline runs end-to-end
- Artifacts persisted to disk

---

### Phase 2 — Online Processing Pipeline (Week 2)
**Duration:** 7 days

- **Day 1–2:** File ingestion and 200-word chunking (`src/ingestion.py`)
- **Day 3–4:** Representation projection pipeline + FAISS VectorStore (`src/synthesis_engine.py`)
- **Day 5–6:** Hierarchical Reasoning 4-layer pipeline (`src/reasoning.py`)
- **Day 7:** LLM wrapper and `generate_narrative()` integration

**Deliverables:**
- Full `Upload → Embed → Project → Cluster → Reason → Synthesize` pipeline
- End-to-end test with sample data

---

### Phase 3 — Evaluation, Frontend & Documentation (Week 3)
**Duration:** 7 days

- **Day 1–2:** Evaluation layer (`src/evaluator.py`) — 4 metrics
- **Day 3–4:** Streamlit app — 3 tabs (Data Processing, Report, Latent Space)
- **Day 5–6:** Auto-narrative on file upload; latent space PCA visualization
- **Day 7:** README, env_template, `.gitignore`, GitHub push

**Deliverables:**
- Fully working web UI
- All evaluation metrics displayed
- Code pushed to GitHub with clean commit history

---

## 9. EVALUATION METRICS

### 9.1 Representation Quality

| Metric | Formula | Target |
|--------|---------|--------|
| DAE Reconstruction Loss | MSE(recon, clean) | < 0.05 |
| VAE ELBO | Recon + β·KL | < 0.08 |
| Cluster Silhouette Score | sklearn.metrics | > 0.35 |

### 9.2 Retrieval Quality

| Metric | Method | Target |
|--------|--------|--------|
| Cluster Precision | Same-cluster docs in top-5 | > 0.6 |
| Retrieval Relevance | Cosine sim (query, retrieved) | > 0.65 |

### 9.3 Narrative Quality

| Metric | Method | Target |
|--------|--------|--------|
| VAE Confidence | Latent reconstruction probability | > 0.7 |
| Evidence Coverage | % context docs referenced | > 0.6 |
| Faithfulness (FIIT) | LLM auditor score | > 0.7 |
| Anomaly Likelihood | Centroid distance, normalised | < 0.3 |

### 9.4 FIIT Framework Compliance

| Dimension | Measurement |
|-----------|------------|
| **Fluency** | Readability — natural language structure |
| **Interactivity** | Presence of action statements / findings |
| **Information** | Factual alignment with source documents |
| **Tone** | Consistent analytical, professional voice |

---

## 10. NOVEL CONTRIBUTIONS

### 10.1 Architectural Novelty

**1. Frozen Backbone + Deep Generative Head**
Unlike pure RAG systems using embedding models directly for retrieval, HNS learns task-specific latent structure on top of the frozen embedding backbone using a DAE → VAE chain. This is analogous to fine-tuning but operates in representation space, not weight space.

**2. Cluster-Aware Retrieval**
Standard FAISS retrieval ranks purely by L2 distance in embedding space. Our cluster-boosted scoring incorporates unsupervised topical coherence into ranking without any supervised signal.

**3. Hierarchical Reasoning Before Synthesis**
Rather than passing retrieved documents directly to an LLM, HNS runs a 4-layer structured reasoning pipeline (Plan → Retrieve → Link → Validate) that organises and quality-checks evidence before synthesis. This reduces hallucination by grounding the LLM in validated, structured context.

**4. Latent-Space Evaluation Metrics**
Rather than relying solely on human evaluation or ROUGE scores, HNS derives objective quality metrics from the learned VAE and K-Means models — providing automatic, unsupervised quality feedback.

### 10.2 Experimental Validation

Proposed experimental comparisons:

| Experiment | Baseline | Ours |
|-----------|----------|------|
| Retrieval precision | Raw FAISS (no cluster boost) | Cluster-aware FAISS |
| Narrative quality | Flat RAG → LLM | Hierarchical reasoning → LLM |
| Representation robustness | Raw embeddings | DAE → VAE embeddings |
| Latent structure quality | PCA of raw embeddings | PCA of VAE latents |

---

## 11. HOW TO USE

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Configure API Key
```bash
cp env_template.txt env_config.txt
# Edit env_config.txt → paste your HuggingFace token
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Step 3 — Train Models (One-Time, ~2–5 min)
```bash
python train.py
```
Saves all artifacts to `trained_models/`.

### Step 4 — Launch App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

### Step 5 — Using the App

| Tab | What to do |
|-----|-----------|
| **📊 Data Processing** | Upload files in sidebar → Click "Process Files" → Auto-report generated |
| **📝 Analysis & Report** | Type a specific query → Click "Generate Report" → View reasoning internals |
| **🔬 Latent Space Explorer** | View PCA 2D scatter of your data coloured by cluster |

---

## 12. RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| HF API rate limits (30 req/min) | Cache responses; use short prompts |
| Large file uploads | 200-word chunking limits memory per segment |
| DAE/VAE convergence failure | Monitor losses; reduce lr or increase epochs |
| LLM hallucination | Evidence grounding in prompt; validation flags |
| Slow CPU training | Batch size 64; synthetic corpus (no downloads) |
| GitHub secret scanning | `env_config.txt` in `.gitignore`; env_template for contributors |

---

## 13. MINIMUM VIABLE PRODUCT (MVP)

### Must-Have ✅
- Working DAE + VAE + K-Means training pipeline
- CSV / PDF / TXT / LOG ingestion
- FAISS-based retrieval
- Hierarchical reasoning (all 4 layers)
- Mistral-7B narrative synthesis
- Streamlit 3-tab UI

### Extensions 🚀
- Real-world dataset integration (live API scraping)
- Fine-tuned embedding model on domain data
- Multi-language support
- Streaming LLM response display
- Export report as PDF

---

## 14. DELIVERABLES CHECKLIST

### Code
- [x] Complete Python codebase (`src/`, `train.py`, `app.py`)
- [x] `requirements.txt`
- [x] `README.md` with setup instructions
- [x] `.gitignore` (excludes keys, model binaries, cache)
- [x] GitHub repository with 10+ meaningful commits

### Documentation
- [x] Comprehensive project specification (this document)
- [x] Module-level docstrings on all files
- [ ] Jupyter notebooks per module (EDA, Training, Reasoning, Generation)
- [ ] Project report (15–20 pages, IEEE format)
- [ ] Presentation slides (20–30 slides)
- [ ] Demo video (5–10 min walkthrough)

### Academic
- [ ] Literature review summary (10+ references)
- [ ] Experimental results table
- [ ] Ablation study (DAE vs no-DAE; cluster-boost vs no-boost)
- [ ] Evaluation metrics table

---

## 15. REFERENCES

1. Kingma, D.P. & Welling, M. (2013). *Auto-Encoding Variational Bayes.* ICLR 2014.

2. Vincent, P., et al. (2010). *Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network.* JMLR.

3. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020, Meta AI.

4. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT Networks.* EMNLP 2019.

5. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022, Google Brain.

6. Izacard, G. & Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.* EACL 2021.

7. Johnson, J., et al. (2019). *Billion-Scale Similarity Search with GPUs.* IEEE Transactions (FAISS).

8. Johnson, J., et al. (2017). *Faiss: A Library for Efficient Similarity Search.* Meta AI Research.

9. Jiang, A.Q., et al. (2023). *Mistral 7B.* arXiv:2310.06825.

10. Brown, T., et al. (2020). *Language Models are Few-Shot Learners (GPT-3).* NeurIPS 2020, OpenAI.

---

## 16. APPENDIX — CONFIGURATION

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `HUGGINGFACE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Any HF instruction model |
| `HUGGINGFACE_API_KEY` | *(set in env_config.txt)* | HuggingFace token |
| `OPENAI_API_KEY` | *(optional)* | Required if using OpenAI |

---

*Built with PyTorch · HuggingFace · FAISS · Streamlit*
*Last Updated: February 2026*
*Author: KarthikMallareddy*
*Project: HNS — Hierarchical Narrative Synthesis*
