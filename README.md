# 🧠 Hierarchical Narrative Synthesis

> **Deep Generative AI · Latent Representation Learning · Hierarchical Reasoning**

A two-phase intelligent document analysis system that trains deep generative models offline, then processes heterogeneous user data — CSV, PDF, TXT, LOG — through a multi-layer reasoning pipeline to produce structured narrative reports.

---

## 🎯 Motive

Traditional RAG (Retrieval-Augmented Generation) systems simply embed documents, search by similarity, and pass results to an LLM. This works for simple Q&A but fails for:

- **Heterogeneous data** — mixing financial CSVs, server logs, and research PDFs
- **Noisy real-world documents** — OCR errors, inconsistent formatting
- **Deep cross-source reasoning** — understanding how a server anomaly in a log relates to a revenue drop in a CSV
- **Structured, evidence-backed reports** — not just answers but auditable narratives

This system addresses all four by introducing **learned latent representations**, **cluster-aware retrieval**, and a **4-layer hierarchical reasoning pipeline** before synthesis.

---

## 🏛️ Architecture — 2-Phase Design

```
┌─────────────────────────────────────────────────┐
│           PHASE 1 — OFFLINE TRAINING             │
│                                                  │
│  Synthetic Corpus (2300 segments)                │
│       ↓                                          │
│  EmbeddingModel (all-MiniLM-L6-v2, frozen)      │
│       ↓  384-d vectors                            │
│  DenoisingAutoencoder  ← trained from scratch    │
│       ↓  noise-robust 384-d                       │
│  VariationalAutoencoder ← trained from scratch   │
│       ↓  64-d latent vectors                      │
│  K-Means Clustering (5 clusters)                 │
│       ↓                                          │
│  Artifacts saved to trained_models/              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│           PHASE 2 — ONLINE INFERENCE             │
│                                                  │
│  User uploads CSV / PDF / TXT / LOG              │
│       ↓  ingestion + chunking                    │
│  Projection Pipeline                             │
│  (Embed → DAE → VAE → Cluster)                   │
│       ↓  indexed in FAISS                        │
│  Hierarchical Reasoning (4 layers)               │
│       ↓                                          │
│  Mistral-7B Narrative Synthesis                  │
│       ↓                                          │
│  Evaluation Metrics                              │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Semantic Embeddings** | `sentence-transformers` — `all-MiniLM-L6-v2` (384-d) |
| **Denoising Autoencoder** | `PyTorch` — custom encoder-decoder, MSE loss |
| **Variational Autoencoder** | `PyTorch` — reparameterization trick, KL divergence + MSE |
| **Clustering** | `scikit-learn` — K-Means (5 clusters) |
| **Vector Store** | `FAISS` — IndexFlatL2, exact L2 similarity search |
| **LLM** | `Mistral-7B-Instruct-v0.2` via HuggingFace Inference API |
| **PDF Parsing** | `pdfminer.six` |
| **CSV Processing** | `pandas` |
| **Frontend** | `Streamlit` |
| **Visualization** | `plotly`, `scikit-learn` PCA |

---

## 🗂️ Project Structure

```
GenAI/
├── train.py                    # Offline training orchestrator
├── app.py                      # Streamlit web application
├── env_config.txt              # API keys (gitignored)
├── env_template.txt            # Template for new contributors
│
├── src/
│   ├── models.py               # EmbeddingModel, DAE, VAE, LatentClusterer
│   ├── ingestion.py            # File parsers + chunker
│   ├── synthesis_engine.py     # Projection pipeline + FAISS VectorStore
│   ├── reasoning.py            # 4-layer HierarchicalReasoner
│   ├── evaluator.py            # Quality metrics
│   ├── llm_wrapper.py          # HuggingFace / OpenAI API abstraction
│   └── training/
│       ├── __init__.py
│       └── data_loader.py      # Synthetic multi-domain corpus builder
│
└── trained_models/             # Saved artifacts (gitignored)
    ├── dae_model.pth
    ├── vae_model.pth
    ├── clusterer.pkl
    ├── training_embeddings.npy
    ├── latent_vectors.npy
    └── training_metadata.pkl
```

---

## 🧩 How Each Component Works

### 1. Data Ingestion (`src/ingestion.py`)
Handles 4 file types:
- **CSV** → loaded with `pandas`, converted to string via `df.to_string()`
- **PDF** → text extracted with `pdfminer.six`
- **TXT / LOG** → decoded as UTF-8

All outputs are chunked into **200-word segments** — uniform input for the embedding model.

### 2. Semantic Embedding (`src/models.py → EmbeddingModel`)
Uses `all-MiniLM-L6-v2` from HuggingFace — a transformer pre-trained on 1 billion+ sentence pairs. Converts each text segment into a **384-dimensional vector** where semantically similar text is geometrically close. Weights are **frozen** (not fine-tuned).

### 3. Denoising Autoencoder (`src/models.py → DenoisingAutoencoder`)
- **Architecture**: 384 → 256 → 128 → 256 → 384
- **Training**: Gaussian noise added to input (σ=0.3), model learns to reconstruct the clean original
- **Purpose**: Smooths inconsistencies from heterogeneous data (OCR noise, CSV formatting artifacts, log syntax), producing robust representations

### 4. Variational Autoencoder (`src/models.py → VariationalAutoencoder`)
- **Architecture**: 384 → 256 → μ, σ (64-d each) → reparameterize → 64-d → 256 → 384
- **Loss**: MSE reconstruction + KL divergence (weighted 0.001)
- **Purpose**: Compresses 384-d to 64-d probabilistic latent space. The smooth, continuous latent space enables meaningful cluster discovery and anomaly detection

### 5. Latent Clustering (`src/models.py → LatentClusterer`)
K-Means (k=5) fitted on the 64-d VAE latent vectors. Assigns every document to one of 5 structural topic clusters. Used during retrieval to boost documents from the same cluster as the query.

### 6. Vector Store (`src/synthesis_engine.py → VectorStore`)
FAISS `IndexFlatL2` stores raw 384-d embeddings for fast nearest-neighbour search. Metadata (original text content) stored separately in a `.pkl` file. Indexed per session — reset on each new file upload.

### 7. Hierarchical Reasoning (`src/reasoning.py → HierarchicalReasoner`)

| Layer | Method | What it does |
|-------|--------|-------------|
| 1 — Planning | `decompose_query()` | LLM breaks query into 2–4 focused sub-questions |
| 2 — Retrieval | `cluster_aware_retrieve()` | FAISS search + -0.3 distance bonus for same-cluster docs |
| 3 — Evidence | `link_evidence()` | Groups docs by source type: CSV, PDF, LOG, text |
| 4 — Validation | `validate_evidence()` | Score variance → confidence; flags cross-source conflicts |

### 8. Narrative Synthesis (`src/synthesis_engine.py → generate_narrative()`)
Constructs a structured prompt containing the query, retrieved evidence (organised by source), validation metadata (confidence, conflicts), then calls **Mistral-7B-Instruct-v0.2** via HuggingFace Inference API. Output is a Markdown report with:
- Executive Summary
- Key Findings
- Supporting Evidence
- Anomalies & Conflicts

### 9. Evaluation (`src/evaluator.py`)

| Metric | How |
|--------|-----|
| **VAE Confidence** | `1 - avg_reconstruction_loss / 10` |
| **Anomaly Likelihood** | Distance from nearest cluster centroid |
| **Evidence Coverage** | % of context docs referenced in narrative |
| **Faithfulness** | LLM auditor scores narrative vs source (0–1) |

---

## 🚀 How to Use

### Prerequisites
- Python 3.10+
- Anaconda (recommended)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Copy the template and add your HuggingFace token:
```bash
cp env_template.txt env_config.txt
```
Edit `env_config.txt`:
```
HUGGINGFACE_API_KEY=hf_your_actual_key_here
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2
LLM_PROVIDER=huggingface
```
Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 3. Train the Models (One-Time)
```bash
python train.py
```
This takes ~2–5 minutes and saves all model artifacts to `trained_models/`.

### 4. Launch the App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 5. Using the App

**Tab 1 — Data Processing:**
1. Upload your files (CSV, PDF, TXT, LOG) in the sidebar
2. Click **"🚀 Process Files"**
3. The system runs the full projection pipeline and auto-generates a narrative report

**Tab 2 — Analysis & Report:**
1. Type a specific query (e.g. *"What are the critical server events?"*)
2. Click **"Generate Report"**
3. Expand "Reasoning Internals" to see sub-queries, evidence map, and confidence
4. Expand "Evaluation Metrics" for quality scores

**Tab 3 — Latent Space Explorer:**
- PCA 2D scatter plot of your data coloured by cluster
- Training loss curves for DAE and VAE

---

## 🔑 What Makes It Hierarchical

**1. Representation Hierarchy** — Data abstracted through 5 levels:
```
Raw Text → 384-d Embedding → Denoised 384-d → 64-d Latent → Cluster Label
```

**2. Reasoning Hierarchy** — 4 ordered processing layers:
```
Plan → Retrieve → Link Evidence → Validate → Synthesize
```

**3. Narrative Hierarchy** — LLM produces structured tiered report:
```
Executive Summary → Key Findings → Evidence → Anomalies
```

---

## 🧪 Sample Use Cases

| Data | Query | What You Get |
|------|-------|-------------|
| `sales_q1.csv` + `server_logs.log` | "Are server errors affecting revenue?" | Cross-source analysis linking downtime to sales dips |
| `research_paper.pdf` | "Summarise the methodology" | Section-aware extraction with evidence citations |
| Multiple CSVs | "Compare performance across quarters" | Trend analysis with anomaly flags |

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `huggingface` | `huggingface` or `openai` |
| `HUGGINGFACE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | Any HF instruction model |
| `OPENAI_API_KEY` | — | Required if using OpenAI provider |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with PyTorch · HuggingFace · FAISS · Streamlit*
