# Project Walkthrough: Hierarchical Narrative Synthesis

This project implements an advanced AI system for processing heterogeneous data into structured narratives using unsupervised representation learning.

## Key Features Implemented

### 1. Heterogeneous Data Ingestion
- **Support**: CSV, PDF, LOG, and TXT files.
- **Location**: `src/ingestion.py`
- **Demo**: Upload `sample_financials.csv` and `sample_logs.log` simultaneously.

### 2. Autoencoder Representation Learning
- **Innovation**: Instead of standard RAG, we train a custom Autoencoder on your specific data.
- **Benefit**: Learns latent concepts and filters noise, moving from 384 dimensions to 128 dimensions.
- **Location**: `src/synthesis_engine.py`

### 3. Latent Space Visualization
- **Feature**: A "Latent Space Explorer" that plots your data clusters in 2D.
- **Tech**: PCA (Principal Component Analysis) + Plotly.
- **Demo**: View the "Latent Space Explorer" tab after processing files.

### 4. Generative Synthesis
- **Feature**: Creates structured, hierarchical reports using OpenAI's GPT models.
- **System Prompt**: "You are a helpful AI assistant tasked with hierarchical narrative synthesis."

### 5. Self-Evaluation & Audit (Backend Depth)
- **Innovative Feature**: The system runs a "hidden" second pass where it acts as a strict Auditor.
- **Task**: It cross-checks the generated report against the source context to detect hallucinations.
- **Output**: Provides a **Faithfulness Score** and a detailed critique of the report's accuracy.

## How to Run & Verify

1. **Environment Setup**:
   - Ensure `OPENAI_API_KEY` is in `env_config.txt`.
   - Install dependencies: `pip install -r requirements.txt`.

2. **Launch Application**:
   - Run `streamlit run app.py`.

3. **Verify the Pipeline**:
   - **Step A**: Upload the `sample_*.` files.
   - **Step B**: Click "Process Files" (watch the Autoencoder train).
   - **Step C**: Use the "Analysis & Report" tab to ask: *"What are the critical server events and how do they impact the financial outlook?"*
   - **Step D**: Expand "View System Internals" to see the reconstruction loss and retrieved chunks.
   - **Step E**: Use "Latent Space Explorer" to see the data clusters.

## Architecture Diagram Alignment
The system follows the provided multi-level architecture:
- **Level 1**: Ingestion & Normalization (`src/ingestion.py`)
- **Level 2**: Reasoning & Knowledge Processing (`src/models.py`, `src/synthesis_engine.py`)
- **Level 3**: Generative Synthesis (`src/synthesis_engine.py`)
- **Level 4**: Evaluation & UI (`app.py`, `src/visualize.py`)
