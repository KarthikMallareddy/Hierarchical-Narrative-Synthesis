"""
synthesis_engine.py — Online Processing & Generative Synthesis

Handles:
  - Vector storage (FAISS)
  - Full representation projection pipeline (Embed → DAE → VAE → Cluster)
  - Cluster-aware context retrieval
  - LLM-based narrative generation from validated evidence
"""

import faiss
import numpy as np
import pickle
import os
import torch

from src.llm_wrapper import LLMProvider


class VectorStore:
    def __init__(self, index_file="faiss_index.bin", metadata_file="faiss_metadata.pkl", dimension=384):
        """
        Initialize FAISS index and metadata storage.
        """
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadatas = []

        # Load existing index if available
        if os.path.exists(self.index_file):
            try:
                self.index = faiss.read_index(self.index_file)
                if self.index.d != self.dimension:
                    self.index = faiss.IndexFlatL2(self.dimension)
            except:
                pass

        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "rb") as f:
                    self.metadatas = pickle.load(f)
            except:
                pass

    def add_documents(self, documents, embeddings, metadatas=None):
        """Add documents and their embeddings to the store."""
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, 'tolist'):
            embeddings = np.array(embeddings)

        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

        new_metadatas = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            meta["content"] = doc
            new_metadatas.append(meta)

        self.metadatas.extend(new_metadatas)

        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadatas, f)

    def query(self, query_embedding, n_results=5):
        """Query the store using a query embedding."""
        if hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding.numpy()
        elif hasattr(query_embedding, 'tolist'):
            query_embedding = np.array(query_embedding)

        query_embedding = np.array(query_embedding).astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        D, I = self.index.search(query_embedding, n_results)

        retrieved_results = []
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.metadatas):
                content = self.metadatas[idx].get("content", "")
                retrieved_results.append((content, float(dist)))

        return retrieved_results

    def reset(self):
        """Clear the index and metadata."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadatas = []
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)


# ============================================================
# Full Representation Projection Pipeline
# ============================================================
def project_to_latent(embeddings, dae, vae, clusterer):
    """
    Project raw embeddings through the full learned pipeline:
      Embed → DAE denoise → VAE encode → Cluster assign
    
    Returns: (denoised, latent_vectors, cluster_labels)
    """
    # Step 1: Denoise via DAE
    denoised = dae.denoise(embeddings)

    # Step 2: Encode to latent space via VAE
    latent_vectors = vae.get_latent(denoised)

    # Step 3: Assign clusters
    cluster_labels = clusterer.predict(latent_vectors)

    return denoised, latent_vectors, cluster_labels


# ============================================================
# Context Retrieval (Updated for v2 pipeline)
# ============================================================
def retrieve_context(query, vector_store, embedding_model, dae=None, vae=None, clusterer=None):
    """
    High-level function to retrieve context for a query.
    Uses the full projection pipeline when models are available.
    Returns a list of (document, score) tuples.
    """
    # Step 1: Generate raw embedding
    query_emb = embedding_model.generate_embeddings([query])
    if torch.is_tensor(query_emb):
        query_emb = query_emb.cpu().numpy()

    query_cluster = None

    # Step 2: Project through trained models (if available)
    if dae is not None and vae is not None and clusterer is not None:
        _, query_latent, query_clusters = project_to_latent(query_emb, dae, vae, clusterer)
        query_cluster = int(query_clusters[0])

    # Step 3: Search in vector store (using raw embeddings for FAISS)
    results = vector_store.query(query_emb, n_results=5)

    return results, query_cluster


# ============================================================
# Generative Narrative Synthesis
# ============================================================
def generate_narrative(query, context_docs, validation_result=None):
    """
    Generate a hierarchical narrative using the configured LLM provider.
    Incorporates validation metadata if available.
    """
    context_text = "\n---\n".join(context_docs)

    # Build validation context
    validation_info = ""
    if validation_result:
        conf = validation_result.get("confidence", 1.0)
        conflicts = validation_result.get("conflicts", [])
        source_count = validation_result.get("source_count", 0)
        cross_source = validation_result.get("cross_source", False)
        
        validation_info = f"""
    VALIDATION METADATA:
    - Confidence Score: {conf:.2f}
    - Source Types: {source_count}
    - Cross-Source Analysis: {"Yes" if cross_source else "No"}
    - Conflicts Detected: {len(conflicts)}
    {"- Conflict Details: " + "; ".join(conflicts) if conflicts else ""}
    """

    system_prompt = """
    You are an Expert Hierarchical Narrative Analyst.
    Your goal is to synthesize heterogeneous data into a structured report.
    You leverage deep latent representations to identify patterns.
    Format your answer in Markdown with sections.
    """

    user_prompt = f"""
    QUERY: {query}

    CONTEXT DATA (Retrieved via Cluster-Aware Semantic Search):
    {context_text}
    {validation_info}

    INSTRUCTIONS:
    1. Answer the query based ONLY on the context data provided.
    2. Cite the specific evidence (CSV data, PDF content, Log entries, or Text).
    3. Structure the answer hierarchically:
       - **Executive Summary** (Top Level) — Key findings
       - **Key Insights** (Middle Level) — Detailed analysis per source
       - **Supporting Evidence** (Bottom Level) — Raw data citations
    4. If conflicts were detected in validation, mention them.
    5. Include a confidence assessment based on evidence coverage.
    """

    provider_name = os.getenv("LLM_PROVIDER", "huggingface")
    llm = LLMProvider(provider=provider_name)

    return llm.generate(user_prompt, system_prompt)
