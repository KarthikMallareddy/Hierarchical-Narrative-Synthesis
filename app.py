"""
Hierarchical Narrative Synthesis — Streamlit Application (v2)

Two-Phase Architecture:
  Phase 1: Offline Training (run train.py separately)
  Phase 2: Online User Data Processing + Hierarchical Reasoning + Synthesis
"""

import streamlit as st
import os
import numpy as np
import torch
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv("env_config.txt")

# Page configuration
st.set_page_config(
    page_title="Hierarchical Narrative Synthesis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
    }
    .stButton>button {
        background-color: #2563EB;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #475569;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Loading Helpers
# ============================================================
ARTIFACTS_DIR = "trained_models"


@st.cache_resource
def load_trained_models():
    """Load pre-trained model artifacts if available."""
    models = {"loaded": False}
    
    try:
        from src.models import EmbeddingModel, DenoisingAutoencoder, VariationalAutoencoder, LatentClusterer
        
        # Load metadata first
        meta_path = os.path.join(ARTIFACTS_DIR, "training_metadata.pkl")
        if not os.path.exists(meta_path):
            return models
        
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load Embedding Model
        embedding_model = EmbeddingModel()
        
        # Load DAE
        dae = DenoisingAutoencoder(input_dim=metadata["embedding_dim"])
        dae.load_state_dict(torch.load(
            os.path.join(ARTIFACTS_DIR, "dae_model.pth"),
            map_location=torch.device('cpu'),
            weights_only=True
        ))
        dae.eval()
        
        # Load VAE
        vae = VariationalAutoencoder(
            input_dim=metadata["embedding_dim"],
            latent_dim=metadata["latent_dim"]
        )
        vae.load_state_dict(torch.load(
            os.path.join(ARTIFACTS_DIR, "vae_model.pth"),
            map_location=torch.device('cpu'),
            weights_only=True
        ))
        vae.eval()
        
        # Load Clusterer
        clusterer = LatentClusterer.load(os.path.join(ARTIFACTS_DIR, "clusterer.pkl"))
        
        models = {
            "loaded": True,
            "embedding_model": embedding_model,
            "dae": dae,
            "vae": vae,
            "clusterer": clusterer,
            "metadata": metadata,
        }
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models


def main():
    st.title("🧠 Hierarchical Narrative Synthesis")
    st.caption("Deep Generative AI — Latent Representation Learning + Hierarchical Reasoning")

    # =========================================================
    # SIDEBAR — System Status & Controls
    # =========================================================
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Load models
        models = load_trained_models()
        
        if models["loaded"]:
            st.success("✅ Pre-trained models loaded")
            meta = models["metadata"]
            st.markdown(f"""
            **Training Stats:**
            - Corpus: **{meta['corpus_size']}** segments
            - Embedding: **{meta['embedding_dim']}**-d
            - Latent: **{meta['latent_dim']}**-d
            - Clusters: **{meta['n_clusters']}**
            - DAE Loss: `{meta['dae_final_loss']:.4f}`
            - VAE Loss: `{meta['vae_final_loss']:.4f}`
            """)
        else:
            st.warning("⚠️ No pre-trained models found")
            st.markdown("""
            Run training first:
            ```
            python train.py
            ```
            """)
        
        st.divider()
        st.header("📁 Upload Your Data")
        uploaded_files = st.file_uploader(
            "Upload CSV, PDF, TXT, or LOG files",
            type=["csv", "pdf", "txt", "log"],
            accept_multiple_files=True
        )

    # =========================================================
    # MAIN AREA — Tabs
    # =========================================================
    tab1, tab2, tab3 = st.tabs([
        "📊 Data Processing", "📝 Analysis & Report", "🔬 Latent Space Explorer"
    ])

    # ---------------------------------------------------------
    # TAB 1: Data Processing
    # ---------------------------------------------------------
    with tab1:
        if uploaded_files:
            st.subheader("📄 Uploaded Files")
            cols = st.columns(len(uploaded_files))
            for i, f in enumerate(uploaded_files):
                with cols[i]:
                    st.markdown(f"**{f.name}** ({f.size} bytes)")

            if st.button("🚀 Process Files", type="primary"):
                with st.spinner("Processing..."):
                    from src.ingestion import process_uploaded_files
                    from src.synthesis_engine import VectorStore
                    
                    # Step 1: Ingest
                    st.write("**Step 1:** Extracting text segments...")
                    documents = process_uploaded_files(uploaded_files)
                    st.session_state.documents = documents
                    st.write(f"  → {len(documents)} segments extracted")

                    if not models["loaded"]:
                        st.error("❌ Models not loaded. Run `python train.py` first.")
                        return

                    emb_model = models["embedding_model"]
                    dae = models["dae"]
                    vae = models["vae"]
                    clusterer = models["clusterer"]

                    # Step 2: Embed
                    st.write("**Step 2:** Generating semantic embeddings...")
                    embeddings = emb_model.generate_embeddings(documents)
                    if torch.is_tensor(embeddings):
                        embeddings = embeddings.cpu().numpy()
                    embeddings = np.array(embeddings).astype(np.float32)
                    st.session_state.raw_embeddings = embeddings

                    # Step 3: DAE Denoise
                    st.write("**Step 3:** Denoising via DAE...")
                    denoised = dae.denoise(embeddings)
                    st.session_state.denoised = denoised

                    # Step 4: VAE Encode
                    st.write("**Step 4:** Projecting to latent space via VAE...")
                    latent = vae.get_latent(denoised)
                    st.session_state.latent_vectors = latent

                    # Step 5: Cluster Assignment
                    st.write("**Step 5:** Assigning clusters via K-Means...")
                    clusters = clusterer.predict(latent)
                    st.session_state.cluster_labels = clusters

                    # Step 6: Store in Vector Store
                    st.write("**Step 6:** Building session knowledge repository...")
                    vector_store = VectorStore(dimension=embeddings.shape[1])
                    vector_store.reset()
                    vector_store.add_documents(documents, embeddings)
                    st.session_state.vector_store = vector_store

                    # Show cluster distribution
                    unique, counts = np.unique(clusters, return_counts=True)
                    st.write("**Cluster Distribution:**")
                    cluster_data = {f"Cluster {c}": int(n) for c, n in zip(unique, counts)}
                    st.bar_chart(cluster_data)

                    st.success(f"✅ Processing complete! {len(documents)} segments → {len(set(clusters))} clusters")

                    # ===== AUTO-GENERATE NARRATIVE =====
                    st.divider()
                    st.subheader("📝 Auto-Generated Narrative Report")
                    with st.spinner("Generating narrative from processed data..."):
                        from src.synthesis_engine import retrieve_context, generate_narrative
                        from src.reasoning import HierarchicalReasoner

                        auto_query = "Provide a comprehensive hierarchical summary of all the uploaded data. Identify key patterns, anomalies, and relationships across all sources."

                        results, query_cluster = retrieve_context(
                            auto_query, vector_store,
                            emb_model, dae, vae, clusterer
                        )
                        context_docs = [doc for doc, score in results]

                        # Run reasoning
                        reasoner = HierarchicalReasoner()
                        query_emb = emb_model.generate_embeddings([auto_query])
                        if torch.is_tensor(query_emb):
                            query_emb = query_emb.cpu().numpy()

                        reasoning_result = reasoner.reason(
                            auto_query, query_emb, query_cluster,
                            vector_store, clusters
                        )

                        narrative = generate_narrative(
                            auto_query, context_docs,
                            reasoning_result['validation']
                        )
                        st.session_state.auto_narrative = narrative
                        st.markdown(narrative)
        else:
            st.info("👈 Upload files from the sidebar to begin.")


    # ---------------------------------------------------------
    # TAB 2: Analysis & Report
    # ---------------------------------------------------------
    with tab2:
        query = st.text_area("🔎 Enter your analysis query:", height=100,
                             placeholder="e.g., What are the critical server events and how do they impact the financial outlook?")
        
        if st.button("Generate Report", type="primary") and query:
            if "vector_store" not in st.session_state:
                st.error("Process files first (Tab 1).")
                return

            with st.spinner("Running hierarchical reasoning pipeline..."):
                from src.synthesis_engine import retrieve_context, generate_narrative
                from src.reasoning import HierarchicalReasoner
                from src.evaluator import evaluate_narrative

                emb_model = models["embedding_model"]
                dae = models["dae"]
                vae = models["vae"]
                clusterer = models["clusterer"]

                # Retrieve with cluster awareness
                results, query_cluster = retrieve_context(
                    query, st.session_state.vector_store,
                    emb_model, dae, vae, clusterer
                )
                context_docs = [doc for doc, score in results]

                # Hierarchical Reasoning
                st.write("🧠 **Hierarchical Reasoning Pipeline**")
                reasoner = HierarchicalReasoner()

                query_emb = emb_model.generate_embeddings([query])
                if torch.is_tensor(query_emb):
                    query_emb = query_emb.cpu().numpy()

                reasoning_result = reasoner.reason(
                    query, query_emb, query_cluster,
                    st.session_state.vector_store,
                    st.session_state.get("cluster_labels")
                )

                with st.expander("🔍 View Reasoning Internals"):
                    st.write(f"**Sub-queries:** {reasoning_result['sub_queries']}")
                    st.write(f"**Evidence Sources:** {list(reasoning_result['evidence_map'].keys())}")
                    st.write(f"**Validation Confidence:** {reasoning_result['validation']['confidence']:.2f}")
                    if reasoning_result['validation']['conflicts']:
                        st.warning("Conflicts: " + "; ".join(reasoning_result['validation']['conflicts']))

                # Generate Narrative
                st.write("📝 **Generating Narrative...**")
                narrative = generate_narrative(
                    query, context_docs,
                    reasoning_result['validation']
                )
                st.markdown(narrative)

                # Evaluate
                with st.expander("📊 Evaluation Metrics"):
                    eval_metrics = evaluate_narrative(
                        narrative, context_docs,
                        vae=vae, clusterer=clusterer,
                        user_latent=st.session_state.get("latent_vectors"),
                        user_embeddings=st.session_state.get("denoised")
                    )
                    
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        val = eval_metrics.get("vae_confidence", "N/A")
                        st.metric("VAE Confidence", f"{val}" if isinstance(val, str) else f"{val:.2f}")
                    with metric_cols[1]:
                        val = eval_metrics.get("anomaly_likelihood", "N/A")
                        st.metric("Anomaly Score", f"{val}" if isinstance(val, str) else f"{val:.2f}")
                    with metric_cols[2]:
                        val = eval_metrics.get("evidence_coverage", "N/A")
                        st.metric("Evidence Coverage", f"{val}" if isinstance(val, str) else f"{val:.0%}")
                    with metric_cols[3]:
                        val = eval_metrics.get("faithfulness_score", "N/A")
                        st.metric("Faithfulness", f"{val}" if isinstance(val, str) else f"{val:.2f}")

    # ---------------------------------------------------------
    # TAB 3: Latent Space Explorer
    # ---------------------------------------------------------
    with tab3:
        if "latent_vectors" in st.session_state:
            st.subheader("🔬 Latent Space Visualization")
            
            latent = st.session_state.latent_vectors
            clusters = st.session_state.cluster_labels
            
            # PCA to 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent)
            
            import plotly.express as px
            import pandas as pd
            
            df = pd.DataFrame({
                "PC1": latent_2d[:, 0],
                "PC2": latent_2d[:, 1],
                "Cluster": [f"Cluster {c}" for c in clusters],
            })
            
            fig = px.scatter(
                df, x="PC1", y="PC2", color="Cluster",
                title="VAE Latent Space (PCA Projection)",
                template="plotly_dark",
                width=800, height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}")
            
            # Training Loss Curves
            if models["loaded"]:
                meta = models["metadata"]
                st.subheader("📉 Training Loss Curves")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(meta.get("dae_losses", []), use_container_width=True)
                    st.caption("Denoising Autoencoder Loss")
                with col2:
                    st.line_chart(meta.get("vae_losses", []), use_container_width=True)
                    st.caption("Variational Autoencoder Loss")
        else:
            st.info("Process files first to visualize the latent space.")


if __name__ == "__main__":
    main()
