import streamlit as st
import os
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

# Custom CSS for "Premium" look
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        color: white;
        background-color: #2563EB;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Home", "Upload Data", "Analysis & Report", "Latent Space Explorer", "System Architecture"])

    if app_mode == "Home":
        st.title("🧠 Hierarchical Narrative Synthesis")
        st.markdown("### From Heterogeneous Data to Narrative Insights")
        st.info("Welcome! This system ingests CSVs, PDFs, and Logs to generate structured narrative reports using Generative AI.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1. Upload")
            st.caption("Upload your raw data files.")
        with col2:
            st.markdown("#### 2. Process")
            st.caption("AI extracts and validates info.")
        with col3:
            st.markdown("#### 3. Generate")
            st.caption("Receive a comprehensive report.")

    elif app_mode == "Upload Data":
        st.title("📂 Data Upload & Ingestion")
        uploaded_files = st.file_uploader("Upload CSV, PDF, or Text files", accept_multiple_files=True)
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files.")
            if st.button("Process Files"):
                with st.spinner("Ingesting and embedding data..."):
                    from src.ingestion import ingest_file
                    from src.models import EmbeddingModel, Autoencoder
                    from src.synthesis_engine import VectorStore
                    
                    # Initialize components
                    if 'embedding_model' not in st.session_state:
                        st.session_state.embedding_model = EmbeddingModel()
                    
                    documents = []
                    metadatas = []
                    
                    for file in uploaded_files:
                        content = ingest_file(file)
                        if content and not content.startswith("Error"):
                            documents.append(content)
                            metadatas.append({"filename": file.name})
                        else:
                            st.error(f"Failed to process {file.name}: {content}")

                    if documents:
                        # 1. Generate Raw Embeddings
                        raw_embeddings = st.session_state.embedding_model.generate_embeddings(documents)
                        
                        # 2. Train Autoencoder (Representation Learning)
                        st.info("Training Autoencoder for Representation Learning...")
                        input_dim = raw_embeddings.shape[1] # 384 for all-MiniLM-L6-v2
                        encoding_dim = 128
                        
                        # Initialize Autoencoder
                        if 'autoencoder' not in st.session_state:
                             st.session_state.autoencoder = Autoencoder(input_dim, encoding_dim)
                        
                        # Train on the current batch
                        losses = st.session_state.autoencoder.train_model(raw_embeddings, epochs=50)
                        st.session_state.last_losses = losses
                        
                        # Show training curve
                        st.line_chart(losses)
                        st.caption(f"Autoencoder optimized. Final Loss: {losses[-1]:.4f}")
                        
                        # 3. Get Latent Representations (Compressed Embeddings)
                        latent_embeddings = st.session_state.autoencoder.get_latent_representation(raw_embeddings)
                        
                        # 4. Store in Vector DB (using compressed dimension)
                        if 'vector_store' not in st.session_state:
                            # Re-initialize vector store with the compressed dimension
                            st.session_state.vector_store = VectorStore(dimension=encoding_dim)
                            
                        st.session_state.vector_store.add_documents(documents, latent_embeddings, metadatas)
                        
                        # Store for visualization
                        st.session_state.last_latent_embeddings = latent_embeddings
                        st.session_state.last_filenames = [m['filename'] for m in metadatas]
                        
                        st.success("Processing complete! Data stored in Knowledge Base (Latent Space).")
                    else:
                        st.warning("No valid content extracted from files.")

    elif app_mode == "Analysis & Report":
        st.title("📊 Analysis & Narrative Generation")
        query = st.text_area("Enter your analysis query:", "Summarize the key findings from the uploaded documents.")
        
        if st.button("Generate Report"):
            if 'vector_store' not in st.session_state or 'embedding_model' not in st.session_state or 'autoencoder' not in st.session_state:
                st.error("Please upload and process data first!")
            else:
                with st.spinner("Reasoning and synthesizing..."):
                    from src.synthesis_engine import retrieve_context, generate_narrative
                    
                    # 1. Retrieve Context
                    # returns list of (doc, score)
                    with st.status("🔍 Local Retrieval & Concept Mapping...", expanded=True) as status:
                        st.write("Mapping query to Latent Space...")
                        context_results = retrieve_context(query, st.session_state.vector_store, st.session_state.embedding_model, st.session_state.autoencoder)
                        
                        st.write(f"Found {len(context_results)} relevant evidence chunks.")
                        for i, (doc, score) in enumerate(context_results):
                            # Distance in FAISS: lower is better (L2)
                            confidence = "High" if score < 0.5 else "Medium" if score < 1.0 else "Low"
                            st.write(f"Chunk {i+1}: {confidence} Confidence (Distance: {score:.4f})")
                        
                        status.update(label="Local Analysis Complete!", state="complete", expanded=False)

                    context_docs = [r[0] for r in context_results]
                    
                    # 2. Generate Narrative
                    with st.spinner("Synthesizing Narrative Report..."):
                        narrative = generate_narrative(query, context_docs)
                    
                    # 3. SELF-EVALUATION (Backend Depth)
                    with st.spinner("🕵️‍♂️ Running Consistency & Faithfulness Check..."):
                        from src.evaluator import evaluate_narrative
                        eval_results = evaluate_narrative(narrative, context_docs)
                    
                    st.subheader("Generated Narrative")
                    st.markdown(narrative)
                    
                    # Display Evaluation Metrics
                    with st.container():
                        st.divider()
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            score = eval_results.get('faithfulness_score', 0)
                            st.metric("Faithfulness Score", f"{score*100:.1f}%", 
                                      help="How well the report sticks to the source data (Lower = Potential Hallucination)")
                        with col2:
                            if eval_results.get('is_valid'):
                                st.success("✅ Consistency Check Passed: No major contradictions found.")
                            else:
                                st.warning("⚠️ Accuracy Note: Some potential inconsistencies detected.")
                            
                            with st.expander("View Quality Audit Notes"):
                                st.write(eval_results.get('critique', "No critique provided."))
                                if eval_results.get('issues_found'):
                                    st.write("**Specific Issues:**")
                                    for issue in eval_results.get('issues_found'):
                                        st.write(f"- {issue}")

                    st.success("Analysis complete with multi-stage verification.")
                    
                    # --- VISUALIZATION OF THE "BRAIN" ---
                    with st.expander("🧠 View System Internals (Why this is not just simple RAG)"):
                        st.markdown("### 1. Representation Learning (Autoencoder)")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Raw Embedding Size", "384 dimensions", "Dense Vector")
                        with col2:
                            st.metric("Latent Space Size", "128 dimensions", "Compressed Knowledge")
                        
                        # Show Training History
                        if 'last_losses' in st.session_state:
                             st.area_chart(st.session_state.last_losses)
                             st.caption("Autoencoder Convergence (Loss vs Epochs)")

                        st.markdown("### 2. Hierarchical Retrieval (Evidence)")
                        st.info("The system retrieved these specific chunks from the latent space to form its answer:")
                        for i, (doc, score) in enumerate(context_results):
                            st.markdown(f"**Evidence Chunk {i+1}** (Distance: `{score:.4f}`)")
                            st.text_area(f"Content {i+1}", doc, height=100, key=f"chunk_{i}")

    elif app_mode == "Latent Space Explorer":
        st.title("🌌 Latent Space Explorer")
        st.markdown("### Visualizing the 'Mind' of the AI")
        st.info("This visualization projects the high-dimensional internal knowledge (128d) into 2D space. Points that are closer together are conceptually similar.")
        
        if 'vector_store' not in st.session_state:
            st.warning("Please upload and process data first to visualize the knowledge base.")
        else:
            # We need to extract embeddings from FAISS (a bit tricky, for now we will re-embed the last batch for demo purposes if stored)
            # Or better, we can modify the upload to store them in session state for viz
            if 'last_latent_embeddings' in st.session_state and 'last_filenames' in st.session_state:
                from src.visualize import plot_latent_space
                fig = plot_latent_space(st.session_state.last_latent_embeddings, st.session_state.last_filenames)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data points to visualize (need at least 2).")
            else:
                st.warning("No data found in current session. Go to 'Upload Data' and process some files!")

    elif app_mode == "System Architecture":
        st.title("🧩 System Architecture")
        st.markdown("""
        **1. Data Ingestion Layer**: Handles parsing of CSV, PDF, and LOG files.
        **2. Knowledge Processing**: Generates embeddings and learns representations via Autoencoders.
        **3. Generative Synthesis**: Uses LLMs to draft narrative reports based on retrieved context.
        **4. Evaluation**: Checks consistency and quality of the output.
        """)

if __name__ == "__main__":
    main()
