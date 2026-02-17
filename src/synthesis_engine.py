import faiss
import numpy as np
import pickle
import os

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
                # Check if loaded index matches requested dimension
                if self.index.d != self.dimension:
                    # Reset if dimension doesn't match
                    self.index = faiss.IndexFlatL2(self.dimension)
            except:
                pass # Fallback to new index
        
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "rb") as f:
                    self.metadatas = pickle.load(f)
            except:
                pass
# ... (add_documents and query remain mostly the same, ensuring dimension checks if needed)

    def add_documents(self, documents, embeddings, metadatas=None):
        """
        Add documents and their embeddings to the store.
        """
        # Convert embeddings to numpy array
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        elif hasattr(embeddings, 'tolist'): # Check if it's a tensor/list
             embeddings = np.array(embeddings)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Handle metadata and document storage
        # We need to store the actual text content to retrieve it later
        current_metadata_len = len(self.metadatas)
        
        new_metadatas = []
        for i, doc in enumerate(documents):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            meta["content"] = doc # Store content in metadata
            new_metadatas.append(meta)
            
        self.metadatas.extend(new_metadatas)

        # Persist to disk
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadatas, f)

    def query(self, query_embedding, n_results=5):
        """
        Query the store using a query embedding.
        """
        if hasattr(query_embedding, 'numpy'):
            query_embedding = query_embedding.numpy()
        elif hasattr(query_embedding, 'tolist'):
            query_embedding = np.array(query_embedding)
            
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        D, I = self.index.search(query_embedding, n_results)
        
        retrieved_results = []
        
        # I[0] contains the indices, D[0] contains the distances
        for dist, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.metadatas):
                content = self.metadatas[idx].get("content", "")
                retrieved_results.append((content, float(dist)))
        
        return retrieved_results

def retrieve_context(query, vector_store, embedding_model, autoencoder=None):
    """
    High-level function to retrieve context for a query.
    Returns a list of (document, score) tuples.
    """
    # 1. Generate Raw Embedding for the query
    query_emb = embedding_model.generate_embeddings([query])
    
    # 2. Compress using Autoencoder (if provided)
    if autoencoder:
        query_emb = autoencoder.get_latent_representation(query_emb)
    
    # 3. Search in vector store
    results = vector_store.query(query_emb, n_results=3)
    
    return results

from src.llm_wrapper import LLMProvider

def generate_narrative(query, context_docs):
    """
    Generate a hierarchical narrative using the configured LLM provider.
    """
    context_text = "\n---\n".join(context_docs)
    
    system_prompt = """
    You are an Expert Hierarchical Narrative Analyst.
    Your goal is to synthesize heterogeneous data into a structured report.
    Format your answer in Markdown with sections.
    """
    
    user_prompt = f"""
    QUERY: {query}
    
    CONTEXT DATA:
    {context_text}
    
    INSTRUCTIONS:
    1. Answer the query based ONLY on the context.
    2. Cite the specific evidence (PDF, CSV, Log).
    3. Structure the answer hierarchically:
       - Executive Summary (Top Level)
       - Key Insights (Middle Level)
       - Supporting Evidence (Bottom Level)
    """
    
    # Initialize provider (can be toggled in env)
    provider_name = os.getenv("LLM_PROVIDER", "huggingface") 
    llm = LLMProvider(provider=provider_name)
    
    return llm.generate(user_prompt, system_prompt)
