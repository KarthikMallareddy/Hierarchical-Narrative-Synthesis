"""
train.py — Offline Training Pipeline

Orchestrates the full Phase 1 training:
  1. Load/download multi-domain training corpus
  2. Generate 384-d semantic embeddings
  3. Train Denoising Autoencoder on embeddings
  4. Train VAE on DAE-denoised embeddings
  5. Fit K-Means on VAE latent vectors
  6. Save all model artifacts to trained_models/
"""

import os
import sys
import torch
import numpy as np
import pickle
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import EmbeddingModel, DenoisingAutoencoder, VariationalAutoencoder, LatentClusterer
from src.training.data_loader import build_training_corpus


ARTIFACTS_DIR = "trained_models"


def train_pipeline(
    # Synthetic dataset configuration (reduced - now supplements only)
    wiki_n=200,
    arxiv_n=100,
    tabular_n=100,
    web_n=100,
    # External dataset configuration (PRIMARY per user request)
    use_external=True,
    external_ratio=0.8,
    social_n=400,
    ext_arxiv_n=300,
    uci_n=250,
    pubmed_n=300,
    # Model configuration
    dae_epochs=30,
    vae_epochs=30,
    n_clusters=5,
    latent_dim=64,
):
    """
    EXTERNAL-FOCUSED training pipeline per user request.
    
    Args:
        wiki_n, arxiv_n, tabular_n, web_n: Reduced synthetic (supplement only)
        use_external: Default TRUE - user wants external datasets
        external_ratio: Default 0.8 (80% external, 20% synthetic for balance)
        social_n: Kaggle Social Media Engagement samples
        ext_arxiv_n: arXiv ML Papers via API samples
        uci_n: UCI ML Repository dataset samples  
        pubmed_n: PubMed Central medical literature samples
        dae_epochs, vae_epochs: Training epochs for autoencoders
        n_clusters: Number of K-means clusters
        latent_dim: VAE latent space dimensionality
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # =========================================================
    # STEP 1: Build Training Corpus (EXTERNAL-FOCUSED per user request)
    # =========================================================
    print("=" * 60)
    print("PHASE 1 — EXTERNAL-FOCUSED OFFLINE TRAINING")
    if use_external:
        print("🌐 EXTERNAL DATA MODE: Real-world datasets prioritized")
        print(f"   Target: {(1-external_ratio)*100:.0f}% synthetic, {external_ratio*100:.0f}% external")
        print(f"   📱 Social Media: {social_n} samples | 📚 arXiv: {ext_arxiv_n} samples")
        print(f"   🏛️  UCI ML Repo: {uci_n} samples | 🏥 PubMed: {pubmed_n} samples")
    else:
        print("📝 SYNTHETIC FALLBACK MODE: External data loading failed")
    print("=" * 60)
    
    corpus = build_training_corpus(
        wiki_n=wiki_n,
        arxiv_n=arxiv_n,
        tabular_n=tabular_n,
        web_n=web_n,
        use_external=use_external,
        external_ratio=external_ratio,
        social_n=social_n,
        ext_arxiv_n=ext_arxiv_n,
        uci_n=uci_n,
        pubmed_n=pubmed_n
    )

    # =========================================================
    # STEP 2: Generate Semantic Embeddings (Model 1)
    # =========================================================
    print("\n🔹 MODEL 1 — Generating Semantic Embeddings...")
    embedding_model = EmbeddingModel()
    
    # Process in batches to avoid memory issues
    batch_size = 128
    all_embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        emb = embedding_model.generate_embeddings(batch)
        if torch.is_tensor(emb):
            emb = emb.cpu().numpy()
        all_embeddings.append(emb)
        print(f"    Embedded batch {i//batch_size + 1}/{(len(corpus)-1)//batch_size + 1}")
    
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"    ✅ Embeddings shape: {embeddings.shape}")
    
    # Save embeddings
    np.save(os.path.join(ARTIFACTS_DIR, "training_embeddings.npy"), embeddings)

    # =========================================================
    # STEP 3: Train Denoising Autoencoder (Model 2)
    # =========================================================
    print(f"\n🔹 MODEL 2 — Training Denoising Autoencoder ({dae_epochs} epochs)...")
    dae = DenoisingAutoencoder(input_dim=embeddings.shape[1])
    dae_losses = dae.train_model(embeddings, epochs=dae_epochs)
    print(f"    Loss: {dae_losses[0]:.4f} → {dae_losses[-1]:.4f}")
    
    # Save DAE
    torch.save(dae.state_dict(), os.path.join(ARTIFACTS_DIR, "dae_model.pth"))
    
    # Get denoised embeddings
    denoised = dae.denoise(embeddings)
    print(f"    ✅ Denoised embeddings shape: {denoised.shape}")

    # =========================================================
    # STEP 4: Train Variational Autoencoder (Model 3)
    # =========================================================
    print(f"\n🔹 MODEL 3 — Training Variational Autoencoder ({vae_epochs} epochs, latent_dim={latent_dim})...")
    vae = VariationalAutoencoder(input_dim=denoised.shape[1], latent_dim=latent_dim)
    vae_losses = vae.train_model(denoised, epochs=vae_epochs)
    print(f"    Loss: {vae_losses[0]:.4f} → {vae_losses[-1]:.4f}")
    
    # Save VAE
    torch.save(vae.state_dict(), os.path.join(ARTIFACTS_DIR, "vae_model.pth"))
    
    # Get latent vectors
    latent_vectors = vae.get_latent(denoised)
    np.save(os.path.join(ARTIFACTS_DIR, "latent_vectors.npy"), latent_vectors)
    print(f"    ✅ Latent vectors shape: {latent_vectors.shape}")

    # =========================================================
    # STEP 5: Fit K-Means Clustering (Model 4)
    # =========================================================
    print(f"\n🔹 MODEL 4 — Fitting K-Means Clustering (k={n_clusters})...")
    clusterer = LatentClusterer(n_clusters=n_clusters)
    labels = clusterer.fit(latent_vectors)
    
    # Count per cluster
    unique, counts = np.unique(labels, return_counts=True)
    for c, n in zip(unique, counts):
        print(f"    Cluster {c}: {n} samples")
    
    # Save clusterer
    clusterer.save(os.path.join(ARTIFACTS_DIR, "clusterer.pkl"))

    # =========================================================
    # SAVE TRAINING METADATA
    # =========================================================
    metadata = {
        "corpus_size": len(corpus),
        "embedding_dim": embeddings.shape[1],
        "latent_dim": latent_dim,
        "n_clusters": n_clusters,
        "use_external": use_external,
        "external_ratio": external_ratio if use_external else 0.0,
        "synthetic_counts": {
            "wiki": wiki_n,
            "arxiv": arxiv_n, 
            "tabular": tabular_n,
            "web": web_n
        },
        "external_counts": {
            "social_media": social_n,
            "arxiv_papers": ext_arxiv_n,
            "uci_ml_repo": uci_n,
            "pubmed_central": pubmed_n
        } if use_external else {},
        "data_sources": [
            "Kaggle: Social Media Engagement",
            "arXiv: ML Papers (API)",
            "UCI ML Repository",
            "PubMed Central"
        ] if use_external else ["Synthetic only"],
        "dae_final_loss": dae_losses[-1],
        "vae_final_loss": vae_losses[-1],
        "dae_losses": dae_losses,
        "vae_losses": vae_losses,
        "training_time_seconds": time.time() - start_time,
    }
    with open(os.path.join(ARTIFACTS_DIR, "training_metadata.pkl"), 'wb') as f:
        pickle.dump(metadata, f)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"✅ TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"   Artifacts saved to: {ARTIFACTS_DIR}/")
    print(f"   - dae_model.pth")
    print(f"   - vae_model.pth")
    print(f"   - clusterer.pkl")
    print(f"   - training_embeddings.npy")
    print(f"   - latent_vectors.npy")
    print(f"   - training_metadata.pkl")
    print(f"{'=' * 60}")

    return {
        "dae": dae,
        "vae": vae,
        "clusterer": clusterer,
        "metadata": metadata,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hierarchical Narrative Synthesis Models - EXTERNAL DATA FOCUSED")
    
    # External dataset parameters (PRIMARY)
    parser.add_argument("--external", action="store_true", default=True, 
                       help="Use external datasets (default: True)")
    parser.add_argument("--no-external", action="store_true", 
                       help="Force synthetic-only mode")
    parser.add_argument("--ratio", type=float, default=0.8, 
                       help="External data ratio (default: 0.8 = 80% external)")
    
    # External dataset sizes  
    parser.add_argument("--social", type=int, default=400, 
                       help="Social media engagement samples")
    parser.add_argument("--arxiv-api", type=int, default=300, 
                       help="arXiv papers via API")
    parser.add_argument("--uci", type=int, default=250, 
                       help="UCI ML Repository samples")
    parser.add_argument("--pubmed", type=int, default=300, 
                       help="PubMed Central samples")
    
    # Synthetic dataset sizes (supplements only, reduced defaults)
    parser.add_argument("--wiki", type=int, default=200, 
                       help="Wikipedia-style segments (supplement)")
    parser.add_argument("--arxiv", type=int, default=100, 
                       help="Synthetic arXiv-style segments (supplement)")
    parser.add_argument("--tabular", type=int, default=100, 
                       help="Tabular data segments (supplement)")
    parser.add_argument("--web", type=int, default=100, 
                       help="Web text segments (supplement)")
    
    args = parser.parse_args()
    
    # Handle external/synthetic mode selection
    if args.no_external:
        args.external = False
        print("🚨 SYNTHETIC-ONLY MODE FORCED")
        print("   All data will be synthetically generated")
    
    if args.external:
        print("🌐 EXTERNAL DATA MODE SELECTED (Recommended)")
        print("   Training on real-world datasets:")
        print("   📱 Kaggle Social Media Engagement")
        print("   📚 arXiv ML Papers (API - no download needed)")  
        print("   🏛️  UCI ML Repository datasets")
        print("   🏥 PubMed Central medical literature")
        print()
        print("   Required setup (optional datasets will fallback):")
        print("   1. pip install datasets (for arXiv API)")
        print("   2. Kaggle setup for social media data (optional)")
        print("   3. UCI/PubMed downloads (optional)")
        print()
        
        response = input("Continue with external data training? (Y/n): ")
        if response.lower() == 'n':
            print("Switching to synthetic-only mode...")
            args.external = False
    
    # Run training
    results = train_pipeline(
        # Synthetic parameters (supplements)
        wiki_n=args.wiki,
        arxiv_n=args.arxiv,
        tabular_n=args.tabular,
        web_n=args.web,
        # External parameters (primary)
        use_external=args.external,
        external_ratio=args.ratio,
        social_n=args.social,
        ext_arxiv_n=args.arxiv_api,
        uci_n=args.uci,
        pubmed_n=args.pubmed
    )
    
    print("\n🎯 QUICK START COMMANDS:")
    print("   # External data (recommended, default):")
    print("   python train.py")
    print()
    print("   # Synthetic only (fast, fully reproducible):")
    print("   python train.py --no-external")
    print()
    print("   # Custom external dataset sizes:")
    print("   python train.py --social 600 --arxiv-api 400 --uci 300 --pubmed 500")
    print()
    print("   # Mixed with custom ratio:")
    print("   python train.py --ratio 0.9 --social 800")
