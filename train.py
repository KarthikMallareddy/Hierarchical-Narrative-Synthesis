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
    wiki_n=500,
    arxiv_n=300,
    tabular_n=200,
    web_n=300,
    dae_epochs=30,
    vae_epochs=30,
    n_clusters=5,
    latent_dim=64,
):
    """
    Full offline training pipeline.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    start_time = time.time()

    # =========================================================
    # STEP 1: Build Training Corpus
    # =========================================================
    print("=" * 60)
    print("PHASE 1 — OFFLINE TRAINING")
    print("=" * 60)
    
    corpus = build_training_corpus(
        wiki_n=wiki_n,
        arxiv_n=arxiv_n,
        tabular_n=tabular_n,
        web_n=web_n
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
    train_pipeline()
