import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# ============================================================
# MODEL 1 — Semantic Embedding Model (384-d)
# ============================================================
class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        """
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_list):
        """
        Generate embeddings for a list of texts.
        Returns a numpy array of embeddings.
        """
        embeddings = self.model.encode(text_list, convert_to_tensor=True)
        return embeddings


# ============================================================
# MODEL 2 — Denoising Autoencoder (DAE)
# ============================================================
class DenoisingAutoencoder(nn.Module):
    """
    Feedforward encoder-decoder network.
    Training objective: Reconstruct clean embeddings from corrupted inputs.
    Loss: Mean squared reconstruction error.
    """
    def __init__(self, input_dim=384, encoding_dim=256, noise_factor=0.3):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
        )

    def add_noise(self, x):
        """Add Gaussian noise to input for denoising objective."""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def denoise(self, x):
        """Get denoised representation (full forward pass on clean input)."""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self.forward(x).numpy()

    def get_encoded(self, x):
        """Get encoder output (intermediate representation)."""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self.encoder(x).numpy()

    def train_model(self, data, epochs=50, lr=0.001):
        """
        Train the DAE: corrupt inputs with noise, reconstruct clean originals.
        Returns list of per-epoch losses.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        else:
            data = data.clone().detach().float()

        self.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            noisy_data = self.add_noise(data)
            outputs = self.forward(noisy_data)
            loss = criterion(outputs, data)  # Reconstruct clean from noisy
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        return losses


# ============================================================
# MODEL 3 — Variational Autoencoder (VAE)
# ============================================================
class VariationalAutoencoder(nn.Module):
    """
    Encoder → latent mean & variance → reparameterization → decoder.
    Loss: Reconstruction (MSE) + KL Divergence.
    """
    def __init__(self, input_dim=384, hidden_dim=256, latent_dim=64):
        super(VariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        """Returns mu and log_var for the latent distribution."""
        h = self.encoder_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z from q(z|x) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x):
        """Get latent representation (mu) for input data."""
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu.numpy()

    def get_reconstruction_probability(self, x):
        """
        Compute reconstruction probability for anomaly detection.
        Higher loss = more anomalous.
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            recon, mu, logvar = self.forward(x)
            recon_loss = nn.functional.mse_loss(recon, x, reduction='none').mean(dim=1)
        return recon_loss.numpy()

    @staticmethod
    def vae_loss(recon_x, x, mu, logvar):
        """VAE loss = Reconstruction + KL Divergence."""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def train_model(self, data, epochs=50, lr=0.001):
        """
        Train the VAE. Returns list of per-epoch losses.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        else:
            data = data.clone().detach().float()

        self.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon, mu, logvar = self.forward(data)
            loss = self.vae_loss(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.item() / len(data))

        return losses


# ============================================================
# MODEL 4 — Latent Clustering (K-Means)
# ============================================================
class LatentClusterer:
    """
    Wraps sklearn K-Means for unsupervised structure discovery
    on VAE latent vectors.
    """
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.centroids = None
        self.labels = None

    def fit(self, latent_vectors):
        """Fit K-Means on VAE latent vectors."""
        if torch.is_tensor(latent_vectors):
            latent_vectors = latent_vectors.numpy()
        self.model.fit(latent_vectors)
        self.centroids = self.model.cluster_centers_
        self.labels = self.model.labels_
        return self.labels

    def predict(self, latent_vectors):
        """Assign cluster labels to new latent vectors."""
        if torch.is_tensor(latent_vectors):
            latent_vectors = latent_vectors.numpy()
        return self.model.predict(latent_vectors)

    def get_distance_to_centroid(self, latent_vectors):
        """Compute distance of each vector to its assigned centroid."""
        if torch.is_tensor(latent_vectors):
            latent_vectors = latent_vectors.numpy()
        labels = self.model.predict(latent_vectors)
        distances = []
        for i, label in enumerate(labels):
            dist = np.linalg.norm(latent_vectors[i] - self.centroids[label])
            distances.append(dist)
        return np.array(distances)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
