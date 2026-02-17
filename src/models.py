import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

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

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid() # Use Sigmoid if input is normalized to [0, 1], else remove or use appropriate activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, data, epochs=50, lr=0.001):
        """
        Train the autoencoder on the provided data (embeddings).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Convert data to tensor if needed
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        else:
            data = data.clone().detach().float() # Fix: Detach from inference graph
            
        self.train()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses

    def get_latent_representation(self, x):
        """
        Get the compressed latent representation (encoder output).
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            encoded = self.encoder(x)
        return encoded.numpy()

