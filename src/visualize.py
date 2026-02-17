import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def plot_latent_space(embeddings, filenames):
    """
    Reduce dimensionality of embeddings to 2D and plot them using Plotly.
    """
    # Convert to numpy if tensor
    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()
    
    # Needs at least 2 samples for PCA
    if len(embeddings) < 2:
        return None
        
    # Use PCA to reduce to 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(embeddings)
    
    df = pd.DataFrame(components, columns=['x', 'y'])
    df['filename'] = filenames
    
    fig = px.scatter(df, x='x', y='y', color='filename', 
                     title='Latent Space Visualization (Compressed Knowledge)',
                     hover_data=['filename'],
                     labels={'x': 'Latent Dimension 1', 'y': 'Latent Dimension 2'})
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1E3A8A')
    )
    
    return fig
