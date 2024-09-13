# vizualization functions


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.model_selection import learning_curve, validation_curve
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import torch
# from models import AE, VAE, TGEM_Encoder



def generate_latent_space(X_data, encoder, batch_size=128):
    if isinstance(X_data, pd.DataFrame):
        x_index = X_data.index
        X_data = torch.tensor(X_data.to_numpy(), dtype=torch.float32)
    Z = torch.tensor([])
    encoder.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    with torch.inference_mode():
        for i in range(0, len(X_data), batch_size):
            # print(i, len(X_data))
            X_batch = X_data[i:i+batch_size].to(device)
            Z_batch = encoder.transform(X_batch)
            Z_batch = Z_batch.cpu()
            Z = torch.cat((Z, Z_batch), dim=0)
        Z = Z.detach().numpy()
        Z = pd.DataFrame(Z, index=x_index)
    encoder.to('cpu')
    return Z


def generate_umap_embedding(Z_data, **kwargs):
    """Generate UMAP plot of data"""
    reducer = umap.UMAP(**kwargs)
    embedding = reducer.fit_transform(Z_data)
    embedding = pd.DataFrame(embedding, index=Z_data.index)
    return embedding

def generate_pca_embedding(Z_data, **kwargs):
    """Generate PCA plot of data"""
    pca = PCA(**kwargs)
    embedding = pca.fit_transform(Z_data)
    embedding = pd.DataFrame(embedding, index=Z_data.index)
    return embedding

