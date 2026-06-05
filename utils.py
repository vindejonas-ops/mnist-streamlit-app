import numpy as np
import torch
from sklearn.decomposition import PCA
from model import AutoEncoder

@torch.no_grad()
def extract_latent_space(model_path="model_weights.pth", data_path="data/mnist_train.csv"):
    """提取隐空间表示，用于可视化"""
    import pandas as pd
    
    # 加载数据
    df = pd.read_csv(data_path)
    labels = np.load("labels.npy") if os.path.exists("labels.npy") else df["label"].values
    
    X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0
    X_tensor = torch.tensor(X)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(latent_dim=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 提取隐向量
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128)
    
    latents = []
    recons = []
    
    for (x,) in loader:
        x = x.to(device)
        x_hat, z = model(x)
        latents.append(z.cpu().numpy())
        recons.append(x_hat.cpu().numpy())
    
    latent = np.vstack(latents)
    recon = np.vstack(recons)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latent)
    
    return {
        'latent': latent,
        'latent_2d': z_2d,
        'recon': recon,
        'original': X,
        'labels': labels,
        'pca': pca
    }