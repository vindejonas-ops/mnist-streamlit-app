import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import AutoEncoder

def train_model():
    # 加载数据
    print("Loading data...")
    df = pd.read_csv("data/mnist_train.csv")
    
    labels = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0
    
    # 转为Tensor
    X_tensor = torch.tensor(X)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = AutoEncoder(latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    epochs = 500
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, z = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "model_weights.pth")
    print("Model saved to model_weights.pth")
    
    # 保存标签供后续使用
    np.save("labels.npy", labels)
    print("Labels saved to labels.npy")

if __name__ == "__main__":
    train_model()