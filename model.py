import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        
        # 编码器：784 -> 256 -> 64 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 解码器：16 -> 64 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()  # 输出像素值0-1
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z  # 返回重建结果和隐向量