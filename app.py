import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from model import AutoEncoder
from sklearn.decomposition import PCA

# 页面配置
st.set_page_config(
    page_title="MNIST AutoEncoder Visualization",
    page_icon="🔢",
    layout="wide"
)

# 缓存加载模型和数据
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(latent_dim=16).to(device)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.eval()
    return model, device

@st.cache_data
def load_data():
    df = pd.read_csv("data/mnist_train.csv")
    labels = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0
    return X, labels

@st.cache_data
def get_latent_and_recon(_model, X, _device, batch_size=128):
    X_tensor = torch.tensor(X)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    latents = []
    recons = []
    
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(_device)
            x_hat, z = _model(x_batch)
            latents.append(z.cpu().numpy())
            recons.append(x_hat.cpu().numpy())
    
    return np.vstack(latents), np.vstack(recons)

# 主标题
st.title("🔢 MNIST AutoEncoder Latent Space Visualization")
st.markdown("""
This app visualizes the **2D latent space** of MNIST digits using an AutoEncoder.
Each point represents a digit image compressed into 16 dimensions, then reduced to 2D via PCA.
""")

# 加载模型和数据
with st.spinner("Loading model and data... (first time may take a while)"):
    model, device = load_model()
    X, labels = load_data()

# 提取隐空间
with st.spinner("Extracting latent space representations..."):
    latent, recon = get_latent_and_recon(model, X, device)

# PCA降维
pca = PCA(n_components=2)
z_2d = pca.fit_transform(latent)

# 创建DataFrame用于绘图
df_plot = pd.DataFrame({
    'x': z_2d[:, 0],
    'y': z_2d[:, 1],
    'label': labels.astype(str),
    'idx': range(len(labels))
})

# 侧边栏控制
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("---")

# 选择数字过滤
selected_digits = st.sidebar.multiselect(
    "Select digits to display:",
    options=list(range(10)),
    default=list(range(10)),
    format_func=lambda x: f"Digit {x}"
)

# 样本数量限制（防止浏览器卡顿）
max_samples = st.sidebar.slider(
    "Max samples to display:",
    min_value=1000,
    max_value=len(X),
    value=6000,
    step=1000
)

# 根据选择过滤
mask = df_plot['label'].isin([str(d) for d in selected_digits])
df_filtered = df_plot[mask].sample(min(max_samples, mask.sum())) if mask.sum() > max_samples else df_plot[mask]

# 主界面：散点图
st.subheader("📊 2D Latent Space (PCA Projection)")

fig = px.scatter(
    df_filtered,
    x='x',
    y='y',
    color='label',
    title=f"Latent Space Visualization ({len(df_filtered)} samples)",
    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
    hover_data={'idx': True, 'label': True},
    color_discrete_sequence=px.colors.qualitative.Bold,
    opacity=0.7
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(height=600)

st.plotly_chart(fig, width="stretch")

# 样本查看区域
st.markdown("---")
st.subheader("🔍 Inspect Individual Digits")

col1, col2 = st.columns([1, 2])

with col1:
    idx = st.number_input("Select sample index:", 0, len(labels)-1, 0)
    st.write(f"**Label:** {labels[idx]}")

with col2:
    # 显示原始图像和重建图像
    col_orig, col_recon = st.columns(2)
    
    with col_orig:
        st.write("**Original**")
        img_orig = X[idx].reshape(28, 28)
        st.image(img_orig, width=150, clamp=True)
    
    with col_recon:
        st.write("**Reconstruction**")
        img_recon = recon[idx].reshape(28, 28)
        st.image(img_recon, width=150, clamp=True)

# 显示隐向量
st.markdown("---")
st.subheader("📈 Latent Vector (16 dimensions)")
latent_df = pd.DataFrame(latent[idx].reshape(1, -1), columns=[f"z{i+1}" for i in range(16)])
st.dataframe(latent_df.round(3), width="stretch")

# 页脚
st.markdown("---")
st.markdown("""
<<small>Built with Streamlit | MNIST AutoEncoder with 16-dim latent space</small>
""", unsafe_allow_html=True)