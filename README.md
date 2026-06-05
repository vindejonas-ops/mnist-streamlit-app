# 🔢 MNIST AutoEncoder Streamlit App

Visualize the latent space of MNIST digits using an AutoEncoder.

## Features

- Interactive 2D PCA visualization of 16-dim latent space
- Filter by digit class
- Inspect original vs reconstructed images
- View latent vector values

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train model (first time)
python train.py

# Run app
streamlit run app.py