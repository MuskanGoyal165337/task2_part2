# streamlit_app.py
# Streamlit app for clustering on synthetic datasets

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px

# Optionally import hdbscan
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False

st.set_page_config(layout="wide")
st.title("Clustering Playground — Synthetic Datasets")

# Sidebar: dataset selection and params
st.sidebar.header("Dataset options")
dataset_type = st.sidebar.selectbox("Dataset type", ["make_blobs", "make_moons", "make_circles"])
n_samples = st.sidebar.slider("Number of samples", min_value=500, max_value=2000, value=1000, step=100)
n_features = st.sidebar.selectbox("n_features (for blobs only)", [2,3])  # make_moons is 2D
random_state = 42

if dataset_type == "make_blobs":
    n_centers = st.sidebar.slider("Number of centers (blobs)", 2, 8, 4)
    cluster_std = st.sidebar.slider("Cluster std (spread)", 0.2, 2.0, 0.5)
    X, y_true = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features,
                           cluster_std=cluster_std, random_state=random_state)
elif dataset_type == "make_moons":
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.05)
    X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    n_features = 2
elif dataset_type == "make_circles":
    noise = st.sidebar.slider("Noise", 0.0, 0.5, 0.05)
    factor = st.sidebar.slider("factor (scale)", 0.1, 0.99, 0.5)
    X, y_true = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    n_features = 2

# Preprocessing options
st.sidebar.header("Feature Scaling")
scaler_choice = st.sidebar.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])

if scaler_choice == "StandardScaler":
    scaler = StandardScaler()
elif scaler_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
elif scaler_choice == "RobustScaler":
    scaler = RobustScaler()
else:
    scaler = None

if scaler is not None:
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

# Algorithm selection
st.sidebar.header("Clustering Algorithm")
algo = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN", "Agglomerative"] + (["HDBSCAN"] if HDBSCAN_AVAILABLE else []))

# Dynamic hyperparameters
if algo == "KMeans":
    k = st.sidebar.slider("k (clusters)", 2, 12, 4)
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
elif algo == "DBSCAN":
    eps = st.sidebar.slider("eps", 0.01, 2.0, 0.2)
    min_samples = st.sidebar.slider("min_samples", 3, 20, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif algo == "Agglomerative":
    k = st.sidebar.slider("k (clusters)", 2, 12, 4)
    linkage = st.sidebar.selectbox("linkage", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
elif algo == "HDBSCAN" and HDBSCAN_AVAILABLE:
    min_cluster_size = st.sidebar.slider("min_cluster_size", 5, 50, 10)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

# Fit & predict
with st.spinner("Clustering..."):
    try:
        labels = model.fit_predict(X_scaled)
    except Exception as e:
        st.error(f"Error running model: {e}")
        labels = np.array([-1]*X.shape[0])

# Compute silhouette (only if more than 1 cluster and not all noise)
unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

sil = None
if n_clusters > 1:
    try:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(X_scaled[mask], labels[mask])
    except Exception as e:
        sil = None

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

# Compute metrics only if clusters exist
dbi = None
ch = None

mask = labels != -1 

if n_clusters > 1 and np.unique(labels[mask]).size > 1:
    try:
        sil = silhouette_score(X_scaled[mask], labels[mask])
        # Davies-Bouldin Index (lower is better)
        dbi = davies_bouldin_score(X_scaled[mask], labels[mask])
        
        # Calinski-Harabasz Index (higher is better)
        ch = calinski_harabasz_score(X_scaled[mask], labels[mask])
    except:
        pass
else:
    sil=None

st.sidebar.markdown(f"**Clusters found:** {n_clusters}")
st.sidebar.markdown(f"**Silhouette Score:** {sil if sil is not None else 'N/A'}")
st.sidebar.markdown(f"**Davies-Bouldin Index:** {dbi if dbi is not None else 'N/A'}")
st.sidebar.markdown(f"**Calinski-Harabasz Score:** {ch if ch is not None else 'N/A'}")


# Visualization
st.header("Clustering Visualization")
# If features > 2, reduce to 2D with PCA for plotting
if X_scaled.shape[1] > 2:
    pca = PCA(n_components=2, random_state=random_state)
    X_vis = pca.fit_transform(X_scaled)
else:
    X_vis = X_scaled

df_vis = pd.DataFrame({"x": X_vis[:,0], "y": X_vis[:,1], "label": labels.astype(str)})
fig = px.scatter(df_vis, x='x', y='y', color='label', title=f"{algo} result (n_clusters={n_clusters})",
                 height=600, width=900)
st.plotly_chart(fig, use_container_width=True)

# Show raw data table if user wants
if st.checkbox("Show data table"):
    st.dataframe(df_vis.head(1000))

# st.markdown("## Notes")
# st.markdown("""
# - Use different scalers to see sensitivity.  
# - For DBSCAN, tune eps and min_samples carefully.  
# - HDBSCAN (if available) can find clusters with varying density and label noise (-1).
# """)

# # synthetic_experiments.py
# from sklearn.datasets import make_blobs, make_moons
# import matplotlib.pyplot as plt

# # Example: vary cluster_std for make_blobs
# for std in [0.2, 0.5, 1.0, 2.0]:
#     X, y = make_blobs(n_samples=1000, centers=4, cluster_std=std, random_state=42)
#     plt.figure(figsize=(4,4))
#     plt.scatter(X[:,0], X[:,1], c=y, s=5)
#     plt.title(f"make_blobs cluster_std={std}")
#     plt.show()

# # make_moons: vary noise
# for noise in [0.01, 0.05, 0.1, 0.2]:
#     X, y = make_moons(n_samples=1000, noise=noise, random_state=42)
#     plt.figure(figsize=(4,4))
#     plt.scatter(X[:,0], X[:,1], c=y, s=5)
#     plt.title(f"make_moons noise={noise}")
#     plt.show()
