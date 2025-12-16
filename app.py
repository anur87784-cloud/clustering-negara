import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Clustering Negara Berdasarkan Sosial Ekonomi")

# Load dataset
df = pd.read_csv("Country-data.csv")
st.subheader("Dataset")
st.dataframe(df.head())

# Pilih fitur
features = ['income', 'child_mort', 'life_expec', 'gdpp', 'total_fer']
X = df[features]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Pilih jumlah cluster
k = st.slider("Pilih jumlah cluster", 2, 6, 3)

# KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]

# Plot
st.subheader("Peta Cluster")
fig, ax = plt.subplots()
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster',
    data=df,
    palette='viridis',
    ax=ax
)
st.pyplot(fig)

# Ringkasan cluster
st.subheader("Rata-rata Tiap Cluster")
st.dataframe(df.groupby('Cluster')[features].mean())
