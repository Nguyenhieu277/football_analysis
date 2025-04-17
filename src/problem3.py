import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("data/results.csv")

percent_cols = ['Won%', 'Save%', 'CS%', 'PSave%']
for col in percent_cols:
    df[col] = (
        df[col]
        .replace('N/a', np.nan)
        .str.replace(r'[^\d.]', '', regex=True)
        .pipe(pd.to_numeric, errors='coerce')
        / 100
    )

df['GA90'] = pd.to_numeric(df['GA90'], errors='coerce')
stats_columns = df.columns[4:].tolist()
for stat in stats_columns:
    df[stat] = pd.to_numeric(df[stat], errors='coerce')

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(df[stats_columns])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_imputed)

inertia = []
silhouette_scores = []

for k in range(2, 71):
    kmeans = KMeans(n_clusters = k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

optimal_k = 7
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title(f'Player Clusters (K={optimal_k})')
plt.colorbar(scatter)
plt.show()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(2, 71), inertia, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(range(2, 71), silhouette_scores, 'rx-')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()