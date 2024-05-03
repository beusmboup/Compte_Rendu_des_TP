import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Charger le jeu de données Iris
iris = load_iris()
X = iris.data  # Les caractéristiques

# Créer et entraîner le modèle K-means avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Obtenir les centres de clusters
centroids = kmeans.cluster_centers_

# Afficher les clusters
plt.figure(figsize=(8, 6))

plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.8, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering of Iris Dataset')
plt.legend()
plt.show()
