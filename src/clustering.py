import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def evaluate_kmeans(X_scaled, k_range=range(2, 11)):
    inertias = []
    silhouette_scores = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    best_k = list(k_range)[int(np.argmax(silhouette_scores))]

    return list(k_range), inertias, silhouette_scores, best_k

def train_kmeans(X_scaled, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    return model, labels
