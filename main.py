from pathlib import Path

import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, validate_data, prepare_features
from src.clustering import evaluate_kmeans, train_kmeans
from src.visualization import (
    plot_eda,
    plot_relationships,
    plot_correlation,
    plot_clustered_data,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Mall_Customers.csv"

def main():
    # 1. Load and inspect data
    df = load_data(DATA_PATH)
    validate_data(df)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary statistics:")
    print(df.describe())

    # 2. EDA
    plot_eda(df)
    plot_relationships(df)
    plot_correlation(df)

    # 3. Feature selection + scaling
    _, X_scaled, scaler = prepare_features(df)

    # 4. Find optimal number of clusters
    k_values, inertias, silhouette_scores, best_k = evaluate_kmeans(X_scaled)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()

    print("\nSilhouette scores:")
    for k, score in zip(k_values, silhouette_scores):
        print(f"k={k}: {score:.3f}")

    print(f"\nSelected number of clusters: {best_k}")

    # 5. Train final K-Means model
    kmeans, labels = train_kmeans(X_scaled, best_k)
    df["Cluster"] = labels

    # 6. Visualize clusters
    plot_clustered_data(df, kmeans, scaler)

    # 7. Segment profiling
    profile = (
        df.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Average_Age=("Age", "mean"),
            Average_Income=("Annual Income (k$)", "mean"),
            Average_Spending=("Spending Score (1-100)", "mean"),
        )
        .round(2)
    )

    print("\nCustomer Segment Profile:")
    print(profile)

    # 8. Save clustered data
    output_path = BASE_DIR / "clustered_customers.csv"
    df.to_csv(output_path, index=False)
    print(f"\nClustered dataset saved to: {output_path}")

if __name__ == "__main__":
    main()
