import matplotlib.pyplot as plt
import seaborn as sns


def _finish(fig, show):
    fig.tight_layout()
    if show:
        plt.show()
        plt.close(fig)
    return fig


def plot_eda(df, show=True):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    sns.histplot(df["Age"], bins=15, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Age Distribution")

    sns.countplot(data=df, x="Gender", ax=axes[0, 1])
    axes[0, 1].set_title("Customer Distribution by Gender")

    sns.histplot(df["Annual Income (k$)"], bins=15, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Annual Income Distribution")

    sns.histplot(df["Spending Score (1-100)"], bins=15, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title("Spending Score Distribution")

    return _finish(fig, show)


def plot_relationships(df, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        s=70,
        ax=axes[0],
    )
    axes[0].set_title("Annual Income vs Spending Score")

    sns.scatterplot(
        data=df,
        x="Age",
        y="Spending Score (1-100)",
        s=70,
        ax=axes[1],
    )
    axes[1].set_title("Age vs Spending Score")

    return _finish(fig, show)


def plot_correlation(df, show=True):
    columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")

    return _finish(fig, show)


def plot_clustered_data(df, kmeans, scaler, show=True):
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster",
        palette="tab10",
        s=90,
        ax=ax,
    )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        c="black",
        label="Centroids",
    )
    ax.set_title("Customer Segments")
    ax.legend(title="Cluster")

    return _finish(fig, show)
