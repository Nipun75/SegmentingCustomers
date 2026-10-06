import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path

from src.clustering import evaluate_kmeans, train_kmeans
from src.data_preprocessing import load_data, prepare_features


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Mall_Customers.csv"


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
)

st.title("👥 Customer Segmentation Dashboard")
st.caption("K-Means clustering using Annual Income and Spending Score — no PCA required.")

df = load_data(DATA_PATH)
_, X_scaled, scaler = prepare_features(df)

# Dataset overview
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", len(df))
col2.metric("Average Age", f"{df['Age'].mean():.1f}")
col3.metric("Average Income", f"{df['Annual Income (k$)'].mean():.1f}k")
col4.metric("Average Spending", f"{df['Spending Score (1-100)'].mean():.1f}")

with st.expander("View customer data"):
    st.dataframe(df, use_container_width=True)

# EDA
st.header("2. Exploratory Data Analysis")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df["Age"], bins=15, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Age Distribution")

sns.countplot(data=df, x="Gender", ax=axes[0, 1])
axes[0, 1].set_title("Customer Distribution by Gender")

sns.histplot(df["Annual Income (k$)"], bins=15, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Annual Income Distribution")

sns.histplot(df["Spending Score (1-100)"], bins=15, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Spending Score Distribution")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

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

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Model selection
st.header("3. Select the Number of Clusters")

k_values, inertias, silhouette_scores, best_k = evaluate_kmeans(X_scaled)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, inertias, marker="o")
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, silhouette_scores, marker="o")
    ax.set_title("Silhouette Score")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(k_values)
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)

st.success(f"Recommended number of clusters: **{best_k}**")

scores_df = {
    "K": k_values,
    "Silhouette Score": [round(score, 3) for score in silhouette_scores],
}
with st.expander("View silhouette scores"):
    st.dataframe(scores_df, use_container_width=True)

# Final clustering
st.header("4. Customer Segmentation")

kmeans, labels = train_kmeans(X_scaled, best_k)
df["Cluster"] = labels

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="tab10",
    s=90,
    ax=ax,
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
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
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Segment profile
st.header("5. Customer Segment Profile")

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

st.dataframe(profile, use_container_width=True)

# Business insights
st.header("6. Business Insights")

for cluster, row in profile.iterrows():
    income = row["Average_Income"]
    spending = row["Average_Spending"]

    if income >= df["Annual Income (k$)"].median() and spending >= df["Spending Score (1-100)"].median():
        label = "High-value customers"
        recommendation = "Use loyalty programs, premium offers, and personalized campaigns."
    elif income >= df["Annual Income (k$)"].median() and spending < df["Spending Score (1-100)"].median():
        label = "Potential customers"
        recommendation = "Use targeted promotions and personalized offers to increase engagement."
    elif income < df["Annual Income (k$)"].median() and spending >= df["Spending Score (1-100)"].median():
        label = "Budget-conscious active customers"
        recommendation = "Use affordable bundles, discounts, and frequent-purchase incentives."
    else:
        label = "Low-engagement customers"
        recommendation = "Use awareness campaigns, introductory offers, and re-engagement strategies."

    st.subheader(f"Cluster {cluster}: {label}")
    st.write(
        f"Average income: {income:.1f}k | "
        f"Average spending score: {spending:.1f}"
    )
    st.write(recommendation)

# Download
st.header("7. Export Results")

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download clustered customer data",
    data=csv_data,
    file_name="clustered_customers.csv",
    mime="text/csv",
)
