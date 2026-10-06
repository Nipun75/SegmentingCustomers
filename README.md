# Customer Segmentation Using K-Means

This project segments mall customers using **K-Means clustering**.

## Dataset

The project uses `Mall_Customers.csv`, containing 200 customers with:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Methodology

1. Load and inspect the data
2. Exploratory Data Analysis (EDA) with graphs
3. Select clustering features: Annual Income and Spending Score
4. Standardize the selected features
5. Use the Elbow Method and Silhouette Score to evaluate the number of clusters
6. Train K-Means
7. Visualize customer clusters
8. Profile each segment using age, income, and spending score
9. Translate segments into business recommendations

**PCA is intentionally not used** because this dataset has only a small number of relevant features and dimensionality reduction is unnecessary for the core objective.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```
jupyter notebook notebooks/customer_segmentation.ipynb
```

## Project goal

Identify groups of customers with similar income and spending behavior so that businesses can create more targeted marketing and customer-retention strategies.
