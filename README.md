# Customer Segmentation Using K-Means

This project segments mall customers using **K-Means clustering** in a standard VS Code/Python project structure.

## Dataset

`Mall_Customers.csv` contains 200 customers with:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Project Structure

```
SegmentingCustomers/
├── Mall_Customers.csv
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── src/
    ├── __init__.py
    ├── data_preprocessing.py
    ├── clustering.py
    └── visualization.py
```

## Workflow

1. Load and inspect the dataset
2. Exploratory Data Analysis (EDA)
3. Feature selection
4. Feature scaling
5. Elbow Method + Silhouette Score
6. K-Means clustering
7. Cluster visualization
8. Customer segment profiling
9. Save clustered customer data

### Features used for clustering

- Annual Income (k$)
- Spending Score (1-100)

CustomerID is not used as a model feature because it is only an identifier. Age and Gender are retained for analysis and segment profiling.

**PCA is not used** because this dataset has only two primary clustering features, so dimensionality reduction is unnecessary.

## Run in VS Code

Open the repository folder in VS Code.

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

The program displays the EDA graphs, Elbow Method, Silhouette Score, final customer clusters, and segment profile. It also creates `clustered_customers.csv`.
