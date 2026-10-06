# Customer Segmentation using K-Means

🚀 **[Live Demo](https://segmentingcustomersgit-r6hb3wwbk9krffturcbcgg.streamlit.app/)**

Customer segmentation project using K-Means clustering...
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
├── app.py
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
9. Business insights and recommendations
10. Export clustered customer data

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

### Run the Python version

```bash
python main.py
```

This runs the complete ML workflow locally and creates `clustered_customers.csv`.

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard provides:

- Dataset overview and metrics
- EDA visualizations
- Elbow Method
- Silhouette Score
- Automatic selection of the best K using the highest silhouette score
- Customer cluster visualization
- Segment profiles
- Business recommendations
- Downloadable clustered customer data

## Deploy the dashboard

The repository is structured so `app.py` can be deployed as the Streamlit app entry point. Connect the GitHub repository to Streamlit Community Cloud and select:

- Repository: `Nipun75/SegmentingCustomers`
- Branch: `main`
- Main file: `app.py`

The dependencies are provided in `requirements.txt`.

## Notes

- The model uses standardized Annual Income and Spending Score features.
- K-Means uses a fixed `random_state=42` for reproducible results.
- PCA is intentionally not included.
