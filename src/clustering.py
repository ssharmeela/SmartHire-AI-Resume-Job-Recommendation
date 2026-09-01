import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

JOBS = "../data/processed/jobs_clean.csv"


def run(n_clusters=5):

    print("=" * 60)
    print("SMART HIRE - JOB CLUSTERING")
    print("=" * 60)

    # Load job dataset
    df = pd.read_csv(JOBS).fillna("")

    print("\nDataset Shape:", df.shape)

    # Use job_text because this is the column in jobs_clean.csv
    X_text = df["job_text"]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    X = vectorizer.fit_transform(X_text)

    print("TF-IDF Shape:", X.shape)

    # K-Means
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    # Silhouette Score
    score = silhouette_score(X, labels)

    print("\n" + "=" * 60)
    print("CLUSTERING RESULTS")
    print("=" * 60)

    print("Number of Clusters:", n_clusters)
    print(f"Silhouette Score: {score:.4f}")

    # Add cluster labels
    df["cluster"] = labels

    print("\nCluster Distribution:")
    print(df["cluster"].value_counts().sort_index())

    # 2D visualization
    svd = TruncatedSVD(
        n_components=2,
        random_state=42
    )

    X_2d = svd.fit_transform(X)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        alpha=0.5
    )

    plt.title("SmartHire - Job Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    plt.show()

    return df