# CUSTOMER SEGMENTATION USING K-MEANS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================
os.makedirs("outputs", exist_ok=True)
# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv("data/Mall_Customers.csv")
print("\nDataset Loaded Successfully!")
print(df.head())
# ==========================================
# DATASET INFORMATION
# ==========================================
print("\nDataset Shape:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
# ==========================================
# DATA CLEANING
# ==========================================
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.to_csv("outputs/cleaned_data.csv", index=False)
print("\nData Cleaning Completed!")
# ==========================================
# EXPLORATORY DATA ANALYSIS
# ==========================================
# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.savefig("outputs/age_distribution.png")
plt.show()
# Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Annual Income (k$)"], kde=True)
plt.title("Annual Income Distribution")
plt.savefig("outputs/income_distribution.png")
plt.show()
# Spending Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Spending Score (1-100)"], kde=True)
plt.title("Spending Score Distribution")
plt.savefig("outputs/spending_distribution.png")
plt.show()
# Correlation Heatmap
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()
# ==========================================
# FEATURE SELECTION
# ==========================================
X = df[["Annual Income (k$)",
        "Spending Score (1-100)"]]
# ==========================================
# FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ==========================================
# ELBOW METHOD
# ==========================================
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.savefig("outputs/elbow_method.png")
plt.show()
# ==========================================
# K-MEANS MODEL
# ==========================================
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
clusters = kmeans.fit_predict(X_scaled)
df["Cluster"] = clusters
# ==========================================
# SAVE SEGMENTED DATA
# ==========================================
df.to_csv(
    "outputs/segmented_customers.csv",
    index=False
)
# ==========================================
# CLUSTER VISUALIZATION
# ==========================================
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)
plt.title("Customer Segmentation")
plt.savefig("outputs/customer_segments.png")
plt.show()
# ==========================================
# CLUSTER SUMMARY
# ==========================================
print("\nCluster Summary:")
summary = df.groupby("Cluster").agg({
    "Age":"mean",
    "Annual Income (k$)":"mean",
    "Spending Score (1-100)":"mean"
})
print(summary)
summary.to_csv(
    "outputs/cluster_summary.csv"
)
print("\nProject Completed Successfully!")
print("\nFiles saved inside outputs folder.")