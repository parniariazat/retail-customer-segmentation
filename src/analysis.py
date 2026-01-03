import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import seaborn as sns



df = pd.read_excel("data/online_retail_dataset.xlsx")  # Load dataset

df.head()
df.info()

df = df[df['CustomerID'].notnull()]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']


reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)


rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(rfm.head())
print(rfm.describe())


rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
rfm_features.loc[:, 'Monetary'] = np.log1p(rfm_features['Monetary'])

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

print(rfm_scaled[:5])


inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.savefig("figures/elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()


silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rfm_scaled)
    score = silhouette_score(rfm_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by k')
plt.tight_layout()
plt.savefig("figures/silhouette_score.png", dpi=300, bbox_inches="tight")
plt.show()



kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print(rfm.head())

KMeans(n_clusters=4, random_state=42, n_init=10)

cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'})

print(cluster_profile)


cluster_profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'CustomerID': 'count'
}).rename(columns={'CustomerID': 'CustomerCount'})

print(cluster_profile)

cluster_names = {
    3: 'High-Value Loyal Customers',
    2: 'Growth Potential Customers',
    1: 'At-Risk Customers',
    0: 'Low-Value / Occasional Customers'
}

rfm['Segment'] = rfm['Cluster'].map(cluster_names)

print(rfm.head())


plt.figure(figsize=(8,6))
sns.scatterplot(
    data=rfm,
    x='Frequency',
    y='Monetary',
    hue='Segment',
    alpha=0.7
)
plt.title('Customer Segments based on Frequency and Monetary Value')
plt.tight_layout()
plt.savefig("figures/segments_scatter.png", dpi=300, bbox_inches="tight")
plt.show()


plt.figure(figsize=(8,6))
sns.boxplot(
    data=rfm,
    x='Segment',
    y='Recency'
)
plt.title('Recency Distribution by Customer Segment')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("figures/recency_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()


