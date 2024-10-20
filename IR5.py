import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Plot the dendrogram
sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Perform Agglomerative Clustering
n_clusters = 5
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
df['Cluster'] = hc.fit_predict(X)

# Plot the clusters
for cluster in range(n_clusters):
    plt.scatter(df[df['Cluster'] == cluster]['Annual Income (k$)'],
                df[df['Cluster'] == cluster]['Spending Score (1-100)'],
                label=f'Cluster {cluster}')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
