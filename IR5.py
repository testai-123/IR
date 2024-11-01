import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

plt.axhline(y=7, color='r', linestyle='--')  # Adjust y value for better visualization

plt.title('Dendrogram for Iris Dataset with Cut Line')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distances')
plt.show()

# Perform Agglomerative Clustering
n_clusters = 3  # As there are 3 species of Iris
hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels = hc.fit_predict(X)

# Create a DataFrame for visualization
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['Cluster'] = labels

# Plot the clusters using the first two features (sepal length and sepal width)
plt.figure(figsize=(10, 5))
for cluster in range(n_clusters):
    plt.scatter(iris_df[iris_df['Cluster'] == cluster]['sepal length (cm)'],
                iris_df[iris_df['Cluster'] == cluster]['sepal width (cm)'],
                label=f'Cluster {cluster}')

plt.title('Agglomerative Clustering on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
