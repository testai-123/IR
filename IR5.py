import pandas as pd 
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

# Load 
iris = load_iris()
x = iris.data
true_labels = iris.target

# Get Number of Clusters
plt.figure(figsize=(10,5))
dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
plt. axhline(y=7,color='r', linestyle='--')

plt.title("Dendrogram for Iris Dataset")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

# Train and Predict
n = 3 
hc = AgglomerativeClustering(n_clusters=n,linkage='ward')
labels = hc.fit_predict(x)

# Accuracy
ari = adjusted_rand_score(true_labels, labels)
print(f'Adjusted Rand Index: {ari:.3f}')

df = pd.DataFrame(data=x,columns=iris.feature_names)
df['Cluster'] = labels

# Plot 
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='Cluster', palette='Set1')
plt.title('Agglomerative Clustering on Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
