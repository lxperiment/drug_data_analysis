import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = '1'

# Set the number of samples and features
n_samples = 1000
n_features = 4

# Create an empty array to store the data
data = np.empty((n_samples, n_features))

# Generate random data for each feature
for i in range(n_features):
    data[:, i] = np.random.normal(size=n_samples)

# Create 5 clusters with different densities and centroids
cluster1 = data[:200, :] + np.random.normal(size=(200, n_features), scale=0.5)
cluster2 = data[200:400, :] + np.random.normal(size=(200, n_features), scale=1) + np.array([5, 5, 5, 5])
cluster3 = data[400:600, :] + np.random.normal(size=(200, n_features), scale=1.5) + np.array([-5, -5, -5, -5])
cluster4 = data[600:800, :] + np.random.normal(size=(200, n_features), scale=2) + np.array([5, -5, 5, -5])
cluster5 = data[800:, :] + np.random.normal(size=(200, n_features), scale=2.5) + np.array([-5, 5, -5, 5])

# Combine the clusters into one dataset
X = np.concatenate((cluster1, cluster2, cluster3, cluster4, cluster5))

# Plot the data
plt.scatter(X[:, 0], X[:, 1])


df = pd.DataFrame(X, columns=["feature_1", "feature_2", "feature_3", "feature_4"])
cluster_id = np.concatenate((np.zeros(200), np.ones(200), np.full(200, 2), np.full(200, 3), np.full(200, 4)))
df["cluster_id"] = cluster_id
df

from sklearn.cluster import AffinityPropagation

# Fit the model:
af = AffinityPropagation(preference=-563, random_state=0).fit(X)
cluster_centers_indices = af.cluster_centers_indices_
af_labels = af.labels_
n_clusters_ = len(cluster_centers_indices)

# Print number of clusters:
print(n_clusters_)

import matplotlib.pyplot as plt
from itertools import cycle

plt.close("all")
plt.figure(1)
plt.clf()

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
for k, col in zip(range(n_clusters_), colors):
    class_members = af_labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + ".")
    plt.plot(
      cluster_center[0],
      cluster_center[1],
       "o",
     markerfacecolor=col,
     markeredgecolor="k",
     markersize=14,
)
for  x in X[class_members]:
    plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()