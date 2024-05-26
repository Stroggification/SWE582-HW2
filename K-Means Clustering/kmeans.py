import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('data.npy')
labels = np.load('label.npy')

#Plot the data using a scatter plot. Assign different colors to different labels.
def plot_data(data, labels, title):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_data = data[labels == label]
        plt.scatter(label_data[:, 0], label_data[:, 1], label=f'Label {label}')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

plot_data(data, labels, 'Original Data with Labels')

#  k-means algorithm 
def kmeans(data, k, max_iters=100):
    # Randomly initialize the centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    
    for _ in range(max_iters):
        # Assign clusters
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        
        # Update centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return clusters, centroids


k = len(np.unique(labels))
clusters, centroids = kmeans(data, k)

# Plot the final clustering 
plot_data(data, clusters, 'Final Clustering Assignments')