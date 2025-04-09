import arff
import random
import numpy as np
import matplotlib.pyplot as plt
import time

def initialize_centroids(data, k):
    return random.sample(data, k)


def assign_clusters(data, centroids):
    clusters = [[] for j in centroids]
    for point in data:
        distances = [euclidean_distance(point, cent) for cent in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters


def get_new_centroids(clusters, data):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            centroid = []
            for dim in zip(*cluster):
                average = sum(dim) / len(cluster)
                centroid.append(average)
            new_centroids.append(centroid)
        else:
            new_centroids.append(random.choice(data))
    return new_centroids


def euclidean_distance(p1, p2):
    squared_differences = []
    for dim1, dim2 in zip(p1, p2):
        squared_differences.append((dim1 - dim2)**2)
    
    distance = (sum(squared_differences))**0.5
    return distance


def sum_squared_distances(clusters, centroids):
    ssd = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            ssd += euclidean_distance(point, centroids[i]) ** 2
    return ssd


def kmeans(data, k, epsilon, max_iters):
    centroids = initialize_centroids(data, k)
    prev_ssd = float('inf')

    for i in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = get_new_centroids(clusters, data)
        ssd = sum_squared_distances(clusters, new_centroids)
        
        if abs(prev_ssd - ssd) < epsilon:
            break
        prev_ssd = ssd
        centroids = new_centroids

    return centroids, clusters, ssd


def load_arff_data(file):
    with open(file, 'r') as f:
        dataset = arff.load(f)
    
    data = []
    for row in dataset['data']:
        data.append([float(value) for value in row])
    
    return data


data = load_arff_data("iris.arff")


k_input = int(input("Choose k value: "))
epsilon_input = float(input("Choose epsilon value: "))
max_iters_input = int(input("Choose max iterations: "))

kmeans(data, k_input, epsilon_input, max_iters_input)


centroids, clusters, ssd = kmeans(data, k_input, epsilon_input, max_iters_input)

print("\nFinal Centroids:")
for centroid in centroids:
    print(centroid)

print(f"\nFinal Sum of Squared Distances: {ssd}")


def evaluate_runtime_vs_k(data, k_max, epsilon, max_iters):
    k_values = range(2, k_max + 1)
    runtimes = []
    
    for k in k_values:
        start = time.time()
        kmeans(data, k, epsilon, max_iters)
        end = time.time()
        runtimes.append(end - start)
    
    plt.plot(k_values, runtimes, marker='o', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Number of Clusters')
    plt.show()

def evaluate_ssd_vs_k(data, k_max, epsilon, max_iters):
    k_values = range(2, k_max + 1)
    ssds = []
    
    for k in k_values:
        _, _, ssd = kmeans(data, k, epsilon, max_iters)
        ssds.append(ssd)
    
    plt.plot(k_values, ssds, marker='o', color='r')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Sum of Squared Distances (SSD)')
    plt.title('SSD vs Number of Clusters (Elbow Method)')
    plt.show()

evaluate_runtime_vs_k(data, k_input, epsilon_input, max_iters_input)
evaluate_ssd_vs_k(data, k_input, epsilon_input, max_iters_input)

