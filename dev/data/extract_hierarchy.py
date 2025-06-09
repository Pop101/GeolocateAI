import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl
import csv
from geopy.distance import geodesic

# Load the HDBSCAN model and data
with open('./hdbscan_model.pkl', 'rb') as f:
    clusterer = pkl.load(f)

# Load clustered coords
df_with_clusters = pl.read_csv('./clustered_coords.csv')
print("\nOriginal cluster distribution:")
print(df_with_clusters.group_by('cluster').agg(pl.len()).sort('len', descending=True))

# Get number of points for each level
n_points = len(np.unique(clusterer.labels_))
points_at_level = [
    int(n_points ** 0.75),
    int(n_points ** 0.5),
]
print("\nNumber of points at each level:")
print(f"Level 0: {n_points}")
for i, x in enumerate(points_at_level, start=1):
    print(f"Level {i}: {x} (factor: {n_points / x if i == 1 else n_points / points_at_level[i-2]:.2f})")

def calculate_centroids(df, cluster_col):
    """Calculate centroids and sizes for clusters in a dataframe."""
    centroids = []
    cluster_ids = []
    
    for cluster in df[cluster_col].unique():
        cluster_df = df.filter(pl.col(cluster_col) == cluster)
        centroid = cluster_df.select(pl.col(['lat', 'lon'])).mean().row(0)
        centroids.append(centroid)
        cluster_ids.append(cluster)
    
    return pl.DataFrame({
        'lat': np.array(centroids)[:, 0],
        'lon': np.array(centroids)[:, 1],
        'cluster': cluster_ids
    })

# Create hierarchical levels using iterative K-means
print("\nCreating hierarchical levels...")
df_with_clusters = df_with_clusters.rename({'cluster': 'cluster_0'})

# Initial centroids from level 0
centroids = calculate_centroids(df_with_clusters, 'cluster_0')
print(f"Finished level 0: {len(centroids)} clusters\n")

# For each level, run k-means on the centroids to create a hierarchical clustering
for level, n_clusters in enumerate(points_at_level, start=1):
    print(f"Processing level {level-1} -> {level} (target: {n_clusters} clusters)")
    
    # First k-means on current centroids w/o noise points
    centroids_no_noise = centroids.filter(pl.col('cluster') != -1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(
        centroids_no_noise
        .select(pl.col(['lat', 'lon']))
    )
    
    # Apply mapping to create new level
    df_with_clusters = df_with_clusters.with_columns(
        pl.col(f'cluster_{level-1}')
        .map_elements(lambda x: -1 if x == -1 else cluster_labels[x], return_dtype=pl.Int64)
        .alias(f'cluster_{level}')
    )
    
    centroids = calculate_centroids(df_with_clusters, f'cluster_{level}')
    print(f"Finished level {level}: {len(centroids)} clusters\n")

# Save results
df_with_clusters.write_csv('./hierarchical_clustered_coords.csv')

# Make Centroid & Avg Distance to Centroid df for all levels of clustering
with open('./hierarchical_cluster_centroids.csv', 'w', newline='') as cluster_file:
    cluster_writer = csv.writer(cluster_file)
    cluster_writer.writerow(
        [f'cluster_{level}' for level in range(len(points_at_level) + 1)] + 
        ['lat', 'lon', 'avg_dist']
    )
    
    for cluster_level in range(len(points_at_level) + 1):
        for cluster in df_with_clusters[f'cluster_{cluster_level}'].unique():
            cluster_df = df_with_clusters.filter(pl.col(f'cluster_{cluster_level}') == cluster)
            centroid = cluster_df.select(pl.col('lat').mean(), pl.col('lon').mean()).row(0)
        
            distances = []
            for row in cluster_df.iter_rows(named=True):
                point = (row['lat'], row['lon'])
                centroid_point = (centroid[0], centroid[1])
                distance = geodesic(point, centroid_point).kilometers
                distances.append(distance)
            
            avg_dist = np.mean(distances)
            cluster_writer.writerow(
                [cluster if (cluster_level == i) else None for i in range(len(points_at_level) + 1)] +
                [centroid[0], centroid[1], avg_dist]
            )