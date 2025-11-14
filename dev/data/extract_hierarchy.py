import polars as pl
import numpy as np
from sklearn.cluster import KMeans
import pickle as pkl
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
    return df.group_by(cluster_col).agg([
        pl.col('lat').mean().alias('lat'),
        pl.col('lon').mean().alias('lon')
    ]).rename({cluster_col: 'cluster'})

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
        .map_elements(lambda x, labels=cluster_labels: -1 if x == -1 else labels[x], return_dtype=pl.Int64)
        .alias(f'cluster_{level}')
    )
    
    centroids = calculate_centroids(df_with_clusters, f'cluster_{level}')
    print(f"Finished level {level}: {len(centroids)} clusters\n")

# Save results
df_with_clusters.write_csv('./hierarchical_clustered_coords.csv')
print("Hierarchical clustered coordinates saved to ./hierarchical_clustered_coords.csv")

# Create hierarchical structure mapping (parent -> children)
print("\nCreating hierarchical structure mapping...")
cluster_columns = [f'cluster_{level}' for level in range(len(points_at_level) + 1)]
hierarchy_df = pl.DataFrame(schema={col: pl.Int64 for col in cluster_columns} | {'num_items': pl.UInt32, 'children': pl.List(pl.Int64)})
for level in range(len(points_at_level), 0, -1):
    parent_col = f'cluster_{level}'
    child_col = f'cluster_{level - 1}'
    higher_cols = [f'cluster_{l}' for l in range(level + 1, len(cluster_columns))]
    
    grouped = (
        df_with_clusters.select(higher_cols + [parent_col, child_col]).group_by(parent_col) # group of all children by parent
        .agg([
            pl.col(child_col).unique().implode().alias('children'),           # unique children into list
            pl.len().alias('num_items')                                       # number of items in the group
        ] + [pl.col(hc).first().alias(hc) for hc in higher_cols])             # get first value for higher level clusters (first is fine since they are the same)
    )
    
    # Match hierarchy_df schema
    missing_cols = [col for col in cluster_columns if col not in grouped.columns]
    grouped = grouped.with_columns([pl.lit(None).alias(col) for col in missing_cols])
    grouped = grouped.select(hierarchy_df.columns)
    
    # vstack to hierarchy_df
    hierarchy_df = pl.concat([hierarchy_df, grouped], how='vertical')

# Level 0 - most granular level
hierarchy_columns = [f'cluster_{level}' for level in range(len(points_at_level) + 1)]
grouped = (
    df_with_clusters.select(hierarchy_columns).group_by('cluster_0')
    .agg([
        pl.lit([]).alias('children'),
        pl.len().alias('num_items')
    ] + [pl.col(hc).first().alias(hc) for hc in hierarchy_columns if hc != 'cluster_0'])
).select(hierarchy_df.columns)
hierarchy_df = pl.concat([hierarchy_df, grouped], how='vertical')

# Level N - least granular level (root) - (None, ... None) -> children - [highest_level_clusters]
highest_level_col = f'cluster_{len(points_at_level)}'
highest_level_col_values = df_with_clusters.select(highest_level_col).unique().to_series().to_list()
grouped = pl.DataFrame({
    'children': [highest_level_col_values],
    'num_items': [len(highest_level_col_values)]
} | {col: [None] for col in cluster_columns}
).select(hierarchy_df.columns).cast(hierarchy_df.schema)
hierarchy_df = pl.concat([hierarchy_df, grouped], how='vertical')

# Serialize children lists to strings for CSV compatibility
hierarchy_df = hierarchy_df.unique().sort(cluster_columns)
hierarchy_df = hierarchy_df.with_columns(pl.col('children').map_elements(lambda x: str(list(x)), return_dtype=str).alias('children'))

hierarchy_df.write_csv('./hierarchical_structure.csv')
print("Hierarchical structure saved to ./hierarchical_structure.csv")

# Make Centroid & Avg Distance to Centroid df for all levels of clustering
print("\nCalculating centroids and mean/std distances...")

def geodesic_distance_km(lat1, lon1, lat2, lon2) -> float:
    """Calculate geodesic distance in kilometers using geopy."""
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers

def distance_stats_to_centroid(struct) -> list[float]:
    """Calculate mean & stdev of distance to centroid from lat/lon lists."""
    centroid_lat = struct['centroid_lat']
    centroid_lon = struct['centroid_lon']
    lats = struct['all_lats']
    lons = struct['all_lons']
    
    if len(lats) == 0:
        return [0.0, 0.0]

    distances = [
        geodesic_distance_km(lat, lon, centroid_lat, centroid_lon)
        for lat, lon in zip(lats, lons)
    ]
    return [float(np.mean(distances)), float(np.std(distances))]


cluster_level_columns = [f'cluster_{level}' for level in range(len(points_at_level) + 1)]
centroid_df = pl.DataFrame(schema={col: pl.Int64 for col in cluster_level_columns} | {'lat': pl.Float64, 'lon': pl.Float64, 'mean_dist': pl.Float64, 'std_dist': pl.Float64})

for cluster_level, cluster_col in enumerate(cluster_level_columns):
    level_centroids = (
        df_with_clusters.group_by(cluster_col).agg([
            pl.col('lat').mean().alias('centroid_lat'),
            pl.col('lon').mean().alias('centroid_lon'),
            pl.col('lat').alias('all_lats'),
            pl.col('lon').alias('all_lons')
        ])
        .with_columns(
            pl.struct(['centroid_lat', 'centroid_lon', 'all_lats', 'all_lons'])
            .map_elements(distance_stats_to_centroid, return_dtype=pl.List(pl.Float64))
            .alias('dist_stats')
        )
        .with_columns([
            pl.col('dist_stats').list.get(0).alias('mean_dist'),
            pl.col('dist_stats').list.get(1).alias('std_dist')
        ])
        .drop(['all_lats', 'all_lons', 'dist_stats'])
        .rename({'centroid_lat': 'lat', 'centroid_lon': 'lon'})
    )

    # Match centroid_df schema
    missing_cols = [col for col in df_with_clusters.columns if col not in level_centroids.columns]
    level_centroids = level_centroids.with_columns([pl.lit(None).alias(col) for col in missing_cols])
    level_centroids = level_centroids.select(centroid_df.columns)

    # vstack to centroid_df
    centroid_df = pl.concat([centroid_df, level_centroids], how='vertical')

centroid_df.write_csv('./hierarchical_cluster_centroids.csv')
print("Cluster centroids and mean/std distances saved to ./hierarchical_cluster_centroids.csv")