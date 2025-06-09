import hdbscan
import polars as pl
import pickle as pkl

# Step 1: load data
df = pl.read_csv('./path_to_coords.csv') # n_rows=300 for testing
unique_coords_df = df.select(['lat', 'lon']).unique()
unique_coords = unique_coords_df.select(['lat', 'lon']).to_numpy()

# Step 2: Train HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    prediction_data=True,
    #metric='haversine',      # cannot use with prims_kdtree
    algorithm='prims_kdtree', # must use due to high memory cost
    approx_min_span_tree=True
)
clusterer.fit(unique_coords)

# Save the trained hdbscan
with open('./hdbscan_model.pkl', 'wb') as f:
    pkl.dump(clusterer, f)
    
# Step 3: Label and Save
unique_coords_df = unique_coords_df.with_columns(pl.Series(name='cluster', values=clusterer.labels_))
df_with_clusters = df.join(
    unique_coords_df,
    on=['lat', 'lon'],
    how='left'
)
df_with_clusters.write_csv('./clustered_coords.csv')