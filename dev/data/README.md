# Generating the Data

Table of Contents:

# Downloading the Data

## 2016 Placing

This data comes from MediaEval's 2016 placing challenge,
a competition with the same goals as this repository!

Unfortunately, it is a subset of a subset (but it is the largest we can get):

First, download the data [here](https://www.kaggle.com/datasets/habedi/large-dataset-of-geotagged-images/data) to any folder.

Then, symlink the data to `2016_placing/2016_placing`

Next, unpack the messagepack format. [This script](./2016_placing/messagepack_unpack.py) (`2016_placing/messagepack_unpack.py`) should help, but **requires changes to paths**

Finally, run [2016_placing/move_data.py](./2016_placing/move_data.py) to create the needed CSV

## Google Landmark

This data comes from google landmark, a dataset containing images of a very large amount of wikipedia landmarks.

First, download the dataset to any folder using the [script provided by google](https://github.com/cvdfoundation/google-landmark).

*Sometimes, the script does not delete the .tar files it extracts. Save your disk space, delete them yourself*

Next, symlink the data to `google_landmark/google_landmark`

The data does not provide lat/lon data, instead, it provides a wikipedia landmark ID. To create a mapping from landmark ID to lat/lon, run [google_landmark/landmark_to_coords.py](./google_landmark/landmark_to_coords.py). **This will take ~36 hours**

Now, run [google_landmark/landmark_join.py](./google_landmark/landmark_join.py) to create the final CSV file.

## Flickr Geotag

This data advertises 14 million geotagged photos. However, it seems the researchers screwed up the upload, and we can really only get around 5 million. Still, this is rather large dataset!

First, download the .tar file [here](https://qualinet.github.io/databases/image/world_wide_scale_geotagged_image_dataset_for_automatic_image_annotation_and_reverse_geotagging/)

Next, extract it until you have three folders: "1", "2" and "spacial-index" along with some CSV and PDF files.

Then, symlink this folder to `ww_geotag/geotag`

Now, just run the script [ww_geotag/exif_to_csv.py](./ww_geotag/exif_to_csv.py) to create the needed CSV.

# Combining and Clustering

Now that we have all three datasets indexed, we need to do something with them!

## Combining the Data

If you ran all the above successfully, this should be as easy as running [./combine_data.py](./combine_data.py)! This will create `./path_to_coords.csv`

## Clustering the Data

Now, we cluster the data using HDBScan. Run the script [./train_hdbscan.py](./train_hdbscan.py) to use the combined data to train the model. **This will also take ~36 hours**.

This should have created `./clustered_coords.csv` and `./hdbscan_model.pkl`

To split these into hierarchies that reduce the size of the total classification task, run [./extract_hierarchy.py](./extract_hierarchy.py). This creates `./hierarchical_cluster_centroids.csv` and `./hierarchical_clustered_coords.csv` and should take 30 minutes.

**We now have all data ready for model training!**

# Data Schema

## Combined Data
After combination, the data is stored in `./path_to_coords.csv` with the following columns:

- path: The path to the image, from the location of this readme
- lat: The latitude of the image
- lon: The longitude of the image

Example rows:
| path                                               | lat       | lon       |
|----------------------------------------------------|-----------|-----------|
| ./2016_placing/./2016_placing/ba/08/6468190417.jpg | 41.906    | 12.455    |
| ./2016_placing/./2016_placing/5a/c7/8582299397.jpg | 48.211072 | 16.36736  |
| ./2016_placing/./2016_placing/81/90/8694156503.jpg | 43.942876 | 12.774091 |

## Clustered Coord
After clustering, the data is stored in `./clustered_coords.csv` with the following columns:

- path: The path to the image, from the location of this readme
- lat: The latitude of the image
- lon: The longitude of the image
- cluster: The cluster ID assigned by HDBScan. -1 indicates noise points. All clusters are unique integers are complete (form a range from 0 to N-1)

Example rows:
| path                                               | lat       | lon       | cluster |
|----------------------------------------------------|-----------|-----------|---------|
| ./2016_placing/./2016_placing/ba/08/6468190417.jpg | 41.906    | 12.455    | 116305  |
| ./2016_placing/./2016_placing/5a/c7/8582299397.jpg | 48.211072 | 16.36736  | -1      |
| ./2016_placing/./2016_placing/81/90/8694156503.jpg | 43.942876 | 12.774091 | 104163  |

## Hierarchical Clustered Coord
After hierarchical clustering, the data is stored in `./hierarchical_clustered_coords.csv` with the following columns:

- path: The path to the image, from the location of this readme
- lat: The latitude of the image
- lon: The longitude of the image
- cluster_0: The cluster ID assigned at the first hierarchy level. Identical to the cluster ID from `./clustered_coords.csv`. -1 indicates noise points.
- cluster_1: The cluster ID assigned at the second hierarchy level. -1 indicates noise points.
- cluster_N: The cluster ID assigned at the Nth hierarchy level. -1 indicates noise points.

Example rows:
| path                                               | lat       | lon       | cluster_0 | cluster_1 | cluster_2 |
|----------------------------------------------------|-----------|-----------|-----------|-----------|-----------|
| ./2016_placing/./2016_placing/ba/08/6468190417.jpg | 41.906    | 12.455    | 116305    | 3949      | 190       |
| ./2016_placing/./2016_placing/5a/c7/8582299397.jpg | 48.211072 | 16.36736  | -1        | -1        | -1        |
| ./2016_placing/./2016_placing/81/90/8694156503.jpg | 43.942876 | 12.774091 | 104163    | 4058      | 190       |

## Hierarchical Cluster Centroids
The centroids of each cluster at each hierarchy level are stored in `./hierarchical_cluster_centroids.csv` with the following columns:

- cluster_0: The cluster ID at the first hierarchy level.
- cluster_N: The cluster ID at the Nth hierarchy level.
- lat: The latitude of the cluster centroid
- lon: The longitude of the cluster centroid
- avg_dist: The average distance (in kilometers) of all points in the cluster to the cluster's centroid

Note that the CSV is sparce, there is at most one cluster ID per row.

Example rows:
| cluster_0 | cluster_1 | cluster_2 | lat                 | lon                | avg_dist           |
|-----------|-----------|-----------|---------------------|--------------------|--------------------|
| -1        |           |           | 37.98282543892728   | -4.766215391611326 | 4942.40434241566   |
|           | 6059      |           | 40.71250321867015   | 14.513979240818365 | 7.3515186607786    |
|           |           | 350       | -13.368196074535884 | 41.955632998337464 | 245.01873738933548 |

## Hierarchical Structure
The hierarchical structure of the underlying data is stored in `./hierarchical_structure.csv` with the following columns:

- cluster_0: The cluster ID at the first hierarchy level.
- cluster_N: The cluster ID at the Nth hierarchy level.
- num_points: The number of data points in this cluster
- children: A list of all child cluster IDs (at the next hierarchy level) contained in this cluster. If cluster_0 is not null, this will be empty (recall, cluster 0 is the most granular level)

Note that the CSV is right-triangular, clusters will always be filled from left to right. If cluster_0 is set, you can assume all other cluster_N columns are set. If cluster_1 is set, you can assume cluster_0 is not set.

Example rows:
| cluster_0 | cluster_1 | cluster_2 | num_items | children               |
|-----------|-----------|-----------|-----------|------------------------|
|           |           | 350       | 2562034   | [345, 21]              |
|           | 6059      | 10        | 123       | [350, 351, 352, 353]   |
| 1         | 5         | 9         | 1523      | []                     |
