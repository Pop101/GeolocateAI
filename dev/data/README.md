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
