# join landmark_coordinats landmark id to train.csv landmark id

import polars as pl
import os

BASE_DIR = './google_landmark'

# Read the CSV files
train_df = pl.read_csv(os.path.join(BASE_DIR, 'train.csv'))
lmrk_coord = pl.read_csv('./landmark_coordinates.csv')
lmrk_coord.columns = ['landmark_id', 'lat', 'lon']

# join
df = train_df.join(lmrk_coord, on='landmark_id', how='left')

# add path
df = df.with_columns(
    (
        pl.lit(BASE_DIR) + 
        os.path.sep + 
        pl.col('id').str.slice(0, 1) + 
        os.path.sep + 
        pl.col('id').str.slice(1, 1) + 
        os.path.sep + 
        pl.col('id').str.slice(2, 1) + 
        os.path.sep + 
        pl.col('id') + 
        '.jpg'
    ).alias('path')
)

# save to csv
df.write_csv('./landmark_path_to_coord.csv')