import polars as pl
import os

BASE_DIR = './2016_placing'

data = pl.read_csv(os.path.join(BASE_DIR, 'data.csv'))

# rewrite "path" column to include ./2016_placing/
data = data.with_columns(
    (pl.lit(BASE_DIR) +
    os.path.sep +
    pl.col('path')).alias('path')
)

data.write_csv('./2016_placing_path_to_coord.csv')