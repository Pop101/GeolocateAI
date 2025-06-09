import polars as pl

# Combine into single df


# Prep 2016 placing
pl2016_df = pl.read_csv('./2016_placing/2016_placing_path_to_coord.csv')
pl2016_df = pl2016_df.select(['path', 'latitude', 'longitude'])
pl2016_df = pl2016_df.with_columns(
    (pl.lit('./2016_placing/') + pl.col('path')).alias('path')
)

# Prep geotag
gtag_df = pl.read_csv('./ww_geotag/geotag_path_to_coord.csv')
gtag_df = gtag_df.select(['File Path',  'Latitude', 'Longitude'])
gtag_df = gtag_df.with_columns(
    (pl.lit('./ww_geotag/') + pl.col('File Path')).alias('File Path')
)

# Prep Google Landmark
lmrk_df = pl.read_csv('./google_landmark/landmark_path_to_coord.csv')
lmrk_df = lmrk_df.select(['path', 'lat', 'lon'])
lmrk_df = lmrk_df.with_columns(
    (pl.lit('./google_landmark/') + pl.col('path')).alias('path')
)

# Combine into single df
pl2016_df.columns = lmrk_df.columns
gtag_df.columns = lmrk_df.columns

df = pl.concat([pl2016_df, lmrk_df, gtag_df], how="vertical")
df = df.filter(
    pl.col('lat').is_not_null() & \
    pl.col('lon').is_not_null() & \
    pl.col('lat').is_finite() & \
    pl.col('lon').is_finite()
)
df.write_csv('./path_to_coords.csv')