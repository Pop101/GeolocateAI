from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import Base, TAGS, GPS
import os
import csv
import traceback

BASE_DIR = './geotag'
output_file = 'geotag_path_to_coord.csv'

def extract_lat_lon(img:Image) -> tuple[float, float]:
    exif_data = img._getexif()
    if not exif_data:
        return None, None
    
    if not exif_data.get(Base.GPSInfo):
        return None, None
    
    lat_ref  = exif_data[Base.GPSInfo][GPS.GPSLatitudeRef]
    lat_data = exif_data[Base.GPSInfo][GPS.GPSLatitude]
    lat      = lat_data[0] + lat_data[1]/60 + lat_data[2]/3600
    lat      = lat * (-1 if lat_ref == 'S' else 1)
    lat = lat._numerator / lat._denominator
    
    lon_ref  = exif_data[Base.GPSInfo][GPS.GPSLongitudeRef]
    lon_data = exif_data[Base.GPSInfo][GPS.GPSLongitude]
    lon      = lon_data[0] + lon_data[1]/60 + lon_data[2]/3600
    lon      = lon * (-1 if lon_ref == 'W' else 1)
    lon = lon._numerator / lon._denominator
    
    return lat, lon
    
with open(output_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File Path', 'Latitude', 'Longitude'])
    
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.jpeg'):
                try:
                    image = Image.open(os.path.join(root, f))
                    lat, lon = extract_lat_lon(image)
                    csv_writer.writerow([os.path.join(root, f), lat, lon])
                    if lat and lon:
                        print("Found lat/lon for {os.path.join(root, f)}")
                except (UnidentifiedImageError, KeyError, TypeError) as e:
                    print(f"Error processing {os.path.join(root, f)}:\t\t{e}")