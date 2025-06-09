import os
import csv
import msgpack
from PIL import Image
import piexif

# Global variables
TARGET_DIR = "/mnt/e/data/flickr/shards"  # Change this to your source directory
BASE_PATH = "/mnt/e/data/flickr/"         # Change this to your output directory

# Ensure base directory exists
os.makedirs(BASE_PATH, exist_ok=True)

# Create CSV file
with open(os.path.join(BASE_PATH, 'data.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['path', 'latitude', 'longitude'])
    
    # Process all files in the target directory
    for filename in os.listdir(TARGET_DIR):
        # Filter for messagepack files
        if not filename.endswith('.msg'):
            continue
            
        file_path = os.path.join(TARGET_DIR, filename)
        
        # Parse messagepack file
        with open(file_path, 'rb') as f:
            # Use Unpacker to handle potential multiple objects
            unpacker = msgpack.Unpacker(f, raw=False)
            
            try:
                for record in unpacker:
                    # Skip non-dict records or missing required fields
                    if not isinstance(record, dict) or not all(k in record for k in ['id', 'latitude', 'longitude', 'image']):
                        continue
                    
                    image_id = record['id']
                    lat = record['latitude'] 
                    lon = record['longitude']
                    
                    # Create directory structure and save image
                    image_path = os.path.join(BASE_PATH, image_id)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    
                    # Save image
                    with open(image_path, 'wb') as img_file:
                        img_file.write(record['image'])
                    
                    # Open image with PIL
                    img = Image.open(image_path)
                    exif = img.getexif()
                    
                    # Convert coordinates to EXIF DMS format
                    def to_dms(coord):
                        deg = int(abs(coord))
                        min_val = int((abs(coord) - deg) * 60)
                        sec = ((abs(coord) - deg) * 60 - min_val) * 60
                        return (deg, 1), (min_val, 1), (int(sec * 100), 100)
                    
                    # Set GPS EXIF data
                    lat_ref = 'N' if lat >= 0 else 'S'
                    lon_ref = 'E' if lon >= 0 else 'W'
                    
                    # Create new EXIF data
                    exif_data = img.getexif()

                    # Create a new GPS IFD
                    gps_ifd = dict(
                        {
                            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
                            piexif.GPSIFD.GPSLatitude: to_dms(lat),
                            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
                            piexif.GPSIFD.GPSLongitude: to_dms(lon)
                        }
                    )

                    # Create exif bytes
                    exif_dict = {"GPS": gps_ifd}
                    exif_bytes = piexif.dump(exif_dict)

                    # Save with EXIF
                    img.save(image_path, exif=exif_bytes)
                    
                    # Write to CSV
                    writer.writerow([image_id, lat, lon])
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
                raise e

print(f"Processing complete. Data saved to {BASE_PATH}")