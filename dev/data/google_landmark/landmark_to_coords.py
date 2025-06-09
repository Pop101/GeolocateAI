from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import requests
import time
import re

from lxml import html

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
BASE_DELAY = 0.3

LAST_REQUEST_TIME = 0
def get(url: str, max_retries=5) -> requests.Response:
    global LAST_REQUEST_TIME
    current_time = time.monotonic()
    delay = BASE_DELAY - (current_time - LAST_REQUEST_TIME)
    if delay > 0: time.sleep(delay)
    LAST_REQUEST_TIME = time.monotonic()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
    
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    
    response = session.get(url, headers=headers, allow_redirects=True, timeout=30)
    response.raise_for_status()
    return response

def parse_coordinates(coord_string):
    """
    Parse coordinates in the format "41° 18′ 40″ N, 72° 55′ 43″ W"
    including support for decimal seconds like "22° 16′ 22.08″ N"
    """
    # Updated pattern to handle decimal seconds
    pattern = r'(\d+)°\s+(\d+)′\s+(\d+(?:\.\d+)?)″\s+([NSEW]),\s+(\d+)°\s+(\d+)′\s+(\d+(?:\.\d+)?)″\s+([NSEW])'
    match = re.search(pattern, coord_string)
    
    if not match:
        raise ValueError("Coordinate string does not match expected format")
    
    # Extract values from the matched groups
    lat_deg = int(match.group(1))
    lat_min = int(match.group(2))
    lat_sec = float(match.group(3))  # Changed to float to handle decimals
    lat_dir = match.group(4)
    
    lon_deg = int(match.group(5))
    lon_min = int(match.group(6))
    lon_sec = float(match.group(7))  # Changed to float to handle decimals
    lon_dir = match.group(8)
    
    # Convert to decimal degrees
    lat_decimal = lat_deg + (lat_min / 60) + (lat_sec / 3600)
    lon_decimal = lon_deg + (lon_min / 60) + (lon_sec / 3600)
    
    # Adjust sign based on direction
    if lat_dir == 'S':
        lat_decimal = -lat_decimal
    if lon_dir == 'W':
        lon_decimal = -lon_decimal
    
    return (lat_decimal, lon_decimal)

import polars as pl

df = pl.read_csv('google_landmark/train_label_to_category.csv')
print(df)
print(df.columns)

# Prepare result dataframe
result_data = {"id": [], "lat": [], "lon": []}

try:
    wip_df = pl.read_csv('./landmark_coordinates.csv')
    result_data['id'] = wip_df['id'].to_list()
    result_data['lat'] = wip_df['lat'].to_list()
    result_data['lon'] = wip_df['lon'].to_list()
    
    start_index = max(wip_df['id'].to_list())
    df = df.filter(pl.col('landmark_id') > start_index)
except:
    print("Cannot resume, starting from scratch.")
    import time
    time.sleep(5)


# Process each row
for row in df.iter_rows(named=True):
    landmark_id = row["landmark_id"]
    wiki_url = row["category"]
    
    try:
        print(f"Processing ID {landmark_id}, URL: {wiki_url}")
        
        # Get the Wikipedia page
        response = get(wiki_url)
        tree = html.fromstring(response.content)
        
        # Find geohack link
        geohack_elements = tree.xpath("//a[contains(@href,'geohack')]")
        
        if geohack_elements:
            geohack_url = geohack_elements[0].text_content().strip()
            lat, lon = parse_coordinates(geohack_url)
            
            # Add to result
            result_data["id"].append(landmark_id)
            result_data["lat"].append(lat)
            result_data["lon"].append(lon)
            
            print(f"  Found coordinates: Lat={lat}, Lon={lon}")
            print(f"  parse object: {geohack_url}")
            if lat == float('nan'): raise ValueError("Invalid coordinates")
        else:
            print(f"  No geohack link found for ID {landmark_id}")
            result_data["id"].append(landmark_id)
            result_data["lat"].append(float('nan'))
            result_data["lon"].append(float('nan'))
            
    except Exception as e:
        print(f"  Error processing ID {landmark_id}: {e}")
        if "geohack_url" in locals():
            print(f"  parse object: {geohack_url}")
        result_data["id"].append(landmark_id)
        result_data["lat"].append(float('nan'))
        result_data["lon"].append(float('nan'))

    # Create and save the result dataframe
    result_df = pl.DataFrame(result_data)

    # Save to CSV
    result_df.write_csv("landmark_coordinates.csv")
    print("Results saved to landmark_coordinates.csv")