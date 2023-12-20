import laspy
import numpy as np
from scipy.interpolate import griddata
from pyproj import Transformer, CRS
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import DBSCAN
import logging
import json
import os
import time
import rasterio
import sys
import pygmt
import utm
from rasterio.warp import calculate_default_transform, reproject, Resampling
from joblib import Parallel, delayed
import joblib.externals.loky

zone_to_process = 'zone_3.1'

zone_files_list = f"./download_lists/{zone_to_process}_files.txt"
zone_laz_dir = f"/media/dimitri/SSD2/Split_NRW/{zone_to_process}/"
zone_DEM_dir = f"/media/dimitri/SSD2/Split_NRW/{zone_to_process}_DEM/"

log_dir = "./logs/"
index_file = "./assets/index.json"
results_dir = "./results/"

backup_DEM_file = './DEM_data/urn_eop_DLR_CDEM10_Copernicus_DSM_04_N50_00_E006_00_V8239-2020_1__DEM1__coverage_20231204210410.tif'

# Constants for laz processing

SSFACTOR = 2 # Subsampling factor for points cloud

lastReturnNichtBoden = 20
brueckenpunkte = 17
unclassified = 1

class_ok = [brueckenpunkte, lastReturnNichtBoden, unclassified]

dst_crs = 'EPSG:4326'

# Initialise logging
# Set at DEBUG if necessary or else INFO

logging.basicConfig(filename=f'{log_dir}/data_processing_{zone_to_process}.log', 
                    level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

def load_file_to_list(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Strip newline characters from each line
        lines = [line.strip() for line in lines]
    return lines

# Create a dictionary for quick lookup from the JSON data (much quicker than recursively looking up in the JSON)

def create_lookup_dict(json_data):

    lookup_dict = {}
    for dataset in json_data.get('datasets', []):
        for file in dataset.get('files', []):
            lookup_dict[file['name']] = (file['size'], file['timestamp'])
    return lookup_dict

def calculate_size(filenames, lookup_dict):
    total_files = 0
    total_size = 0
    not_found_files = []

    for filename in filenames:
        file_info = lookup_dict.get(filename)
        if file_info:
            total_files += 1
            total_size += int(file_info[0])  # file_info[0] is the size
        else:
            not_found_files.append(filename)

    return total_files, round(total_size / (1024**3), 2), not_found_files  # Size in GB and list of not found files

def check_files_exist(file_list, directory):
    missing_files = []
    for file in file_list:
        file_path = os.path.join(directory, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    return missing_files

def utm_to_latlon(x, y):
    # Convert lat/lon to UTM coordinates
    lat, lon = utm.to_latlon(x, y, 32, 'U')

    return lat, lon

def latlon_to_utm(lat, lon):
    # Convert lat/lon to UTM coordinates
    utm_x, utm_y, _, _ = utm.from_latlon(lat, lon, 32, 'U')

    return utm_x, utm_y

# Load the index file and create a lookup dictionary

with open(index_file, 'r') as file:
    data = json.load(file)
lookup_dict = create_lookup_dict(data)
logging.info("Index file loaded.")

# Load .laz files list and calculate number of files and size

laz_list = load_file_to_list(zone_files_list)
index_info = calculate_size(laz_list, lookup_dict)
logging.info(f"Loaded .laz file list {zone_files_list}, found {index_info[0]} files, size is {index_info[1]} GB.")

# Check that all .laz files in the list exist in the index and .laz directory

if index_info[2]:
    logging.error(f"The following files were not found in the index: {index_info[2]}")

missing_laz = check_files_exist(laz_list, zone_laz_dir)

if not missing_laz:
    logging.info("All .laz files are present in the LAZ directory.")
else:
    logging.error("The following .laz files were not found:", missing_laz)

# Create a DEM file list and perform verifications

def convert_filenames(laz_files):
    dem_files = []
    for file in laz_files:
        # Split the file name to extract the necessary parts
        parts = file.split('_')
        # Construct the new file name with the desired format
        new_file = f"dgm1_32_{parts[2]}_{parts[3]}_1_nw.tif"
        dem_files.append(new_file)
    return dem_files

dem_list = convert_filenames(laz_list)

missing_DEM = check_files_exist(dem_list, zone_DEM_dir)

if not missing_DEM:
    logging.info("All DEM .tif files are present in the DEM directory.")
else:
    logging.error("The following DEM .tif files were not found:", missing_DEM)

def process_laz(laz_file):

    process_id = os.getpid()

    highest_points = pd.DataFrame()

    laz_file_path = zone_laz_dir + laz_file

    with laspy.open(laz_file_path) as file:
        las = file.read()
    
    logging.debug(f"File {laz_file} loaded")

    class_val = las.classification[::SSFACTOR]

    mask = (np.isin(class_val, class_ok))

    points = np.vstack((las.x[::SSFACTOR][mask], las.y[::SSFACTOR][mask], las.z[::SSFACTOR][mask])).transpose()

    if len(points) == 0:
        logging.info(f"Found no points to process and no obstacles.")
        return highest_points

    DEM_file = zone_DEM_dir + convert_filenames([laz_file])[0]

    with rasterio.open(DEM_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        temp_DEM_file = f'./temp/temp_DEM_file{process_id}.tif'

        with rasterio.open(temp_DEM_file, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    
    logging.debug("Saved temporary reprojected DEM file")


    transformer = Transformer.from_pipeline(
    f"+proj=pipeline "
    f"+step +inv +proj=utm +zone=32 +ellps=WGS84 "  # Convert from UTM Zone 32N to geographic coordinates
    f"+step +proj=vgridshift +grids=@{temp_DEM_file},{backup_DEM_file} +multiplier = -1 "  # Vertical grid shift to remove the DEM elevation
    )

    intermediate_time_start = time.time()
    transformed_points = np.array([transformer.transform(xi, yi, zi) for xi, yi, zi in zip(points[:,0], points[:,1], points[:,2])])
    intermediate_time_stop = time.time()
    intermediate_execution_time = intermediate_time_stop - intermediate_time_start

    logging.info(f"Performed vgridshift for {laz_file} in {round(intermediate_execution_time, 1)} seconds")

    df = pd.DataFrame(
    data={
        "x": points[:,0], #np.array(las.x), # We need UTM coordinates
        "y": points[:,1], #np.array(las.y), # 
        "z": points[:,2],
        "h": transformed_points[:,2]
    }
    )

    size_df = sys.getsizeof(df)
    logging.debug(f"Size of the DataFrame: {np.ceil(size_df / (1024*1024))} MB")

    inf_rows = df.isin([np.inf, -np.inf]).any(axis=1)
    inf_rows_df = df[inf_rows]
    df = df[~inf_rows]
    logging.debug(f'Removed {len(inf_rows_df)} rows to the dataframe.')

    region = pygmt.info(data=df[["x", "y"]], spacing=1)  # West, East, South, North

    x_min, x_max, y_min, y_max = list(region)

    # Filtering the DataFrame
    condition = (abs(df['x'] - x_min) < 1.5) | \
                (abs(df['x'] - x_max) < 1.5) | \
                (abs(df['y'] - y_min) < 1.5) | \
                (abs(df['y'] - y_max) < 1.5)

    df_filtered = df[~condition]

    # Number of rows removed
    logging.debug(f'Removed another {len(df) - len(df_filtered)} points on the edge.')

    df_trimmed = pygmt.blockmedian(
        data=df_filtered[["x", "y", "h"]],
        T=0.9999,  # 99.99th quantile, i.e. the highest point
        spacing="1+e", # 1+e for 1 m # 0.1 increases the size of df but more accurate?
        region=region,
    )

    size_df_trimmed = sys.getsizeof(df_trimmed)
    logging.debug(f"Size of the trimmed dataframe: {np.ceil(size_df_trimmed / (1024*1024))} MB")

    high_points = df_trimmed[df_trimmed['h'] > 60] # Default = 60

    if high_points.empty:
        logging.info(f"Found no obstacles.")
        os.remove(temp_DEM_file)
        return highest_points
    else:
        # Assuming that points within 100m of each other belong to the same obstacle
        clustering = DBSCAN(eps=45, min_samples=2).fit(high_points[['x', 'y', 'h']]) # TODO: no error if no cluster found # Default = 50

        # Add the cluster labels to the high_points DataFrame
        high_points = high_points.copy()
        high_points['cluster'] = clustering.labels_

        # Filter out noise points (DBSCAN labels noise as -1)
        obstacles = high_points[high_points['cluster'] != -1]

        if obstacles.empty:
            logging.info(f"Found no obstacles.")
            os.remove(temp_DEM_file)
            return highest_points
        else:
            # Find the highest point in each obstacle cluster
            highest_points = obstacles.loc[obstacles.groupby('cluster')['h'].idxmax()]

            # The resulting DataFrame 'highest_points' contains the coordinates of the highest point of each obstacle
            highest_points.reset_index(drop=True, inplace=True)

            # Apply the conversion function to the DataFrame to create new columns 'lat' and 'lon'
            highest_points['lat'], highest_points['lon'] = zip(*highest_points.apply(lambda row: utm_to_latlon(row['x'], row['y']), axis=1))

            highest_points['gnd_elev'] = highest_points.apply(lambda row: round(-1 * transformer.transform(row['x'], row['y'], 0)[2], 2), axis=1)
            highest_points['source'] = laz_file
            highest_points['timestamp'] = lookup_dict.get(laz_file)[1]

            logging.info(f"Found {len(highest_points)} obstacles.")

            os.remove(temp_DEM_file)
            return highest_points
        
joblib.externals.loky.process_executor._MAX_MEMORY_LEAK_SIZE = int(9e11)

logging.info("Starting to process...")
start_time = time.time()

results_df = pd.DataFrame()

individual_dfs = Parallel(n_jobs=5, timeout = 99999)(delayed(process_laz)(laz_file) for laz_file in tqdm(laz_list))

results_df = pd.concat(individual_dfs, ignore_index=True)

results_df.reset_index(drop=True, inplace=True)

end_time = time.time()

# Compute execution time
execution_time = end_time - start_time

logging.info(f"Processed all files. Execution time: {round(execution_time, 1)} seconds")

results_df.to_excel(f'{results_dir}/results_parallel_{zone_to_process}.xlsx', index=False)