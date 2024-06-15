from os import listdir
import os
from os.path import isfile, join
from zipfile import ZipFile
import pandas as pd
import numpy as np
import constants
def main(dir_path):
    csv_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.csv')]
    for file in csv_files:
        print(file)
        data = pd.read_csv(f'{dir_path}/{file}')
        os.remove(f'{dir_path}/{file}')
        data1 = data.iloc[2:202]
        data2 = data.iloc[203:402]
        data1.to_csv(f'{dir_path}/{file[:-4]}_1.csv')
        data2.to_csv(f'{dir_path}/{file[:-4]}_2.csv')

def extract_zip_into_dir(zip_path, dir_path):
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dir_path)
        # move all extracted files to the root of the directory
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                os.rename(os.path.join(root, file), f'{dir_path}/{file.split("/")[-1]}')      
        

def run_split_csv(measurement_, zip_name_):
    extract_zip_into_dir(constants.PATH_TO_MEASUREMENTS_ZIPS + zip_name_,
                         constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement_}')
    main(constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement_}')

if __name__ == "__main__":
    """
    given a measurement, and a zip containing a measurement, 
    this code extracts samples of that measurement (because we dont need all of the points)
    to the directory - ../measurements/extracted_measurements/measurement_{THE MEASUREMENT NUM YOU ENTERED}
    """
    # ======= you can modify the parameters bellow
    measurement = 3
    zip_name = "measurament_3.zip"
    # ============================================

    # ================== do not change if you are not rafa =====================================
    extract_zip_into_dir(constants.PATH_TO_MEASUREMENTS_ZIPS + zip_name,
                         constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement}')
    main(constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement}')
