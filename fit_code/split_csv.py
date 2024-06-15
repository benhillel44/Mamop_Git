from os import listdir
import os
from os.path import isfile, join
from zipfile import ZipFile
import pandas as pd
import numpy as np

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
        

if __name__ == "__main__":
    measurement = 6
    extract_zip_into_dir("C:\\Users\\TLP-278\\Downloads\\Telegram Desktop\\measurement_6_second_half.zip", f'measurement_{measurement}')
    main(f'measurement_{measurement}')
