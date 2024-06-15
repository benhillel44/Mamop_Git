import split_csv, fit_sin, fit_RLC

if __name__ == '__main__':
    # TODO - fix this, doesnt work.
    """
    run the code start to end:
    (1) extract samples from zip to csv's
    (2) fit each csv a sin curve to get the amplitude over frequency
    (3) fit the RLC to get the amplitude over frequency
    """
    measurement_num = 3 # measurement num by which the files be saved as
    measurement_zip_name = "measurament_3.zip" # the name of the zip placed under measurements/measurements_zips

    split_csv.run_split_csv(measurement_num, zip_name_=measurement_zip_name)
    fit_sin.run_fit_sin(measurement_num)
    fit_RLC.run_fit_RLC(measurement_num)
