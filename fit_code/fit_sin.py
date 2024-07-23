import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fmin
import math
from os import listdir
import os
import constants



def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    
    if A < 0:
        A = -A
        p += np.pi

    p %= (2.*np.pi)

    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}


def proccess_data(data, filename, csv_path, is_channelA_milli=False, is_channelB_milli=False):
    x = data['Time'].astype(float)
    channelA = data['Channel A'].astype(float) / 1000 if is_channelA_milli else data['Channel A'].astype(float)
    channelB = data['Channel B'].astype(float) / 1000 if is_channelB_milli else data['Channel B'].astype(float)

    sin_paramsA = fit_sin(x, channelA)
    sin_paramsB = fit_sin(x, channelB)

    if save_sin_fit_graphs:
        plt.scatter(x, channelA, s=10)
        plt.scatter(x, channelB, s=10)
        plt.plot(x, sin_paramsA["fitfunc"](x), color='green')
        plt.plot(x, sin_paramsB["fitfunc"](x), color='blue')
        plt.title(f'{filename}')
        # plt.show()
        plt.savefig(f'{filename}.png') #_freqA_{sin_paramsA['freq']}_freqB_{sin_paramsB['freq']}
        plt.clf()


    sin_paramsA.pop('fitfunc')
    sin_paramsB.pop('fitfunc')
    sin_paramsA.pop('rawres')
    sin_paramsB.pop('rawres')
    sin_paramsA.pop('maxcov')
    sin_paramsB.pop('maxcov')

    # change the names of the columns of both sin params  
    keys = list(sin_paramsA.keys())
    for key in keys:
        sin_paramsA[f'{key}_A'] = sin_paramsA.pop(key)
        sin_paramsB[f'{key}_B'] = sin_paramsB.pop(key)

    
    return sin_paramsA, sin_paramsB



def main(file_input_dir, csv_output_dir, graph_output_dir,
         is_channelA_milli=False, is_channelB_milli=False):
    csv_name = f'{file_input_dir.split("/")[-1]}.csv'
    # create the directory if it does not exist
    if not os.path.exists(graph_output_dir):
        os.makedirs(graph_output_dir)

    df_data_total = []

    files = listdir(file_input_dir)
    for file in files:
        if file.endswith('.csv'):
            data = pd.read_csv(f'{file_input_dir}/{file}')

            # remove the first row from data
            data = data.iloc[2:202]
            try:
                sin_paramsA, sin_paramsB = proccess_data(data, f'{graph_output_dir}\\{file[:-4]}', f'{csv_dir}/{csv_name}'
                                                         , is_channelA_milli=is_channelA_milli
                                                         , is_channelB_milli=is_channelB_milli)
                sin_paramsA.update(sin_paramsB)
            except:
                print(f'Error with file {file}')
                continue

            # remove abviously wrong data with freq over 1000 kHz
            if sin_paramsA["freq_A"] > 1000 or sin_paramsA["freq_B"] > 1000:
                continue

            # remove data if the frequencies are off by more than 10%
            if abs(sin_paramsA["freq_A"] - sin_paramsA["freq_B"]) > 0.1 * sin_paramsA["freq_A"]:
                continue
            

            sin_paramsA["phase_diff"] = (sin_paramsA["phase_A"] - sin_paramsA["phase_B"] ) % (2*np.pi)
            sin_paramsA["amp_ratio"] = sin_paramsA["amp_A"] / sin_paramsA["amp_B"]

            # create a dataframe with the sin_paramsA and sin_paramsB
            df_data = pd.DataFrame([sin_paramsA])
            df_data_total.append(df_data)

            
    # print(df_data_total)
    # save it as csv 
    df_data_total = pd.concat(df_data_total, ignore_index=True)
    df_data_total.to_csv(f'{csv_output_dir}/{csv_name}')

def run_fit_sin(measurement_):
    file_dir_ = constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement_}'
    graph_dir_ = constants.PATH_TO_FIT_GRAPHS
    csv_dir_ = constants.PATH_TO_MEASUREMENT_CSV
    main(file_dir_, csv_dir_, graph_dir_)

if __name__ == "__main__":
    # ======= you can modify the parameters bellow
    measurement = 12
    save_sin_fit_graphs = False
    is_channelA_milli = True
    is_channelB_milli = True
    # ============================================

    file_dir = constants.PATH_TO_EXTRACTED_MEASUREMENTS + f'measurement_{measurement}'
    graph_dir = constants.PATH_TO_FIT_GRAPHS
    csv_dir = constants.PATH_TO_MEASUREMENT_CSV
    main(file_dir, csv_dir, graph_dir,
         is_channelA_milli=is_channelA_milli,
         is_channelB_milli=is_channelB_milli)

