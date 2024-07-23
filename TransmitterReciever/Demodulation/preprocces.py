import sys

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from .utils import fft, ifft, LPF, HPF, plot
from numpy import pi
from ..Main.constants import *


def load_data(path_to_csv) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)
    df.dropna()
    # ===> Transfare to standart units - if mV => V
    for channel in df.columns:
        if df[channel][0].startswith('(mV'):
            df[channel][0] = 0
            df[channel] = df[channel].astype(float) / 1000.0

    # drop the first row (that includes the (mv))
    df = df.drop([0])
    return df


# def denoise(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
def denoise(time, x) -> (np.ndarray, np.ndarray):
    # (1) FFT to the data
    freq, fx = fft(time, x)

    # show the FFT
    # plot(x=freq, y=fx.real, title='FFT of the Channel A over Time', x_axsis_range=[-800,800])

    # keep only the high frequencies
    f_0 = CARRIER_FREQ / 1000
    cutoff_freq = f_0 * (95 / 100)
    fx[np.abs(freq) < cutoff_freq] = 0

    cutoff_freq = f_0 * (105 / 100)
    fx[np.abs(freq) > cutoff_freq] = 0


    # show the FFT
    # plot(x=freq, y=fx.real, title='FFT of the Channel A over Time After HPF', x_axsis_range=[-800,800])

    denoised_x = ifft(fx).real

    # show the Denoised DAta
    plot(x=time, y=denoised_x, title='Denoised Data')

    return time, denoised_x


def hetrodine_detection(t, x, f_0) -> (np.ndarray, np.ndarray):
    # multiply the signal by sin(f_0t)
    x *= np.sin(2 * pi * f_0 * t)
    t, x = LPF(t, x, f_0)

    return t, x


def envelope_detection(t, x, f_0) -> (np.ndarray, np.ndarray):
    print(t.shape, x.shape)
    # in a more - Aravy way
    sample_freq = 1 / (t[100] - t[99])
    print("sample freq", sample_freq)
    n_points_AOI = int(1 / ((1 / f_0 - 1 / sample_freq) * 2 * pi))
    print(n_points_AOI)
    n_points_AOI = 100
    print(f">> preforming Arabic envelope detection with local maxima of {n_points_AOI} points")
    x_AOF_maximums = np.array([np.max(x[i:i + n_points_AOI]) for i in range(0, len(x) - n_points_AOI)])

    return t, x_AOF_maximums


def run_preprocess_csv(path_to_csv: str, f_0: float) -> (np.ndarray, np.ndarray):
    df_data = load_data(path_to_csv)
    F_0 = f_0
    # plot(x=df_data['Time'], y=df_data['Channel A'], title='Raw Signal Data')
    t = np.array(df_data['Time'].astype(float))
    channel = np.array(df_data['Channel A'].astype(float))
    t, x = denoise(t, channel)
    t, x = envelope_detection(t, x, F_0)
    # plot(x=t, y=np.abs(x), title='After envelope detection')
    return t, x


def run_preprocess_list(time, voltage, f_0: float) -> (np.ndarray, np.ndarray):
    F_0 = f_0
    plot(x=time, y=voltage, title='Raw Signal Data')
    t, x = denoise(time, voltage)
    t, x = envelope_detection(time, x, F_0)
    plot(x=t, y=np.abs(x), title='After envelope detection')
    return t, x


if __name__ == '__main__':
    run_preprocess_csv('../data/H.csv', 600)
