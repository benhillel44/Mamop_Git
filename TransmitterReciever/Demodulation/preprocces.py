import sys

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from utils import fft, ifft, LPF, HPF, plot
from numpy import pi
from demodulator import *

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

def denoise(df: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # (1) FFT to the data
    t = df['Time'].astype(float)
    x = df['Channel A'].astype(float)
    freq, fx = fft(t, x)

    # show the FFT
    # plot(x=freq, y=fx.real, title='FFT of the Channel A over Time', x_axsis_range=[-800,800])

    # keep only the high frequencies (above 500kHz)
    fx[np.abs(freq)<500] = 0

    # show the FFT
    # plot(x=freq, y=fx.real, title='FFT of the Channel A over Time', x_axsis_range=[-800,800])

    denoised_x = ifft(fx).real

    # show the Denoised DAta
    # plot(x=t, y=denoised_x, title='Denoised Data')

    return t, denoised_x


def envelope_detection(t, x, f_0) -> (np.ndarray, np.ndarray):
    # multiply the signal by sin(f_0t)
    x *= np.sin(2*pi*f_0*t)
    t, x = LPF(t, x, f_0)

    return t, x


def main() :
    df_data = load_data('../data/H.csv')
    F_0 = 600
    plot(x=df_data['Time'], y=df_data['Channel A'], title='Raw Signal Data')

    t, x = denoise(df_data)

    t, x = envelope_detection(t, x, F_0)

    plot(x=t, y=np.abs(x), title='After envelope detection')

if __name__ == '__main__':
    main()