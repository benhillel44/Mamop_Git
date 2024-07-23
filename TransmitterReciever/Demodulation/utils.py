import numpy as np
import plotly.graph_objects as go


def fft(x, y) -> (np.ndarray, np.ndarray):
    yf = np.fft.fft(y)
    freq = np.fft.fftfreq(len(y), x[2]-x[1])
    return freq, yf


def ifft(yf) -> np.ndarray:
    y = np.fft.ifft(yf)
    return y


def LPF(t, x, f_max) -> (np.ndarray, np.ndarray):
    freq, fx = fft(t, x)
    # keep only the high frequencies (above 500kHz)
    fx[np.abs(freq) > f_max] = 0
    denoised_x = ifft(fx).real
    return t, denoised_x


def HPF(t, x, f_min) -> (np.ndarray, np.ndarray):
    freq, fx = fft(t, x)
    # keep only the high frequencies (above 500kHz)
    fx[np.abs(freq) < f_min] = 0
    denoised_x = ifft(fx).real
    return t, denoised_x


def plot(x: np.ndarray, y: np.ndarray, title: str,
         x_axsis_range=None , y_axsis_range=None,
         y_axis_title='y Value', x_axis_title='x Value'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=''))
    fig.update_layout(title_text=title)
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=y_axis_title)

    if x_axsis_range is not None:
        fig.update_xaxes(range=x_axsis_range)
    if y_axsis_range is not None:
        fig.update_yaxes(range=y_axsis_range)
    fig.show()

