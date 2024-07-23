import numpy as np
import plotly.graph_objects as go
from .utils import *
from scipy.signal import find_peaks

class AdaptiveASKDecoder:
    def __init__(self, tps, n_symbols):
        self.tps = tps
        self.n_symbols = n_symbols
        self.signal_max = None
        self.signal_min = None

    def detect_symbol(self, sx):
        possible_symbols = (np.array([(i/(self.n_symbols-1)) for i in range(self.n_symbols)])
                            * (self.signal_max - self.signal_min) + self.signal_min)
        n = len(sx)
        mean_val = np.mean(sx[int(n/4):-int(n/4)])
        return np.argmin(np.abs(mean_val - possible_symbols))

    def detect_simbol_chunks(self, t, x):

        # (1) detect sharp changes in local maximums over AOI
        dx = np.array([x[i + 1] - x[i] for i in range(0, len(x) - 1)])
        dx = np.abs(dx)
        # plot(x=t, y=dx, title="dx (derivative)")

        n_avr = 300
        dx = np.convolve(dx, np.ones(n_avr) / n_avr, 'valid')

        # find the largest value in dx and filter values above  max / n_symbols
        max_change = np.max(dx)
        minimal_valid_change = max_change / self.n_symbols
        # HPF
        dx[dx < minimal_valid_change * 0.8] = 0

        # running avrage again
        n_avr = int(n_avr / 2)
        dx = np.convolve(dx, np.ones(n_avr) / n_avr, 'valid')

        plot(x=t, y=dx, title="dx after running average")

        # find main peaks
        peaks, _ = find_peaks(dx, width=n_avr)
        dx_peaks = np.zeros(x.shape)
        dx_peaks[peaks] = 1

        # plot(x=t, y=dx_peaks, title="peaks found with scipy")

        # finnaly, always assume we the first symbol is starting at 0
        dx_peaks[0] = 1

        return t, dx_peaks

    def decode(self, t, x) -> np.array:
        # (0) initiate signal max and min
        self.signal_max, self.signal_min = np.max(x), np.min(x)

        # detect chunks of simbols
        t, dx_simbols = self.detect_simbol_chunks(t, x)

        decoded_message = []
        tps_ls = []
        # for each 2 following '1' in dx_symbols, check what is the symbol there
        indexes_of_symbol_chuncks = [i for i in range(len(dx_simbols)) if dx_simbols[i] > 0]
        for i in range(len(indexes_of_symbol_chuncks)-1):
            start = indexes_of_symbol_chuncks[i]+1
            end = indexes_of_symbol_chuncks[i+1]

            # how many times has the simble appeared
            n = round((t[end] - t[start]) / self.tps)
            tps_ls.append((t[end] - t[start]) / n)

            # detect the simbol type
            symbol_type = self.detect_symbol(x[start:end])

            for _ in range(n):
                decoded_message.append(symbol_type)

        print(f"decoded message with given {self.tps} (tps) time per symbol: \n{decoded_message}")
        print(f"measured time per symbol:\n mean={np.mean(tps_ls)}\n var={np.var(tps_ls)}")
        return np.array(decoded_message)



