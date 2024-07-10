import numpy as np
import plotly.graph_objects as go
from utils import *
from scipy.signal import find_peaks

class AdaptiveASKDecoder:
    def __init__(self, tps, n_symbols):
        self.tps = tps
        self.n_symbols = n_symbols
        self.signal_max = None
        self.signal_min = None

        # array of symbols we can map to e.g. if n=4 we have [0, 1/3, 2/3, 1]
        self.technical_symbols = [i/(n_symbols-1) for i in range(n_symbols)]
        # array of symbols as they are received, effectively 'technical_symbols'*signal_max
        self.effective_symbols = None
        # a map between the two
        self.effective_to_technical_symbol = {}

        # number of intervals used in the heuristic method.
        self.heuristic_divide = 25

    def initialise_symbols(self, x):
        """
        Initialises 'self.signal_max' and related symbol parameters specific to received data.

        Parameters
        _________________
        x: np.ndarray - values array

        Return
        _________________
        nothing
        """
        self.signal_max = max(x)
        self.effective_symbols = []
        for i in range(self.n_symbols):
            technical_value = self.technical_symbols[i]
            effective_value = technical_value * self.signal_max
            self.effective_symbols.append(effective_value)
            self.effective_to_technical_symbol[effective_value] = technical_value


    def calc_heuristic(self, t, x) -> float:
        """
        Calculates a heuristic value for symbol data. Current implementation is based
         on dividing the data into self.heuristic_divide intervals,
         and averaging the maximum value on each interval.

        Parameters
        _________________
        st: np.ndarray - time array
        sx: np.ndarray - values array

        Return
        ________________
        heuristic: int - the heuristic value for the given symbol data.
        """

        sum_of_peaks = 0
        interval_length = np.floor(len(t)/self.heuristic_divide)
        for i in range(self.heuristic_divide):
            curr_interval_values = [x[idx] for idx, time in enumerate(t[i*interval_length:(i+1)*interval_length])]
            curr_peak = max(curr_interval_values)
            sum_of_peaks += curr_peak

        average_peak = sum_of_peaks/self.heuristic_divide
        return average_peak


    def nearest_effective_symbol(self, heuristic) -> float:
        """
        Rounds heuristic guess to nearest effective symbol (i.e. normalised to received amplitude).
        Assumes self.effective_symbols has been initialised.

        Parameters
        _________________
        heuristic: float - heuristic value given to symbol data in question

        Return
        ________________
        nearest effective (i.e. normalised) symbol

        """
        dist = lambda symbol: abs(symbol-heuristic)
        return min(self.effective_symbols, key=dist)

    def detect_symbol(self, st, sx) -> int:
        """
        Given subsection of time and values, return the most likely symbol for
        this section. Assume that the possible symbols are i * (self.signal_max - self.signal_min) / tps

        Parameters
        _________________
        st: np.ndarray - subsection of the time array
        sx: np.ndarray - subsection of the values array

        Return
        ________________
        symbol: int - the most likely (non-normalised) symbol corresponding to that section

        """
        h = self.calc_heuristic(st, sx)
        effective_symbol = self.nearest_effective_symbol(h)
        symbol = self.effective_to_technical_symbol[effective_symbol]
        return symbol


    def detect_simbol_chunks(self, t, x):
        # Hyperparameter - n_points_AOI: the number of points we consider at once -> at best should be the number of points in
        # one cycle of the carrier signal
        n_points_AOI = 100
        x_AOF_maximums = np.array([np.max(x[i:i + n_points_AOI]) for i in range(0, len(x) - n_points_AOI)])

        # (1) detect sharp changes in local maximums over AOI

        dx = np.array([x_AOF_maximums[i + 1] - x_AOF_maximums[i] for i in range(0, len(x_AOF_maximums) - 1)])
        dx = np.abs(dx)
        # plot(x=t, y=dx, title="DX of AOI transform")

        n_avr = n_points_AOI
        dx = np.convolve(dx, np.ones(n_avr) / n_avr, 'valid')

        # find the largest value in dx and filter values above  max / n_symbols
        max_change = np.max(dx)
        minimal_valid_change = max_change / self.n_symbols
        # HPF
        dx[dx < minimal_valid_change * 0.8] = 0

        # running avrage again
        n_avr = int(n_points_AOI / 2)
        dx = np.convolve(dx, np.ones(n_avr) / n_avr, 'valid')

        # plot(x=t, y=dx, title="DX of AOI transform")

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
        self.initialise_symbols(x)

        # detect chunks of simbols
        t, dx_simbols = self.detect_simbol_chunks(t, x)

        decoded_message = []
        print(t)
        # for each 2 following '1' in dx_symbols, check what is the symbol there
        indexes_of_symbol_chuncks = [i for i in range(len(dx_simbols)) if dx_simbols[i] > 0]
        for i in range(len(indexes_of_symbol_chuncks)):
            start = indexes_of_symbol_chuncks[i]+1
            end = indexes_of_symbol_chuncks[i+1]+1

            # how many times has the simble appeared
            n = round((t[end] - t[start]) / self.n_symbols)

            symbol_type = self.detect_symbol(t[start:end], x[start:end])
            print(f"symbol type - {symbol_type}")

            for _ in range(n):
                decoded_message.append(symbol_type)

        print(f"decoded_message: {decoded_message}")
        return np.array(decoded_message)



