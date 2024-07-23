import preprocces
import demodulator
from utils import *

def main():
    F_0 = 600  # the carrier frequency
    TPS = 0.05  # this is the estimated time per symbol
    n_symbols = 4  # the num of levels in the modulation
    t, x = preprocces.run_preprocess_csv('H.csv', F_0)
    decoder = demodulator.AdaptiveASKDecoder(TPS, n_symbols)
    code = decoder.decode(t, x)


if __name__ == "__main__":
    main()