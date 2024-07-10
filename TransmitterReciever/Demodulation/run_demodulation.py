import preprocces
import demodulator
from utils import *

def main():
    F_0 = 600
    t, x = preprocces.run_preprocess('../data/H.csv', F_0)
    decoder = demodulator.AdaptiveASKDecoder(0.06, 4)
    code = decoder.decode(t, x)


if __name__ == "__main__":
    main()