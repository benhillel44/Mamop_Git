import numpy as np
import plotly.graph_objects as go
from utils import *


def decode_4ASK(t, x, tps) -> (np.ndarray, np.ndarray):
    possible_values = np.array((0, 0.333, 0.666, 1)).astype(float) * (np.max(x) - np.min(x))
    find_closest_symbol = lambda xx: np.argmin([np.abs(xx - possible_symbol) for possible_symbol in possible_values])

    code = np.zeros(x.shape)
    curr_index = 0 # the number of symbols decoded
    max_time = np.max(t)
    while curr_index * tps < max_time:
        code[curr_index*tps < t < (curr_index+1)*tps] = (
            find_closest_symbol(np.mean(x[curr_index*tps < t < (curr_index+1)*tps])))
        curr_index += 1
    return t, code



