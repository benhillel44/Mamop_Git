import numpy as np
from constants import *

import TransmitterReciever.Demodulation.utils as utils

def create_symbol_array(binary_str, bits_per_symbol):
    """
    Creates a symbol array from a binary string.
    """
    # Check for valid binary string
    if not all(char in ('0', '1') for char in binary_str):
        raise ValueError("Binary string must contain only '0' and '1' characters.")

    # Calculate the number of symbols
    symbol_count = len(binary_str) // bits_per_symbol
    if len(binary_str) % bits_per_symbol != 0:
        padding = bits_per_symbol - (len(binary_str) % bits_per_symbol)
        binary_str += '0' * padding
        symbol_count += 1

    max_val = 2 ** bits_per_symbol - 1

    # Convert binary string to symbols
    symbols = []
    for i in range(0, len(binary_str), bits_per_symbol):
        symbol = binary_str[i:i + bits_per_symbol]
        symbols.append(int(symbol, 2) / max_val)

    return symbols, symbol_count


def ask_modulate(binary_str, bits_per_symbol, wanted_sin_frequency=6e5):
    """
    Performs ASK modulation on a binary string.

    Args:
      binary_str: The binary string to modulate (string of '0's and '1's).
      bits_per_symbol: The number of bits per symbol (e.g., 2 for 4-ASK).
      samples_per_symbol_ratio: The ratio of samples per symbol.

    Returns:
      The modulated ASK signal as a NumPy array.

    """

    symbols, symbol_count = create_symbol_array(binary_str, bits_per_symbol)

    # Define constants
    fc = wanted_sin_frequency  # Carrier frequency in Hz
    fs = fc * SAMPLES_PER_SIN  # Sampling frequency, 8 times the carrier frequency

    max_time = TIME_PER_SYMBOL * symbol_count
    sample_count = int(max_time * fs)
    padded_sample_count = 2 ** int(np.ceil(np.log2(sample_count)))
    if padded_sample_count > 2 ** 15:
        print("Sample amount too lare for picoscope limitation (too many symbols) - 32K, rest will be clipped")
        print("Either decrease the binary string length or increase modulation depth")
        padded_sample_count = 2 ** 15
    padding_length = padded_sample_count - sample_count

    times = np.linspace(0, max_time, num=sample_count, endpoint=False)
    padding_time = padding_length/fs
    total_time = max_time + padding_time
    padded_times = np.linspace(0, total_time, num=padded_sample_count, endpoint=False)

    padding_array = np.zeros(padding_length)
    sampled_symbol_array = np.array([symbols[int(t // TIME_PER_SYMBOL)] for t in times])

    raw_signal = np.concatenate((sampled_symbol_array, padding_array))

    # Generate carrier wave
    carrier_wave = np.sin(2 * np.pi * fc * padded_times)

    # Modulate carrier with symbols
    result = raw_signal * carrier_wave
    pico_freq = 1 / total_time
    print("Max time: ", max_time)
    print("total time: ", total_time)

    utils.plot(padded_times, raw_signal, title= "Raw signal")
    # utils.plot(times, sampled_symbol_array, title= "unpaded Raw signal")
    # utils.plot(padded_times, result, title= "result")

    print("pico freq should be", pico_freq, "Hz")
    return result, pico_freq


def save_result_to_csv(result):
    # Save to CSV file
    csv_filename = 'AM.csv'
    with open(csv_filename, 'w') as csvfile:
        for value in result:
            csvfile.write(f"{value}\n")

    print(f"Sine wave data multiplied by bit pattern saved to {csv_filename}")


if __name__ == '__main__':
    # header 01010
    # data 01101000
    result, freq = ask_modulate("001100110001101000", bits_per_symbol=2, wanted_sin_frequency=6e5)
