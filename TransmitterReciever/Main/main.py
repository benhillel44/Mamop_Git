import time

import TransmitterReciever.Demodulation.preprocces as preprocess
from TransmitterReciever.Demodulation.demodulator import AdaptiveASKDecoder
from write_read_awg import PicoSender
from constants import *
from picture_to_chunks import picture_to_chunks, chunk_to_bytes


if __name__ == "__main__":
    chunks = picture_to_chunks()
    byte_chunks = [chunk_to_bytes(chunk) for chunk in chunks]

    pico_sender = PicoSender()
    binary_string = byte_chunks[0]
    print(f"binary string in bytes: {binary_string}")
    # transform this to a binary string
    binary_string = ''.join(format(byte, '08b') for byte in binary_string)
    print(f"binary string in binary: {binary_string}")
    print(f"binary string in base {2**BITS_PER_SYMBOL}: {int(binary_string, 2)}")
    pico_sender.send(binary_string)
    time.sleep(1)
    t, x = pico_sender.receive()
    t, x = preprocess.run_preprocess_list(t, x, CARRIER_FREQ / 1000)
    time.sleep(1)
    pico_sender.stop()

    decoder = AdaptiveASKDecoder(0.1, 2**BITS_PER_SYMBOL)
    code = decoder.decode(t, x)
    print(code)