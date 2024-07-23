import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time
import am_mod
from constants import *


class PicoSender:
    def __init__(self):
        # Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}
        self.f_carrier = CARRIER_FREQ

        # Open PicoScope 2000 Series device
        # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(self.chandle), None)
        assert_pico_ok(self.status["openunit"])

        self.enabled = 1
        self.disabled = 0
        self.analogue_offset = 0.0

    def setup(self):
        # Set up channel A
        # handle = chandle
        # channel = PS2000A_CHANNEL_A = 0
        # enabled = 1
        # coupling type = PS2000A_DC = 1
        # range = PS2000A_2V = 7
        # analogue offset = 0 V
        channel_range = CHANNEL_RANGE # ps.PS2000A_RANGE['PS2000A_10V']
        self.status["setChA"] = ps.ps2000aSetChannel(self.chandle,
                                                ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                                self.enabled,
                                                ps.PS2000A_COUPLING['PS2000A_DC'],
                                                channel_range,
                                                self.analogue_offset)
        assert_pico_ok(self.status["setChA"])

        # Set up channel B
        # handle = chandle
        # channel = PS2000A_CHANNEL_B = 1
        # enabled = 1
        # coupling type = PS2000A_DC = 1
        # range = PS2000A_2V = 7
        # analogue offset = 0 V
        self.status["setChB"] = ps.ps2000aSetChannel(self.chandle,
                                                ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                                self.enabled,
                                                ps.PS2000A_COUPLING['PS2000A_DC'],
                                                channel_range,
                                                self.analogue_offset)
        assert_pico_ok(self.status["setChB"])
        # Size of capture
        self.totalSamples = SIZE_OF_ONE_BUFFER * NUM_OF_BUFFERS_TO_CAPTURE

        # Create buffers ready for assigning pointers for data collection
        self.bufferAMax = np.zeros(shape=SIZE_OF_ONE_BUFFER, dtype=np.int16)
        self.bufferBMax = np.zeros(shape=SIZE_OF_ONE_BUFFER, dtype=np.int16)

        memory_segment = 0

        # Set data buffer location for data collection from channel A
        # handle = chandle
        # source = PS2000A_CHANNEL_A = 0
        # pointer to buffer max = ctypes.byref(bufferAMax)
        # pointer to buffer min = ctypes.byref(bufferAMin)
        # buffer length = maxSamples
        # segment index = 0
        # ratio mode = PS2000A_RATIO_MODE_NONE = 0
        self.status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(self.chandle,
                                                             ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                                             self.bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                             None,
                                                             SIZE_OF_ONE_BUFFER,
                                                             memory_segment,
                                                             ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
        assert_pico_ok(self.status["setDataBuffersA"])

        # Set data buffer location for data collection from channel B
        # handle = chandle
        # source = PS2000A_CHANNEL_B = 1
        # pointer to buffer max = ctypes.byref(bufferBMax)
        # pointer to buffer min = ctypes.byref(bufferBMin)
        # buffer length = maxSamples
        # segment index = 0
        # ratio mode = PS2000A_RATIO_MODE_NONE = 0
        self.status["setDataBuffersB"] = ps.ps2000aSetDataBuffers(self.chandle,
                                                             ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                                             self.bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                             None,
                                                             SIZE_OF_ONE_BUFFER,
                                                             memory_segment,
                                                             ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
        assert_pico_ok(self.status["setDataBuffersB"])

        # Begin streaming mode:
        sampleInterval = ctypes.c_int32(112)
        sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_NS']
        # We are not triggering:
        maxPreTriggerSamples = 0
        autoStopOn = 1
        # No downsampling:
        downsampleRatio = 1
        self.status["runStreaming"] = ps.ps2000aRunStreaming(self.chandle,
                                                        ctypes.byref(sampleInterval),
                                                        sampleUnits,
                                                        maxPreTriggerSamples,
                                                        self.totalSamples,
                                                        autoStopOn,
                                                        downsampleRatio,
                                                        ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'],
                                                        SIZE_OF_ONE_BUFFER)
        assert_pico_ok(self.status["runStreaming"])

        actualSampleInterval = sampleInterval.value
        self.actualSampleIntervalNs = actualSampleInterval  # * 1000

        print("Capturing at sample interval %s ns" % self.actualSampleIntervalNs)

        # We need a big buffer, not registered with the driver, to keep our complete capture in.
        self.bufferCompleteA = np.zeros(shape=self.totalSamples, dtype=np.int16)
        self.bufferCompleteB = np.zeros(shape=self.totalSamples, dtype=np.int16)
        self.nextSample = 0
        self.autoStopOuter = False
        self.wasCalledBack = False
        # Convert the python function into a C function pointer.
        self.cFuncPtr = ps.StreamingReadyType(self.streaming_callback)

    def streaming_callback(self, handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
        self.wasCalledBack = True
        destEnd = self.nextSample + noOfSamples
        sourceEnd = startIndex + noOfSamples
        self.bufferCompleteA[self.nextSample:destEnd] = self.bufferAMax[startIndex:sourceEnd]
        self.bufferCompleteB[self.nextSample:destEnd] = self.bufferBMax[startIndex:sourceEnd]
        # print(bufferAMax[startIndex:sourceEnd])
        self.nextSample += noOfSamples
        if autoStop:
            self.autoStopOuter = True

    def run_signal_gen(self, binary_string):
        waveform_list, pico_freq = am_mod.ask_modulate(binary_string, BITS_PER_SYMBOL, self.f_carrier)
        waveform_list = [int(x * 32767) for x in waveform_list]

        wavetype = ctypes.c_int16(0)
        sweepType = ctypes.c_int32(0)
        triggertype = ctypes.c_int32(0)
        triggerSource = ctypes.c_int32(0)
        waveform_length = ctypes.c_int32(len(waveform_list))
        waveform_length2 = ctypes.c_uint32(len(waveform_list))
        c_waveform_data = (ctypes.c_int16 * len(waveform_list))(*waveform_list)

        phase = ctypes.c_uint32(0)

        ps.ps2000aSigGenFrequencyToPhase(
            self.chandle,
            int(pico_freq),
            0,
            waveform_length2,
            ctypes.byref(phase)
        )

        self.status["SetSigGenArbitrary"] = ps.ps2000aSetSigGenArbitrary(self.chandle, 0,
                                                                    2000000,
                                                                    phase,
                                                                    phase,
                                                                    0,
                                                                    0,
                                                                    c_waveform_data,
                                                                    waveform_length,
                                                                    sweepType,
                                                                    0,  # ps.PS2000A_EXTRA_OPERATIONS["PS2000A_ES_OFF"],
                                                                    0,  # ps.PS2000A_INDEX_MODE['PS2000A_SINGLE'],
                                                                    ctypes.c_uint32(0),  # shots
                                                                    ctypes.c_uint32(0),  # sweeps
                                                                    0,
                                                                    # ps.PS2000A_SIGGEN_TRIG_TYPE["PS2000A_SIGGEN_RISING"],
                                                                    0,
                                                                    # ps.PS2000A_SIGGEN_TRIG_SOURCE["PS2000A_SIGGEN_NONE"],
                                                                    ctypes.c_int16(0)  # threshold
                                                                    )

        # status["SetSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(chandle, 0, 2000000, wavetype, 6e5, 6e5, 0, 1, sweepType, 0, 0,
        #                                                         0, triggertype, triggerSource, 1)
        assert_pico_ok(self.status["SetSigGenArbitrary"])

    def grab_data(self):
        # Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
        while self.nextSample < self.totalSamples and not self.autoStopOuter:
            self.wasCalledBack = False
            self.status["getStreamingLastestValues"] = ps.ps2000aGetStreamingLatestValues(self.chandle, self.cFuncPtr, None)
            if not self.wasCalledBack:
                # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
                # again.
                time.sleep(0.01)

        print("Done grabbing values.")

    def stop(self):
        self.status["stop"] = ps.ps2000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])

        # Disconnect the scope
        # handle = chandle
        self.status["close"] = ps.ps2000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])

    def receive(self) -> (np.ndarray, np.ndarray):
        self.setup()
        self.grab_data()
        # Create time data
        time_data = np.linspace(0, (self.totalSamples - 1) * self.actualSampleIntervalNs, self.totalSamples) * 1e-6
        return time_data, np.array(self.bufferCompleteA)

    def send(self, binary_data: str):
        self.run_signal_gen(binary_data)



import TransmitterReciever.Demodulation.preprocces as preprocess
from TransmitterReciever.Demodulation.demodulator import AdaptiveASKDecoder

pico_sender = PicoSender()
binary_string = "111110101100011010001000"
pico_sender.send(binary_string)
t, x = pico_sender.receive()
t, x = preprocess.run_preprocess_list(t, x, CARRIER_FREQ)
pico_sender.stop()

decoder = AdaptiveASKDecoder(0.1, 2**BITS_PER_SYMBOL)
code = decoder.decode(t, x)
print(code)

