#
# Copyright (C) 2018-2019 Pico Technology Ltd. See LICENSE file for terms.
#
# PS2000 Series (A API) STREAMING MODE EXAMPLE
# This example demonstrates how to call the ps2000a driver API functions in order to open a device, setup 2 channels and collects streamed data (1 buffer).
# This data is then plotted as mV against time in ns.

import ctypes
import numpy as np
from picosdk.ps2000a import ps2000a as ps
import matplotlib.pyplot as plt
from picosdk.functions import adc2mV, assert_pico_ok
import time
import am_mod


waveform_list, pico_freq = am_mod.ask_modulate("01001011", 2)
waveform_list = [int(x * 32767) for x in waveform_list]


# Create chandle and status ready for use
chandle = ctypes.c_int16()
status = {}

# Open PicoScope 2000 Series device
# Returns handle to chandle for use in future API functions
status["openunit"] = ps.ps2000aOpenUnit(ctypes.byref(chandle), None)
assert_pico_ok(status["openunit"])

enabled = 1
disabled = 0
analogue_offset = 0.0

wavetype = ctypes.c_int16(0)
sweepType = ctypes.c_int32(0)
triggertype = ctypes.c_int32(0)
triggerSource = ctypes.c_int32(0)
waveform_length = ctypes.c_int32(len(waveform_list))
waveform_length2 = ctypes.c_uint32(len(waveform_list))
c_waveform_data = (ctypes.c_int16 * len(waveform_list))(*waveform_list)

phase = ctypes.c_uint32(0)

ps.ps2000aSigGenFrequencyToPhase(
    chandle,
    int(pico_freq),
    0,
    waveform_length2,
    ctypes.byref(phase)
)

status["SetSigGenArbitrary"] = ps.ps2000aSetSigGenArbitrary(chandle, 0,
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
                                                            0,  # ps.PS2000A_SIGGEN_TRIG_TYPE["PS2000A_SIGGEN_RISING"],
                                                            0,  # ps.PS2000A_SIGGEN_TRIG_SOURCE["PS2000A_SIGGEN_NONE"],
                                                            ctypes.c_int16(0)  # threshold
                                                            )

# status["SetSigGenBuiltIn"] = ps.ps2000aSetSigGenBuiltIn(chandle, 0, 2000000, wavetype, 6e5, 6e5, 0, 1, sweepType, 0, 0,
#                                                         0, triggertype, triggerSource, 1)
print(status["SetSigGenArbitrary"])
assert_pico_ok(status["SetSigGenArbitrary"])

# Set up channel A
# handle = chandle
# channel = PS2000A_CHANNEL_A = 0
# enabled = 1
# coupling type = PS2000A_DC = 1
# range = PS2000A_2V = 7
# analogue offset = 0 V
channel_range = 7 #ps.PS2000A_RANGE['PS2000A_10V']
status["setChA"] = ps.ps2000aSetChannel(chandle,
                                        ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                        enabled,
                                        ps.PS2000A_COUPLING['PS2000A_DC'],
                                        channel_range,
                                        analogue_offset)
assert_pico_ok(status["setChA"])

# Set up channel B
# handle = chandle
# channel = PS2000A_CHANNEL_B = 1
# enabled = 1
# coupling type = PS2000A_DC = 1
# range = PS2000A_2V = 7
# analogue offset = 0 V
status["setChB"] = ps.ps2000aSetChannel(chandle,
                                        ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                        enabled,
                                        ps.PS2000A_COUPLING['PS2000A_DC'],
                                        channel_range,
                                        analogue_offset)
assert_pico_ok(status["setChB"])

# Size of capture
sizeOfOneBuffer = 500
numBuffersToCapture = 50

totalSamples = sizeOfOneBuffer * numBuffersToCapture

# Create buffers ready for assigning pointers for data collection
bufferAMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)
bufferBMax = np.zeros(shape=sizeOfOneBuffer, dtype=np.int16)

memory_segment = 0

# Set data buffer location for data collection from channel A
# handle = chandle
# source = PS2000A_CHANNEL_A = 0
# pointer to buffer max = ctypes.byref(bufferAMax)
# pointer to buffer min = ctypes.byref(bufferAMin)
# buffer length = maxSamples
# segment index = 0
# ratio mode = PS2000A_RATIO_MODE_NONE = 0
status["setDataBuffersA"] = ps.ps2000aSetDataBuffers(chandle,
                                                     ps.PS2000A_CHANNEL['PS2000A_CHANNEL_A'],
                                                     bufferAMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersA"])

# Set data buffer location for data collection from channel B
# handle = chandle
# source = PS2000A_CHANNEL_B = 1
# pointer to buffer max = ctypes.byref(bufferBMax)
# pointer to buffer min = ctypes.byref(bufferBMin)
# buffer length = maxSamples
# segment index = 0
# ratio mode = PS2000A_RATIO_MODE_NONE = 0
status["setDataBuffersB"] = ps.ps2000aSetDataBuffers(chandle,
                                                     ps.PS2000A_CHANNEL['PS2000A_CHANNEL_B'],
                                                     bufferBMax.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                     None,
                                                     sizeOfOneBuffer,
                                                     memory_segment,
                                                     ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'])
assert_pico_ok(status["setDataBuffersB"])

# Begin streaming mode:
sampleInterval = ctypes.c_int32(112)
sampleUnits = ps.PS2000A_TIME_UNITS['PS2000A_NS']
# We are not triggering:
maxPreTriggerSamples = 0
autoStopOn = 1
# No downsampling:
downsampleRatio = 1
status["runStreaming"] = ps.ps2000aRunStreaming(chandle,
                                                ctypes.byref(sampleInterval),
                                                sampleUnits,
                                                maxPreTriggerSamples,
                                                totalSamples,
                                                autoStopOn,
                                                downsampleRatio,
                                                ps.PS2000A_RATIO_MODE['PS2000A_RATIO_MODE_NONE'],
                                                sizeOfOneBuffer)
assert_pico_ok(status["runStreaming"])

actualSampleInterval = sampleInterval.value
actualSampleIntervalNs = actualSampleInterval# * 1000

print("Capturing at sample interval %s ns" % actualSampleIntervalNs)

# We need a big buffer, not registered with the driver, to keep our complete capture in.
bufferCompleteA = np.zeros(shape=totalSamples, dtype=np.int16)
bufferCompleteB = np.zeros(shape=totalSamples, dtype=np.int16)
nextSample = 0
autoStopOuter = False
wasCalledBack = False


def streaming_callback(handle, noOfSamples, startIndex, overflow, triggerAt, triggered, autoStop, param):
    global nextSample, autoStopOuter, wasCalledBack
    wasCalledBack = True
    destEnd = nextSample + noOfSamples
    sourceEnd = startIndex + noOfSamples
    bufferCompleteA[nextSample:destEnd] = bufferAMax[startIndex:sourceEnd]
    bufferCompleteB[nextSample:destEnd] = bufferBMax[startIndex:sourceEnd]
    # print(bufferAMax[startIndex:sourceEnd])
    nextSample += noOfSamples
    if autoStop:
        autoStopOuter = True


# Convert the python function into a C function pointer.
cFuncPtr = ps.StreamingReadyType(streaming_callback)

# Fetch data from the driver in a loop, copying it out of the registered buffers and into our complete one.
while nextSample < totalSamples and not autoStopOuter:
    wasCalledBack = False
    status["getStreamingLastestValues"] = ps.ps2000aGetStreamingLatestValues(chandle, cFuncPtr, None)
    if not wasCalledBack:
        # If we weren't called back by the driver, this means no data is ready. Sleep for a short while before trying
        # again.
        time.sleep(0.01)

print("Done grabbing values.")

# Find maximum ADC count value
# handle = chandle
# pointer to value = ctypes.byref(maxADC)
maxADC = ctypes.c_int16()
status["maximumValue"] = ps.ps2000aMaximumValue(chandle, ctypes.byref(maxADC))
assert_pico_ok(status["maximumValue"])

# Convert ADC counts data to mV
adc2mVChAMax = adc2mV(bufferCompleteA, channel_range, maxADC)
adc2mVChBMax = adc2mV(bufferCompleteB, channel_range, maxADC)

# Create time data
time = np.linspace(0, (totalSamples - 1) * actualSampleIntervalNs, totalSamples)

# Plot data from channel A and B
# plt.plot(time, adc2mVChAMax[:])
plt.plot(time, bufferCompleteA)
# plt.plot(time, adc2mVChBMax[:])
plt.xlabel('Time (ns)')
plt.ylabel('Voltage (mV)')
plt.show()

# Stop the scope
# handle = chandle
status["stop"] = ps.ps2000aStop(chandle)
assert_pico_ok(status["stop"])

# Disconnect the scope
# handle = chandle
status["close"] = ps.ps2000aCloseUnit(chandle)
assert_pico_ok(status["close"])

# Display status returns
print(status)
