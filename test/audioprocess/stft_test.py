# -*- coding: utf8 -*-
import numpy as np
from scipy.signal import stft
import copy
from collections import OrderedDict

def calc_stft(signal, sample_rate=16000, frame_size=0.025, frame_stride=0.01, winfunc=np.hamming, NFFT=512):

    # Calculate the number of frames from the signal
    frame_length = frame_size * sample_rate
    frame_step = frame_stride * sample_rate
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = 1 + int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    # zero padding
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(signal, z)

    # Slice the signal into frames from indices
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    # Get windowed frames
    frames *= winfunc(frame_length)
    # Compute the one-dimensional n-point discrete Fourier Transform(DFT) of
    # a real-valued array by means of an efficient algorithm called Fast Fourier Transform (FFT)
    tmp = np.fft.rfft(frames, NFFT)
    mag_frames = np.absolute(tmp)
    # Compute power spectrum
    pow_frames = (1.0 / NFFT) * ((mag_frames) ** 2)

    return pow_frames

if __name__ == '__main__':
    import scipy.io.wavfile
    import matplotlib.pyplot as plt

    # Read wav file
    # "OSR_us_000_0010_8k.wav" is downloaded from http://www.voiptroubleshooter.com/open_speech/american.html
    sample_rate, signal = scipy.io.wavfile.read(r"D:\cats\bugs\1812\750andmusic.wav")
    # Get speech data in the first 2 seconds
    # signal = signal[0:int(2. * sample_rate)]
    signal = signal[int(0.4 * sample_rate):int(1. * sample_rate)]

    # Calculate the short time fourier transform
    pow_spec = calc_stft(signal, sample_rate, NFFT=sample_rate)

    # start_time = 0
    # last_valid_frame = 0
    # result = OrderedDict()
    # number = 1
    # for index_time, powers in enumerate(pow_spec):
    #     index_time = index_time/100
    #     max_power_index = np.argmax(powers)
    #     max_power = powers[max_power_index]
    #     # Normally the power should be bigger than 5000 when there is voice
    #     if max_power > 5000:
    #         if not abs(max_power_index - last_valid_frame) <= 5 or not (index_time - start_time) > 1:
    #             duration = str(start_time) + '-' + str(index_time)
    #         else:
    #             start_time = index_time
    #             duration = str(start_time) + '-' + str(index_time)
    #         result[number] = duration
    #         last_valid_frame = max_power_index

    start_time = 0
    last_valid_frame = 0
    result = {}
    for index_time, powers in enumerate(pow_spec):
        max_power_index = np.argmax(powers)
        max_power = powers[max_power_index]
        # Normally the power should be bigger than 5000 when there is voice
        if max_power > 5000:
            if max_power_index == last_valid_frame:
                duration = str(start_time) + '-' + str(index_time)
            else:
                start_time = index_time
                duration = str(start_time) + '-' + str(index_time)
            result[max_power_index] = duration
            last_valid_frame = max_power_index
    # f, t, Zxx = stft(signal, fs=sample_rate, nperseg=512, nfft=sample_rate)
    plt.imshow(pow_spec)
    plt.tight_layout()
    plt.show()