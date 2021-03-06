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

def detect_mix(pow_spec):
    start_time = 0
    last_valid_frame = 0
    result = OrderedDict()
    number = 0
    for index_time, powers in enumerate(pow_spec):
        max_power_index = np.argmax(powers)
        max_power = powers[max_power_index]
        # Normally the power should be bigger than 5000 when there is voice
        if max_power > 5000:
            if abs(max_power_index - last_valid_frame) < 5:
                duration = str(start_time) + '-' + str(index_time)
            else:
                start_time = index_time
                number += 1
                duration = str(start_time) + '-' + str(index_time)
            if number == 0:
                number += 1
            result[number] = duration
            last_valid_frame = max_power_index


if __name__ == '__main__':
    import scipy.io.wavfile

    # Read wav file
    # "OSR_us_000_0010_8k.wav" is downloaded from http://www.voiptroubleshooter.com/open_speech/american.html
    sample_rate, signal = scipy.io.wavfile.read(r"D:\cats\bugs\1812\8000+music+8000.wav")
    # sample_rate, signal = scipy.io.wavfile.read(r"D:\cats\bugs\1812\400-800-1200-1600.wav")
    # Get speech data in the first 2 seconds
    # signal = signal[0:int(2. * sample_rate)]
    # signal = signal[int(0.4 * sample_rate):int(1. * sample_rate)]

    # Calculate the short time fourier transform
    pow_spec = calc_stft(signal, sample_rate, NFFT=sample_rate)

    start_time = 0
    last_end_time = 0
    last_valid_freq = 0
    last_valid_freq_power = 0
    result = OrderedDict()
    number = 0
    is_multi_freq = False
    first_time = True
    new_section = True
    valid_frequencies = []
    max_valid_freq = 0
    for index_time, powers in enumerate(pow_spec):
        index_time = index_time/100.0
        max_power_freq_index = np.argmax(powers)
        max_power = powers[max_power_freq_index]
        max_valid_freq = max_valid_freq if (max_valid_freq > max_power_freq_index) else max_power_freq_index
        # Normally the power should be bigger than 5000 when there is voice
        if max_power > 5000:
            if abs(max_power_freq_index - last_valid_freq) < 5:
                if powers[max_valid_freq] < max_power * 0.7:
                    # this means it's in multi_freq_section
                    pass
                else:
                    if new_section:
                        number += 1
                        start_time = index_time
                        new_section = False
                    if is_multi_freq:
                        new_section = True
                        is_multi_freq = False
            else:
                if first_time:
                    start_time = index_time
                    first_time = False
                else:
                    if not is_multi_freq:
                        is_multi_freq = True
                        new_section = True
                    if new_section:
                        number += 1
                        start_time = index_time
                        new_section = False
                        if powers[last_valid_freq] > last_valid_freq_power:
                            print('--------------error----------: ' + str(last_valid_freq))

            duration = str(start_time) + '-' + str(index_time)

            if number != 0:
                result[number] = duration
            last_valid_freq = max_power_freq_index
            last_valid_freq_power = max_power



    # f, t, Zxx = stft(signal, fs=sample_rate, nperseg=512, nfft=sample_rate)
    print("over")