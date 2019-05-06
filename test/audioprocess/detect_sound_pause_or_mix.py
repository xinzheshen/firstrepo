#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.io.wavfile
from collections import OrderedDict
import sys
import time
import argparse


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
    result = OrderedDict()

    start_time = 0
    last_valid_freq = 0
    last_valid_freq_power = 0
    number = 0
    first_time = True
    for index_time, powers in enumerate(pow_spec):
        index_time = index_time/100.0
        max_power_freq_index = np.argmax(powers)
        max_power = powers[max_power_freq_index]
        # Normally the power should be bigger than 5000 when there is voice
        if max_power > 5000:
            valid_frequencies = np.where(powers > max_power * 0.9)
            max_freq = max(valid_frequencies[0])
            min_freq = min(valid_frequencies[0])

            # The audio is single freq, but the got data may not be single.
            if abs(max_power_freq_index - last_valid_freq) < 50:
                duration = str(start_time) + '-' + str(index_time)
            else:
                start_time = index_time
                number += 1
                duration = str(start_time) + '-' + str(index_time)
                if first_time:
                    first_time = False
                elif powers[last_valid_freq] >= last_valid_freq_power:
                    sys.stderr.write('The time of mixing sound: ' + str(index_time))
                    exit(100)
            if number == 0:
                number += 1
            result[number] = duration
            last_valid_freq = max_power_freq_index
            last_valid_freq_power = max_power
    return result


def detect_pause(pow_spec):
    result = OrderedDict()

    start_time = 0
    last_valid_freq = 0
    last_valid_freq_power = 0
    number = 0
    is_multi_freq = False
    first_time = True
    new_section = True
    max_valid_freq = 0
    for index_time, powers in enumerate(pow_spec):
        index_time = index_time/100.0
        max_power_freq_index = np.argmax(powers)
        max_power = powers[max_power_freq_index]
        max_valid_freq = max_valid_freq if (max_valid_freq > max_power_freq_index) else max_power_freq_index
        # Normally the power should be bigger than 5000 when there is voice
        if max_power > 5000:
            valid_frequencies = np.where(powers > max_power * 0.05)
            max_freq = max(valid_frequencies[0])
            min_freq = min(valid_frequencies[0])

            if abs(max_power_freq_index - last_valid_freq) < 50:
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
                        tmp = powers[last_valid_freq]
                        number += 1
                        start_time = index_time
                        new_section = False
                        if powers[last_valid_freq] >= last_valid_freq_power or (max_freq - min_freq) > 5000:
                            sys.stderr.write('The time of mixing sound: ' + str(index_time))
                            exit(100)

            duration = str(start_time) + '-' + str(index_time)

            if number != 0:
                result[number] = duration
            last_valid_freq = max_power_freq_index
            last_valid_freq_power = max_power
    return result


def process_argv():
    parser = argparse.ArgumentParser(prog='silence_detect')
    parser.add_argument('--file', '-f', help='The path of the audio file.', required=True)
    parser.add_argument('--allSingleFreq', '-a', help='If the audio is all single frequency.', choices=['true', 'false'], default='true')
    return parser.parse_args()


if __name__ == '__main__':
    arg = process_argv()
    # sample_rate, signal = scipy.io.wavfile.read(r"D:\cats\bugs\1812\400-800-1200-1600.wav")
    sample_rate, signal = scipy.io.wavfile.read(arg.file)
    pow_spec = calc_stft(signal, sample_rate, NFFT=sample_rate)
    if arg.allSingleFreq == 'true':
        result = detect_mix(pow_spec)
    else:
        result = detect_pause(pow_spec)
    print ('The index and start-end time:')
    for key, value in result.items():
        print (str(key) + ' : ' + value)
