#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
"""
Check the volume of audio file.
File requirement:
    wav format
    single channel
    duration less than 30 seconds
"""
import logging
import os
import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

# 倍频程上下限
OCTAVE_ARRAY = (11.2, 22.4, 45, 90, 180, 355, 710, 1400, 2800, 5600, 11200, 22400)
# 中心频率
CENTER_FREQ_ARRAY = (16, 31.5, 63, 125, 250, 500, 100, 2000, 4000, 8000, 16000)

error_code_dict = {1: 'The file should be ended with .wav',
                   2: 'The threshold value should be between 0 and 1',
                   3: 'The file should be single channel audio',
                   4: 'The duration of file should be less than 30 seconds',
                   5: 'The file provided is not found'
                   }

logging.basicConfig(level=logging.ERROR,#控制台打印的日志级别
                    # filename='check_volume.log',
                    # filemode='a',#模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                                   # a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


class CheckVolumeError(Exception):
    def __init__(self, error_code, error_information):
        super().__init__(self)
        self.error_code = error_code
        self.error_information = error_information

    def __str__(self):
        return " Error code " + str(self.error_code) + " : " + self.error_information


def check_volume(file_path, threshold_value=0, file_extension="_without_silence"):
    if not str(file_path).endswith(".wav"):
        logging.error(error_code_dict[1])
        raise CheckVolumeError(1, error_code_dict[1])

    if threshold_value < 0 or threshold_value > 1:
        logging.error(error_code_dict[2])
        raise CheckVolumeError(2, error_code_dict[2])

    params, wave_data = open_autio_file(file_path)

    channel_num, sample_width, frame_rate, frame_num = params[:4]

    indexs_min, indexs_max, wave_data_without_silence = filter_audio_file(wave_data, threshold_value)

    generate_audio_file(params[:3], wave_data_without_silence.tostring(), file_path, file_extension)

    times = np.arange(0, frame_num) * (1.0 / frame_rate)
    times_without_silence = times[indexs_min:indexs_max]
    out_put = file_path[:file_path.rindex(".")]
    plot_figure(times, wave_data, "Time(s)", "Amplitude", "original wave figure", 511)

    plot_figure(times_without_silence, wave_data_without_silence,
                "Time(s)", "Amplitude", "without silence wave figure", 513)

    frequencies, powers, SPL_result = calculate_freq_domain(wave_data_without_silence, frame_rate)

    plot_figure(frequencies / 1000, powers, "Frequency (kHz)", "Power (dB)",
                "frequency spectrum figure", 515, out_put + ".png")

    return SPL_result


def open_autio_file(file_path):
    if not str(file_path).endswith(".wav"):
        logging.error(error_code_dict[1])
        raise CheckVolumeError(1, error_code_dict[1])

    if not os.path.isfile(file_path):
        logging.error(error_code_dict[5])
        raise CheckVolumeError(4, error_code_dict[5])

    f = wave.open(file_path, 'rb')

    params = f.getparams()
    channel_num, sample_width, frame_rate, frame_num = params[:4]
    if channel_num != 1:
        logging.error(error_code_dict[3])
        raise CheckVolumeError(3, error_code_dict[3])
    # 限制读入的音频文件时长,应小于30秒
    # if int(frame_num / frame_rate) > 30:
    #     logging.error(error_code_dict[4])
    #     raise CheckVolumeError(4, error_code_dict[4])

    wave_data_str = f.readframes(frame_num)  # 读取音频，字符串格式

    wave_data = np.fromstring(wave_data_str, dtype=np.int16)  # 将字符串转化为int
    return params, wave_data


def filter_audio_file(wave_data, threshold_value=0):
    if threshold_value < 0 or threshold_value > 1:
        logging.error(error_code_dict[2])
        raise CheckVolumeError(2, error_code_dict[2])

    wave_data_normalization = wave_data * 1.0 / max(abs(wave_data))  # wave幅值归一化
    indexs = np.where(abs(wave_data_normalization) > threshold_value)
    indexs_min = min(indexs[0])
    indexs_max = max(indexs[0])
    wave_data_without_silence = wave_data[indexs_min:indexs_max]
    return indexs_min, indexs_max, wave_data_without_silence


def generate_audio_file(params, wave_data_str, file_path, file_extension="_without_silence"):
    out_put = file_path[:file_path.rindex(".")] + file_extension + file_path[file_path.rindex("."):]
    channel_num, sample_width, frame_rate = params
    wf = wave.open(out_put, 'wb')
    wf.setnchannels(channel_num)
    wf.setsampwidth(sample_width)
    wf.setframerate(frame_rate)
    wf.writeframes(wave_data_str)
    wf.close()


def plot_figure(x_data, y_data, x_label, y_label, title, subplot_location=None, output=None):
    if subplot_location is not None:
        plt.subplot(subplot_location)
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)  # 标尺，on：有，off:无。
    if output is not None:
        plt.savefig(output, dpi=300)


def calculate_freq_domain(wave_data, frame_rate):
    num = len(wave_data)
    p = fft(wave_data)

    half_num = int(np.ceil(num / 2))  # 因为对称，取一半进行处理
    p = p[0:half_num]
    p = abs(p)

    p = p / float(num)  # 除以采样点数，去除幅度对信号长度或采样频率的依赖
    p = p ** 2

    # 奇nfft排除奈奎斯特点
    if num % 2 > 0:  # fft点数为奇
        p[1:len(p)] = p[1:len(p)] * 2
    else:  # fft点数为偶
        p[1:len(p) - 1] = p[1:len(p) - 1] * 2

    frequencies = np.arange(0, half_num, 1.0) * (frame_rate / num)
    temp = [np.log10(x) for x in p]
    powers = [10 * x for x in temp]  # 得到对应每个频率的能量（分贝）值

    # print("Average Result : " + '%f' % (sum(powers) / half_num))

    SPL_original_array = np.zeros((len(OCTAVE_ARRAY) - 1, 1))
    SPL_array = np.zeros((len(OCTAVE_ARRAY) - 1, 1))

    start_index = 0
    for i in range(len(OCTAVE_ARRAY) - 1):
        count = 0
        energy_sum = 0
        for j in range(start_index, len(frequencies)):
            if frequencies[j] >= OCTAVE_ARRAY[i] and frequencies[j] < OCTAVE_ARRAY[i + 1]:
                count += 1
                energy_sum += p[j]
                start_index = j
        if count != 0:
            SPL_original_array[i] = 10 * np.log10(energy_sum / count)

    # 计算A率加权后的声压级（SPL）
    SPL_sum = 0
    for i in range(len(CENTER_FREQ_ARRAY)):
        SPL_array[i] = SPL_original_array[i] + calculate_weight_A(CENTER_FREQ_ARRAY[i])
        SPL_sum += pow(10, SPL_array[i] / 10)

    SPL_result = 10 * np.log10(SPL_sum)
    # print("Weighted Result : " + '%f' % SPL_result)

    return frequencies, powers, SPL_result[0]


def calculate_weight_A(freq):
    ra = 12200.0 ** 2 * freq ** 4 / (
                (freq ** 2 + 20.6 ** 2) * pow((freq ** 2 + 107.7 ** 2) * (freq ** 2 + 737.9 ** 2), 0.5) * (
                    freq ** 2 + 12200 ** 2))
    A = 2.0 + 20 * np.log10(ra)
    return A


def get_error_message_from_error_code(error_code):
    try:
        return error_code_dict[error_code]
    except Exception:
        return "The error code is invalid "


if __name__ == '__main__':
    file_path = r"D:\cats\bugs\1812\750andmusic.wav"
    threshold_value = 0.04  # 音频中静音部分的阈值
    print("volume (dB) : " + '%f' % check_volume(file_path, threshold_value))
