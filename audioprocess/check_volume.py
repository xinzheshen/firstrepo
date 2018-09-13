#!/usr/bin/python3.5
# -*- coding: utf-8 -*-
"""
Check the volume of audio file.
File requirement:
    wav format
    single channel
    duration less than 30 seconds
"""
from enum import Enum
import logging
import time
import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

# 倍频程上下限
OCTAVE_ARRAY = (11.2, 22.4, 45, 90, 180, 355, 710, 1400, 2800, 5600, 11200, 22400)
# 中心频率
CENTER_FREQ_ARRAY = (16, 31.5, 63, 125, 250, 500, 100, 2000, 4000, 8000, 16000)

error_code_enum = Enum('error_code', (
    'The file is not ended with .wav',
    'The threshold value should be between 0 and 1',
    'The file should be single channel',
    'The file is not found',
    'The duration of file should be less than 30 seconds'
))

logging.basicConfig(level=logging.ERROR,#控制台打印的日志级别
                    # filename='check_volume.log',
                    # filemode='a',#模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                                   # a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )


def check_volume(file_path, threshold_value=0, file_extension="_without_silence.wav"):
    if not str(file_path).endswith(".wav"):
        logging.error(error_code_enum(1))
        exit(1)
    if threshold_value < 0 or threshold_value > 1:
        logging.error(error_code_enum(2))
        exit(2)

    try:
        params, wave_data = open_file(file_path)

        channel_num, sample_width, frame_rate, frame_num = params[:4]

        indexs_min, indexs_max, wave_data_without_silence = \
            generate_audiofile_without_silence(params, wave_data, file_path, threshold_value, file_extension)

        time = np.arange(0, frame_num) * (1.0 / frame_rate)
        time_without_silence = time[indexs_min:indexs_max]
        out_put = file_path[:file_path.rindex(".")]
        plot_figure(time, wave_data, "Time(s)", "Amplitude", "original wave figure", 511)

        plot_figure(time_without_silence, wave_data_without_silence,
                    "Time(s)", "Amplitude", "without silence wave figure", 513)

        frequences, powers, SPL_result = calculate_freq_domain(wave_data_without_silence, frame_rate)

        plot_figure(frequences / 1000, powers, "Frequency (kHz)", "Power (dB)", "frequency spectrum figure", 515, out_put + ".png", True)

        return SPL_result

    except Exception as e:
        logging.error(e)


def open_file(file_path):
    try:
        time_start_openfile=time.time()
        f = wave.open(file_path, 'rb')
        time_middle_openfile=time.time()

        params = f.getparams()
        channel_num, sample_width, frame_rate, frame_num = params[:4]
        if channel_num != 1:
            logging.error(error_code_enum(3))
            exit(3)
        # 限制读入的音频文件时长,应小于30秒
        if int(frame_num / frame_rate) > 30:
            logging.error(error_code_enum(5))
            exit(5)

        wave_data_str = f.readframes(frame_num)  # 读取音频，字符串格式

        time_end_openfile=time.time()
        print("open file cost", time_middle_openfile - time_start_openfile)
        print("read file cost", time_end_openfile - time_middle_openfile)
        wave_data = np.fromstring(wave_data_str, dtype=np.int16)  # 将字符串转化为int
        return params, wave_data
    except FileNotFoundError as e:
        logging.error(e)
        exit(4)
    except Exception as e:
        logging.error(e)


def generate_audiofile_without_silence(params, wave_data, file_path, threshold_value=0, file_extension="_without_silence.wav"):
    # filter
    wave_data_normalization = wave_data * 1.0 / max(abs(wave_data))  # wave幅值归一化
    indexs = np.where(abs(wave_data_normalization) > threshold_value)
    indexs_min = min(indexs[0])
    indexs_max = max(indexs[0])
    wave_data_without_silence = wave_data[indexs_min:indexs_max]
    channel_num, sample_width, frame_rate = params[:3]

    # generate new file
    out_put = file_path[:file_path.rindex(".")] + file_extension
    wf = wave.open(out_put, 'wb')
    wf.setnchannels(channel_num)
    wf.setsampwidth(sample_width)
    wf.setframerate(frame_rate)
    wf.writeframes(wave_data_without_silence.tostring())
    wf.close()

    return indexs_min, indexs_max, wave_data_without_silence


def plot_figure(x_data, y_data, x_label, y_label, title, subplot_location=None, output='default.png', save=False):
    try:
        if subplot_location is not None:
            plt.subplot(subplot_location)
        #plt.figure()
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid('on')  # 标尺，on：有，off:无。
        if save:
            time_start_save = time.time()
            plt.savefig(output, dpi=800)
            time_end_save = time.time()
            print('save plot cost', time_end_save - time_start_save)
    except Exception as e:
        logging.error(e)


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

    frequences = np.arange(0, half_num, 1.0) * (frame_rate / num)
    temp = [np.log10(x) for x in p]
    powers = [10 * x for x in temp]  # 得到对应每个频率的能量（分贝）值

    # print("Average Result : " + '%f' % (sum(powers) / half_num))

    SPL_original_array = np.zeros((len(OCTAVE_ARRAY) - 1, 1))
    SPL_array = np.zeros((len(OCTAVE_ARRAY) - 1, 1))

    time_start_cal=time.time()
    start_index = 0
    for i in range(len(OCTAVE_ARRAY) - 1):
        count = 0
        energy_sum = 0
        for j in range(start_index, len(frequences)):
            if frequences[j] >= OCTAVE_ARRAY[i] and frequences[j] < OCTAVE_ARRAY[i + 1]:
                count += 1
                energy_sum += p[j]
                start_index = j
        if count != 0:
            SPL_original_array[i] = 10 * np.log10(energy_sum / count)
    time_end_cal=time.time()
    print('cal cost', time_end_cal - time_start_cal)

    # 计算A率加权后的声压级（SPL）
    SPL_sum = 0
    for i in range(len(CENTER_FREQ_ARRAY)):
        SPL_array[i] = SPL_original_array[i] + calculate_weight_A(CENTER_FREQ_ARRAY[i])
        SPL_sum += pow(10, SPL_array[i] / 10)

    SPL_result = 10 * np.log10(SPL_sum)
    # print("Weighted Result : " + '%f' % SPL_result)

    return frequences, powers, SPL_result


def calculate_weight_A(freq):
    ra = 12200.0 ** 2 * freq ** 4 / (
                (freq ** 2 + 20.6 ** 2) * pow((freq ** 2 + 107.7 ** 2) * (freq ** 2 + 737.9 ** 2), 0.5) * (
                    freq ** 2 + 12200 ** 2))
    A = 2.0 + 20 * np.log10(ra)
    return A


def get_error_message_from_exit_code(exit_code):
    try:
        return error_code_enum(exit_code)
    except Exception as e:
        return "The exit code is invalid "

