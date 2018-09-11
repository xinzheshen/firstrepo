import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft


#倍频程上下限
OCTAVE_ARRAY = (11.2, 22.4, 45, 90, 180, 355, 710, 1400, 2800, 5600, 11200, 22400)
#中心频率
CENTER_FREQ_ARRAY = (16, 31.5, 63, 125, 250, 500, 100, 2000, 4000, 8000, 16000)

def main():
    file_path = "/home/shenxinzhe/Downloads/NAutoAudioFromBramIphone_180904/break.wav"
    threshold_value = 0.04  # 音频中静音部分的阈值
    f = wave.open(file_path, 'rb')
    params = f.getparams()
    channel_num, sample_width, frame_rate, frame_num = params[:4]
    wave_data_str = f.readframes(frame_num)  # 读取音频，字符串格式
    wave_data = np.fromstring(wave_data_str, dtype=np.int16)  # 将字符串转化为int

    wave_data_normalization = wave_data*1.0/max(abs(wave_data))  # wave幅值归一化

    pidxs = np.where(abs(wave_data_normalization) > threshold_value)
    pidxs_min = min(pidxs[0])
    pidxs_max = max(pidxs[0])
    wave_data_without_silence = wave_data[pidxs_min:pidxs_max]

    # 生成新文件
    extension = "_without_silence.wav"
    out_put = file_path[:file_path.rindex(".")] + extension
    generate_audio_file(wave_data_without_silence.tostring(), channel_num, sample_width, frame_rate, out_put)

    # plot the wave
    time = np.arange(0, frame_num)*(1.0 / frame_rate)
    time_without_silence = time[pidxs_min:pidxs_max]
    plt.subplot(3, 1, 1)
    plt.plot(time, wave_data_normalization)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("original wave plot")
    plt.grid('on')  # 标尺，on：有，off:无。
    # plt.ylim(-0.1, 0.1)
    plt.subplot(3, 1, 2)
    plt.plot(time_without_silence, wave_data_without_silence)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("without silence wave plot")
    plt.grid('on')  # 标尺，on：有，off:无。

def caculate_freq_domain(wave_data, frame_rate):
    n = len(wave_data)
    p = fft(wave_data)

    nUniquePts = int(np.ceil((n+1)/2))
    p = p[0:nUniquePts]
    p = abs(p)

    p = p / float(n)    #除以采样点数，去除幅度对信号长度或采样频率的依赖
    p = p**2            #求平方得到能量

    #乘2（详见技术手册）
    #奇nfft排除奈奎斯特点
    if n % 2 > 0:       #fft点数为奇
        p[1:len(p)] = p[1:len(p)]*2
    else:               #fft点数为偶
        p[1:len(p)-1] = p[1:len(p)-1] * 2

    freqArray = np.arange(0, nUniquePts, 1.0) * (frame_rate / n)
    yData = [np.log10(x) for x in p]
    yData = [10 * x for x in yData]

    print(sum(yData)/nUniquePts)




    SPLOriginalArray = np.zeros((len(OCTAVE_ARRAY) - 1,1))
    SPLArray = np.zeros((len(OCTAVE_ARRAY) - 1,1))
    for i in range(len(OCTAVE_ARRAY) - 1):
        count = 0
        energySum = 0
        for j in range(len(freqArray)):
            if freqArray[j] >= OCTAVE_ARRAY[i] and freqArray[j] < OCTAVE_ARRAY[i+1]:
                count += 1
                energySum += p[j]
        if count != 0:
            SPLOriginalArray[i] = 10 * np.log10(energySum / count)

    #计算A率加权后的声压级（SPL）
    SPLSum = 0
    for i in range(len(CENTER_FREQ_ARRAY)):
        SPLArray[i] = SPLOriginalArray[i] + calWeightA(CENTER_FREQ_ARRAY[i])
        SPLSum += pow(10, SPLArray[i] / 10)

    SPLResult = 10 * np.log10(SPLSum)
    print("SPLRusult")
    print(SPLResult)

    plt.subplot(4, 1, 3)
    plt.plot(freqArray/1000, yData , color='k')
    plt.xlabel('Freqency (kHz)')
    plt.ylabel('Power (dB)')
    plt.grid('on')
    #plt.ylim(-50, 50)
    plt.show()


#存成新文件
def generate_audio_file(data_str, channel_num, sample_width, frame_rate, out_put):
    wf = wave.open(out_put, 'wb')
    wf.setnchannels(channel_num)
    wf.setsampwidth(sample_width)
    wf.setframerate(frame_rate)
    wf.writeframes(data_str)
    wf.close()


def calWeightA(freq):
    ra = 12200.0**2 * freq**4 / ((freq**2 + 20.6**2) * pow((freq**2 + 107.7**2) * (freq**2 + 737.9**2), 0.5) * (freq**2 + 12200**2))
    A = 2.0 + 20*np.log10(ra)
    return A
