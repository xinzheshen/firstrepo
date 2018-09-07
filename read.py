import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np
import math
import os


'''
filepath = "./data/" #添加路径
filename= os.listdir(filepath) #得到文件夹下的所有文件名称
'''
#f = wave.open('/home/shenxinzhe/Downloads/chime-audios/chime-audios/ChimeRecord-4 3.wav', 'rb')
f = wave.open('/home/shenxinzhe/codeforpython/test/output_25b.wav', 'rb')
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
strData = f.readframes(nframes)#读取音频，字符串格式
waveData = np.fromstring(strData, dtype=np.int16)#将字符串转化为int

#计算幅值平方和的常数对数的十倍

#waveData = waveData*1.0/(max(abs(waveData)))#wave幅值归一化
#用于处理多声道
#waveData = np.reshape(waveData, [nframes, nchannels])
# plot the wave
time = np.arange(0, nframes)*(1.0 / framerate)

#plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, waveData)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
#plt.title("Single channel wavedata")
plt.grid('on')#标尺，on：有，off:无。
plt.ylim(-40000, 40000)
#plt.savefig("./output.jpg")
# plt.show()
'''
plt.figure()
plt.subplot(5,1,1)
plt.plot(time,waveData[:,0])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-1 wavedata")
plt.grid('on')#标尺，on：有，off:无。
plt.subplot(5,1,3)
plt.plot(time,waveData[:,1])
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Ch-2 wavedata")
plt.grid('on')#标尺，on：有，off:无。
'''

#绘制频谱图
n = len(waveData)
p = fft(waveData)

nUniquePts = int(math.ceil((n+1)/2))
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

freqArray = np.arange(0, nUniquePts, 1.0) * (framerate / n)
yData = [math.log10(x) for x in p]
yData = [10 * x for x in yData]

print(sum(yData)/nUniquePts)

def calWeightA(freq):
    ra = 12200.0**2 * freq**4 / ((freq**2 + 20.6**2) * pow((freq**2 + 107.7**2) * (freq**2 + 737.9**2), 0.5) * (freq**2 + 12200**2))
    A = 2.0 + 20*np.log10(ra)
    return A
#倍频程上下限
octaveArray = (11.2, 22.4, 45, 90, 180, 355, 710, 1400, 2800, 5600, 11200, 22400)
#中心频率
centerFreqArray = (16, 31.5, 63, 125, 250, 500, 100, 2000, 4000, 8000, 16000)
SPLOriginalArray = np.zeros((len(octaveArray) - 1,1))
SPLArray = np.zeros((len(octaveArray) - 1,1))
for i in range(len(octaveArray) - 1):
    count = 0
    energySum = 0
    for j in range(len(freqArray)):
        if freqArray[j] >= octaveArray[i] and freqArray[j] < octaveArray[i+1]:
            count += 1
            energySum += p[j]
    if count != 0:
        SPLOriginalArray[i] = 10 * np.log10(energySum / count)

#计算A率加权后的声压级（SPL）
SPLSum = 0
for i in range(len(centerFreqArray)):
    SPLArray[i] = SPLOriginalArray[i] + calWeightA(centerFreqArray[i])
    SPLSum += pow(10, SPLArray[i] / 10)

SPLResult = 10 * np.log10(SPLSum)
print("SPLRusult")
print(SPLResult)

plt.subplot(2, 1, 2)
plt.plot(freqArray/1000, yData , color='k')
plt.xlabel('Freqency (kHz)')
plt.ylabel('Power (dB)')
plt.grid('on')
plt.ylim(-30, 80)
plt.show()





