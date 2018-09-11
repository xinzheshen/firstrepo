import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack as fft

timeStep = 0.02
timeVec = np.arange(0, 20, timeStep)
sig = np.sin(np.pi / 5 * timeVec) + 0.1 * np.random.randn(timeVec.size)
plt.subplot(2, 1, 1)
plt.plot(timeVec, sig)


n = sig.size
sig_fft = fft.fft(sig)
sampleFreq = fft.fftfreq(n, timeStep)
pidxs = np.where(sampleFreq > 0)
freqs = sampleFreq[pidxs]
power = np.abs(sig_fft)[pidxs]
freq = freqs[power.argmax()]
sig_fft[np.abs(sampleFreq) > freq] = 0
main_sig = fft.ifft(sig_fft)
plt.subplot(2, 1, 2)
plt.plot(timeVec, main_sig, linewidth = 3)
plt.show()
