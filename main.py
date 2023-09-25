import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import wave
import struct
import pylab
import pdb
from scipy.io import wavfile


def Filter(B,B1,frq_DS,Yf_DS):
    fBWIndex = (np.abs(frq_DS - B)).argmin()
    B = frq_DS[fBWIndex]
    fBWIndex2 = (np.abs(frq_DS - B1)).argmin()
    B1 = frq_DS[fBWIndex2]
    for cnt in range(len(frq_DS)):
        if ~((-1 * B1 > (frq_DS[cnt]) > -1 * B) or (B1 < (frq_DS[cnt]) < B)):
            Yf_DS[cnt] = Y_DS[cnt] * 0
    return Yf_DS


################################################
filenameWave = 'FDMAMixedAudio12.wav'
BWrange = 100000
################################################

rate, data = wavfile.read(filenameWave)
if len(data.shape) > 1:
    data = data[:, 0]

##############################################
## Generate Signal and add save it to text file
Fs = rate*10;
Ts = 1.0 / Fs;  # sampling interval
t = np.arange(0, len(data) * Ts, Ts)  # time vector
y = [float(x) for x in data]

n = len(y)  # length of the signal
k = np.arange(n)
T = n / Fs
frq = k / T  # two sides frequency range
fcen = frq[int(len(frq) / 2)]
frq_DS = frq - fcen
frq_SS = frq[range(int(n / 2))]  # one side frequency range

Y = np.fft.fft(y)  # fft computing and normalization
yinv = np.fft.ifft(Y).real  # ifft computing and normalization
Y_DS = np.roll(Y, int(n / 2))
Y_SS = Y[range(int(n / 2))]

yinv = np.array(yinv)

fcenIndex = (np.abs(frq_DS)).argmin()
RangeIndex = (np.abs(frq_DS - BWrange)).argmin() - fcenIndex

RangeIndexMin = fcenIndex - RangeIndex
if RangeIndexMin < 0:
    RangeIndexMin = 0

RangeIndexMax = fcenIndex + RangeIndex
if RangeIndexMax > len(frq_DS) - 1:
    RangeIndexMax = len(frq_DS) - 1

fig, ax = plt.subplots(2, 1, figsize=(16, 6))
ax[0].plot(t, yinv)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
ax[1].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Y_DS[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plt.show()

###############################################################################

y = np.array(y)
y_int = y.astype(np.int16)

yinv = np.array(yinv)
yinv_int = yinv.astype(np.int16)


####(signal1)
Yf_DS1 = np.copy(Y_DS)
Yf_DS1 = Filter(27600,27000, frq_DS, Yf_DS1)

Yf1 = np.roll(Yf_DS1, int(n / 2))
yinv1 = np.fft.ifft(Yf1).real  # ifft computing and normalization
yinv1 = np.array(yinv1)
yinv1_int = yinv1.astype(np.int16)

demod = 20 * np.cos(2 * np.pi * 27280 * t)

yout1 = demod * yinv1
Yf1 = np.fft.fft(yout1)
Yf1 = np.roll(Yf1, int(n / 2))

Yf1 = Filter(2500,0,frq_DS,Yf1)

yout1 = np.fft.ifft(Yf1).real
yout1 = yout1.astype(np.int16)

wavfile.write('./Filtered1.wav', Fs, yout1)

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf1[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(t, yout1, 'g')  # plotting the spectrum
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')

plt.show()

####(signal2)
Yf_DS2 = np.copy(Y_DS)
Yf_DS2 = Filter(36500,36200, frq_DS, Yf_DS2)

Yf2 = np.roll(Yf_DS2, int(n / 2))
yinv2 = np.fft.ifft(Yf2).real  # ifft computing and normalization
yinv2 = np.array(yinv2)
yinv2_int = yinv2.astype(np.int16)

demod = 20 * np.cos(2 * np.pi * 36420 * t)

yout2 = demod * yinv2
Yf2 = np.fft.fft(yout2)
Yf2 = np.roll(Yf2, int(n / 2))

Yf2 = Filter(3000,0,frq_DS,Yf2)

yout2 = np.fft.ifft(Yf2).real
yout2 = yout2.astype(np.int16)

wavfile.write('./Filtered2.wav', Fs, yout2)

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf2[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(t, yout2, 'g')  # plotting the spectrum
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')

plt.show()

#####(siganl3)
Yf_DS3 = np.copy(Y_DS)
Yf_DS3 = Filter(46500, 44500,  frq_DS, Yf_DS3)

Yf3 = np.roll(Yf_DS3, int(n / 2))
yinv3 = np.fft.ifft(Yf3).real  # ifft computing and normalization
yinv3 = np.array(yinv3)
yinv3_int = yinv3.astype(np.int16)

demod = 5 * np.cos(2 * np.pi * 45200 * t)

yout3 = demod * yinv3
Yf3 = np.fft.fft(yout3)
Yf3 = np.roll(Yf3, int(n / 2))

Yf3 = Filter(3000,0,frq_DS,Yf3)

yout3 = np.fft.ifft(Yf3).real
yout3 = yout3.astype(np.int16)

wavfile.write('./Filtered3.wav', Fs, yout3)

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf3[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(t, yout3, 'g')  # plotting the spectrum
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')

plt.show()

#####(siganl4)
Yf_DS4 = np.copy(Y_DS)
Yf_DS4 = Filter( 56000,53000, frq_DS, Yf_DS4)

Yf4 = np.roll(Yf_DS4, int(n / 2))
yinv4 = np.fft.ifft(Yf4).real  # ifft computing and normalization
yinv4 = np.array(yinv4)
yinv4_int = yinv4.astype(np.int16)

demod = 10 * np.cos(2 * np.pi * 55000 * t)

yout4 = demod * yinv4
Yf4 = np.fft.fft(yout4)
Yf4 = np.roll(Yf4, int(n / 2))


yout4 = np.fft.ifft(Yf4).real
yout4 = yout4.astype(np.int16)

wavfile.write('./Filtered4.wav', Fs, yout4)

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf4[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(t, yout4, 'g')  # plotting the spectrum
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')

plt.show()

#####(siganl5)
Yf_DS5 = np.copy(Y_DS)
Yf_DS5 = Filter( 68000,60000, frq_DS, Yf_DS5)

Yf5 = np.roll(Yf_DS5, int(n / 2))
yinv5 = np.fft.ifft(Yf5).real  # ifft computing and normalization
yinv5 = np.array(yinv5)
yinv5_int = yinv5.astype(np.int16)

demod = 10 * np.cos(2 * np.pi * 63700 * t)

yout5 = demod * yinv5
Yf5 = np.fft.fft(yout5)
Yf5 = np.roll(Yf5, int(n / 2))


yout5 = np.fft.ifft(Yf5).real
yout5 = yout5.astype(np.int16)

wavfile.write('./Filtered5.wav', Fs, yout4)

fig, ax = plt.subplots(2, 1, figsize=(16, 9))
ax[0].plot(frq_DS[RangeIndexMin:RangeIndexMax], abs(Yf5[RangeIndexMin:RangeIndexMax]), 'r')  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(t, yout5, 'g')  # plotting the spectrum
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Amplitude')

plt.show()