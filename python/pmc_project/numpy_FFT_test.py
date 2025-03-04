#練習使用numpy模組的FFT (Fast Fourier Transform)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#FFT計算子程式
def daqfft(y,fs,blocksize):
    Y = np.fft.fft(y,blocksize)/blocksize
    Y=(np.abs(Y)[0:int(np.fix(blocksize/2))+1])*2   #FFT振幅
    f=np.arange(0,len(Y),1)*fs/blocksize   #FFT頻率軸
    results=[f,Y]
    return results


#建立模擬訊號
fs=50   #採樣頻率
t = np.arange(0, 10.0+1/fs, 1/fs)
f1=3  #模擬訊號3hz
y = np.sin(2 * np.pi *f1* t)



#加窗後FFT
window=signal.get_window('hann',len(y))
f,Y=daqfft(y*window,fs,len(y))
Y=Y*2



#畫出時域圖
plt.figure(1)
L1= plt.plot(t, y)
plt.setp(L1, linestyle='-')
plt.grid('on')
plt.xlabel('Time (sec)') 
plt.ylabel('Peak')


#畫出頻域圖
plt.figure(3)
L2= plt.plot(f, Y)
plt.setp(L2, linestyle='-')
plt.grid('on')
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Peak')

plt.ion()
plt.show()
