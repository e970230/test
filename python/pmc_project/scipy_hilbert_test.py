#測試scipy模組的hilbert轉換

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


f1=2
f2=15
fs=100
t=np.arange(0,4,1/fs)
y=(np.cos(2*np.pi*f1*t)*0.3)*(np.cos(2*np.pi*f2*t))


#進行Hilbert transform
analytic_signal = signal.hilbert(y)
envelope = np.abs(analytic_signal)


#畫出時域圖
plt.figure(1)
L1= plt.plot(t,y,t,envelope)
plt.setp(L1, linestyle='-')
plt.grid('on')
plt.xlabel('Time (sec)') 
plt.ylabel('Peak')
plt.show()

