#練習使用scipy模組中的Hilbert轉換找出訊號Envelope

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


#FFT計算子程式
def daqfft(y,fs,blocksize):
    Y = np.fft.fft(y,blocksize)/blocksize
    Y=(np.abs(Y)[0:int(np.fix(blocksize/2))+1])*2   #FFT振幅
    f=np.arange(0,len(Y),1)*fs/blocksize   #FFT頻率軸
    results=[f,Y]
    return results


#建立模擬訊號
f1=2     #模擬訊號y1頻率
f2=15    #模擬訊號y2頻率
fs=100   #採樣頻率
t=np.arange(0,4,1/fs)
y1=np.cos(2*np.pi*f1*t)*0.5
y2=np.cos(2*np.pi*f2*t)*0.5
y=y1*y2


#進行Hilbert transform
analytic_signal = signal.hilbert(y)
envelope = np.abs(analytic_signal)


#加窗後FFT
window=signal.get_window('hann',len(envelope))
f,Y=daqfft(envelope*window,fs,len(envelope))
Y=Y*2




#將Envelope執行detrend後再FFT
envelope_d=signal.detrend(envelope,type='constant')
f,Y_d=daqfft(envelope_d*window,fs,len(envelope))
Y_d=Y_d*2





# Two subplots, the axes array is 1-d
fig=plt.figure(1)
ax0 = fig.add_subplot(211)
ax0.plot(t, y1, label='y1')
ax0.set_xlabel("Time (sec)")
ax0.legend()
ax1 = fig.add_subplot(212)
ax1.plot(t, y2, label='y2')
ax1.set_xlabel("Time (sec)")






#畫出時域圖
plt.figure(2)
L2= plt.plot(t,y,t,envelope)
plt.setp(L2, linestyle='-')
plt.grid('on')
plt.xlabel('Time (sec)') 
plt.ylabel('Peak')



#畫出頻域圖
plt.figure(3)
L3= plt.plot(f, Y)
plt.setp(L3, linestyle='-')
plt.grid('on')
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Peak')


#畫出頻域圖
plt.figure(4)
L4= plt.plot(f, Y_d)
plt.setp(L4, linestyle='-')
plt.grid('on')
plt.xlabel('Frequency (Hz)') 
plt.ylabel('Peak')


#detrend後的Envelope與模擬載波時域比較圖
plt.figure(5)
L5=plt.plot(t,y1,t,envelope_d)
plt.setp(L5, linestyle='-')
plt.grid('on')
plt.xlabel('Time (sec)') 
plt.ylabel('Peak')



plt.ion()
plt.show()
