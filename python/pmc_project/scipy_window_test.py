#測試呼叫訊號處理中的窗函數


import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

window_size=128  #窗長度
window=signal.get_window('hann',window_size)  #獲取hann窗; np.array([])形式

print(type(window))


t = np.arange(0, window_size, 1)

#畫出時域圖
plt.figure(1)
L1= plt.plot(t,window)
plt.setp(L1, linestyle='-')
plt.grid('on')
plt.xlabel('Time (sec)') 
plt.ylabel('Peak')
plt.show()
